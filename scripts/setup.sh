#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"
export PATH="$HOME/.local/bin:$PATH"

clone_github() {
  local repo="$1"
  local dest="$2"
  local official="https://github.com/${repo}.git"
  local mirrored="${GITHUB_MIRROR_PREFIX}https://github.com/${repo}.git"
  rm -rf "$dest"
  echo "克隆 $repo （仓内 vendor 缺失时的回退；优先 GitHub 镜像）"
  if ! git -c http.version=HTTP/1.1 clone --depth 1 "$mirrored" "$dest"; then
    echo "镜像失败，回退官方地址"
    rm -rf "$dest"
    git -c http.version=HTTP/1.1 clone --depth 1 "$official" "$dest"
  fi
}

ensure_vendor() {
  # usage: ensure_vendor <path> <github-owner/repo> <smoke-file-relative>
  local dest="$1"
  local repo="$2"
  local smoke="$3"
  if [[ -f "$dest/$smoke" && "${FORCE_VENDOR_CLONE:-}" != "1" ]]; then
    echo "使用仓内 vendor：$dest"
    return
  fi
  if [[ "${FORCE_VENDOR_CLONE:-}" == "1" ]]; then
    echo "FORCE_VENDOR_CLONE=1，重新拉取 $repo"
  else
    echo "仓内缺少 $dest/$smoke，回退从 GitHub 拉取 $repo"
  fi
  clone_github "$repo" "$dest"
}

link_venv() {
  # usage: link_venv <project-link> <cache-target>
  local link="$1"
  local target="$2"
  mkdir -p "$(dirname "$target")"
  if [[ -L "$link" ]]; then
    rm -f "$link"
  elif [[ -d "$link" ]]; then
    if [[ -x "$target/bin/python" ]]; then
      local stale="${link}.stale-$(date +%s)"
      echo "Caches 已有可用环境，移走旧目录 → $(basename "$stale")"
      mv "$link" "$stale"
    else
      echo "迁移已有 venv：$link → $target"
      mv "$link" "$target"
    fi
  fi
  ln -sfn "$target" "$link"
  if [[ -e "$target" ]]; then
    chflags -R nohidden "$target" 2>/dev/null || true
  fi
  echo "venv 链接：$link → $target"
}

if ! command -v uv >/dev/null 2>&1; then
  echo "安装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12

# 主环境：装到 Library/Caches，项目内只留符号链接
export UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/main"
link_venv "$ROOT/.venv" "$UV_PROJECT_ENVIRONMENT"
uv sync --index-url "$UV_INDEX_URL"

ensure_vendor vendor/ltx-2-mlx dgrauet/ltx-2-mlx \
  packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/cli.py
(
  cd vendor/ltx-2-mlx
  export UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/ltx"
  link_venv "$(pwd)/.venv" "$UV_PROJECT_ENVIRONMENT"
  uv sync --index-url "$UV_INDEX_URL"
)

# 仓内 LTX 已带 Gemma 分阶段加载改动；脚本幂等，缺标记时补上
if [[ ! -f "$ROOT/scripts/apply_ltx_gemma_patch.py" ]]; then
  echo "缺少 scripts/apply_ltx_gemma_patch.py" >&2
  exit 1
fi
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/main" \
  uv run python "$ROOT/scripts/apply_ltx_gemma_patch.py"
MARKER="NOVEL_VIDEO_STAGED_GEMMA_LOAD"
for f in \
  vendor/ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/blocks.py \
  vendor/ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/text_encoders/gemma/encoders/base_encoder.py
do
  if ! grep -q "$MARKER" "$f"; then
    echo "LTX Gemma 分阶段加载标记缺失：$f" >&2
    exit 1
  fi
done
echo "已确认 LTX Gemma 分阶段加载"

mkdir -p tools
download_bin() {
  local name="$1"
  local url="$2"
  if [[ -x "tools/$name" ]]; then
    echo "使用仓内 tools/$name"
    return
  fi
  echo "仓内缺少 tools/$name，下载 $url"
  local tmp
  tmp="$(mktemp -d)"
  curl -L --fail --retry 3 --retry-delay 2 -o "$tmp/$name.zip" "$url"
  unzip -o "$tmp/$name.zip" -d "$tmp"
  local bin
  bin="$(find "$tmp" -type f -name "$name" | head -n 1)"
  if [[ -z "$bin" ]]; then
    echo "压缩包里没有 $name" >&2
    exit 1
  fi
  mv "$bin" "tools/$name"
  chmod +x "tools/$name"
  rm -rf "$tmp"
}

download_bin ffmpeg "https://evermeet.cx/ffmpeg/ffmpeg-129150-g7ddf7bc8a48.zip"
download_bin ffprobe "https://evermeet.cx/ffmpeg/ffprobe-129150-g7ddf7bc8a48.zip"
xattr -dr com.apple.quarantine tools/ffmpeg tools/ffprobe 2>/dev/null || true

ensure_vendor vendor/musicgen-mlx andrade0/musicgen-mlx generate.py
(
  cd vendor/musicgen-mlx
  export UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/musicgen"
  link_venv "$(pwd)/.venv" "$UV_PROJECT_ENVIRONMENT"
  uv sync --index-url "$UV_INDEX_URL"
)
if [[ ! -x "$NOVEL_VIDEO_CACHE/venvs/ltx/bin/ltx-2-mlx" ]]; then
  echo "LTX CLI 未安装：$NOVEL_VIDEO_CACHE/venvs/ltx/bin/ltx-2-mlx" >&2
  exit 1
fi
if [[ ! -x "$NOVEL_VIDEO_CACHE/venvs/musicgen/bin/python" ]]; then
  echo "MusicGen Python 未安装：$NOVEL_VIDEO_CACHE/venvs/musicgen/bin/python" >&2
  exit 1
fi
if [[ ! -f vendor/musicgen-mlx/generate.py ]]; then
  echo "MusicGen 源码不完整：缺少 generate.py" >&2
  exit 1
fi

echo "预下载权重（一律走 HF_ENDPOINT=$HF_ENDPOINT，已禁用 XET；权重不入库）"
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/main" uv run python - <<'PY'
from pipeline.mirrors import apply_mirrors
apply_mirrors()
from huggingface_hub import snapshot_download
import os
print("HF_ENDPOINT=", os.environ.get("HF_ENDPOINT"))
print("HF_HUB_DISABLE_XET=", os.environ.get("HF_HUB_DISABLE_XET"))

snapshot_download(
    "dgrauet/ltx-2.3-mlx-q4",
    allow_patterns=[
        "transformer-distilled-1.1.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
        "spatial_upscaler_x2_v1_1_config.json",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "connector.safetensors",
        "config.json",
        "embedded_config.json",
        "quantize_config.json",
        "split_model.json",
        "LICENSE",
        "README.md",
        ".gitattributes",
    ],
)
snapshot_download("mlx-community/gemma-3-12b-it-4bit")
snapshot_download("mflux-community/flux2-klein-4b-mflux-q4")
snapshot_download("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
snapshot_download("facebook/musicgen-small")
print("weights ready")
PY

echo "冒烟：transformers 导入计时（应在数秒内，不应数分钟）"
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/main" uv run python - <<'PY'
import time
t=time.time()
import transformers
print(f"main transformers {time.time()-t:.2f}s {transformers.__version__}")
PY
env -u VIRTUAL_ENV "$NOVEL_VIDEO_CACHE/venvs/ltx/bin/python" - <<'PY'
import time
t=time.time()
import transformers
print(f"ltx transformers {time.time()-t:.2f}s {transformers.__version__}")
PY
env -u VIRTUAL_ENV "$NOVEL_VIDEO_CACHE/venvs/musicgen/bin/python" - <<'PY'
import time
t=time.time()
import transformers
print(f"musicgen transformers {time.time()-t:.2f}s {transformers.__version__}")
PY

echo "环境就绪。镜像：HF=$HF_ENDPOINT  PyPI=$UV_INDEX_URL"
echo "venv 缓存：$NOVEL_VIDEO_CACHE/venvs"
echo "下一步：./scripts/run.sh doctor"
echo "演示分镜：mkdir -p output/demo && cp input/demo_storyboard.json output/demo/storyboard.json"
