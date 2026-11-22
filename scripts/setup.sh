#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"
export PATH="$HOME/.local/bin:$PATH"

# ---------- 国内镜像辅助 ----------

github_mirror_urls() {
  # usage: github_mirror_urls <owner/repo.git 或完整 https://github.com/...>
  local target="$1"
  local official
  if [[ "$target" == https://github.com/* ]]; then
    official="$target"
  else
    official="https://github.com/${target}"
  fi
  local prefix
  for prefix in ${=GITHUB_MIRROR_PREFIXES}; do
    print -r -- "${prefix}${official}"
  done
}

clone_github() {
  local repo="$1"
  local dest="$2"
  local url
  rm -rf "$dest"
  echo "克隆 $repo（仅国内 GitHub 镜像，不回退海外）"
  for url in $(github_mirror_urls "${repo}.git"); do
    echo "  尝试 $url"
    if git -c http.version=HTTP/1.1 clone --depth 1 "$url" "$dest"; then
      return 0
    fi
    rm -rf "$dest"
  done
  echo "所有国内 GitHub 镜像均失败：$repo" >&2
  echo "可设置 GITHUB_MIRROR_PREFIXES 换代理后重试。" >&2
  exit 1
}

curl_cn() {
  # 带重试的 curl；调用方保证 URL 已是国内镜像
  curl -L --fail --retry 5 --retry-delay 2 --connect-timeout 20 --max-time 0 "$@"
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
    echo "仓内缺少 $dest/$smoke，回退从国内镜像拉取 $repo"
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

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo "使用已有 uv：$(command -v uv) ($(uv --version 2>/dev/null || true))"
    return
  fi
  echo "安装 uv（清华 PyPI，不走 astral.sh）"
  local py
  for py in python3 python3.14 python3.13 python3.12 python3.11; do
    if command -v "$py" >/dev/null 2>&1; then
      "$py" -m pip install --user -i "$UV_INDEX_URL" uv
      export PATH="$HOME/.local/bin:$HOME/Library/Python/3.14/bin:$HOME/Library/Python/3.13/bin:$HOME/Library/Python/3.12/bin:$PATH"
      if command -v uv >/dev/null 2>&1; then
        return
      fi
    fi
  done
  echo "无法通过国内 PyPI 安装 uv，请先安装 Python 3.11+ 后再重试。" >&2
  exit 1
}

pick_python() {
  # 优先本机已有解释器；缺了才用 npmmirror 安装（默认装 3.14）
  local cand
  if [[ -n "${NOVEL_VIDEO_PYTHON}" ]]; then
    cand=("${NOVEL_VIDEO_PYTHON}")
  else
    cand=(3.14 3.13 3.12 3.11)
  fi
  local v
  for v in $cand; do
    if uv python find "$v" >/dev/null 2>&1; then
      export UV_PYTHON="$v"
      echo "使用已有 Python $v：$(uv python find "$v")"
      return
    fi
  done
  local install_ver="${NOVEL_VIDEO_PYTHON:-3.14}"
  echo "本机无可用 Python，经国内镜像安装 $install_ver"
  echo "UV_PYTHON_INSTALL_MIRROR=$UV_PYTHON_INSTALL_MIRROR"
  uv python install "$install_ver"
  export UV_PYTHON="$install_ver"
  echo "已安装 Python $install_ver：$(uv python find "$install_ver")"
}

# ---------- 主流程 ----------

ensure_uv
pick_python

# 主环境：装到 Library/Caches，项目内只留符号链接
export UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/main"
link_venv "$ROOT/.venv" "$UV_PROJECT_ENVIRONMENT"
uv sync --python "$UV_PYTHON" --index-url "$UV_INDEX_URL"

ensure_vendor vendor/ltx-2-mlx dgrauet/ltx-2-mlx \
  packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/cli.py
(
  cd vendor/ltx-2-mlx
  export UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/ltx"
  link_venv "$(pwd)/.venv" "$UV_PROJECT_ENVIRONMENT"
  uv sync --python "$UV_PYTHON" --index-url "$UV_INDEX_URL"
)

# 仓内 LTX 已带 Gemma 分阶段加载改动；脚本幂等，缺标记时补上
if [[ ! -f "$ROOT/scripts/apply_ltx_gemma_patch.py" ]]; then
  echo "缺少 scripts/apply_ltx_gemma_patch.py" >&2
  exit 1
fi
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/main" \
  UV_PYTHON="$UV_PYTHON" \
  uv run --python "$UV_PYTHON" python "$ROOT/scripts/apply_ltx_gemma_patch.py"
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
  if [[ -x "tools/$name" ]]; then
    echo "使用仓内 tools/$name"
    return
  fi
  echo "仓内缺少 tools/$name，尝试经清华 Homebrew 镜像安装"
  if ! command -v brew >/dev/null 2>&1; then
    echo "未找到 brew，无法从国内源补 $name。" >&2
    echo "请把可执行文件放到 tools/$name 后重跑 setup。" >&2
    exit 1
  fi
  if [[ "$name" == "ffmpeg" || "$name" == "ffprobe" ]]; then
    brew install ffmpeg
  fi
  local bin
  bin="$(command -v "$name" || true)"
  if [[ -z "$bin" || ! -x "$bin" ]]; then
    echo "brew 安装后仍找不到 $name" >&2
    exit 1
  fi
  cp "$bin" "tools/$name"
  chmod +x "tools/$name"
}

download_bin ffmpeg
download_bin ffprobe
xattr -dr com.apple.quarantine tools/ffmpeg tools/ffprobe 2>/dev/null || true

ensure_vendor vendor/musicgen-mlx andrade0/musicgen-mlx generate.py
(
  cd vendor/musicgen-mlx
  export UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/musicgen"
  link_venv "$(pwd)/.venv" "$UV_PROJECT_ENVIRONMENT"
  uv sync --python "$UV_PYTHON" --index-url "$UV_INDEX_URL"
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
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/main" \
  UV_PYTHON="$UV_PYTHON" \
  uv run --python "$UV_PYTHON" python - <<'PY'
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
        # A2V 对白条件：编码要 audio_vae；_load_decoders 仍会打开 vocoder
        "audio_vae.safetensors",
        "vocoder.safetensors",
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
# musicgen-mlx 只读 Audiocraft 的 state_dict.bin / compression_state_dict.bin。
# pytorch_model.bin 与 model.safetensors 是同一套权重的 transformers 格式，约 4.5GB，不加载。
snapshot_download(
    "facebook/musicgen-small",
    allow_patterns=[
        "state_dict.bin",
        "compression_state_dict.bin",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "generation_config.json",
        "preprocessor_config.json",
        "README.md",
        ".gitattributes",
    ],
)
# MusicGen 的 T5Conditioner 运行时只读这四份；整仓还有 pytorch/tf/flax，用不到
snapshot_download(
    "t5-base",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "spiece.model",
    ],
)
print("weights ready")
PY

echo "冒烟：transformers 导入计时（应在数秒内，不应数分钟）"
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$NOVEL_VIDEO_CACHE/venvs/main" \
  UV_PYTHON="$UV_PYTHON" \
  uv run --python "$UV_PYTHON" python - <<'PY'
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

echo "环境就绪。Python=$UV_PYTHON  镜像：HF=$HF_ENDPOINT  PyPI=$UV_INDEX_URL  PythonInstall=$UV_PYTHON_INSTALL_MIRROR"
echo "venv 缓存：$NOVEL_VIDEO_CACHE/venvs"
echo "下一步：./scripts/run.sh doctor"
echo "演示成片：./scripts/run.sh produce --storyboard input/demo_storyboard.json --novel input/novel.txt --out output/demo"
