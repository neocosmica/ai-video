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
  echo "克隆 $repo （优先 GitHub 镜像）"
  if ! git -c http.version=HTTP/1.1 clone --depth 1 "$mirrored" "$dest"; then
    echo "镜像失败，回退官方地址"
    rm -rf "$dest"
    git -c http.version=HTTP/1.1 clone --depth 1 "$official" "$dest"
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  echo "安装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12
uv sync --index-url "$UV_INDEX_URL"

if [[ ! -d vendor/ltx-2-mlx/.git ]]; then
  clone_github dgrauet/ltx-2-mlx vendor/ltx-2-mlx
fi
(cd vendor/ltx-2-mlx && uv sync --index-url "$UV_INDEX_URL")
chflags -R nohidden .venv vendor/ltx-2-mlx/.venv 2>/dev/null || true

mkdir -p tools
download_bin() {
  local name="$1"
  local url="$2"
  if [[ -x "tools/$name" ]]; then
    return
  fi
  echo "下载 $name"
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

if [[ ! -d vendor/musicgen-mlx/.git ]]; then
  clone_github andrade0/musicgen-mlx vendor/musicgen-mlx
fi
(cd vendor/musicgen-mlx && uv sync --index-url "$UV_INDEX_URL")
chflags -R nohidden vendor/musicgen-mlx/.venv 2>/dev/null || true

echo "预下载权重（一律走 HF_ENDPOINT=$HF_ENDPOINT，已禁用 XET）"
uv run python - <<'PY'
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

echo "环境就绪。镜像：HF=$HF_ENDPOINT  PyPI=$UV_INDEX_URL  GitHub前缀=$GITHUB_MIRROR_PREFIX"
