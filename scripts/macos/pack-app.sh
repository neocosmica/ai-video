#!/bin/zsh
# 把运行依赖同步进 AI短视频.app/Contents/Resources/runtime/
# 不含 projects/、.git、缓存、日志；模型与 venv 仍在用户缓存。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$ROOT/AI短视频.app"
RUNTIME="$APP/Contents/Resources/runtime"

if [[ ! -d "$APP/Contents" ]]; then
  echo "找不到 $APP" >&2
  exit 1
fi

echo "同步运行依赖 → $RUNTIME"
mkdir -p "$RUNTIME"

rsync -a --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'server.log' \
  --exclude 'server.pid' \
  --exclude '.venv/' \
  --exclude '.venv' \
  --exclude '.x86_64-backup/' \
  --exclude '*.wrong-arch.bak' \
  --exclude '.ffmpeg-dl/' \
  --exclude '.ffprobe-dl/' \
  --exclude '.ffmpeg-dl.*/' \
  --exclude '.ffprobe-dl.*/' \
  "$ROOT/pipeline" \
  "$ROOT/scripts" \
  "$ROOT/webui" \
  "$ROOT/vendor" \
  "$ROOT/tools" \
  "$RUNTIME/"

# 顶层配置与锁文件
for f in config.json pyproject.toml uv.lock; do
  if [[ -f "$ROOT/$f" ]]; then
    cp -f "$ROOT/$f" "$RUNTIME/$f"
  fi
done

# 同步启动器源文件 → App/MacOS（可执行）
if [[ -f "$ROOT/scripts/macos/launcher" ]]; then
  mkdir -p "$APP/Contents/MacOS"
  cp -f "$ROOT/scripts/macos/launcher" "$APP/Contents/MacOS/launcher"
fi
chmod +x "$APP/Contents/MacOS/launcher" 2>/dev/null || true
if [[ -x "$ROOT/scripts/macos/menubar-helper" ]]; then
  mkdir -p "$RUNTIME/scripts/macos"
  cp -f "$ROOT/scripts/macos/menubar-helper" "$RUNTIME/scripts/macos/menubar-helper"
  chmod +x "$RUNTIME/scripts/macos/menubar-helper"
fi

# 打包前校验：ffmpeg/ffprobe 必须可执行且含本机架构（发行包面向 Apple Silicon）
host_arch="$(uname -m)"
for name in ffmpeg ffprobe; do
  bin="$RUNTIME/tools/$name"
  if [[ ! -x "$bin" ]]; then
    echo "打包失败：缺少 $bin（先 ./scripts/setup.sh）" >&2
    exit 1
  fi
  bin_archs="$(/usr/bin/lipo -archs "$bin" 2>/dev/null || true)"
  if [[ -z "$bin_archs" || "$bin_archs" != *"$host_arch"* ]]; then
    echo "打包失败：$bin 架构是 ${bin_archs:-未知}，本机是 $host_arch。" >&2
    echo "未装 Rosetta 的 Mac 会报 Bad CPU type in executable。请换成含 $host_arch 的版本后再打包。" >&2
    exit 1
  fi
  # 真跑一次，堵住「文件在但 CPU type 不对 / 损坏」
  if ! "$bin" -version >/dev/null 2>&1; then
    echo "打包失败：$bin -version 失败（架构或损坏）。" >&2
    exit 1
  fi
done
echo "已校验 tools/ffmpeg、ffprobe：$host_arch（含 -version）"

# 体积提示
du -sh "$RUNTIME" | awk '{print "runtime 体积：" $1}'
echo "完成。双击 App 将优先使用包内 runtime。"
