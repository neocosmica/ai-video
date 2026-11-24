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

# 确保 launcher / menubar 可执行
chmod +x "$APP/Contents/MacOS/launcher" 2>/dev/null || true
if [[ -x "$ROOT/scripts/macos/menubar-helper" ]]; then
  mkdir -p "$RUNTIME/scripts/macos"
  cp -f "$ROOT/scripts/macos/menubar-helper" "$RUNTIME/scripts/macos/menubar-helper"
  chmod +x "$RUNTIME/scripts/macos/menubar-helper"
fi

# 体积提示
du -sh "$RUNTIME" | awk '{print "runtime 体积：" $1}'
echo "完成。双击 App 将优先使用包内 runtime。"
