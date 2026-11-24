#!/bin/zsh
# 从 scripts/macos/console-app-icon.png 重建「AI短视频.app」图标。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MASTER="$ROOT/scripts/macos/console-app-icon.png"
APP="$ROOT/AI短视频.app"
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"

if [[ ! -f "$MASTER" ]]; then
  echo "缺少 $MASTER" >&2
  exit 1
fi

for s in 16 32 128 256 512; do
  sips -z "$s" "$s" "$MASTER" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$MASTER" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done

mkdir -p "$APP/Contents/Resources"
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
rm -rf "$(dirname "$ICONSET")"
# 逼 Finder 刷新图标缓存
touch "$APP"
echo "已写入 $APP/Contents/Resources/AppIcon.icns"
