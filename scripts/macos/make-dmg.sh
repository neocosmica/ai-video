#!/bin/zsh
# 生成可拖进「应用程序」的安装镜像：dist/AI短视频.dmg
# 默认先跑 pack-app.sh；跳过打包：SKIP_PACK=1 ./scripts/macos/make-dmg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$ROOT/AI短视频.app"
DIST="$ROOT/dist"
VOLNAME="安装 AI短视频"
DMG_NAME="AI短视频.dmg"
STAGE="$DIST/dmg-stage"
RW_DMG="$DIST/AI短视频.rw.dmg"
FINAL_DMG="$DIST/$DMG_NAME"

if [[ ! -d "$APP/Contents" ]]; then
  echo "找不到 $APP" >&2
  exit 1
fi

if [[ "${SKIP_PACK:-}" != "1" ]]; then
  echo "先同步 App 内 runtime…"
  "$ROOT/scripts/macos/pack-app.sh"
fi

if [[ ! -x "$APP/Contents/Resources/runtime/scripts/run.sh" ]]; then
  echo "App 内还没有 runtime，请先跑 ./scripts/macos/pack-app.sh" >&2
  exit 1
fi

echo "准备 DMG 内容…"
rm -rf "$STAGE" "$RW_DMG" "$FINAL_DMG"
mkdir -p "$STAGE"

# 拷贝 App（保留签名位与可执行权限）
ditto "$APP" "$STAGE/AI短视频.app"
ln -s /Applications "$STAGE/Applications"

echo "创建读写镜像…"
hdiutil create \
  -ov \
  -volname "$VOLNAME" \
  -srcfolder "$STAGE" \
  -fs HFS+ \
  -format UDRW \
  "$RW_DMG" >/dev/null

echo "挂载并设置窗口布局…"
# 卷名可含空格：从 attach 输出里抠 /Volumes/… 整段
ATTACH_OUT="$(hdiutil attach -readwrite -noverify -noautoopen "$RW_DMG")"
MOUNT_POINT="$(print -r -- "$ATTACH_OUT" | sed -n 's#.*\(/Volumes/.*\)$#\1#p' | tail -1)"
if [[ -z "$MOUNT_POINT" || ! -d "$MOUNT_POINT" ]]; then
  echo "挂载失败：" >&2
  print -r -- "$ATTACH_OUT" >&2
  exit 1
fi
echo "已挂载：$MOUNT_POINT"

cleanup_mount() {
  hdiutil detach "$MOUNT_POINT" -force >/dev/null 2>&1 || true
}
trap cleanup_mount EXIT

# Finder：图标网格 + Applications 快捷方式并排，方便拖进去
/usr/bin/osascript <<APPLESCRIPT >/dev/null || true
tell application "Finder"
  activate
  open POSIX file "$MOUNT_POINT"
  delay 1
  set win to front window
  set current view of win to icon view
  set toolbar visible of win to false
  set statusbar visible of win to false
  set bounds of win to {200, 160, 760, 520}
  set opts to icon view options of win
  set arrangement of opts to not arranged
  set icon size of opts to 96
  try
    set position of item "AI短视频.app" of win to {140, 180}
    set position of item "Applications" of win to {420, 180}
  end try
  close win
end tell
APPLESCRIPT

# 让 .DS_Store 落盘
sync
sleep 1
hdiutil detach "$MOUNT_POINT" >/dev/null
trap - EXIT

echo "压缩为只读镜像…"
hdiutil convert "$RW_DMG" -format UDZO -imagekey zlib-level=9 -o "$FINAL_DMG" >/dev/null
rm -f "$RW_DMG"
rm -rf "$STAGE"

du -sh "$FINAL_DMG" | awk '{print "已生成：" $2 "（" $1 "）"}'
echo "打开镜像后，把「AI短视频」拖到「Applications」即可安装。"
