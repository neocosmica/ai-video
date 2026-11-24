#!/bin/zsh
# 编译菜单栏助手 → scripts/macos/menubar-helper
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/scripts/macos/MenubarHelper.swift"
OUT="$ROOT/scripts/macos/menubar-helper"

if ! command -v swiftc >/dev/null 2>&1; then
  echo "缺少 swiftc，无法编译菜单栏助手" >&2
  exit 1
fi

swiftc -O -parse-as-library -o "$OUT" "$SRC"
chmod +x "$OUT"
echo "已编译 $OUT"
