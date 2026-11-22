#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"
export PATH="$HOME/.local/bin:$ROOT/tools:$PATH"
exec uv run python -m pipeline "$@"
