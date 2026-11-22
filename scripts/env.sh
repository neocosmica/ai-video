#!/bin/zsh
# 国内镜像 + 本地缓存环境。setup.sh / run.sh 都会 source 这份文件。
# 原则：所有外网下载只走国内镜像，不回退海外官方地址。

# Hugging Face → hf-mirror（关掉 XET，避免走到海外 CDN）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-$HF_ENDPOINT}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HUGGINGFACE_HUB_DISABLE_TELEMETRY=1

# PyPI / uv → 清华
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

# uv 托管 Python 安装包 → npmmirror（镜像 astral python-build-standalone）
# 官方基址：https://github.com/astral-sh/python-build-standalone/releases/download
export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://cdn.npmmirror.com/binaries/python-build-standalone}"

# GitHub 克隆 / release 下载：只用国内代理前缀（空格分隔，按序尝试）
export GITHUB_MIRROR_PREFIXES="${GITHUB_MIRROR_PREFIXES:-https://ghfast.top/ https://mirror.ghproxy.com/ https://ghproxy.net/}"
# 兼容旧变量名：取列表第一个
export GITHUB_MIRROR_PREFIX="${GITHUB_MIRROR_PREFIX:-${${=GITHUB_MIRROR_PREFIXES}[1]}}"

# Homebrew（若用 brew 补 ffmpeg）→ 清华
export HOMEBREW_API_DOMAIN="${HOMEBREW_API_DOMAIN:-https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles/api}"
export HOMEBREW_BOTTLE_DOMAIN="${HOMEBREW_BOTTLE_DOMAIN:-https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles}"
export HOMEBREW_BREW_GIT_REMOTE="${HOMEBREW_BREW_GIT_REMOTE:-https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git}"
export HOMEBREW_CORE_GIT_REMOTE="${HOMEBREW_CORE_GIT_REMOTE:-https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git}"

# 优先本机已有解释器；3.14 / 3.13 / 3.12 / 3.11 均可（项目 requires-python >=3.11）
export NOVEL_VIDEO_PYTHON="${NOVEL_VIDEO_PYTHON:-}"

# 强制 Python 子进程也看见这些变量
export PYTHONUNBUFFERED=1

# 重型 venv 装到 Library/Caches，项目内只留符号链接
export NOVEL_VIDEO_CACHE="${NOVEL_VIDEO_CACHE:-$HOME/Library/Caches/novel-video}"
mkdir -p "$NOVEL_VIDEO_CACHE/venvs"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$NOVEL_VIDEO_CACHE/venvs/main}"
