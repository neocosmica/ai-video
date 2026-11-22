#!/bin/zsh
# 国内镜像环境。setup.sh / run.sh 都会 source 这份文件。

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

# GitHub 克隆前缀（失败时脚本会回退官方地址）
export GITHUB_MIRROR_PREFIX="${GITHUB_MIRROR_PREFIX:-https://ghfast.top/}"

# 强制 Python 子进程也看见这些变量
export PYTHONUNBUFFERED=1
