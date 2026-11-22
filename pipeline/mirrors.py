from __future__ import annotations

import os


HF_MIRROR = "https://hf-mirror.com"
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
GITHUB_MIRROR_PREFIX = "https://ghfast.top/"


def apply_mirrors() -> None:
    """把 Hugging Face / PyPI 相关环境变量钉到国内镜像。可重复调用。"""
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)
    os.environ.setdefault("HF_HUB_ENDPOINT", os.environ["HF_ENDPOINT"])
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("UV_DEFAULT_INDEX", PYPI_MIRROR)
    os.environ.setdefault("UV_INDEX_URL", PYPI_MIRROR)
    os.environ.setdefault("PIP_INDEX_URL", PYPI_MIRROR)
    os.environ.setdefault("PIP_TRUSTED_HOST", "pypi.tuna.tsinghua.edu.cn")
    os.environ.setdefault("GITHUB_MIRROR_PREFIX", GITHUB_MIRROR_PREFIX)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """给子进程一份已打好镜像的环境变量。"""
    apply_mirrors()
    env = dict(base or os.environ)
    env["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", HF_MIRROR)
    env["HF_HUB_ENDPOINT"] = os.environ.get("HF_HUB_ENDPOINT", env["HF_ENDPOINT"])
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HUGGINGFACE_HUB_DISABLE_TELEMETRY"] = "1"
    env["UV_DEFAULT_INDEX"] = os.environ.get("UV_DEFAULT_INDEX", PYPI_MIRROR)
    env["UV_INDEX_URL"] = os.environ.get("UV_INDEX_URL", PYPI_MIRROR)
    env["PIP_INDEX_URL"] = os.environ.get("PIP_INDEX_URL", PYPI_MIRROR)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("VIRTUAL_ENV", None)
    return env


apply_mirrors()
