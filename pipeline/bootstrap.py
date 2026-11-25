"""本机依赖与模型就绪检查 / 补装。

供 ``scripts/setup.sh``、网页启动与 ``doctor`` 共用同一份清单。
分镜语言模型（Qwen3-4B / mlx-lm）也在这里，避免只装进网页却忘了 setup。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pipeline.mirrors import apply_mirrors, child_env
from pipeline.util import ROOT

apply_mirrors()

# 与 config.json / setup 历史清单对齐；allow_patterns=None 表示整仓
WEIGHT_SPECS: list[dict[str, Any]] = [
    {
        "id": "llm",
        "label": "分镜语言模型 Qwen3-4B",
        "repo": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "allow_patterns": None,
        "required_files": (),
        # 只有 tokenizer/config 的残缺快照不能算就绪（mlx 会报 No safetensors）
        "weight_globs": ("model*.safetensors",),
    },
    {
        "id": "flux",
        "label": "文生图 FLUX.2 Klein 4B",
        "repo": "mflux-community/flux2-klein-4b-mflux-q4",
        "allow_patterns": None,
        "required_files": (),
    },
    {
        "id": "gemma",
        "label": "视频提示词编码器 Gemma 3 12B",
        "repo": "mlx-community/gemma-3-12b-it-4bit",
        "allow_patterns": None,
        "required_files": (),
        "weight_globs": ("model*.safetensors",),
    },
    {
        "id": "tts",
        "label": "对白 TTS Qwen3-TTS",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "allow_patterns": None,
        "required_files": (),
        "weight_globs": ("model*.safetensors",),
    },
    {
        "id": "ltx",
        "label": "视频 LTX-2.3",
        "repo": "dgrauet/ltx-2.3-mlx-q4",
        "allow_patterns": [
            "transformer-distilled-1.1.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
            "connector.safetensors",
            "config.json",
            "embedded_config.json",
            "quantize_config.json",
            "split_model.json",
            "LICENSE",
            "README.md",
            ".gitattributes",
        ],
        "required_files": ("audio_vae.safetensors", "vocoder.safetensors", "config.json"),
    },
    {
        "id": "musicgen",
        "label": "配乐 MusicGen small",
        "repo": "facebook/musicgen-small",
        "allow_patterns": [
            "state_dict.bin",
            "compression_state_dict.bin",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "generation_config.json",
            "preprocessor_config.json",
            "README.md",
            ".gitattributes",
        ],
        "required_files": ("state_dict.bin", "compression_state_dict.bin"),
    },
    {
        "id": "t5",
        "label": "配乐条件编码器 T5-base",
        "repo": "t5-base",
        "allow_patterns": [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "spiece.model",
        ],
        "required_files": ("model.safetensors", "config.json"),
    },
]


@dataclass
class SetupState:
    status: str = "unknown"  # unknown|ready|checking|installing|failed
    pct: float = 0.0
    message: str = ""
    error: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "ready": self.status == "ready",
                "pct": round(float(self.pct), 1),
                "message": self.message,
                "error": self.error,
                "items": list(self.items),
                "updated_at": self.updated_at,
            }

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.updated_at = time.time()


_STATE = SetupState()
_WORKER: threading.Thread | None = None
_WORKER_LOCK = threading.Lock()


def _weight_ok(spec: dict[str, Any]) -> tuple[bool, str]:
    from pipeline.hf_cache import resolve_snapshot, snapshot_ready

    snap = resolve_snapshot(str(spec["repo"]))
    return snapshot_ready(
        snap,
        required_files=tuple(spec.get("required_files") or ()),
        weight_globs=tuple(spec.get("weight_globs") or ()),
    )


def _pkg_items() -> list[dict[str, Any]]:
    from pipeline.config import Settings
    from pipeline.runtime import ltx_cli, main_venv_path, musicgen_venv_path

    settings = Settings.load()
    items: list[dict[str, Any]] = []

    def add(iid: str, label: str, ok: bool, detail: str = "") -> None:
        items.append({"id": iid, "kind": "pkg", "label": label, "ok": ok, "detail": detail})

    main_py = main_venv_path() / "bin" / "python"
    add("venv-main", "主环境 Python", main_py.exists(), str(main_py) if main_py.exists() else "请跑 setup")

    mlx_ok = False
    mlx_detail = ""
    try:
        import mlx_lm  # noqa: F401

        mlx_ok = True
        mlx_detail = getattr(mlx_lm, "__version__", "installed")
    except Exception as exc:
        mlx_detail = str(exc)
    add("mlx-lm", "分镜依赖 mlx-lm", mlx_ok, mlx_detail)

    try:
        import mflux  # noqa: F401

        add("mflux", "文生图依赖 mflux", True, "installed")
    except Exception as exc:
        add("mflux", "文生图依赖 mflux", False, str(exc))

    from pipeline.media_bins import binary_runnable

    # Settings.load 已做过包内→系统回退；这里再试跑，堵住「文件在但不可执行」假阳性
    ff_ok, ff_detail = binary_runnable(settings.ffmpeg)
    fp_ok, fp_detail = binary_runnable(settings.ffprobe)
    if ff_ok and fp_ok:
        note = str((settings.budget_detail or {}).get("ffmpeg") or ff_detail)
        add("ffmpeg", "ffmpeg", True, note)
    else:
        add("ffmpeg", "ffmpeg", False, ff_detail if not ff_ok else fp_detail)

    try:
        cli = ltx_cli(settings.ltx_dir)
        add("ltx-cli", "LTX CLI", True, str(cli))
    except SystemExit as exc:
        add("ltx-cli", "LTX CLI", False, str(exc))

    gen_py = settings.musicgen_dir / "generate.py"
    mg_py = musicgen_venv_path() / "bin" / "python"
    add(
        "musicgen",
        "MusicGen 环境",
        gen_py.is_file() and mg_py.exists(),
        str(gen_py) if gen_py.is_file() else "缺 generate.py",
    )

    marker = "NOVEL_VIDEO_STAGED_GEMMA_LOAD"
    blocks = (
        settings.ltx_dir
        / "packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/blocks.py"
    )
    patched = blocks.is_file() and marker in blocks.read_text(encoding="utf-8", errors="ignore")
    add("ltx-patch", "LTX Gemma 分阶段补丁", patched, "已打" if patched else "未打（请 setup）")
    return items


def _weight_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for spec in WEIGHT_SPECS:
        ok, detail = _weight_ok(spec)
        items.append(
            {
                "id": f"weight-{spec['id']}",
                "kind": "weight",
                "label": spec["label"],
                "ok": ok,
                "detail": detail,
                "repo": spec["repo"],
            }
        )
    return items


def probe(*, refresh_state: bool = True) -> dict[str, Any]:
    """快速检查（不下载）。"""
    items = _pkg_items() + _weight_items()
    missing = [it for it in items if not it["ok"]]
    ready = not missing
    payload = {
        "status": "ready" if ready else "need_install",
        "ready": ready,
        "pct": 100.0 if ready else 0.0,
        "message": "依赖与模型已就绪" if ready else f"还缺 {len(missing)} 项，正在准备安装…",
        "error": "",
        "items": items,
        "missing": [it["id"] for it in missing],
    }
    if refresh_state and _STATE.status not in {"installing", "checking"}:
        _STATE.update(
            status="ready" if ready else "need_install",
            pct=payload["pct"],
            message=payload["message"],
            error="",
            items=items,
        )
    return payload


def setup_status() -> dict[str, Any]:
    snap = _STATE.snapshot()
    if snap["status"] in {"installing", "checking"}:
        return snap
    if snap["status"] in {"ready", "need_install", "failed"} and snap["items"]:
        return snap
    return probe(refresh_state=True)


def _uv_bin() -> str:
    return shutil.which("uv") or str(Path.home() / ".local/bin/uv")


def _run_uv_sync(set_msg: Callable[[str, float], None]) -> None:
    uv = _uv_bin()
    if not Path(uv).exists() and not shutil.which("uv"):
        raise RuntimeError("找不到 uv，请先安装或跑 scripts/setup.sh")
    set_msg("正在同步 Python 依赖（含 mlx-lm）…", max(_STATE.pct, 2.0))
    env = child_env()
    cache = Path.home() / "Library" / "Caches" / "novel-video" / "venvs" / "main"
    env["UV_PROJECT_ENVIRONMENT"] = str(cache)
    cmd = [uv, "sync", "--index-url", env.get("UV_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")]
    # 尽量沿用 setup 的 Python 选择；不强制版本以免打断已有环境
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"uv sync 失败（{proc.returncode}）"
        raise RuntimeError(err[-2000:])
    # 确认分镜依赖
    try:
        import importlib

        importlib.invalidate_caches()
        import mlx_lm  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"uv sync 后仍缺 mlx-lm：{exc}") from exc


def _download_weight(spec: dict[str, Any], set_msg: Callable[[str, float], None], base_pct: float, span: float) -> None:
    from huggingface_hub import snapshot_download

    set_msg(f"正在下载 {spec['label']}…", base_pct)
    kwargs: dict[str, Any] = {"repo_id": spec["repo"]}
    patterns = spec.get("allow_patterns")
    if patterns:
        kwargs["allow_patterns"] = patterns
    snapshot_download(**kwargs)
    ok, detail = _weight_ok(spec)
    if not ok:
        raise RuntimeError(f"{spec['label']} 下载后仍不可用：{detail}")
    set_msg(f"已就绪：{spec['label']}", base_pct + span)


def _install_loop(*, force: bool = False) -> None:
    def set_msg(message: str, pct: float) -> None:
        _STATE.update(status="installing", message=message, pct=min(99.0, max(0.0, pct)), error="")

    try:
        _STATE.update(status="checking", message="正在检查依赖与模型…", pct=1.0, error="")
        report = probe(refresh_state=False)
        items = report["items"]
        _STATE.update(items=items)
        missing = [it for it in items if not it["ok"]]
        # 自动检查说齐了就停。手动「安装依赖」带 force：判断可能失误，仍按清单再拉一遍。
        if not missing and not force:
            _STATE.update(status="ready", pct=100.0, message="依赖与模型已就绪", items=items, error="")
            return

        if not missing and force:
            set_msg("检查显示已就绪，仍重新拉取依赖与模型（已有文件会跳过）…", 3.0)
        else:
            set_msg(f"发现缺 {len(missing)} 项，开始补装…", 3.0)

        pkg_missing = [it for it in missing if it["kind"] == "pkg"]
        # 包/环境缺：先 uv sync；更重的 LTX/MusicGen/ffmpeg 仍建议完整 setup
        need_full_setup = any(
            it["id"] in {"ltx-cli", "musicgen", "ffmpeg", "ltx-patch", "venv-main"}
            for it in pkg_missing
        )
        if need_full_setup:
            set_msg("环境不完整，正在跑 scripts/setup.sh（含依赖与模型）…", 5.0)
            env = child_env()
            proc = subprocess.Popen(
                ["/bin/zsh", str(ROOT / "scripts" / "setup.sh")],
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            lines = 0
            for line in proc.stdout:
                lines += 1
                text = line.strip()
                if text:
                    # setup 很长：用行数做粗进度，卡在 5–90
                    pct = min(90.0, 5.0 + lines * 0.35)
                    set_msg(text[:160], pct)
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"setup.sh 退出码 {code}")
        else:
            if force or any(it["id"] in {"mlx-lm", "mflux"} for it in pkg_missing):
                _run_uv_sync(set_msg)

            # force：全部重拉，不信任「已有」的判断（残缺快照会被补上）
            weight_missing = (
                list(WEIGHT_SPECS)
                if force
                else [spec for spec in WEIGHT_SPECS if not _weight_ok(spec)[0]]
            )
            n = max(1, len(weight_missing))
            for i, spec in enumerate(weight_missing):
                base = 10.0 + (80.0 * i / n)
                span = 80.0 / n
                _download_weight(spec, set_msg, base, span)

        final = probe(refresh_state=False)
        if not final["ready"]:
            still = ", ".join(it["label"] for it in final["items"] if not it["ok"])
            raise RuntimeError(f"补装后仍缺：{still}")
        _STATE.update(
            status="ready",
            pct=100.0,
            message="依赖与模型已就绪",
            items=final["items"],
            error="",
        )
    except Exception as exc:
        _STATE.update(
            status="failed",
            message="依赖安装失败",
            error=f"{exc}",
            pct=max(_STATE.pct, 0.0),
        )
        traceback.print_exc()


def ensure_async(*, force: bool = False) -> dict[str, Any]:
    """后台补装；已在装则返回当前进度。"""
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER and _WORKER.is_alive():
            return setup_status()
        if not force:
            cur = probe(refresh_state=True)
            if cur["ready"]:
                return setup_status()
        _STATE.update(status="installing", message="准备安装…", pct=1.0, error="")
        _WORKER = threading.Thread(
            target=_install_loop,
            kwargs={"force": force},
            name="novel-video-bootstrap",
            daemon=True,
        )
        _WORKER.start()
    return setup_status()


def install_weights_cli() -> int:
    """供 setup.sh 调用：按 WEIGHT_SPECS 拉齐权重（含分镜 LLM）。"""
    apply_mirrors()
    from huggingface_hub import snapshot_download

    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print(f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}")
    for spec in WEIGHT_SPECS:
        print(f"下载 {spec['label']} ← {spec['repo']}", flush=True)
        kwargs: dict[str, Any] = {"repo_id": spec["repo"]}
        if spec.get("allow_patterns"):
            kwargs["allow_patterns"] = spec["allow_patterns"]
        snapshot_download(**kwargs)
        ok, detail = _weight_ok(spec)
        print(f"  → {'OK' if ok else '缺'} {detail}", flush=True)
        if not ok:
            return 1
    print("weights ready", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m pipeline.bootstrap [probe|ensure|install-weights]")
        return 0
    cmd = args[0]
    if cmd == "probe":
        print(json.dumps(probe(), ensure_ascii=False, indent=2))
        return 0
    if cmd == "ensure":
        ensure_async(force=True)
        while True:
            st = setup_status()
            print(f"[{st['status']}] {st['pct']}% {st['message']}", flush=True)
            if st["status"] in {"ready", "failed"}:
                return 0 if st["status"] == "ready" else 1
            time.sleep(1.0)
    if cmd == "install-weights":
        return install_weights_cli()
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
