"""视频：MLX 上的 LTX distilled（a2v / I2V）。Gemma 只做提示词编码。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pipeline.backends.catalog import StageModel
from pipeline.config import Settings
from pipeline.hf_cache import resolve_snapshot
from pipeline.mirrors import child_env
from pipeline.progress import run_tee
from pipeline.runtime import ltx_cli, main_venv_path
from pipeline.util import resolve


class MlxVideoRuntime:
    def __init__(self, settings: Settings, spec: StageModel):
        if spec.runtime != "ltx_distilled":
            raise SystemExit(f"视频运行时不是 ltx_distilled：{spec.runtime}")
        self.settings = settings
        self.spec = spec
        _ensure_ltx_patches(settings)
        self.model_arg = _resolve_model_arg(spec.repo, label="LTX 权重")
        gemma = next((item for item in spec.downloads if item.id == "gemma"), None)
        if gemma is None:
            # LTX-2.5 的 Gemma 4 在权重包里，命令不再另传 Gemma 3。
            self.gemma_arg = None
            self.offline = Path(self.model_arg).exists()
        else:
            self.gemma_arg = _resolve_model_arg(gemma.repo, label="Gemma")
            self.offline = Path(self.model_arg).exists() and Path(self.gemma_arg).exists()

    def render(
        self,
        shot: dict,
        image: Path,
        output: Path,
        log: Path,
        width: int,
        height: int,
        frames: int,
        *,
        audio: Path | None,
        end_image: Path | None = None,
        cond=None,
    ) -> int:
        from pipeline.motion import video_conditioning, video_prompt

        plan = cond or video_conditioning(shot, has_end_frame=bool(end_image and end_image.exists()))
        use_end = bool(plan.use_end_frame and end_image is not None and end_image.exists())
        env = _video_env(self.settings, offline=self.offline)
        cmd = _command(
            self.settings,
            prompt=video_prompt(shot, with_speech=audio is not None, cond=plan),
            image=image,
            output=output,
            seed=int(shot["seed"]),
            width=width,
            height=height,
            frames=frames,
            model=self.model_arg,
            gemma=self.gemma_arg,
            audio=audio,
            end_image=end_image if use_end else None,
            start_strength=plan.start_strength,
            start_binding=plan.start_binding,
            end_strength=plan.end_strength if use_end else None,
            end_binding=plan.end_binding if use_end else None,
            denoise_plan=plan.denoise_plan,
        )
        mode = "a2v" if audio is not None else "i2v"
        shot_id = str(shot.get("id") or "shot")
        tag = f"{shot_id} {mode} {plan.motion_level}"
        if use_end:
            tag += "+end"
        return run_tee(
            cmd,
            log=log,
            env=env,
            progress=tag,
            plan=plan.denoise_plan,
            desc="Denoising",
        )


def _ensure_ltx_patches(settings: Settings) -> None:
    marker = "NOVEL_VIDEO_STAGED_GEMMA_LOAD"
    blocks = (
        settings.ltx_dir
        / "packages"
        / "ltx-pipelines-mlx"
        / "src"
        / "ltx_pipelines_mlx"
        / "utils"
        / "blocks.py"
    )
    encoder = (
        settings.ltx_dir
        / "packages"
        / "ltx-core-mlx"
        / "src"
        / "ltx_core_mlx"
        / "text_encoders"
        / "gemma"
        / "encoders"
        / "base_encoder.py"
    )
    need = any(not path.exists() or marker not in path.read_text(encoding="utf-8") for path in (blocks, encoder))
    if not need:
        return
    patcher = resolve("scripts/apply_ltx_gemma_patch.py")
    if not patcher.exists():
        raise SystemExit(f"LTX 缺少 Gemma 分阶段加载补丁，且找不到 {patcher}")
    print("正在为 vendor/ltx-2-mlx 应用 Gemma 稳定加载补丁…")
    py = main_venv_path() / "bin" / "python"
    if not py.exists():
        py = Path(resolve(".venv")) / "bin" / "python"
    if not py.exists():
        raise SystemExit("找不到主环境 Python，请先运行 scripts/setup.sh")
    env = child_env()
    env.pop("VIRTUAL_ENV", None)
    code = subprocess.call([str(py), str(patcher)], env=env)
    if code != 0:
        raise SystemExit("应用 Gemma 补丁失败")
    for path in (blocks, encoder):
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"补丁后仍缺少标记：{path}")


def _resolve_model_arg(repo_or_path: str, *, label: str) -> str:
    local = resolve_snapshot(repo_or_path)
    if local is not None:
        print(f"使用本地{label}：{local}")
        return str(local)
    print(f"未命中本地缓存，回退 repo id：{repo_or_path}")
    return repo_or_path


def _command(
    settings: Settings,
    *,
    prompt: str,
    image: Path,
    output: Path,
    seed: int,
    width: int,
    height: int,
    frames: int,
    model: str,
    gemma: str | None,
    audio: Path | None,
    end_image: Path | None = None,
    start_strength: float = 1.0,
    start_binding: str = "replace",
    end_strength: float | None = None,
    end_binding: str | None = None,
    denoise_plan: tuple[int, int] = (8, 3),
) -> list[str]:
    # 直接跑 venv 里的入口，禁止每镜 `uv run`
    cli = str(ltx_cli(settings.ltx_dir))
    image_args = [
        "--image",
        str(image),
        "0",
        f"{float(start_strength):.4g}",
        start_binding,
    ]
    if end_image is not None and end_strength is not None and end_binding:
        image_args += [
            "--image",
            str(end_image),
            "last",
            f"{float(end_strength):.4g}",
            end_binding,
        ]
    common_tail = [
        "--model",
        model,
        "--prompt",
        prompt,
        *image_args,
        "-f",
        str(frames),
        "--frame-rate",
        str(settings.video_fps),
        "-H",
        str(height),
        "-W",
        str(width),
        "--seed",
        str(seed),
        "-o",
        str(output),
    ]
    if gemma:
        common_tail[2:2] = ["--gemma", gemma]
    if audio is not None:
        cmd = [cli, "a2v", "--distilled", "--audio", str(audio), *common_tail]
    else:
        cmd = [cli, "generate", "--distilled", *common_tail, "--no-audio"]
    # 步数与进度条分母一致（dynamic 可能是 12+4）
    cmd += ["--stage1-steps", str(denoise_plan[0]), "--stage2-steps", str(denoise_plan[1])]
    if settings.video_low_ram:
        cmd.append("--low-ram")
    return cmd


def _video_env(settings: Settings, *, offline: bool) -> dict[str, str]:
    env = child_env()
    if offline:
        # 本地 snapshot 齐备时禁止 hub 联网刷新，避免卡在 snapshot_download
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    # 短剧 prompt 很短；1024 会放大 Gemma 激活显存
    env.setdefault("LTX2_GEMMA_MAX_LENGTH", "256")
    # 可编辑包 + 防止隐藏 .pth 被跳过：显式 PYTHONPATH
    srcs = [
        settings.ltx_dir / "packages" / "ltx-pipelines-mlx" / "src",
        settings.ltx_dir / "packages" / "ltx-core-mlx" / "src",
    ]
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in srcs)
    # LTX 用 shutil.which("ffmpeg")：把已校验可用的 bin 目录放最前，避开坏的 tools/
    ff_dir = str(Path(settings.ffmpeg).resolve().parent)
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = os.pathsep.join([ff_dir, local_bin, env.get("PATH", "")])
    # 子进程勿继承主项目 VIRTUAL_ENV，否则会指到错误解释器
    env.pop("VIRTUAL_ENV", None)
    return env
