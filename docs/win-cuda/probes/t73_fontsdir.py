"""诊断 Windows 上 ASS 字幕滤镜的 fontsdir 写法。

结论：三种带 fontsdir 的写法（原样 / 正斜杠 / 转义冒号）全部失败，
报 `Error parsing a filter description`（盘符冒号破坏滤镜语法）；
去掉 fontsdir 后 rc=0（ffmpeg 自动加载系统字体）。
对应实现：pipeline/platform.ass_filter()
"""
import os, sys, io, pathlib, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = pathlib.Path(r"C:/Users/annas/avwin/repo")
sys.path.insert(0, str(REPO)); os.chdir(REPO)
PROJ = pathlib.Path(r"C:/Users/annas/probes/proj_demo")
ff = REPO/"tools"/"ffmpeg.exe"
from pipeline.platform import font_dir
FD = font_dir()
print("[1] fontsdir =", FD, "| 含冒号:", ":" in str(FD), flush=True)

variants = [
    ("原样", f"ass=work/subs.ass:fontsdir={FD}"),
    ("正斜杠", "ass=work/subs.ass:fontsdir=" + str(FD).replace("\\", "/")),
    ("转义冒号", "ass=work/subs.ass:fontsdir=" + str(FD).replace(":", "\\:")),
    ("无fontsdir", "ass=work/subs.ass"),
]
print("\n[2] 测试", flush=True)
for desc, vf in variants:
    cmd = [str(ff), "-y", "-i", "work/joined.mp4", "-i", "work/soundtrack.m4a",
           "-vf", vf, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", "-map", "0:v:0", "-map", "1:a:0",
           "work/test_font.mp4"]
    r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print("  [%-10s] rc=%d" % (desc, r.returncode), flush=True)
    if r.returncode != 0:
        for line in (r.stderr or "").splitlines()[-4:]:
            print("      ", line, flush=True)
