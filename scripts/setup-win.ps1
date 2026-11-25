# Windows + NVIDIA 环境安装脚本
#
# 与 scripts/setup.sh 对应，但走 PyTorch/diffusers 原生 CUDA 栈：
#   - 主环境：torch cu128 + diffusers + transformers(>=5) + bitsandbytes
#   - TTS 环境：transformers==4.57.3 + qwen-tts（与主环境版本冲突，必须隔离）
#
# 用法（PowerShell）：
#   .\scripts\setup-win.ps1
#   .\scripts\setup-win.ps1 -CudaTag cu128 -SkipModels
#
param(
    [string]$CudaTag = "cu128",
    [switch]$SkipModels,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

Write-Host "==> 小说成片（Windows CUDA）环境安装" -ForegroundColor Cyan
Write-Host "    项目根：$Root"

# ---------- 镜像 ----------
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_ENDPOINT = $env:HF_ENDPOINT
$env:HF_HUB_DISABLE_XET = "1"
$env:PYPI = "https://pypi.tuna.tsinghua.edu.cn/simple/"
$env:TORCH_INDEX = "https://download.pytorch.org/whl/$CudaTag"
$env:PYTHONUTF8 = "1"

function Run($desc, $cmd) {
    Write-Host "`n>>> $desc" -ForegroundColor Yellow
    & cmd /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "$desc 失败（exit $LASTEXITCODE）" }
}

# ---------- 主环境 ----------
$MainVenv = Join-Path $Root ".venv"
$MainPy = Join-Path $MainVenv "Scripts\python.exe"
if (-not (Test-Path $MainPy)) {
    Run "创建主 venv" "$Python -m venv `"$MainVenv`""
}

Run "升级 pip" "$MainPy -m pip install -U pip -q -i $env:PYPI"
Run "安装 torch $CudaTag" "$MainPy -m pip install torch torchvision torchaudio --index-url $env:TORCH_INDEX"

# triton-windows 是 Windows 上 flash-linear-attention 的前提；
# 缺了它 Qwen3.5 只有 ~12.5 tok/s（实测），装上后 17.8 tok/s
Run "安装 triton-windows" "$MainPy -m pip install triton-windows -i $env:PYPI"
Run "安装推理依赖" "$MainPy -m pip install diffusers transformers accelerate bitsandbytes gguf safetensors huggingface_hub sentencepiece protobuf soundfile scipy einops tqdm pillow imageio imageio-ffmpeg -i $env:PYPI"
Run "安装 flash-linear-attention" "$MainPy -m pip install flash-linear-attention -i $env:PYPI"

# ---------- TTS 独立环境 ----------
$TtsVenv = Join-Path $Root "venvs\tts"
$TtsPy = Join-Path $TtsVenv "Scripts\python.exe"
if (-not (Test-Path $TtsPy)) {
    Run "创建 TTS venv" "$Python -m venv `"$TtsVenv`""
}
Run "TTS: 升级 pip" "$TtsPy -m pip install -U pip -q -i $env:PYPI"
Run "TTS: 安装 torch $CudaTag" "$TtsPy -m pip install torch torchaudio --index-url $env:TORCH_INDEX"
Run "TTS: 安装依赖" "$TtsPy -m pip install transformers==4.57.3 accelerate==1.12.0 soundfile librosa einops onnxruntime sox numpy -i $env:PYPI"
Run "TTS: 安装 qwen-tts" "$TtsPy -m pip install --no-deps qwen-tts -i $env:PYPI"

# ---------- ffmpeg ----------
$Tools = Join-Path $Root "tools"
New-Item -ItemType Directory -Force -Path $Tools | Out-Null
$Ff = Join-Path $Tools "ffmpeg.exe"
$Fp = Join-Path $Tools "ffprobe.exe"
if (-not (Test-Path $Ff) -or -not (Test-Path $Fp)) {
    Write-Host "`n>>> 下载 ffmpeg（gyan.dev 静态版）" -ForegroundColor Yellow
    $zip = Join-Path $env:TEMP "ffmpeg-win.zip"
    Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath (Join-Path $env:TEMP "ffmpeg-win") -Force
    $bin = Get-ChildItem -Path (Join-Path $env:TEMP "ffmpeg-win") -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    $binp = Get-ChildItem -Path (Join-Path $env:TEMP "ffmpeg-win") -Recurse -Filter "ffprobe.exe" | Select-Object -First 1
    Copy-Item $bin.FullName $Ff -Force
    Copy-Item $binp.FullName $Fp -Force
    Write-Host "    ffmpeg -> $Ff"
} else {
    Write-Host "`n使用已有 tools\ffmpeg.exe"
}

# ---------- 校验 ----------
Write-Host "`n==> 冒烟检查" -ForegroundColor Cyan
Run "CUDA 可用性" "$MainPy -c `"import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')`""
Run "后端判定" "$MainPy -m pipeline doctor"

Write-Host "`n环境就绪。" -ForegroundColor Green
Write-Host "下一步："
Write-Host "  .\scripts\run-win.ps1 doctor"
Write-Host "  .\scripts\run-win.ps1 produce --project demo"
if (-not $SkipModels) {
    Write-Host "`n注意：首次 produce 会按需下载权重（LTX-2 19B Q2_K 约 7.5GB、" -ForegroundColor Yellow
    Write-Host "Gemma3 文本编码器约 22GB、FLUX.2 Klein 4B 约 7GB）。" -ForegroundColor Yellow
}
