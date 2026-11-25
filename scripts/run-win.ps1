# Windows 运行入口（对应 scripts/run.sh）
#
# 用法：
#   .\scripts\run-win.ps1                    # 起网页控制台（默认）
#   .\scripts\run-win.ps1 doctor             # 环境诊断
#   .\scripts\run-win.ps1 produce --project demo
#   .\scripts\run-win.ps1 budget --project demo
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

# 镜像与编码
$env:HF_ENDPOINT = if ($env:HF_ENDPOINT) { $env:HF_ENDPOINT } else { "https://hf-mirror.com" }
$env:HF_HUB_ENDPOINT = $env:HF_ENDPOINT
$env:HF_HUB_DISABLE_XET = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Root;$env:PYTHONPATH" } else { $Root }
$env:PATH = "$Root\tools;$env:PATH"

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "找不到主环境：$Py`n请先运行 .\scripts\setup-win.ps1"
}

& $Py -m pipeline @Args
exit $LASTEXITCODE
