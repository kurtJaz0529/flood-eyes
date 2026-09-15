# 慧眼识灾 · 一键快速开始（Windows PowerShell）
# 用法： powershell -ExecutionPolicy Bypass -File scripts/quickstart.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
Write-Host "==> 工作目录：$(Get-Location)" -ForegroundColor Cyan

Write-Host "`n[1/4] 环境自检" -ForegroundColor Cyan
python scripts/check_env.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n依赖不完整，先执行：pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n[2/4] 生成演示样本（已存在则跳过）" -ForegroundColor Cyan
if (Test-Path "data/samples/samples.json") {
    Write-Host "  已存在 data/samples/samples.json，跳过"
} else {
    python data/make_samples.py --n 6 --size 640
}

Write-Host "`n[3/4] 全流程自检" -ForegroundColor Cyan
python tests/test_pipeline.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n自检未通过，请查看上面的报错" -ForegroundColor Red
    exit 1
}

Write-Host "`n[4/4] 完成" -ForegroundColor Green
Write-Host "  启动界面：  python app/main.py        然后打开 http://127.0.0.1:7860"
Write-Host "  训练模型：  python train.py --data data/samples --epochs 20 --arch tiny"
Write-Host "  生成动图：  python scripts/make_demo_gif.py"
Write-Host "  真实数据：  python scripts/download_data.py --dataset sen1floods11"
