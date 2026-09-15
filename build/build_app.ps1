# 慧眼识灾 · 一键打包脚本（Windows）
# 用法：
#   powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile lite      # 精简版（无 torch，~350MB）
#   powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile full      # 完整版（含 U-Net，~900MB）
#   powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile lite -Console   # 带控制台（排错用）

param(
    [ValidateSet("lite", "full")][string]$Profile = "lite",
    [switch]$Console,
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
Write-Host "==> 工作目录：$(Get-Location)" -ForegroundColor Cyan

$env:FLOOD_PROFILE = $Profile
$env:FLOOD_CONSOLE = if ($Console) { "1" } else { "0" }

Write-Host "`n[1/4] 检查 PyInstaller" -ForegroundColor Cyan
python -c "import PyInstaller; print('PyInstaller', PyInstaller.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "未安装 PyInstaller，正在安装..." -ForegroundColor Yellow
    python -m pip install pyinstaller pyinstaller-hooks-contrib -i https://pypi.tuna.tsinghua.edu.cn/simple
}

Write-Host "`n[2/4] 生成应用图标（如缺失）" -ForegroundColor Cyan
if (-not (Test-Path "docs/assets/app_icon.ico")) { python scripts/make_icon.py }

Write-Host "`n[3/4] 打包（profile=$Profile，首次约 3~15 分钟）" -ForegroundColor Cyan
$distPath = if ($Profile -eq "full") { "dist_full" } else { "dist" }
python -m PyInstaller build/flood_eyes.spec --noconfirm --clean --distpath $distPath
if ($LASTEXITCODE -ne 0) { throw "打包失败，请查看上面的报错" }

$dist = Join-Path (Get-Location) (Join-Path $distPath "慧眼识灾")
if (-not (Test-Path $dist)) { throw "没有找到产物目录 $dist" }

# 清掉上次运行留下的输出与日志，保证分发包干净
foreach ($sub in @("outputs", "logs")) {
    $p = Join-Path $dist $sub
    if (Test-Path $p) { Remove-Item $p -Recurse -Force }
}

# 写一份使用说明到产物目录
$readme = @"
慧眼识灾 · 遥感 AI 洪水识别系统  v0.2.0
========================================

【怎么用】
  1. 双击 慧眼识灾.exe
  2. 程序会自动打开一个应用窗口（没有地址栏，看起来像原生软件）
  3. 页签① 选一个示例样本 -> 点"开始识别"
  4. 页签② 做灾前/灾后对比
  5. 结果和 PDF 简报保存在 outputs 文件夹

【包含什么】
  · 6 景合成演示影像（含逐像元真值，用于精度自检）
  · 2 组真实 Sentinel-2 灾前/灾后影像（鄱阳湖 2020、涿州 2023）
  · NDWI + Otsu 基线模型（零依赖，秒级出结果）
  · 中文 PDF 简报导出

【常见问题】
  Q: 双击没反应？
  A: 看 logs\desktop.log；或改用带控制台的版本重新打包（-Console）。
  Q: 端口被占用？
  A: 程序会自动从 7860 往后找可用端口，不影响使用。
  Q: 想用自己的影像？
  A: 页签①直接上传 .tif / .png / .npy（4 波段顺序：蓝、绿、红、近红外）。
  Q: 想用 U-Net 深度模型？
  A: 精简版不含 PyTorch，请使用完整版；并把训练好的权重放到 weights 文件夹。

【数据来源】
  Sentinel-2 L2A（ESA/Copernicus，AWS 公开 COG，Element84 Earth Search STAC）
  许可：Copernicus Sentinel Data Terms and Conditions

【诚实声明】
  本软件为演示版本。合成数据上的精度指标不代表真实洪灾精度；
  真实精度需用 Sen1Floods11 等公开数据集训练后重新评测。
"@
Set-Content -Path (Join-Path $dist "使用说明.txt") -Value $readme -Encoding UTF8

Write-Host "`n[4/4] 打包 ZIP" -ForegroundColor Cyan
$version = "0.2.0"
$zipName = "慧眼识灾_v${version}_$Profile.zip"
$zipPath = Join-Path (Get-Location) (Join-Path $distPath $zipName)
if (-not $NoZip) {
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path $dist -DestinationPath $zipPath -CompressionLevel Optimal
}

$exe = Join-Path $dist "慧眼识灾.exe"
$size = [math]::Round((Get-ChildItem $dist -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "`n==> 完成！" -ForegroundColor Green
Write-Host "   exe   : $exe"
if (Test-Path $zipPath) { Write-Host "   分发包: $zipPath（$([math]::Round((Get-Item $zipPath).Length/1MB,1)) MB）" }
Write-Host "   解压后体积: $size MB"
