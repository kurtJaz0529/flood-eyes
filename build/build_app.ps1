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
    # 走官方 PyPI 并锁定版本：第三方镜像 + 不锁版本的组合，
    # 一旦镜像被投毒就会直接污染最终分发包（PyInstaller 会注入引导代码）。
    $pin = if ($env:PYINSTALLER_VERSION) { $env:PYINSTALLER_VERSION } else { "6.11.1" }
    python -m pip install "pyinstaller==$pin" "pyinstaller-hooks-contrib>=2024.10" --index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 安装失败，请手动执行：python -m pip install pyinstaller==$pin" }
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
    if (Test-Path -LiteralPath $p) {
        $resolvedOutput = (Resolve-Path -LiteralPath $p).Path
        $resolvedDist = (Resolve-Path -LiteralPath $dist).Path
        if (-not $resolvedOutput.StartsWith($resolvedDist + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "清理路径超出构建产物目录" }
        if ((Get-Item -LiteralPath $p).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "不清理链接目录：$p" }
        Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
    }
}

# 写一份使用说明到产物目录
$readme = @"
慧眼识灾 · 遥感自动化与场景适配洪水识别  v0.5.0
========================================

【怎么用】
  1. 双击 慧眼识灾.exe
  2. 程序会自动打开一个应用窗口（没有地址栏，看起来像原生软件）
  3. 选择地貌、识别策略和日期，可上传本地灾前/灾后 GeoTIFF 与 DEM
  4. 展开“多时相遥感监测”使用 NDVI/SAVI/NDWI/MNDWI/NDMI/NBR
  5. 批量区可加入合成演示任务；结果保存在 outputs 文件夹

【包含什么】
  · 6 景合成演示影像（含逐像元真值，用于精度自检）
  · 七类地貌的实验场景适配、NDWI 基线与 GIS 复核图层
  · 六种本地光谱监测；SWIR 指数需要真实短波红外波段
  · 本安装包不包含真实影像缓存与深度学习权重
  · 中文 PDF 简报导出

【常见问题】
  Q: 双击没反应？
  A: 看 logs\desktop.log；或改用带控制台的版本重新打包（-Console）。
  Q: 端口被占用？
  A: 程序会自动从 7860 往后找可用端口，不影响使用。
  Q: 想用自己的影像？
  A: 在本地多光谱区上传 GeoTIFF。六波段预设为 B2/B3/B4/B8/B11/B12。
  Q: 想用 U-Net 深度模型？
  A: 精简版不含 PyTorch，请使用完整版；并把训练好的权重放到 weights 文件夹。

【数据来源】
  Sentinel-2 L2A（ESA/Copernicus，AWS 公开 COG，Element84 Earth Search STAC）
  许可：Copernicus Sentinel Data Terms and Conditions

【诚实声明】
  本软件为演示版本。合成数据上的精度指标不代表真实洪灾精度；
  真实精度需按地区和事件用独立标注重新评测；默认场景参数未经区域标定。
  无需安装 Python；本地处理可离线，在线地图和卫星下载需要网络。
  卸载保留用户成果和日志。详情见 _internal\docs 下的使用说明。
"@
Set-Content -Path (Join-Path $dist "使用说明.txt") -Value $readme -Encoding UTF8

Write-Host "`n[4/4] 打包 ZIP" -ForegroundColor Cyan
$version = "0.5.0"
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
