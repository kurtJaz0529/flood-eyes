# 慧眼识灾 · 一键编译 Windows 安装包
# ===================================
# 用法：
#   powershell -ExecutionPolicy Bypass -File build/build_installer.ps1                 # 精简版 + 完整版
#   powershell -ExecutionPolicy Bypass -File build/build_installer.ps1 -Edition lite   # 只编精简版
#   powershell -ExecutionPolicy Bypass -File build/build_installer.ps1 -Edition full   # 只编完整版
#
# 前置条件：
#   1. 已经用 build/build_app.ps1 打过包（dist/ 或 dist_full/ 存在）
#   2. 没装 Inno Setup 也没关系，脚本会自动下载并静默安装到用户目录

param(
    [ValidateSet("lite", "full", "both")][string]$Edition = "both"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
Write-Host "==> 工作目录：$(Get-Location)" -ForegroundColor Cyan

# ---------- 1. 找/装 Inno Setup 编译器 ----------
function Find-Iscc {
    $candidates = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "$env:USERPROFILE\InnoSetup6\ISCC.exe",
        "$env:USERPROFILE\InnoSetup7\ISCC.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

$iscc = Find-Iscc
if (-not $iscc) {
    Write-Host "`n[1/3] 未找到 Inno Setup，自动下载安装（约 10 MB）..." -ForegroundColor Yellow
    $dl = Join-Path $env:TEMP "innosetup-6.7.3.exe"
    $url = "https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe"
    Write-Host "      下载 $url"
    try {
        Invoke-WebRequest -Uri $url -OutFile $dl -UseBasicParsing
    } catch {
        throw "下载 Inno Setup 失败：$_`n请手动安装：https://jrsoftware.org/isdl.php"
    }
    $dir = Join-Path $env:USERPROFILE "InnoSetup6"
    Write-Host "      静默安装到 $dir"
    Start-Process -FilePath $dl -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART","/SP-","/DIR=$dir" -Wait
    $iscc = Join-Path $dir "ISCC.exe"
}
if (-not (Test-Path $iscc)) { throw "找不到 ISCC.exe，请手动安装 Inno Setup" }
Write-Host "`n[1/3] 编译器：$iscc" -ForegroundColor Green

# ---------- 2. 检查待打包的应用是否已构建 ----------
$targets = @()
if ($Edition -eq "lite" -or $Edition -eq "both") { $targets += "lite" }
if ($Edition -eq "full" -or $Edition -eq "both") { $targets += "full" }

foreach ($t in $targets) {
    $distPath = if ($t -eq "full") { "dist_full\慧眼识灾" } else { "dist\慧眼识灾" }
    if (-not (Test-Path "$distPath\慧眼识灾.exe")) {
        Write-Host "`n[2/3] $t 版应用尚未打包，先执行 build_app.ps1 -Profile $t" -ForegroundColor Yellow
        powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile $t -NoZip
        if ($LASTEXITCODE -ne 0) { throw "$t 版应用打包失败" }
    }
}
Write-Host "`n[2/3] 应用产物检查通过" -ForegroundColor Green

# ---------- 3. 编译安装包 ----------
Write-Host "`n[3/3] 编译安装包（lzma2 最高压缩，每档约 1~4 分钟）" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path dist_installer | Out-Null
foreach ($t in $targets) {
    $t0 = Get-Date
    Write-Host "  -> $t ..."
    & $iscc "/DProfile=$t" "build\installer.iss" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "$t 版安装包编译失败" }
    Write-Host ("     完成，用时 {0:N0}s" -f ((Get-Date) - $t0).TotalSeconds)
}

Write-Host "`n==> 安装包已生成：" -ForegroundColor Green
Get-ChildItem dist_installer\*.exe | Sort-Object Name | ForEach-Object {
    Write-Host ("   {0}  ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB))
}
Write-Host "`n分发提示："
Write-Host "  · 只需把这一个 setup.exe 发给别人，双击安装即可，无需 Python、无需联网"
Write-Host "  · 首次运行 Windows SmartScreen 可能提示'未知发布者'，点'更多信息 -> 仍要运行'"
Write-Host "  · 安装到 %LOCALAPPDATA%\Programs\慧眼识灾，无需管理员权限"
