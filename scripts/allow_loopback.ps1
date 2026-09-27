# 慧眼识灾 · 为本程序放行 Windows 入站连接（本机界面需要）
# ============================================================
# 为什么需要这一步
#   回环超时可能来自防火墙、安全软件或其他进程网络策略，不能仅凭超时确定来源。
#   仅在确认 Windows 防火墙拦截后使用；本脚本不保证能解决其他组件的限制。
#   Python 在 Windows 上用一对回环 TCP 连接模拟 socket.socketpair()，而该调用
#   没有超时；连接被丢弃会让 asyncio 在建事件循环时永久阻塞，界面永远起不来。
#
# 用法
#   方式一：右键本文件 -> 以管理员身份运行（会自行提权）
#   方式二：在管理员 PowerShell 里执行
#             powershell -ExecutionPolicy Bypass -File allow_loopback.ps1
#   指定程序路径：
#             ... -File allow_loopback.ps1 -ExePath "D:\某个目录\慧眼识灾.exe"
#   撤销放行：
#             ... -File allow_loopback.ps1 -Remove
#
# 安全性
#   规则限定指定程序、入站 TCP、本地与远端 IPv4 回环地址（127.0.0.1）。
#   -DryRun 只显示计划，不提权、不修改规则。

param(
    [string]$ExePath,
    [string]$RuleName = "慧眼识灾 本机界面",
    [switch]$Remove,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ---------- 1. 定位程序 ----------
if (-not $ExePath -and -not $Remove) {
    # 打包版布局：<安装目录>\_internal\scripts\allow_loopback.ps1
    # 源码版布局：<仓库>\scripts\allow_loopback.ps1
    $candidates = @(
        (Join-Path (Split-Path $PSScriptRoot -Parent) "慧眼识灾.exe"),
        (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "慧眼识灾.exe"),
        (Join-Path (Split-Path $PSScriptRoot -Parent) "dist\慧眼识灾\慧眼识灾.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { $ExePath = $candidate; break }
    }
}
if (-not $Remove -and (-not $ExePath -or -not (Test-Path -LiteralPath $ExePath -PathType Leaf))) {
    throw "找不到 慧眼识灾.exe，请用 -ExePath 指定完整路径。"
}
if ($ExePath -and -not $Remove) { $ExePath = (Resolve-Path -LiteralPath $ExePath).Path }

# 无副作用预览，用于检查自动定位和规则范围；不请求管理员权限。
if ($DryRun) {
    [ordered]@{ Program=$ExePath; RuleName=$RuleName; Remove=[bool]$Remove;
        Protocol="TCP"; Direction="Inbound"; LocalAddress="127.0.0.1";
        RemoteAddress="127.0.0.1" } | ConvertTo-Json
    exit 0
}

# ---------- 0. 自提权 ----------
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "需要管理员权限，正在请求提权..." -ForegroundColor Yellow
    $forward = @("-ExecutionPolicy", "Bypass", "-NoProfile", "-File", "`"$PSCommandPath`"",
                 "-RuleName", "`"$RuleName`"")
    if ($ExePath) { $forward += @("-ExePath", "`"$ExePath`"") }
    if ($Remove) { $forward += "-Remove" }
    Start-Process -FilePath "powershell.exe" -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList $forward | ForEach-Object { exit $_.ExitCode }
    exit 0
}



# ---------- 2. 撤销 ----------
if ($Remove) {
    $existing = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    if (-not $existing) {
        Write-Host "没有名为「$RuleName」的规则，无需撤销。" -ForegroundColor Green
        exit 0
    }
    $existing | Remove-NetFirewallRule
    Write-Host "已撤销规则「$RuleName」。" -ForegroundColor Green
    exit 0
}

# ---------- 3. 放行（幂等：先删同名旧规则，避免堆积）----------
$stale = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
if ($stale) { $stale | Remove-NetFirewallRule }

New-NetFirewallRule `
    -DisplayName $RuleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -Program $ExePath `
    -LocalAddress 127.0.0.1 `
    -RemoteAddress 127.0.0.1 `
    -Profile Any `
    -Enabled True | Out-Null

$rule = Get-NetFirewallRule -DisplayName $RuleName
$app = $rule | Get-NetFirewallApplicationFilter

Write-Host ""
Write-Host "已放行入站连接：" -ForegroundColor Green
Write-Host ("  规则名：{0}" -f $rule.DisplayName)
Write-Host ("  程序  ：{0}" -f $app.Program)
Write-Host ("  方向  ：{0} / 协议 TCP / 配置文件 {1}" -f $rule.Direction, $rule.Profile)
Write-Host ""
Write-Host "现在可以双击 慧眼识灾.exe 启动界面了。" -ForegroundColor Cyan
Write-Host "如需撤销：本脚本加 -Remove 再运行一次。" -ForegroundColor DarkGray
