# 桌面应用构建与验收（v0.5.0）

本次交付 Windows 10/11 x64 精简版，内置 Python。功能包括七类地貌实验配方、NDWI 基线、六种光谱监测、批处理、独立标注评估和 PDF/GIS 导出。无 PyTorch、训练权重及真实影像缓存；完整版本轮未构建。

操作说明见 [安装与离线运行](安装与离线运行.md)，算法边界见 [场景适配与遥感扩展使用说明](场景适配与遥感扩展使用说明.md)。

## 当前验证边界

源码回归 291 通过、7 跳过（缺权重或真实影像），地图坐标测试通过；源码图形服务的 51 项端到端检查通过。检查覆盖七种场景、GIS/PDF 输出、未知区语义、六种指数、独立队列、评估入口。

**冻结 EXE 的回环超时尚未确定具体拦截组件。** 改名副本与原程序的差异支持按进程过滤的可能性；不能仅据默认防火墙策略认定原因。启动器给出诊断和仅限回环的候选修复命令。随包脚本支持 `-DryRun` 无副作用预览；实际修改规则需要管理员权限。


旧版本的 10/10 验收记录、真实区域面积、启动速度和完整版体积不能替代本版本验收，已从本说明移除。合成测试不证明真实洪灾精度。

## 构建

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile lite
powershell -ExecutionPolicy Bypass -File build/build_installer.ps1 -Edition lite
```

产物是 `dist/慧眼识灾/`、`dist/慧眼识灾_v0.5.0_lite.zip` 和 `dist_installer/慧眼识灾_安装程序_v0.5.0_精简版.exe`。精确大小和 SHA256 以本轮发布清单为准。

`-Profile full` 构建包含 PyTorch 的版本，需要自行提供训练权重。本轮没有验证该版本。`-Console` 可生成带控制台的排错程序；`-NoZip` 跳过 ZIP。

ZIP 和安装器均排除运行产生的 `outputs`、`logs` 及 Python 缓存；ZIP 完整生成后才替换旧文件。不要把用户数据目录当作构建目录。

## 验收命令

```powershell
# 源码界面对照
python scripts/verify_release.py --source --report outputs/release_v0.5.0/source_ui.json
# 冻结界面：本机连接必须可用
python scripts/verify_packaged_app.py --exe "dist/慧眼识灾/慧眼识灾.exe" --report outputs/release_v0.5.0/frozen_ui.json
# 冻结离线计算
python scripts/verify_frozen_cli.py --exe "dist/慧眼识灾/慧眼识灾.exe" --report outputs/release_v0.5.0/frozen_cli.json
```

界面验收使用临时数据目录，独立队列与 API 超时，结束后关闭进程并保存报告。旧脚本入口已转发到新验收器，不再请求已移除的单时相/对比接口。

安装测试由 `scripts/verify_installer.py` 自己编译一份验收专用安装包：编译期用 `/DAppIdValue=` 指定独立卸载登记，用 `/DGroupName=` 指定独立开始菜单组。**注意 `DisableProgramGroupPage=yes` 时 Inno Setup 会忽略运行期的 `/GROUP=` 参数**，只能用编译期宏；早期版本依赖 `/GROUP=`，导致验收安装写进了生产开始菜单组。现在脚本会断言 smoke 组名不等于生产组名，并新增 `production_group_untouched` / `production_group_still_untouched` 两项检查，防止该问题复发。卸载只删除安装清单内的程序文件，保留测试哨兵成果和日志后方可通过。

## 路径和依赖

完整解压绿色包，保持 `慧眼识灾.exe` 与 `_internal` 相邻。默认成果写到程序旁的 `outputs`；不可写时退到 `%LOCALAPPDATA%/HuiYanShiZai`。`--data-dir` 可指定独立可写根目录。

打包保留 GDAL/PROJ 数据、rasterio DLL、Gradio 静态资源、中文报告字体配置与项目动态导入。精简版排除 PyTorch、视频编解码、SciPy/skimage，使用已验证的 OpenCV/NumPy 路径。

## 排错

| 现象 | 检查方法 |
|---|---|
| 本机连接失败 | 运行 `--diagnose 新文件.json`；报告的 `cause=loopback_timeout` 表示回环超时，确认防火墙拦截后再考虑报告中的命令，或用 `_internal\scripts\allow_loopback.ps1` |
| 60 秒未启动 | `logs/desktop.log` 会记录线程调用栈，不据此推断具体根因 |
| 端口占用 | 默认从 7860 自动寻找空闲端口 |
| 输出位置不明确 | 查看日志中的工作目录，或显式使用 `--data-dir` |
| SWIR 指数被拒绝 | 检查真实 B11/B12 与所选波段顺序，不能用可见光替代 |
| 权重未加载 | 精简版不含 PyTorch；完整版还需兼容的训练权重 |
| 在线下载报 `schannel: the revocation status is unknown` | 证书吊销状态无法查询（受限网络访问不到 CRL/OCSP）。日志会给出归因；确认网络可信时可显式设 `FLOOD_ALLOW_UNSAFE_TLS=1` 降级（会记入成果溯源 `tls_verification_disabled`） |

本地处理可离线；底图、在线检索和卫星下载需要网络。安装包未签名；发布状态和剩余限制见本轮验收记录。
