<div align="center">

<img src="docs/assets/logo.png" alt="慧眼识灾标志" width="96">

# 慧眼识灾 · v0.5.0

**遥感洪水识别、光谱监测与 GIS 成果导出**

从地图选点或本地影像出发，完成灾前 / 灾后分析，查看水体变化、有效观测范围与处理依据，并导出 GeoTIFF、JSON 和 PDF 简报。

**当前版本：v0.5.0 · 2026-09-27 最终交付**

[下载安装包](https://github.com/kurtJaz0529/flood-eyes/releases/latest) · [功能概览](#功能概览) · [源码运行](#源码运行) · [验收与模型评测](#验收与模型评测) · [文档](#文档)

</div>

## 安装与使用

Windows 10/11 x64 安装包已内置 Python。两版均支持 CPU 运行，本地影像处理可离线使用；在线地图、卫星检索和影像下载需要可达的互联网。

| 版本 | 下载 | 大小 | 包含内容 |
|---|---|---|---|
| **完整版（推荐）** | [HuiYanShiZai-Setup-v0.5.0-Full.exe](https://github.com/kurtJaz0529/flood-eyes/releases/download/v0.5.0/HuiYanShiZai-Setup-v0.5.0-Full.exe) | 247.6 MiB | 基线、场景适配、光谱监测、批量任务，以及 PyTorch CPU 和实验 TinyUNet 权重 |
| **精简版** | [HuiYanShiZai-Setup-v0.5.0-Lite.exe](https://github.com/kurtJaz0529/flood-eyes/releases/download/v0.5.0/HuiYanShiZai-Setup-v0.5.0-Lite.exe) | 143.9 MiB | 基线、场景适配、光谱监测和批量任务，不含 PyTorch 与深度模型权重 |

[最新版发布页](https://github.com/kurtJaz0529/flood-eyes/releases/latest) · [v0.5.0 发布说明](https://github.com/kurtJaz0529/flood-eyes/releases/tag/v0.5.0) · [SHA256 校验文件](https://github.com/kurtJaz0529/flood-eyes/releases/download/v0.5.0/SHA256SUMS.txt)

1. 下载所需版本的 EXE，双击安装，再从开始菜单或桌面启动“慧眼识灾”。
2. 在地图上选点并点击“使用此地点”，填写灾前、灾后时间窗口，选择地貌与识别策略，点击“开始分析”；也可展开“本地多光谱影像（可选）”上传两期影像。
3. 查看质量提示、有效范围和变化结果，下载成果包。完整版还可展开“U-Net 实验模型 · 本地影像识别”，显式选择“U-Net 深度模型”，进行单景或双时相处理。

完整版内置模型无需另找权重；实验权重不参与自动选模，地图一键流程使用基线或场景适配。需要查看深度模型结果时，请使用独立的 U-Net 入口。

正常安装无需管理员权限。若日志提示本机连接受到 Windows 策略阻断，可在开始菜单运行“修复本机连接（需管理员确认）”；规则仅针对该程序的 `127.0.0.1` TCP 连接，安装路径改变后需重新应用。详细记录见[问题修复与最终交付](docs/问题修复与完整版交付_20260927.md)。

成果默认写入程序可写目录下的 `outputs/`，日志位于 `logs/`；可通过 `--data-dir` 或 `FLOOD_DATA_DIR` 指定数据目录。卸载保留用户成果和日志。

下载后可在 PowerShell 中核对文件哈希：

```powershell
Get-FileHash .\HuiYanShiZai-Setup-v0.5.0-Full.exe -Algorithm SHA256
```

## 功能概览

| 功能 | 当前实现 | 使用边界 |
|---|---|---|
| 在线灾前 / 灾后分析 | 按坐标和日期检索 Sentinel-2，下载窗口影像并生成变化成果 | 当前为点位窗口；受数据可用性、云量和网络影响 |
| 光学基线 | NDWI、Otsu 阈值、近红外闸门及形态学处理 | 无需模型权重；提取水体仍需结合实际地物核查 |
| 场景适配 | 平原、丘陵、山地、城市、海岸、湿地稻田、干旱裸地七类实验配方，可使用 DEM 和 AWEI 辅助复核 | 地貌由用户声明；经验参数尚需区域标定 |
| U-Net 实验模型 | 完整版内置 TinyUNet，提供单景识别与灾前 / 灾后对比 | 使用小型真实标注试验集训练，需显式选择 |
| 多时相光谱监测 | NDVI、SAVI、NDWI、MNDWI、NDMI、NBR，逐期指数、相邻差值和共同有效区 | 使用本地影像；SWIR 指数必须提供真实 B11/B12 波段 |
| 批量任务 | CSV/JSON 导入、逐行校验、串行队列、重试、取消、显式恢复与成果下载 | 与在线一键流程共用处理链；恢复和缓存会核对请求及输入身份 |
| GIS 与报告 | 水体、变化、有效区和复核栅格，叠加图、统计 JSON、中文 PDF 与 ZIP 成果包 | 面积和变化解释依赖配准、投影、质量掩膜与共同有效范围 |
| Sentinel-1 SAR 工具 | 源码命令行提供 SAFE 标定、GCP 仿射定位、双时相对齐及水体变化提取 | 独立处理链；尚未与地图在线选景自动融合 |

本版同时修复了桌面启动的代理绕过、本机连接诊断、Windows 在线 COG 读取、训练标签与 NoData 处理，以及安装 / 卸载验收隔离。在线下载采用保留 TLS 证书验证的 Python HTTPS Range 读取方式。

### 处理演示

![鄱阳湖真实影像处理演示](docs/assets/demo_real_poyang.gif)

历史 Sentinel-2 L2A 灾前 / 灾后影像演示：鄱阳湖，2020-05-19 与 2020-07-15。动图展示处理流程；其中的面积和旧界面不作为当前版本精度验收结果。

### 输入与成果口径

- 光学洪水识别需要真实绿光、近红外及正确的反射率尺度。在线下载为 B2/B3/B4/B8 四波段；MNDWI 还需 SWIR1，NDMI/NBR 分别需要 SWIR1/SWIR2。可使用带正确波段描述的 GeoTIFF，或显式声明 `s2_6band` 等预设。
- 两期影像需要对齐到共同网格；云、云影、NoData 和待复核像元不应当作“无水”。变化面积只在共同有效区域统计，观测不足会明确报告。
- 保留输入影像旁的 SCL 质量侧车。网页上传不会自动携带同名侧车，需要完整质量层时使用本地批量或 API。
- 场景适配的面积计算要求米制投影。SWIR 从 20 m 重采样到 10 m 不增加真实空间细节。
- 新增水面还可能来自潮位、灌溉或季节变化，需结合日期与现场信息判断洪灾；NBR 相邻差值采用“后减前”，不能直接套用传统 dNBR 分级。

| 栅格 | 主要编码 |
|---|---|
| `water_mask.tif` | `0` 有效非水、`1` 水体、`255` 未判定 |
| `change.tif` | `0` 稳定非水、`1` 持续水体、`2` 新增水体、`3` 退水、`255` 未判定 |
| `valid_mask.tif` | `1` 参与判定、`0` 无效或需复核；`0` 是真实类别 |
| `review.tif` / `before_review.tif` | 场景适配的辅助复核原因，编码详见使用说明 |

详细规则见[场景适配与遥感扩展](docs/场景适配与遥感扩展使用说明.md)和[光谱监测说明](docs/光谱时序监测使用说明.md)。

## 源码运行

在项目目录使用独立 Python 环境安装依赖。下面为 Windows PowerShell 示例：

```powershell
git clone https://github.com/kurtJaz0529/flood-eyes.git
cd flood-eyes
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app/desktop.py
```

界面默认使用 `http://127.0.0.1:7860/`。只启动服务时可加 `--headless`。依赖清单包含 PyTorch，首次安装需要下载依赖；后续示例中的 `python` 均指已安装项目依赖的解释器。

仓库包含 `data/samples/` 合成演示影像。真实影像缓存、训练数据、模型权重和打包产物不随源码克隆；源码在没有权重时可运行基线，已发布的完整版安装包另行内置实验权重。合成样本用于检查流程，不支持真实灾害精度结论。

### 获取真实影像

```powershell
python scripts/fetch_real_samples.py --list
python scripts/fetch_real_samples.py --event poyang2020 --size 512
```

脚本保留数据来源与质量信息。抓取成功后，可通过 Python API 处理本地影像：

```python
from src import FloodDetector

result = FloodDetector(mode="baseline").detect("data/real/poyang2020_post.tif")
print(result.summary_text())
```

公开卫星数据通道无需下载账号或密钥，但需网络可达；日期范围内无合格影像时，应调整地点、时间或数据来源。

### 批量与光谱监测

```powershell
# jobs.csv / jobs.json 的字段和示例见自动化处理使用说明
python scripts/run_batch.py jobs.csv --run --out-dir outputs/automation
python scripts/run_batch.py --list --out-dir outputs/automation

# 仓库合成样本：只检查多时相光谱处理流程
python scripts/run_spectral.py --index ndvi --images data/samples/demo01_pre.tif data/samples/demo01_post.tif
```

批量任务至少提供坐标和灾前 / 灾后时间窗口；填入 `local_pre`、`local_post` 可改用本地影像。界面与命令行共用任务记录时，应指向相同的 `outputs/automation` 目录。光谱监测独立于该任务库。完整说明见[自动化处理](docs/自动化处理使用说明.md)。

### SAR 命令行

先核对数据类型，再使用实际的 Sentinel-1 SAFE 目录：

```powershell
python scripts/check_data.py "D:\satellite"
python scripts/prepare_s1_sar.py --pre "D:\satellite\before.SAFE" --post "D:\satellite\after.SAFE" --pol VV --out data/sar_demo
```

SAR 使用回波标定与变化判据，输入不可作为光学反射率代入 NDWI。数据准备见[数据使用指南](docs/数据使用指南.md)。

## 验收与模型评测

### v0.5.0 最终交付记录

以下为 **2026-09-27 已完成的发布验收**，来源为[最终交付记录](docs/问题修复与完整版交付_20260927.md)。安装包上传时已核对其字节数和 SHA256 与该次交付清单一致。

| 检查项目 | 结果 |
|---|---|
| 代码测试 | 366 通过、5 跳过 |
| 完整版 EXE：真实在线、U-Net 与界面功能 | 58 / 58 |
| 精简版 EXE：真实在线与界面功能 | 53 / 53 |
| 两版冻结离线命令 | 各 46 / 46 |
| 两版隔离安装、卸载与成果保留 | 各 18 / 18 |

两版均完成 512×512 真实 Sentinel 灾前 / 灾后下载及 PDF/GIS 导出。以上记录说明已检验相应工程流程，区域识别精度需要独立标注评测。

### 完整版内置实验模型

内置模型为 **TinyUNet**，使用 Sen1Floods11 的 Sentinel-2 **L1C TOA** 四波段 B2/B3/B4/B8 数据；保留真实 `-1/0/1` 标签，未知与 NoData 不参与损失和统计。

| 项目 | 记录 |
|---|---|
| 试验规模 | 11 个事件、44 幅影像 |
| 训练集 | 36 幅 |
| 验证集 | 西班牙 4 幅完整影像 |
| 留出测试集 | 玻利维亚 4 幅完整影像，不参与训练与模型选择 |
| 训练 / 选模 | CPU，80 轮；第 26 轮验证 IoU 最佳，为 0.7615 |
| 留出测试 | IoU **0.8370**、F1 **0.9113**、Precision **0.8615**、Recall **0.9671** |
| 分类阈值 | 固定 0.5，无测试集调参 |

这些指标仅适用于该小型留出事件。训练 L1C 与在线 L2A 数据存在产品差异，不能据此承诺中国地区或其他目标区域精度。合成样本上的旧训练曲线也不代表该模型的真实评测结果。

### 自行训练与评估

下例展示数据准备、按事件划分训练与独立测试的方法，不保证重现已发布权重的分数。下载目录需为空；若原始影像和标签网格不一致，准备脚本会拒绝处理，需要先核查和正确配准。

```powershell
python scripts/prepare_sen1floods.py --out data/sen1floods11/pilot --chips-per-event 4

python train.py --data data/sen1floods11/pilot --arch tiny --epochs 80 --img-size 256 --val-size 512 --val-groups Spain --test-groups Bolivia --device cpu --out weights/pilot_custom.pt --log-dir logs/pilot_custom --data-note "Sen1Floods11 L1C TOA experimental pilot"

python scripts/evaluate_checkpoint.py --weights weights/pilot_custom.pt --data data/sen1floods11/pilot --groups Bolivia --out outputs/pilot_custom_metrics.json
```

训练的 `--data` 指向这一批影像目录，避免把多个副本一起递归纳入。独立真值应保持正确 CRS、网格、标签与事件隔离；原基线和场景适配还可通过 `scripts/evaluate_flood.py` 对同网格的预测 / 真值栅格评测，同时报告覆盖率。

## 构建与检查

在安装依赖的 Windows 环境执行。完整版构建会收集 `weights/*.pt`；源码仓库不附带这些文件，自行构建需准备兼容权重。已发布的完整版安装包已包含试验权重。

```powershell
# 精简版：生成应用目录、便携 ZIP，再编译安装器
powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile lite
powershell -ExecutionPolicy Bypass -File build/build_installer.ps1 -Edition lite

# 完整版
powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile full
powershell -ExecutionPolicy Bypass -File build/build_installer.ps1 -Edition full
```

应用及便携 ZIP 分别写入 `dist/`、`dist_full/`，安装器写入 `dist_installer/`。GitHub Release 当前提供两版安装器与校验文件。编译脚本可安装所需的 PyInstaller / Inno Setup 工具，首次构建需要联网。

开发者检查入口：

```powershell
python -m pip install pytest
python -m pytest -q
node tests/test_map_coordinates.js
python scripts/verify_release.py --source --report outputs/source_ui.json
python scripts/verify_frozen_cli.py --exe "dist/慧眼识灾/慧眼识灾.exe" --report outputs/frozen_cli.json
```

地图坐标检查需要 Node.js。冻结界面、在线任务及隔离安装的完整验收参数见各脚本的 `--help` 和[最终交付记录](docs/问题修复与完整版交付_20260927.md)。

## 项目结构

```text
app/           地图界面、桌面启动器、批量任务与光谱监测入口
src/           波段与质量处理、识别、配准、任务队列、GIS/PDF 导出
scripts/       数据准备、批处理、SAR、模型评估与发布验收工具
build/         PyInstaller 与 Inno Setup 构建配置
data/samples/  合成演示影像与真值
tests/         质量、配准、面积、任务恢复、下载及安装相关回归
docs/          使用说明、验收记录、演示素材
train.py       支持真实标签和事件划分的 U-Net 训练入口
```

`weights/`、真实数据、`outputs/`、运行日志及构建产物按 `.gitignore` 排除。协作与遥感质量审查要求见 [AGENTS.md](AGENTS.md)。

## 文档

| 内容 | 入口 |
|---|---|
| 本版修复、模型说明与最终验收 | [问题修复与完整版交付](docs/问题修复与完整版交付_20260927.md) |
| 批量请求、队列与成果 | [自动化处理使用说明](docs/自动化处理使用说明.md) |
| 地貌配方、SWIR、DEM 与质量口径 | [场景适配与遥感扩展](docs/场景适配与遥感扩展使用说明.md) |
| 多时相指数、差值与数据要求 | [光谱时序监测](docs/光谱时序监测使用说明.md) |
| 数据类型与准备工具 | [数据使用指南](docs/数据使用指南.md) |
| 历史变更与版本库背景 | [CHANGELOG](CHANGELOG.md) · [版本库说明](docs/版本库说明_20260923.md) |
| 演示与答辩素材 | [路演脚本](docs/demo_script.md) · [PPT 大纲](docs/pitch_outline.md) |

早期验收文档与演示素材保留原阶段记录；最新交付状态以本页和 2026-09-27 最终交付记录为准。

## 后续工作

- 使用目标地区、相同产品级别的真实影像与人工标注，开展独立事件验证及分地貌误差分析。
- 扩展区域多边形、跨景拼接和长时间序列处理。
- 完善 SAR 地形与质量处理，再评估光学 / SAR 协同流程。

## 数据与许可

合成演示样本由项目脚本生成。真实光学影像来自 Copernicus Sentinel-2 的公开数据服务，训练试验使用 [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)；数据来源与哈希记录在下载或准备流程生成的清单中，许可和引用方式以原始数据提供方的说明为准。

代码仓库当前尚未提供 `LICENSE` 文件。
