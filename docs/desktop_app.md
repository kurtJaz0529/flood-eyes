# 桌面应用（Windows 应用软件）· 构建与分发

> 目标：评委/应急部门拿到一个 **双击就能用** 的软件，不需要装 Python、不需要配环境、不需要联网。

---

## 一、三种用法怎么选

| 方式 | 体积 | 适用 |
|---|---|---|
| **安装包 setup.exe**（推荐给别的电脑） | 174 / 352 MB | 发给评委、老师、应急部门：双击安装 → 开始菜单点开即用，无需管理员、无需 Python |
| 绿色版 zip（解压即用） | 229 / 401 MB | 不想安装、U 盘直接跑 |
| 源码运行 | — | 二次开发、改算法 |

---

## 二、两个版本怎么选

| | 精简版（推荐分发） | 完整版 |
|---|---|---|
| 包含 PyTorch / U-Net | ❌ | ✅ |
| 识别模型 | NDWI + Otsu 基线 + 近红外闸门 | 基线 + U-Net 语义分割 |
| 解压后体积 | **427 MB** | **878 MB** |
| 分发包（zip） | **229 MB** | **401 MB** |
| 启动时间 | 15~25 秒 | 40~70 秒 |
| 适用场景 | 评委现场演示、U 盘分发、应急终端 | 算法组内部评测、需要深度模型时 |

> **建议**：路演现场用精简版（启动快、体积小、结果与完整版一致）；答辩时如果评委要看 U-Net，再打开完整版。
> 精简版不是"阉割版"—— 真实影像识别、双时相对比、面积统计、PDF 简报导出全都有。

## 三、绿色版怎么运行

```
慧眼识灾/                    ← 解压后整个文件夹
├── 慧眼识灾.exe             ← 双击这个
├── 使用说明.txt
└── _internal/               ← 运行时（不要动）
```

双击 `慧眼识灾.exe` 后：

1. 程序自动找一个可用端口（默认 7860，被占用就往后找）；
2. 用 Edge/Chrome 的 `--app` 模式打开一个**没有地址栏的窗口**，看起来就是原生软件；
3. 同时弹出一个小控制窗，显示服务地址和「打开界面 / 退出程序」按钮；
4. 识别结果、导出简报都保存在 exe 同目录的 `outputs/` 文件夹。

> 首次启动稍慢（要把 400+ MB 运行时加载进内存），之后窗口是秒开的。
> 如果电脑没有 Edge/Chrome，会自动退回默认浏览器打开。

## 四、自己重新打包

```powershell
cd flood-eyes

# 精简版（约 5 分钟，产物在 dist/）
powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile lite

# 完整版（约 10~15 分钟，产物在 dist_full/）
powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile full

# 排错时加控制台，能看到实时日志
powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile lite -Console
```

产物：

```
dist/慧眼识灾_v0.2.0_lite.zip      ← 直接发给评委
dist/慧眼识灾/慧眼识灾.exe          ← 本地运行
```

## 五、打包做了什么（技术细节）

| 问题 | 处理方式 |
|---|---|
| 资源路径：打包后代码在只读的 `_internal/` | `src/paths.py` 区分 **bundle_root（只读资源）** 与 **user_root（可写产物）** |
| GDAL/PROJ 找不到数据文件 | 手动收集 `rasterio/gdal_data`、`rasterio/proj_data`，运行时钩子设置 `GDAL_DATA` / `PROJ_LIB` |
| rasterio 的 delvewheel DLL 加载失败 | 保持 `rasterio/` 与 `rasterio.libs/` 的相对位置（`../rasterio.libs`），与 wheel 内补丁一致 |
| 无控制台时 stdout 为 None，uvicorn 日志配置报错 | 用实现了 `isatty/encoding/fileno` 的 Tee 对象同时写控制台与 `logs/desktop.log` |
| 体积失控（polars 176 MB、OpenCV ffmpeg 80 MB、gradio ffmpeg 30 MB） | 全部排除/过滤（都是视频、DataFrame 组件用的，本项目用不到） |
| 中文 PDF 在冻结环境缺字体 | reportlab 内置 CID 字体 `STSong-Light` + 收集 `_cidfontdata` |
| 精简版没有 torch 也要能跑 | `src/__init__.py` 把深度模型做成可选导入，`FloodDetector` 捕获 ImportError 自动降级到基线 |
| 中文 exe 名 / 图标 / 版本信息 | `docs/assets/app_icon.ico` + `build/version_info.txt`（会出现在 exe 属性里） |

## 六、验证过什么

打包后**不是"能启动"就算完**。仓库自带验收脚本，一条命令跑完：

```powershell
python scripts/verify_packaged_app.py --exe "dist/慧眼识灾/慧眼识灾.exe"
# 验收结果：10/10 项通过
```

它会自动启动 exe（headless）、调用三个接口、解压成果包校验内容。实测结果：

- [x] exe 启动、端口监听、日志写入 `logs/desktop.log`
- [x] 真实影像单时相识别：鄱阳湖 154.37 km²（与源码模式**完全一致**）
- [x] 真实影像双时相对比：涿州 1.33 → 5.26 km²，新增淹没 3.98 km²
- [x] 合成样本识别：demo04 7.63 km²
- [x] 成果包导出：掩膜 / 叠加图 / 热力图 / 对比图 / `stats.json` / **中文 PDF（STSong 字体正常）**
- [x] 打包内资源齐全：`data/samples`、`data/real`、`rasterio/gdal_data`、`rasterio/proj_data`、`rasterio.libs`
- [x] 统计面板 HTML 正常渲染（`heye-hero` 卡片存在）

## 七、常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 双击没反应 | 看 `logs/desktop.log`；或用 `-Console` 重新打包看实时输出 |
| 提示端口被占用 | 正常，程序会自动往后找端口 |
| 杀毒软件报毒 | PyInstaller 打包的 exe 常被误报，添加信任即可；正式提交可做代码签名 |
| 想换成自己的影像 | 页签①直接上传 `.tif/.png/.npy`，4 波段顺序为 **蓝、绿、红、近红外** |
| 想用自己的 U-Net 权重 | 把 `.pt` 放到 exe 同目录的 `weights/`，重启程序，下拉框选"自动" |
| 输出目录在哪 | exe 同目录 `outputs/`（若该目录不可写则退到 `%LOCALAPPDATA%\HuiYanShiZai`） |

## 八、标准安装包（推荐给别的电脑用）

一条命令生成标准 Windows 安装包：

```powershell
powershell -ExecutionPolicy Bypass -File build/build_installer.ps1
```

产物（单文件，直接发给别人）：

| 安装包 | 体积 | 内容 |
|---|---|---|
| `慧眼识灾_安装程序_v0.2.0_精简版.exe` | **174.6 MB** | 基线模型，启动快 |
| `慧眼识灾_安装程序_v0.2.0_完整版.exe` | **351.9 MB** | 额外含 PyTorch + U-Net |

**别人拿到后怎么用**：双击 setup.exe → 下一步 → 完成 → 开始菜单/桌面出现「慧眼识灾」→ 双击即用。
**无需管理员权限、无需装 Python、无需联网。**

安装包做的事：

- 装到 `%LOCALAPPDATA%\Programs\慧眼识灾`（用户级安装，不弹 UAC）
- 开始菜单：`慧眼识灾` / `使用说明` / `识别结果目录`
- 桌面快捷方式（可勾选）
- 注册「添加或删除程序」卸载项（名称 `慧眼识灾 V0.2.0（精简版）`）
- 安装完成后可选立即启动
- 卸载时自动清理 `outputs` / `logs` / `_internal`

### 实测记录（两种安装包都跑过完整流程）

| 验证项 | 精简版 | 完整版 |
|---|---|---|
| 静默安装到全新目录 | ✅ 432 MB | ✅ 955 MB |
| 开始菜单 3 个快捷方式 | ✅ | ✅ |
| 桌面快捷方式 | ✅ | ✅ |
| 卸载注册项（版本/发布者/图标） | ✅ | ✅ |
| **从安装目录运行**（脱离源码树） | ✅ 鄱阳湖 154.37 km² | ✅ U-Net 12.17 km² |
| 成果包写入 `<安装目录>\outputs\` | ✅ | ✅ |
| 卸载后目录/快捷方式全部清理 | ✅ | ✅ |

> **SmartScreen 提示**：安装包未做代码签名时，首次运行会提示「Windows 已保护你的电脑」，
> 点「更多信息 → 仍要运行」即可。正式提交/对外分发建议做代码签名（EV 证书），
> 或者在文档里附一句说明，避免评委被弹窗吓到。

## 九、软件著作权 / 软著登记材料建议

如果团队要申请软件著作权，本仓库可以直接作为材料来源：

| 材料 | 对应文件 |
|---|---|
| 软件名称/版本号 | 慧眼识灾 V0.2（`build/version_info.txt`） |
| 源代码（前 30 页 + 后 30 页） | `src/`、`app/`、`train.py`（约 3600 行） |
| 用户手册 | 本文档 + `README.md` + 产物里的 `使用说明.txt` |
| 软件截图 | `docs/assets/demo_real_poyang.gif`、界面三个页签截图 |
| 运行环境 | Windows 10/11 64 位，无需 Python |
| 技术特点 | 双模型策略、近红外物理闸门、双时相变化检测、BOA 偏移量比对 |

> 提醒：软著登记通常要求提交源代码的连续前后各 30 页（每页 50 行），
> 直接从 `src/` 打印即可；记得把作者/单位信息按学校要求填好。
