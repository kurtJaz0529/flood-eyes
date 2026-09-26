# -*- mode: python ; coding: utf-8 -*-
"""
慧眼识灾 · PyInstaller 打包配置
================================

构建（推荐用 build/build_app.ps1，会自动设好环境变量）：

    set FLOOD_PROFILE=lite   # 精简版：不含 PyTorch，~350 MB，基线模型可用
    set FLOOD_PROFILE=full   # 完整版：含 PyTorch + U-Net，~900 MB
    pyinstaller build/flood_eyes.spec --noconfirm

产物：dist/慧眼识灾/慧眼识灾.exe（one-folder，启动快，双击即用）
"""

import glob
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

SPEC_DIR = os.path.abspath(SPECPATH)  # SPECPATH 本身就是 spec 所在目录
ROOT = os.path.dirname(SPEC_DIR)

# 默认值必须与文件头注释、build_app.ps1 保持一致（都是 lite）。
# 原默认是 full：直接执行 pyinstaller build/flood_eyes.spec 会意外打出 ~900MB
# 的完整包，并先把 PyTorch 一并塞进分发物。
PROFILE = os.environ.get("FLOOD_PROFILE", "lite").strip().lower()
if PROFILE not in ("lite", "full"):
    raise SystemExit(f"[spec] 未知的 FLOOD_PROFILE={PROFILE!r}，只能是 lite 或 full")
LITE = PROFILE == "lite"
CONSOLE = os.environ.get("FLOOD_CONSOLE", "0") == "1"
APP_NAME = os.environ.get("FLOOD_APP_NAME", "慧眼识灾")

print(f"[spec] 打包配置：profile={PROFILE}  console={CONSOLE}  app={APP_NAME}")
if "FLOOD_PROFILE" not in os.environ:
    print("[spec] 注意：未显式设置 FLOOD_PROFILE，已按 lite（精简版，不含 PyTorch）打包；"
          "需要完整版请先 set FLOOD_PROFILE=full")

# --------------------------------------------------------------------------
# 资源
# --------------------------------------------------------------------------
datas = []
binaries = []
hiddenimports = []

# 1) 示例影像（合成 + 真实）
for sub in ("samples", "real"):
    path = os.path.join(ROOT, "data", sub)
    if os.path.isdir(path):
        datas.append((path, os.path.join("data", sub)))

# 1a) 内置洪灾事件库（src/events.py 在运行时从 bundle_root/data/events 读取）
_events = os.path.join(ROOT, "data", "events")
if os.path.isdir(_events):
    datas.append((_events, os.path.join("data", "events")))
else:
    print("[spec] 警告：未找到 data/events，打包后事件库将为空、检索功能不可用")

# 1b) 界面资源与全流程脚本（frozen 后要从 bundle_root 读）
#     注意：heye_map.js 是**数据文件**（运行时按路径读取），不是导入的模块，
#     PyInstaller 不会自动收集——漏掉它打包后地图区域会空白。
for name in ("apple.css", "heye_map.js"):
    p = os.path.join(ROOT, "app", name)
    if os.path.isfile(p):
        datas.append((p, "app"))
    else:
        print(f"[spec] 警告：未找到 app/{name}")
_scripts = os.path.join(ROOT, "scripts")
if os.path.isdir(_scripts):
    for name in ("fetch_real_samples.py", "fetch_s1_rtc.py", "check_data.py"):
        p = os.path.join(_scripts, name)
        if os.path.isfile(p):
            datas.append((p, "scripts"))

# 1c) 应用图标：顶栏 logo（内联为 data URI）与浏览器 favicon 都读它。
#     同样属于"运行时按路径读取"的数据文件，PyInstaller 不会自动收集。
_icon = os.path.join(ROOT, "docs", "assets", "app_icon.png")
if os.path.isfile(_icon):
    datas.append((_icon, os.path.join("docs", "assets")))
else:
    print("[spec] 警告：未找到 docs/assets/app_icon.png，顶栏与浏览器标签将退回文字占位")

# 2) 权重（精简版不需要：没有 torch 也加载不了）
if not LITE:
    for weight in glob.glob(os.path.join(ROOT, "weights", "*.pt")):
        datas.append((weight, "weights"))

# 3) Gradio 全家桶（含前端静态资源，必须全量收集）
for pkg in ("gradio", "gradio_client", "safehttpx", "groovy", "hf_gradio", "pydub"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # pragma: no cover
        print(f"[spec] collect_all({pkg}) 跳过：{exc}")

# 4) rasterio：官方没有 hook，手动收集 GDAL/PROJ 数据与 delvewheel DLL
try:
    import rasterio

    rdir = os.path.dirname(rasterio.__file__)
    for sub in ("gdal_data", "proj_data"):
        p = os.path.join(rdir, sub)
        if os.path.isdir(p):
            datas.append((p, os.path.join("rasterio", sub)))
    libs_dir = os.path.join(os.path.dirname(rdir), "rasterio.libs")
    if os.path.isdir(libs_dir):
        binaries += [(p, "rasterio.libs") for p in glob.glob(os.path.join(libs_dir, "*.dll"))]
    datas += collect_data_files("rasterio", include_py_files=False)
    hiddenimports += collect_submodules("rasterio")
    print(f"[spec] rasterio：gdal_data={os.path.isdir(os.path.join(rdir, 'gdal_data'))} "
          f"dll={len(glob.glob(os.path.join(libs_dir, '*.dll')))}")
except Exception as exc:  # pragma: no cover
    print(f"[spec] rasterio 收集失败：{exc}")

# 5) reportlab 中文 CID 字体
hiddenimports += [
    "reportlab.pdfbase._cidfontdata",
    "reportlab.pdfbase.cidfonts",
    "reportlab.pdfbase.pdfmetrics",
    "reportlab.pdfbase.ttfonts",
]
datas += collect_data_files("reportlab")

# 6) uvicorn 动态导入
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
]

# 7) 项目自身模块（函数内 import，显式兜底）
hiddenimports += ["app", "app.main", "app.components", "app.desktop", "src"]
hiddenimports += ["app.automation", "app.spectral", "src.adaptive_flood",
                  "src.spectral_monitor", "src.evaluation", "src.jobs", "src.pipeline"]
for name in ("场景适配与遥感扩展使用说明.md", "光谱时序监测使用说明.md", "自动化处理使用说明.md"):
    datas.append((os.path.join(ROOT, "docs", name), "docs"))

# --------------------------------------------------------------------------
# 排除（显著减小体积）
# --------------------------------------------------------------------------
excludes = [
    # 只在训练脚本里用到
    "matplotlib",
    "mpl_toolkits",
    # 只作源码兜底，打包版用 OpenCV
    "scipy",
    "skimage",
    # Gradio 的惰性可选依赖（我们的界面用不到）
    "polars",          # 176 MB，DataFrame 组件才用
    "yt_dlp",          # 视频下载组件
    "pygame",          # 小游戏组件
    # 与本项目无关
    "sklearn",
    "IPython",
    "jupyter",
    "notebook",
    "nbformat",
    "pytest",
    "sphinx",
    "tensorflow",
    "paddle",
    "onnx",
    "numba",
    "pyarrow",
    "fastparquet",
    "docutils",
    "PIL.ImageQt",
]

# 丢弃用不到的大块二进制/资源（视频编解码、前端 ffmpeg）
_DROP_BIN_PATTERNS = ("opencv_videoio_ffmpeg", "opencv_ffmpeg")
_DROP_DATA_PATTERNS = (os.path.join("static", "ffmpeg"), "ffmpeg-core.wasm")


def _filter_entries(entries, patterns):
    """按文件名/路径关键字过滤 (src, dest) 列表。"""
    kept, dropped = [], 0
    for entry in entries:
        src = entry[0] if isinstance(entry, (tuple, list)) else entry
        name = str(src).lower()
        if any(str(p).lower() in name for p in patterns):
            dropped += 1
            continue
        kept.append(entry)
    return kept, dropped

binaries, _n = _filter_entries(binaries, _DROP_BIN_PATTERNS)
datas, _m = _filter_entries(datas, _DROP_DATA_PATTERNS)
print(f"[spec] 精简：丢弃二进制 {_n} 个、资源 {_m} 个")

if LITE:
    excludes += [
        "torch",
        "torchvision",
        "torchaudio",
        "segmentation_models_pytorch",
        "timm",
        "sympy",
        "networkx",
        "mpmath",
    ]
else:
    # 完整版：让 PyInstaller 的 torch hook 处理
    hiddenimports += ["torch", "torch.nn", "torch.nn.functional"]
    try:
        import segmentation_models_pytorch  # noqa: F401

        hiddenimports += collect_submodules("segmentation_models_pytorch")
    except Exception as exc:
        print(f"[spec] smp 未安装，跳过：{exc}")

# --------------------------------------------------------------------------
# 打包
# --------------------------------------------------------------------------
block_cipher = None

a = Analysis(
    [os.path.join(ROOT, "app", "desktop.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPEC_DIR, "rthook_paths.py")],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Analysis 之后才能真正过滤掉 OpenCV 自带的 ffmpeg 解码器（约 80 MB，本项目不做视频）
a.binaries, _dropped_bin = _filter_entries(a.binaries, _DROP_BIN_PATTERNS)
print(f"[spec] Analysis 后丢弃二进制 {_dropped_bin} 个")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, "docs", "assets", "app_icon.ico"),
    version=os.path.join(SPEC_DIR, "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
