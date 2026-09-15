"""
慧眼识灾 · 打包运行时钩子
=========================

PyInstaller 在启动主脚本前执行本文件。

做两件事：
    1. 告诉 GDAL / PROJ 去哪里找数据文件（打包后它们在 _internal/rasterio/ 下）
    2. 把工作目录切到可写目录，避免只读安装目录导致导出失败
"""

import os
import sys

_BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))

for _var, _rel in (
    ("GDAL_DATA", ("rasterio", "gdal_data")),
    ("PROJ_LIB", ("rasterio", "proj_data")),
    ("PROJ_DATA", ("rasterio", "proj_data")),
):
    _path = os.path.join(_BASE, *_rel)
    if os.path.isdir(_path):
        os.environ.setdefault(_var, _path)

# GDAL 在冻结环境里更稳的默认值
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

# 让相对路径（如 outputs/）落在 exe 同目录
try:
    _exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    if os.access(_exe_dir, os.W_OK):
        os.chdir(_exe_dir)
except Exception:
    pass
