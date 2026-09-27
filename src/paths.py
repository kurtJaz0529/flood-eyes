"""
慧眼识灾 · 路径解析
===================

同一套代码要同时跑在三种环境里：

    1. 源码运行        python app/main.py
    2. 打包后的 exe    PyInstaller one-folder（资源在 sys._MEIPASS，只读）
    3. 绿色便携运行    exe 放在 U 盘/桌面（产物写到 exe 同目录）

所以把"只读资源"和"可写产物"分开：

    bundle_root()  —— 只读：代码、示例影像、权重、字体
    user_root()    —— 可写：识别结果、导出简报、日志
"""

from __future__ import annotations

import os
import sys
from typing import List


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包环境里。"""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> str:
    """只读资源根目录。"""
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def user_root() -> str:
    """可写目录：优先 exe 同目录（便携），不可写则退到 LOCALAPPDATA。"""
    override = os.environ.get("FLOOD_DATA_DIR")
    if override:
        path = os.path.abspath(os.path.expanduser(override))
        os.makedirs(path, exist_ok=True)
        return path
    if not is_frozen():
        return bundle_root()
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    # 用系统生成的唯一临时文件探测可写性。
    # 原实现固定写 exe_dir/.write_test：多实例并发时一个实例会删掉另一个刚建的
    # 文件导致误判"不可写"；若目录里本来就有同名文件，还会被直接截断并删除。
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(dir=exe_dir, prefix=".huiyan_write_", delete=True) as fh:
            fh.write(b"ok")
        return exe_dir
    except Exception:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        path = os.path.join(base, "HuiYanShiZai")
        os.makedirs(path, exist_ok=True)
        return path


def data_dir(*parts: str) -> str:
    return os.path.join(bundle_root(), "data", *parts)


def samples_dir() -> str:
    return data_dir("samples")


def real_dir() -> str:
    return data_dir("real")


def weights_dirs() -> List[str]:
    """权重搜索顺序：用户目录 -> 打包资源目录。"""
    dirs = [os.path.join(user_root(), "weights"), os.path.join(bundle_root(), "weights")]
    out: List[str] = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def outputs_dir() -> str:
    path = os.path.join(user_root(), "outputs")
    os.makedirs(path, exist_ok=True)
    return path


def logs_dir() -> str:
    path = os.path.join(user_root(), "logs")
    os.makedirs(path, exist_ok=True)
    return path


def resource(*parts: str) -> str:
    """拼一个只读资源路径。"""
    return os.path.join(bundle_root(), *parts)
