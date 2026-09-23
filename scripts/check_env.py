"""
慧眼识灾 · 环境自检
===================

    python scripts/check_env.py

检查 Python 版本、关键依赖、GPU 可用性，并给出缺失项的安装建议。
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import List, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

REQUIRED = [
    ("numpy", "pip install numpy"),
    ("scipy", "pip install scipy"),
    ("rasterio", "pip install rasterio"),
    ("PIL", "pip install pillow"),
    ("torch", "pip install torch torchvision"),
]
OPTIONAL = [
    ("cv2", "pip install opencv-python-headless"),
    ("skimage", "pip install scikit-image"),
    ("gradio", "pip install gradio"),
    ("reportlab", "pip install reportlab"),
    ("segmentation_models_pytorch", "pip install segmentation-models-pytorch"),
    ("matplotlib", "pip install matplotlib"),
]


def _version(mod) -> str:
    return str(getattr(mod, "__version__", "") or "")


def _check(items: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    ok, missing = [], []
    for name, hint in items:
        try:
            mod = importlib.import_module(name)
            ok.append(f"{name:<30s} {_version(mod)}")
        except Exception as exc:
            missing.append(f"{name:<30s} {hint}   ({type(exc).__name__})")
    return ok, missing


def main() -> int:
    print("=" * 70)
    print("慧眼识灾 · 环境自检")
    print("=" * 70)
    print(f"Python : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

    print("\n[必需依赖]")
    ok, missing = _check(REQUIRED)
    for line in ok:
        print(f"  [OK]   {line}")
    for line in missing:
        print(f"  [缺失] {line}")

    print("\n[可选依赖]（缺失不影响 NDWI 基线演示）")
    ok2, missing2 = _check(OPTIONAL)
    for line in ok2:
        print(f"  [OK]   {line}")
    for line in missing2:
        print(f"  [缺失] {line}")

    print("\n[GPU]")
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  torch {torch.__version__}  CUDA 可用：{cuda}")
        if cuda:
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU：{torch.cuda.get_device_name(0)}  显存 {props.total_memory / 1e9:.1f} GB  "
                  f"算力 {props.major}.{props.minor}")
            print("  -> 可以直接训练 U-Net")
        else:
            print("  -> CPU 模式：基线推理秒级可跑；训练建议安装 CUDA 版 torch：")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    except ImportError as exc:
        print(f"  [缺失] torch 未安装（{exc}）")
    except Exception as exc:
        # 区分"没装"和"装了但初始化失败"：原实现把所有异常都报成"未安装"，
        # 会误导用户反复重装（实际可能是 CUDA/驱动/环境变量问题）。
        print(f"  [异常] torch 已安装但初始化失败：{type(exc).__name__}: {exc}")
        print("         这不是缺包问题，请检查 CUDA/驱动 或 torch 版本与 Python 的匹配")

    print("\n[结论]")
    if missing:
        print("  ⚠️ 必需依赖不完整，先执行：pip install -r requirements.txt")
        return 1
    print("  ✅ 必需依赖齐全，可以运行：python app/main.py")
    if missing2:
        print("  ℹ️ 部分可选依赖缺失：界面/PDF 导出可能不可用，但基线算法不受影响")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
