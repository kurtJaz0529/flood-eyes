"""
慧眼识灾 · 遥感 AI 洪水识别系统
================================

src 包对外接口：

    from src import FloodDetector, load_scene, build_model
"""

from .baseline import otsu_threshold, predict as baseline_predict  # noqa: F401
from .infer import FloodDetector, FloodResult, available_weights, quick_detect, resolve_device  # noqa: F401
from .paths import bundle_root, outputs_dir, user_root, weights_dirs  # noqa: F401
from .preprocess import Scene, load_scene, normalize, percentile_stretch  # noqa: F401

# 深度模型依赖 torch / segmentation-models-pytorch，做成可选：
# 精简版（无 torch）仍可用 NDWI 基线完整运行。
try:  # pragma: no cover - 取决于运行环境
    from .model_unet import (  # noqa: F401
        DiceBCELoss,
        TinyUNet,
        build_model,
        load_checkpoint,
        predict_tiled,
        save_checkpoint,
        segmentation_metrics,
    )

    TORCH_AVAILABLE = True
    TORCH_ERROR = ""
except Exception as _exc:  # pragma: no cover
    TORCH_AVAILABLE = False
    TORCH_ERROR = f"{type(_exc).__name__}: {_exc}"

__version__ = "0.4.0"
__all__ = [
    "FloodDetector",
    "FloodResult",
    "Scene",
    "load_scene",
    "normalize",
    "percentile_stretch",
    "baseline_predict",
    "otsu_threshold",
    "quick_detect",
    "available_weights",
    "resolve_device",
    "bundle_root",
    "user_root",
    "weights_dirs",
    "outputs_dir",
    "TORCH_AVAILABLE",
    "TORCH_ERROR",
    "__version__",
]
if TORCH_AVAILABLE:
    __all__ += [
        "build_model",
        "TinyUNet",
        "DiceBCELoss",
        "predict_tiled",
        "segmentation_metrics",
        "save_checkpoint",
        "load_checkpoint",
    ]
