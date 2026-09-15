"""
慧眼识灾 · 统一推理接口
=======================

对外只暴露两个东西：
    FloodDetector  —— 配置一次，反复调用（模型只加载一次）
    FloodResult    —— 一次识别的全部产物（掩膜/概率/叠加图/统计/元数据）

调用示例：
    detector = FloodDetector(mode="auto")
    result = detector.detect("data/samples/demo01_post.tif")
    print(result.summary_text())
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import baseline, postprocess, preprocess
from .paths import weights_dirs
from .preprocess import Scene, estimate_cloud_mask, load_scene, reproject_scene_to, same_geo_grid


def resolve_device(prefer: str = "auto") -> str:
    if prefer and prefer != "auto":
        return prefer
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def available_weights(dirs: Any = None) -> List[str]:
    """扫描权重目录，返回按修改时间倒序的权重文件列表。

    dirs 可以是单个目录、目录列表或 None（默认：用户目录 -> 打包资源目录）。
    """
    if dirs is None:
        search = weights_dirs()
    elif isinstance(dirs, str):
        search = [dirs]
    else:
        search = list(dirs)
    files: List[str] = []
    for d in search:
        if not os.path.isdir(d):
            continue
        for pat in ("*.pt", "*.pth", "*.ckpt"):
            files.extend(glob.glob(os.path.join(d, pat)))
    files = [f for f in files if not f.endswith(".tmp")]
    return sorted(set(files), key=os.path.getmtime, reverse=True)


@dataclass
class FloodResult:
    """一次洪水识别的完整结果。"""

    mask: np.ndarray  # (H,W) bool，最终水体掩膜
    prob: np.ndarray  # (H,W) float32 0~1 置信度
    rgb: np.ndarray  # (H,W,3) uint8 真彩色
    overlay: np.ndarray  # (H,W,3) uint8 掩膜叠加图
    stats: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    change: Optional[Dict[str, Any]] = None
    scene: Optional[Scene] = None
    elapsed_s: float = 0.0

    # -- 文本输出 ----------------------------------------------------------
    @property
    def area_km2(self) -> float:
        return float(self.stats.get("water_area_km2", 0.0))

    def summary_text(self) -> str:
        """一句话结论，直接贴进简报/弹窗。"""
        lines = [
            f"【识别结论】{self.meta.get('model_label', '模型')} 检测到水体面积 "
            f"{self.area_km2:,.2f} km²，占影像面积 {self.stats.get('water_fraction_pct', 0):.2f}%，"
            f"共 {self.stats.get('n_components', 0)} 个连通水域。",
            f"【处理耗时】{self.elapsed_s:.2f} 秒（像元 {self.stats.get('pixel_size_m', 10):g} m）",
        ]
        if self.meta.get("threshold_desc"):
            lines.append(f"【分割依据】{self.meta['threshold_desc']}")
        if self.meta.get("warnings"):
            lines.append("【提示】" + "；".join(self.meta["warnings"]))
        if self.change:
            c = self.change
            lines.append(
                f"【双时相】灾前 {c['before_water_km2']:,.2f} km² → 灾后 {c['after_water_km2']:,.2f} km²，"
                f"新增淹没 {c['new_water_km2']:,.2f} km²，退水 {c['receded_water_km2']:,.2f} km²。"
            )
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "meta": {k: v for k, v in self.meta.items() if k != "scene"},
            "stats": {k: v for k, v in self.stats.items() if k != "components"} | {
                "components": self.stats.get("components", [])[:5]
            },
            "elapsed_s": self.elapsed_s,
        }
        if self.change:
            payload["change"] = {k: v for k, v in self.change.items() if k != "masks"}
        return json.dumps(payload, ensure_ascii=False, indent=indent, default=str)


class FloodDetector:
    """洪水识别器：统一封装"基线 / 深度模型"两条路线。"""

    def __init__(
        self,
        mode: str = "auto",  # auto | baseline | unet
        weights: Optional[str] = None,
        device: str = "auto",
        pixel_size_m: Optional[float] = None,
        min_area_px: int = 120,
        open_radius: int = 2,
        close_radius: int = 3,
        fill_holes: bool = True,
        max_hole_px: int = 500,
        tile: int = 512,
        overlap: int = 64,
        batch_size: int = 4,
        tta: bool = False,
        threshold: float = 0.5,
        softness: float = 0.05,
        band_order: str = "auto",
        min_water_area_km2: float = 0.0,
        mask_clouds: bool = True,
        max_cloud_pct: float = 35.0,
    ):
        self.mode = mode
        self.weights = weights
        self.device = resolve_device(device)
        self.pixel_size_m = None if pixel_size_m is not None and float(pixel_size_m) <= 0 else pixel_size_m
        self.min_area_px = int(min_area_px)
        self.open_radius = int(open_radius)
        self.close_radius = int(close_radius)
        self.fill_holes = bool(fill_holes)
        self.max_hole_px = int(max_hole_px)
        self.tile = int(tile)
        self.overlap = int(overlap)
        self.batch_size = int(batch_size)
        self.tta = bool(tta)
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.band_order = band_order
        self.min_water_area_km2 = float(min_water_area_km2)
        self.mask_clouds = bool(mask_clouds)
        self.max_cloud_pct = float(max_cloud_pct)
        self._model = None
        self._model_meta: Dict[str, Any] = {}
        self._load_error: Optional[str] = None
        self._synthetic_weights: Optional[bool] = None

    # -- 模型加载 ----------------------------------------------------------
    def _weights_looks_synthetic(self, path: str) -> bool:
        if self._synthetic_weights is not None:
            return self._synthetic_weights
        try:
            import torch

            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            note = str((payload.get("meta") or {}).get("data_note") or "")
            self._synthetic_weights = "synthetic" in note.lower()
        except Exception:
            self._synthetic_weights = False
        return self._synthetic_weights

    @property
    def resolved_mode(self) -> str:
        """根据权重是否存在，把 auto 解析成具体模式。

        合成样本上训的权重在 auto 下不用，避免演示数字被当成真实精度。
        显式选择 U-Net 时仍加载（并在结果里警告）。
        """
        if self.mode == "auto":
            path = self._weights_path()
            if not path:
                return "baseline"
            if self._weights_looks_synthetic(path):
                return "baseline"
            return "unet"
        return self.mode

    def _weights_path(self) -> Optional[str]:
        """解析权重路径：显式指定优先，其次按 用户目录 -> 资源目录 自动选最新权重。"""
        if self.weights:
            cand = self.weights
            if not os.path.isabs(cand) and not os.path.isfile(cand):
                for d in weights_dirs():
                    probe = os.path.join(d, self.weights)
                    if os.path.isfile(probe):
                        cand = probe
                        break
            if os.path.isfile(cand):
                return cand
            self._load_error = f"指定的权重文件不存在：{self.weights}"
            return None
        found = available_weights()
        return found[0] if found else None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self.resolved_mode != "unet":
            return False
        path = self._weights_path()
        if not path:
            self._load_error = "未找到权重文件，已自动切换 NDWI 基线"
            return False
        try:
            from .model_unet import load_checkpoint

            self._model, self._model_meta = load_checkpoint(path, device=self.device)
            self._model_meta["weights_path"] = path
            return True
        except ImportError as exc:  # 精简版没有 torch
            self._load_error = (
                f"当前为精简版（未包含 PyTorch：{exc}），U-Net 不可用，已切换 NDWI 基线。"
                "需要深度模型请使用完整版安装包"
            )
            self._model = None
            return False
        except Exception as exc:  # 权重损坏/结构不匹配 -> 退回基线，不让演示中断
            self._load_error = f"权重加载失败({type(exc).__name__}: {exc})，已切换 NDWI 基线"
            self._model = None
            return False

    def model_info(self) -> Dict[str, Any]:
        if self.resolved_mode != "unet":
            info = {"mode": "baseline", "label": "NDWI + Otsu 基线", "device": self.device}
            path = self._weights_path()
            if self.mode == "auto" and path and self._weights_looks_synthetic(path):
                info["warning"] = "当前权重为合成样本训练，自动模式已改用 NDWI 基线；如需 U-Net 请显式选择"
            elif self._load_error:
                info["label"] = "NDWI + Otsu 基线(降级)"
                info["warning"] = self._load_error
            return info
        if not self._ensure_model():
            return {"mode": "baseline", "label": "NDWI + Otsu 基线(降级)", "device": self.device,
                    "warning": self._load_error}
        info = dict(self._model_meta)
        info.update({"mode": "unet", "label": f"U-Net ({info.get('arch', '?')})", "device": self.device})
        return info

    # -- 核心推理 ----------------------------------------------------------
    def detect(
        self,
        image_path: Optional[str] = None,
        nir_path: Optional[str] = None,
        scene: Optional[Scene] = None,
    ) -> FloodResult:
        if scene is None:
            if not image_path:
                raise ValueError("必须提供 image_path 或 scene")
            scene = load_scene(
                image_path,
                band_order=self.band_order,
                nir_path=nir_path,
                pixel_size_m=self.pixel_size_m,
            )
        return self.detect_scene(scene)

    def detect_scene(self, scene: Scene) -> FloodResult:
        t0 = time.perf_counter()
        warnings: List[str] = []
        if not scene.has_nir:
            warnings.append("影像缺少近红外波段，基线使用 (绿-红) 代理指数，精度会下降")
        nodata = scene.nodata_mask
        if nodata is not None and nodata.all():
            warnings.append("影像全部为无效像元")

        cloud = estimate_cloud_mask(scene)
        cloud_pct = 0.0
        if cloud is not None and cloud.any():
            cloud_pct = 100.0 * float(np.count_nonzero(cloud)) / max(cloud.size, 1)
            if cloud_pct >= self.max_cloud_pct:
                warnings.append(
                    f"估计云量 {cloud_pct:.1f}% ≥ {self.max_cloud_pct:g}%，光学识别不可靠，建议改用雷达 SAR"
                )
            elif cloud_pct >= 8.0:
                src = "SCL" if scene.meta.get("scl_source") else "光谱估计"
                warnings.append(f"{src}云量 {cloud_pct:.1f}%，云区已从水体统计中剔除")
            if self.mask_clouds:
                nodata = cloud if nodata is None else (nodata | cloud)

        mode = self.resolved_mode
        use_unet = mode == "unet" and self._ensure_model()
        if mode == "unet" and not use_unet and self._load_error:
            warnings.append(self._load_error)

        if use_unet:
            prob, meta = self._predict_unet(scene, warnings)
            model_label = f"U-Net ({self._model_meta.get('arch', '?')})"
        else:
            _mask_raw, prob, thr_meta = baseline.predict(
                scene.ndwi(),
                method="hybrid",
                fixed_threshold=None,
                softness=self.softness,
                nodata_mask=nodata,
                nir=scene.bands.get("nir"),
            )
            meta = dict(thr_meta)
            meta["threshold_desc"] = baseline.describe_threshold(thr_meta)
            meta["model_label"] = "NDWI + Otsu 基线"
            model_label = meta["model_label"]

        # 后处理
        mask = postprocess.clean_mask(
            prob >= (self.threshold if use_unet else 0.5),
            open_radius=self.open_radius,
            close_radius=self.close_radius,
            min_area_px=self.min_area_px,
            fill_holes=self.fill_holes,
            max_hole_px=self.max_hole_px,
        )
        if nodata is not None:
            mask = mask & ~nodata

        stats = postprocess.area_stats(mask, scene.pixel_size_m, nodata)
        stats.update(postprocess.confidence_stats(prob, mask))

        if self.min_water_area_km2 > 0 and stats["largest_area_km2"] < self.min_water_area_km2:
            warnings.append(
                f"最大连通水域 {stats['largest_area_km2']:.3f} km² 低于告警阈值 "
                f"{self.min_water_area_km2:g} km²，可视为无有效淹没"
            )

        rgb = scene.rgb()
        overlay = postprocess.overlay_mask(rgb, mask)

        meta.update(
            {
                "model_label": model_label,
                "mode": "unet" if use_unet else "baseline",
                "device": self.device if use_unet else "-",
                "pixel_size_m": scene.pixel_size_m,
                "source": os.path.basename(scene.path) if scene.path else "in-memory",
                "shape": list(scene.shape),
                "nir_available": scene.has_nir,
                "cloud_pct": cloud_pct,
                "cloud_source": scene.meta.get("scl_source") or "heuristic",
                "boa_offset": scene.meta.get("boa_offset", 0),
                "band_map": scene.meta.get("band_map"),
                "warnings": warnings,
                "params": {
                    "min_area_px": self.min_area_px,
                    "open_radius": self.open_radius,
                    "close_radius": self.close_radius,
                    "threshold": self.threshold if use_unet else 0.5,
                    "tta": self.tta,
                    "tile": self.tile,
                },
            }
        )

        return FloodResult(
            mask=mask,
            prob=prob,
            rgb=rgb,
            overlay=overlay,
            stats=stats,
            meta=meta,
            scene=scene,
            elapsed_s=time.perf_counter() - t0,
        )

    def _predict_unet(self, scene: Scene, warnings: List[str]) -> Any:
        from .model_unet import predict_tiled

        names = scene.channel_names
        chw = scene.stack(names)
        in_ch = int(self._model_meta.get("in_channels", 4))
        if chw.shape[0] < in_ch:  # 缺波段补 0（例如没有近红外）
            got = chw.shape[0]
            pad = np.zeros((in_ch - got, *chw.shape[1:]), dtype=np.float32)
            chw = np.concatenate([chw, pad], axis=0)
            warnings.append(f"模型需要 {in_ch} 通道，输入仅 {got} 通道，缺失通道已补零")
        elif chw.shape[0] > in_ch:
            chw = chw[:in_ch]

        mean = self._model_meta.get("mean") or preprocess.DEFAULT_MEAN
        std = self._model_meta.get("std") or preprocess.DEFAULT_STD
        prob, _mask, pmeta = predict_tiled(
            self._model,
            chw,
            tile=self.tile,
            overlap=self.overlap,
            batch_size=self.batch_size,
            device=self.device,
            mean=mean,
            std=std,
            threshold=self.threshold,
            tta=self.tta,
        )
        pmeta["threshold_desc"] = f"U-Net 概率阈值 {self.threshold:.2f}"
        pmeta["model_label"] = f"U-Net ({self._model_meta.get('arch', '?')})"
        if self._model_meta.get("trained_on"):
            pmeta["trained_on"] = self._model_meta["trained_on"]
        # 训练数据与真实影像存在域差异时给出明确提示（避免误读结果）
        if self._model_meta.get("data_note") == "synthetic demo data":
            warnings.append(
                "该权重是在合成样本上训练的，仅用于验证流程；真实影像上的分割结果可能过检/漏检，"
                "请以 NDWI 基线结果交叉验证，或用 Sen1Floods11 重新训练"
            )
        return prob, pmeta

    # -- 双时相对比 --------------------------------------------------------
    def compare(
        self,
        before_path: Optional[str] = None,
        after_path: Optional[str] = None,
        before_scene: Optional[Scene] = None,
        after_scene: Optional[Scene] = None,
    ) -> FloodResult:
        """灾前/灾后双时相分析，返回以灾后结果为主体、附带变化统计的结果。

        两景有地理参考且不在同一网格时，先把灾后重投影到灾前栅格再识别。
        """
        if before_scene is None:
            if not before_path:
                raise ValueError("必须提供 before_path 或 before_scene")
            before_scene = load_scene(
                before_path, band_order=self.band_order, pixel_size_m=self.pixel_size_m
            )
        if after_scene is None:
            if not after_path:
                raise ValueError("必须提供 after_path 或 after_scene")
            after_scene = load_scene(
                after_path, band_order=self.band_order, pixel_size_m=self.pixel_size_m
            )

        aligned = False
        geo_ok = (
            before_scene.transform is not None
            and after_scene.transform is not None
            and before_scene.crs is not None
            and after_scene.crs is not None
        )
        if geo_ok and not same_geo_grid(before_scene, after_scene):
            after_scene = reproject_scene_to(after_scene, before_scene)
            aligned = True
        elif before_scene.shape != after_scene.shape:
            raise ValueError(
                f"灾前/灾后影像尺寸不一致：{before_scene.shape} vs {after_scene.shape}，"
                "且缺少地理参考无法自动对齐，请先裁剪到同一范围"
            )

        before = self.detect_scene(before_scene)
        after = self.detect_scene(after_scene)
        if aligned:
            after.meta.setdefault("warnings", []).append("灾后影像已重投影到灾前网格后再对比")

        change = postprocess.change_stats(before.mask, after.mask, after.scene.pixel_size_m)
        change["change_map"] = postprocess.change_map_rgb(before.mask, after.mask, after.rgb)
        change["before_overlay"] = before.overlay
        change["after_overlay"] = after.overlay
        change["before_rgb"] = before.rgb
        change["after_rgb"] = after.rgb
        change["before_result"] = before
        change["after_result"] = after

        after.change = change
        after.elapsed_s += before.elapsed_s
        after.meta["comparison"] = {
            "before_source": os.path.basename(before_path or before_scene.path or "before"),
            "after_source": os.path.basename(after_path or after_scene.path or "after"),
            "reprojected": aligned,
        }
        return after


def quick_detect(image_path: str, mode: str = "auto", **kwargs: Any) -> FloodResult:
    """一行代码跑一次识别（脚本/测试用）。"""
    return FloodDetector(mode=mode, **kwargs).detect(image_path)
