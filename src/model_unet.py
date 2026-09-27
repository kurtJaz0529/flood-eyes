"""
慧眼识灾 · U-Net 语义分割模型
=============================

双通道实现，保证"有没有第三方库都能跑"：

  ① 首选 segmentation-models-pytorch（smp.Unet）
     骨干可选 resnet34 / efficientnet-b0 / mit_b0 ...，可加载 ImageNet 预训练权重，
     是竞赛里"精度上得去"的正路。
  ② 兜底 TinyUNet（本文件内置，纯 PyTorch 实现）
     没有 smp 也能训练/推理，参数量约 2M，CPU 上也能跑通演示。

输入：4 通道 [蓝, 绿, 红, 近红外] 反射率（归一化后）
输出：单通道 logits，sigmoid 后为水体概率
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import DEFAULT_MEAN, DEFAULT_STD, iter_tiles, pad_to_tile, stitch_tiles

EPS = 1e-7


# --------------------------------------------------------------------------
# 兜底模型：TinyUNet
# --------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """轻量 U-Net：4 层编码器 + 4 层解码器 + 跳跃连接。"""

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base: int = 16):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]
        self.inc = DoubleConv(in_channels, chs[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(chs[0], chs[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(chs[1], chs[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(chs[2], chs[3]))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(chs[3], chs[3] * 2))
        self.up3 = nn.ConvTranspose2d(chs[3] * 2, chs[3], 2, stride=2)
        self.dec3 = DoubleConv(chs[3] * 2, chs[3])
        self.up2 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec2 = DoubleConv(chs[2] * 2, chs[2])
        self.up1 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec1 = DoubleConv(chs[1] * 2, chs[1])
        self.up0 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec0 = DoubleConv(chs[0] * 2, chs[0])
        self.head = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        b = self.bottleneck(x3)
        d3 = self.dec3(torch.cat([self.up3(b), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), x0], dim=1))
        return self.head(d0)


# --------------------------------------------------------------------------
# 模型构建
# --------------------------------------------------------------------------

SMP_ENCODERS = ("resnet34", "resnet18", "resnet50", "efficientnet-b0", "mobilenet_v2", "mit_b0", "tu-efficientnet_b0")

# 训练脚本把 info["arch"] 存成 "smp_unet" / "tiny_unet"，build_model 只认 auto/smp/tiny
_ARCH_ALIASES = {
    "smp_unet": "smp",
    "smp-unet": "smp",
    "tiny_unet": "tiny",
    "tiny-unet": "tiny",
    "tinyunet": "tiny",
}


def normalize_arch(arch: Optional[str]) -> str:
    """把 checkpoint 里的 arch 字段收成 build_model 能认的键。"""
    a = (arch or "auto").strip().lower()
    return _ARCH_ALIASES.get(a, a)


def smp_available() -> bool:
    try:
        import segmentation_models_pytorch  # noqa: F401

        return True
    except Exception:
        return False


def build_model(
    arch: str = "auto",
    encoder: str = "resnet34",
    in_channels: int = 4,
    classes: int = 1,
    encoder_weights: Optional[str] = None,
    base: int = 16,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """构建模型。

    arch:
        "auto"  -> 有 smp 用 smp.Unet，否则 TinyUNet
        "smp"   -> 强制 smp.Unet
        "tiny"  -> 强制 TinyUNet
    """
    arch = normalize_arch(arch)
    info: Dict[str, Any] = {"in_channels": in_channels, "classes": classes, "arch_key": arch}
    if arch in ("auto", "smp") and smp_available():
        try:
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
            info.update({"arch": "smp_unet", "encoder": encoder, "encoder_weights": encoder_weights})
            return model, info
        except Exception as exc:  # 权重下载失败等 -> 退回内置模型
            info["smp_fallback_reason"] = f"{type(exc).__name__}: {exc}"
    model = TinyUNet(in_channels=in_channels, out_channels=classes, base=base)
    info.update({"arch": "tiny_unet", "encoder": "scratch", "base": base})
    return model, info


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# --------------------------------------------------------------------------
# 损失函数
# --------------------------------------------------------------------------


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(targets) & (targets >= 0)
    clean = torch.where(valid, targets, torch.zeros_like(targets))
    probs = torch.sigmoid(logits) * valid
    num = 2.0 * (probs * clean).sum(dim=(1, 2, 3)) + EPS
    den = probs.sum(dim=(1, 2, 3)) + clean.sum(dim=(1, 2, 3)) + EPS
    usable = valid.sum(dim=(1, 2, 3)) > 0
    return ((1.0 - num / den) * usable).sum() / usable.sum().clamp(min=1)


class DiceBCELoss(nn.Module):
    """Dice + BCE 组合损失：Dice 治类别不平衡，BCE 给稳定梯度。"""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(targets) & (targets >= 0)
        clean = torch.where(valid, targets, torch.zeros_like(targets))
        raw = F.binary_cross_entropy_with_logits(logits, clean, pos_weight=self.pos_weight, reduction="none")
        bce = (raw * valid).sum() / valid.sum().clamp(min=1)
        return self.bce_weight * bce + self.dice_weight * dice_loss(logits, targets)


# --------------------------------------------------------------------------
# 精度指标
# --------------------------------------------------------------------------


def segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = EPS,
) -> Dict[str, float]:
    """IoU / Dice(F1) / Precision / Recall。pred 可为 bool 或概率（>0.5 视为水）。"""
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    valid = np.isfinite(target) & (target >= 0)
    if not np.isfinite(pred[valid]).all():
        raise ValueError("Nonfinite prediction on valid target pixels")
    pred, target = pred[valid], target[valid] > 0.5
    if pred.dtype != bool:
        pred = pred > 0.5
    pred = pred.astype(bool)

    tp = float(np.count_nonzero(pred & target))
    fp = float(np.count_nonzero(pred & ~target))
    fn = float(np.count_nonzero(~pred & target))
    tn = float(np.count_nonzero(~pred & ~target))

    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "iou": iou,
        "dice": dice,
        "f1": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# --------------------------------------------------------------------------
# 权重读写
# --------------------------------------------------------------------------


def save_checkpoint(path: str, model: nn.Module, meta: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    """读取权重并重建模型结构。meta 里记录了 arch/encoder/in_channels。

    arch 会做别名映射（smp_unet → smp）。结构对不上就抛错，避免把不匹配的权重
    静默装进随机初始化的模型。

    加载只接受 weights_only=True（安全反序列化）：权重可能来自用户目录、网盘
    或第三方，关闭该选项会在 load_state_dict 之前执行 pickle，等于任意代码执行。
    """
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"权重无法安全加载：{path}（{type(exc).__name__}: {exc}）。"
            "本程序只接受标准 torch.save 保存的权重；"
            "若该文件来自旧版本或第三方，请用 train.py 重新导出后再使用。"
        ) from exc
    meta = dict(payload.get("meta", {}))
    arch = normalize_arch(meta.get("arch_key") or meta.get("arch", "auto"))
    model, info = build_model(
        arch=arch,
        encoder=meta.get("encoder", "resnet34"),
        in_channels=int(meta.get("in_channels", 4)),
        classes=int(meta.get("classes", 1)),
        encoder_weights=None,
        base=int(meta.get("base", 16)),
    )
    state = payload["state_dict"]
    n_params = len(model.state_dict())
    try:
        # 严格加载：任何缺失/多余键都判为结构不匹配。
        # 原实现用 strict=False 配"缺失数不超过 10%"的阈值放行，少量层
        # （BN 统计量、head 偏置等）会保持随机初始化，输出掩膜"看着正常"却是错的，
        # 而且没有任何提示。宁可拒绝加载，也不要静默用半随机权重出结果。
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"权重与模型结构不匹配：arch={arch}（模型 {n_params} 个参数张量）。"
            f"请确认该权重由同一份 train.py 导出。底层错误：{exc}"
        ) from exc
    meta.update({
        "model_info": info,
        "arch_key": arch,
        "missing_keys": [],
        "unexpected_keys": [],
    })
    model.to(device).eval()
    return model, meta


# --------------------------------------------------------------------------
# 分块推理
# --------------------------------------------------------------------------


@torch.no_grad()
def predict_tiled(
    model: nn.Module,
    chw: np.ndarray,
    tile: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    device: str = "cpu",
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    threshold: float = 0.5,
    tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """分块滑窗推理。

    返回 (prob, mask, meta)，prob/mask 尺寸与输入一致。
    tta=True 时做水平/垂直翻转平均，边界更稳（速度约 4 倍开销）。
    """
    from .preprocess import normalize

    chw = np.asarray(chw, dtype=np.float32)
    c, h, w = chw.shape
    model.eval()

    padded, (oh, ow) = pad_to_tile(np.moveaxis(chw, 0, -1), tile)
    padded_chw = np.moveaxis(padded, -1, 0)
    tiles: List[np.ndarray] = []
    coords: List[Tuple[int, int]] = []
    for t, (y0, x0) in iter_tiles(np.moveaxis(padded_chw, 0, -1), tile=tile, overlap=overlap):
        tiles.append(np.moveaxis(t, -1, 0))
        coords.append((y0, x0))

    probs: List[np.ndarray] = []
    for i in range(0, len(tiles), batch_size):
        batch = np.stack([normalize(t, mean, std) for t in tiles[i : i + batch_size]], axis=0)
        xb = torch.from_numpy(batch).to(device)
        logits = model(xb)
        p = torch.sigmoid(logits)
        if tta:
            for flip in (2, 3):
                l2 = model(torch.flip(xb, dims=[flip]))
                p = p + torch.flip(torch.sigmoid(l2), dims=[flip])
            p = p / 3.0
        probs.extend(list(p[:, 0].detach().cpu().numpy().astype(np.float32)))

    prob_padded = stitch_tiles(probs, coords, padded_chw.shape[1:])
    prob = prob_padded[:oh, :ow]
    mask = prob >= float(threshold)
    meta = {
        "tile": tile,
        "overlap": overlap,
        "n_tiles": len(tiles),
        "tta": bool(tta),
        "threshold": float(threshold),
        "device": device,
    }
    return prob.astype(np.float32), mask, meta
