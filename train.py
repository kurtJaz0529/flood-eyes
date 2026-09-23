"""
慧眼识灾 · U-Net 训练脚本
=========================

一条命令完成训练 + 验证 + 曲线记录：

    # 用仓库自带的合成样本先跑通链路（CPU 几分钟）
    python train.py --data data/samples --epochs 30 --img-size 256 --arch tiny

    # 用真实数据 Sen1Floods11 训练（推荐，GPU）
    python train.py --data data/sen1floods11 --epochs 60 --img-size 512 \
        --arch smp --encoder resnet34 --encoder-weights imagenet --batch-size 8

设计要点：
    1. 场景级划分训练/验证（同一景的切片不跨集合），避免数据泄漏
    2. Dice + BCE 组合损失，缓解水体/陆地极度不平衡
    3. 归一化统计量随权重一起保存，推理端直接复用，避免"训练-推理不一致"
    4. 记录 IoU / Dice / Precision / Recall 到 CSV + 曲线图，作为大赛佐证材料
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_unet import (  # noqa: E402
    DiceBCELoss,
    build_model,
    count_parameters,
    save_checkpoint,
    segmentation_metrics,
)
from src.preprocess import DEFAULT_MEAN, DEFAULT_STD, load_scene, normalize  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------


def _read_mask(path: str) -> np.ndarray:
    from PIL import Image

    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 127).astype(np.uint8)


def discover_samples(data_dir: str) -> List[Dict[str, str]]:
    """在数据目录中找出 (影像, 掩膜) 配对。

    支持两种组织方式：
        1. 本仓库合成样本：demoXX_post.tif + demoXX_mask.png
        2. Sen1Floods11 风格：*_img.tif + *_mask.tif / *_label.png
    """
    items: List[Dict[str, str]] = []
    files = sorted(os.listdir(data_dir))
    for f in files:
        stem, ext = os.path.splitext(f)
        if ext.lower() not in (".tif", ".tiff"):
            continue
        if not (stem.endswith("_post") or stem.endswith("_img") or stem.endswith("_S2Hand")):
            continue
        base = stem.replace("_post", "").replace("_img", "").replace("_S2Hand", "")
        for cand in (f"{base}_mask.png", f"{base}_mask.tif", f"{base}_label.png", f"{stem}_mask.png"):
            p = os.path.join(data_dir, cand)
            if os.path.isfile(p):
                items.append({"image": os.path.join(data_dir, f), "mask": p, "id": base})
                break
    return items


class FloodDataset(Dataset):
    """按需读取 + 随机裁剪增广。影像 4 通道 [蓝,绿,红,近红外]，掩膜 0/1。"""

    def __init__(
        self,
        items: Sequence[Dict[str, str]],
        img_size: int = 256,
        augment: bool = True,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        band_order: str = "auto",
        repeat: int = 1,
    ):
        self.items = list(items)
        self.img_size = int(img_size)
        self.augment = augment
        self.mean = tuple(mean) if mean else DEFAULT_MEAN
        self.std = tuple(std) if std else DEFAULT_STD
        self.band_order = band_order
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:
        return len(self.items) * self.repeat

    def _load(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        item = self.items[idx % len(self.items)]
        scene = load_scene(item["image"], band_order=self.band_order)
        mask = _read_mask(item["mask"])
        # 通道顺序必须固定为 blue/green/red/nir：模型与归一化均按此顺序定义。
        # 原实现先过滤缺失波段、再把补零通道追加到末尾，一旦中间少一个波段
        # （例如缺 red），实际堆叠会变成 [blue, green, nir, 0] 而模型仍按
        # [blue, green, red, nir] 解释——训练不报错，只是精度莫名下降。
        required = ("blue", "green", "red", "nir")
        missing_bands = [b for b in required if b not in scene.bands]
        if missing_bands:
            raise ValueError(
                f"{item['image']} 缺少波段 {missing_bands}，"
                f"实际只有 {sorted(scene.bands)}；训练要求 4 波段（blue/green/red/nir）齐全。"
            )
        chw = scene.stack(list(required))  # (4,H,W) 反射率，顺序固定
        if mask.shape != chw.shape[1:]:
            raise ValueError(f"{item['image']} 与 {item['mask']} 尺寸不一致：{chw.shape[1:]} vs {mask.shape}")
        return chw, mask.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chw, mask = self._load(idx)
        c, h, w = chw.shape
        s = self.img_size

        if h < s or w < s:  # 影像小于裁剪尺寸 -> 反射补边
            ph, pw = max(0, s - h), max(0, s - w)
            chw = np.pad(chw, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
            c, h, w = chw.shape

        if self.augment:
            y0 = np.random.randint(0, h - s + 1)
            x0 = np.random.randint(0, w - s + 1)
        else:  # 验证集取中心区域
            y0, x0 = (h - s) // 2, (w - s) // 2
        img = chw[:, y0 : y0 + s, x0 : x0 + s]
        msk = mask[y0 : y0 + s, x0 : x0 + s]

        if self.augment:
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k, axes=(1, 2))
                msk = np.rot90(msk, k, axes=(0, 1))
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1]
                msk = msk[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]
                msk = msk[::-1, :]
            # 轻微光谱抖动，提升对时相/大气差异的鲁棒性
            img = img * (1.0 + 0.06 * np.random.randn(img.shape[0], 1, 1).astype(np.float32))

        img = np.ascontiguousarray(img.astype(np.float32))
        msk = np.ascontiguousarray(msk.astype(np.float32))
        x = torch.from_numpy(normalize(img, self.mean, self.std))
        y = torch.from_numpy(msk[None, ...])
        return x, y


def compute_stats(items: Sequence[Dict[str, str]], max_pixels: int = 400_000) -> Tuple[List[float], List[float]]:
    """从训练集统计各通道均值/标准差（只用训练集，避免验证集信息泄漏）。"""
    acc = np.zeros(4, dtype=np.float64)
    acc2 = np.zeros(4, dtype=np.float64)
    n = 0
    for it in items:
        scene = load_scene(it["image"])
        names = [b for b in ("blue", "green", "red", "nir") if b in scene.bands]
        chw = scene.stack(names)
        if chw.shape[0] < 4:
            chw = np.concatenate([chw, np.zeros((4 - chw.shape[0], *chw.shape[1:]), np.float32)], 0)
        flat = chw.reshape(4, -1)[:, :: max(1, chw[0].size // max(1, max_pixels // max(1, len(items))))]
        acc += flat.sum(axis=1)
        acc2 += (flat ** 2).sum(axis=1)
        n += flat.shape[1]
    mean = acc / max(n, 1)
    var = np.maximum(acc2 / max(n, 1) - mean ** 2, 1e-8)
    return mean.astype(float).tolist(), np.sqrt(var).astype(float).tolist()


# --------------------------------------------------------------------------
# 训练 / 验证
# --------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: Optional[Any] = None,
) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()[:, 0]
        gt = y.numpy()[:, 0] > 0.5
        m = segmentation_metrics(prob, gt)
        for k in agg:
            agg[k] += m[k]
    tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
    eps = 1e-7
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def plot_curve(log_path: str, out_path: str) -> Optional[str]:
    """把训练日志画成曲线图（README/答辩材料用）。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
        if not rows:
            return None
        ep = [int(r["epoch"]) for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train loss", color="#d9534f")
        axes[0].plot(ep, [float(r["val_loss"]) for r in rows], label="val loss", color="#0275d8")
        axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ep, [float(r["val_iou"]) for r in rows], label="val IoU", color="#5cb85c")
        axes[1].plot(ep, [float(r["val_dice"]) for r in rows], label="val Dice", color="#f0ad4e")
        axes[1].set_title("Val IoU / Dice"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].plot(ep, [float(r["val_precision"]) for r in rows], label="precision", color="#8e44ad")
        axes[2].plot(ep, [float(r["val_recall"]) for r in rows], label="recall", color="#16a085")
        axes[2].set_title("Precision / Recall"); axes[2].set_xlabel("epoch"); axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path
    except Exception as exc:  # pragma: no cover
        print(f"[warn] 曲线绘制失败：{exc}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="慧眼识灾 U-Net 训练脚本")
    ap.add_argument("--data", default="data/samples", help="数据目录（含 *_post.tif 与 *_mask.png）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--arch", default="auto", choices=["auto", "smp", "tiny"], help="auto=有smp用smp")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--encoder-weights", default=None, help="如 imagenet（需联网下载）")
    ap.add_argument("--base", type=int, default=16, help="TinyUNet 基础通道数")
    ap.add_argument("--val-id", default=None, help="指定验证场景 id（默认按 1/6 划分）")
    ap.add_argument("--val-ratio", type=float, default=1 / 6, help="验证集比例（场景级）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", default="weights/best_model.pt")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--bce-weight", type=float, default=0.5)
    ap.add_argument("--dice-weight", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=0, help=">0 时启用早停")
    ap.add_argument("--resume", default=None, help="从权重继续训练")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if device == "cuda":
        print(f"[env] GPU: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ---- 数据划分 ----
    items = discover_samples(args.data)
    if not items:
        raise SystemExit(
            f"在 {args.data} 中没有找到 (影像, 掩膜) 配对。\n"
            "· 先跑 python data/make_samples.py 生成演示样本；\n"
            "· 或执行 python scripts/download_data.py --dataset sen1floods11 下载真实数据。"
        )
    random.Random(args.seed).shuffle(items)
    if args.val_id:
        val_items = [i for i in items if i["id"] == args.val_id]
        train_items = [i for i in items if i["id"] != args.val_id]
    else:
        n_val = max(1, int(round(len(items) * args.val_ratio)))
        val_items, train_items = items[:n_val], items[n_val:]
    if not train_items:
        train_items = val_items
    print(f"[data] 训练场景 {len(train_items)} 个，验证场景 {len(val_items)} 个：")
    print(f"        train={[i['id'] for i in train_items]}")
    print(f"        val  ={[i['id'] for i in val_items]}")

    # ---- 归一化统计（只统计训练集）----
    mean, std = compute_stats(train_items)
    print(f"[norm] mean={[round(m,4) for m in mean]}")
    print(f"[norm] std ={[round(s,4) for s in std]}")

    train_ds = FloodDataset(train_items, args.img_size, augment=True, mean=mean, std=std, repeat=4)
    val_ds = FloodDataset(val_items, args.img_size, augment=False, mean=mean, std=std, repeat=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # ---- 模型 ----
    model, info = build_model(
        arch=args.arch,
        encoder=args.encoder,
        in_channels=4,
        classes=1,
        encoder_weights=args.encoder_weights,
        base=args.base,
    )
    if "smp_fallback_reason" in info:
        print(f"[warn] smp 不可用，已退回内置 TinyUNet：{info['smp_fallback_reason']}")
    model.to(device)
    print(f"[model] {info}  参数量={count_parameters(model)/1e6:.2f} M")

    if args.resume and os.path.isfile(args.resume):
        # 断点权重可能来自他人分享/网盘，必须安全反序列化：
        # weights_only=False 会在 load_state_dict 之前执行 pickle 中的任意代码。
        try:
            state = torch.load(args.resume, map_location=device, weights_only=True)
        except Exception as exc:
            raise SystemExit(
                f"[model] 断点文件无法安全加载：{args.resume}（{type(exc).__name__}: {exc}）。\n"
                f"        本程序只接受标准 torch.save 保存的检查点；请确认文件来源可信且未损坏。"
            ) from exc
        # 严格加载：缺失/多余键说明结构不一致，静默放行会让部分层保持随机初始化。
        model.load_state_dict(state["state_dict"], strict=True)
        print(f"[model] 已加载 {args.resume} 继续训练")

    criterion = DiceBCELoss(args.bce_weight, args.dice_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    scaler = None
    if device == "cuda":  # 混合精度只在 GPU 上开启
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:  # 兼容旧版 PyTorch
            scaler = torch.cuda.amp.GradScaler()

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train_log.csv")
    fields = ["epoch", "lr", "train_loss", "val_loss", "val_iou", "val_dice", "val_precision", "val_recall", "val_accuracy", "seconds"]
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best_iou, best_epoch, no_improve = -1.0, -1, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = float("nan")
        with torch.no_grad():
            model.eval()
            vl, n = 0.0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vl += float(criterion(model(x), y)) * x.size(0)
                n += x.size(0)
            val_loss = vl / max(n, 1)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        row = {
            "epoch": epoch, "lr": f"{lr:.6f}", "train_loss": f"{train_loss:.5f}", "val_loss": f"{val_loss:.5f}",
            "val_iou": f"{metrics['iou']:.5f}", "val_dice": f"{metrics['dice']:.5f}",
            "val_precision": f"{metrics['precision']:.5f}", "val_recall": f"{metrics['recall']:.5f}",
            "val_accuracy": f"{metrics['accuracy']:.5f}", "seconds": f"{dt:.1f}",
        }
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(row)
        print(
            f"[{epoch:3d}/{args.epochs}] loss {train_loss:.4f}/{val_loss:.4f}  "
            f"IoU {metrics['iou']:.4f}  Dice {metrics['dice']:.4f}  "
            f"P {metrics['precision']:.4f}  R {metrics['recall']:.4f}  {dt:.1f}s"
        )

        if metrics["iou"] > best_iou:
            best_iou, best_epoch, no_improve = metrics["iou"], epoch, 0
            meta = {
                **info,
                "in_channels": 4,
                "classes": 1,
                "img_size": args.img_size,
                "mean": mean,
                "std": std,
                "val_iou": float(metrics["iou"]),
                "val_dice": float(metrics["dice"]),
                "val_precision": float(metrics["precision"]),
                "val_recall": float(metrics["recall"]),
                "best_epoch": epoch,
                "epochs_run": epoch,
                "trained_on": os.path.basename(os.path.normpath(args.data)),
                "train_scenes": [i["id"] for i in train_items],
                "val_scenes": [i["id"] for i in val_items],
                "train_loss": float(train_loss),
                "data_note": "synthetic demo data" if "samples" in args.data else "real imagery",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_checkpoint(args.out, model, meta)
            print(f"        ↑ 保存最优权重 {args.out}（IoU {best_iou:.4f}）")
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"[early-stop] {args.patience} 轮无提升，提前结束")
                break

    print(f"\n[done] 最佳验证 IoU {best_iou:.4f} @ epoch {best_epoch}，总耗时 {time.time()-t_start:.1f}s")
    print(f"[done] 日志 {log_path}")
    curve = plot_curve(log_path, os.path.join(args.log_dir, "train_curve.png"))
    if curve:
        print(f"[done] 曲线图 {curve}")
    summary = {
        "best_iou": best_iou, "best_epoch": best_epoch, "arch": info.get("arch"),
        "encoder": info.get("encoder"), "params_M": count_parameters(model) / 1e6,
        "train_scenes": [i["id"] for i in train_items], "val_scenes": [i["id"] for i in val_items],
        "data_note": "synthetic demo data" if "samples" in args.data else "real imagery",
    }
    with open(os.path.join(args.log_dir, "train_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
