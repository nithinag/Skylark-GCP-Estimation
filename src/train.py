"""Training script for Two-Stage Coarse-to-Fine Aerial GCP Pose Estimation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    CLASS_NAMES,
    IGNORE_INDEX,
    STAGE1_HM_STRIDE,
    STAGE1_SIZE,
    STAGE2_CROP_SIZE,
    STAGE2_HM_STRIDE,
    Stage1CoarseDataset,
    Stage2FineDataset,
    build_group_splits,
    compute_class_weights,
    load_annotations,
    prepare_stage1_cache,
    prepare_stage2_cache,
)
from src.evaluate import evaluate_stage1, evaluate_stage2_oracle
from src.model import (
    GaussianFocalLoss,
    Stage1CoarseModel,
    Stage2FineModel,
    compute_localization_loss,
    load_checkpoint,
    save_checkpoint,
)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _should_use_amp(device: torch.device, no_amp: bool) -> bool:
    """Enable CUDA AMP unless disabled or running on GTX 16xx GPUs (known FP16 NaN issues)."""
    if no_amp or device.type != "cuda":
        return False
    gpu_name = torch.cuda.get_device_name(0)
    if "GTX 16" in gpu_name:
        print(f"Detected {gpu_name}: using FP32 precision for numerical stability.")
        return False
    return True


def train_stage1(
    train_samples: list,
    val_samples: list,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    """Train Stage 1 coarse heatmap localizer on full letterboxed images."""
    if args.cache_dir is not None and not args.smoke_test:
        print(f"Preparing Stage 1 image cache in {args.cache_dir} ...")
        prepare_stage1_cache(train_samples + val_samples, args.data_dir, args.cache_dir, target_size=STAGE1_SIZE)

    train_ds = Stage1CoarseDataset(
        train_samples,
        args.data_dir,
        augment=True,
        cache_dir=args.cache_dir if not args.smoke_test else None,
        seed=args.seed,
    )
    val_ds = Stage1CoarseDataset(
        val_samples,
        args.data_dir,
        augment=False,
        cache_dir=args.cache_dir if not args.smoke_test else None,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_s1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_s1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = Stage1CoarseModel(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_s1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_s1))
    focal_loss_fn = GaussianFocalLoss()
    use_amp = _should_use_amp(device, args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1e9
    no_improve = 0
    best_path = args.weights_dir / "stage1_best.pt"
    history = []
    grad_accum = max(1, args.grad_accum)

    for epoch in range(1, args.epochs_s1 + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"[Stage 1] Epoch {epoch}/{args.epochs_s1}", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar, 1):
            images = batch["image"].to(device, non_blocking=True)
            heatmaps = batch["heatmap"].to(device, non_blocking=True)
            xy_in = batch["xy_in"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                hm_logits = model(images)
                loss, hm_l, coord_l = compute_localization_loss(
                    hm_logits,
                    heatmaps,
                    xy_in,
                    stride=STAGE1_HM_STRIDE,
                    focal_loss_fn=focal_loss_fn,
                    coord_weight=0.05,
                )

            scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", hm=f"{hm_l.item():.4f}", xy=f"{coord_l.item():.2f}")

            if args.smoke_test and steps >= 2:
                break

        scheduler.step()
        avg_loss = running_loss / max(1, steps)

        if args.smoke_test:
            save_checkpoint(best_path, model, optimizer, epoch=epoch, metrics={"train_loss": avg_loss})
            check_model = Stage1CoarseModel(pretrained=False).to(device)
            load_checkpoint(best_path, check_model, map_location=device)
            print(f"[Smoke Test Stage 1] loss={avg_loss:.4f} | checkpoint save/load OK -> {best_path}")
            return best_path

        val_metrics = evaluate_stage1(model, val_loader, device)
        val_score = val_metrics["pck_25px"] - 0.01 * val_metrics["median_err_px"]
        history.append({"epoch": epoch, "train_loss": avg_loss, **val_metrics})

        print(
            f"[Stage 1] Epoch {epoch:02d} | loss={avg_loss:.4f} | "
            f"PCK@10={val_metrics['pck_10px']:.1f}% | PCK@25={val_metrics['pck_25px']:.1f}% | "
            f"PCK@50={val_metrics['pck_50px']:.1f}% | med_err={val_metrics['median_err_px']:.1f}px",
            flush=True,
        )

        if val_score > best_score:
            best_score = val_score
            no_improve = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config={"stage": 1, "input_size": list(STAGE1_SIZE), "hm_stride": STAGE1_HM_STRIDE},
            )
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[Stage 1] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    (args.outputs_dir / "train_history_stage1.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path


def train_stage2(
    train_samples: list,
    val_samples: list,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    """Train Stage 2 multi-task fine localizer and 3-class shape classifier."""
    class_weights = compute_class_weights(train_samples).to(device)
    print(f"Stage 2 inverse-frequency class weights ({CLASS_NAMES}): {class_weights.tolist()}")

    if args.stage2_cache_dir is not None and not args.smoke_test:
        print(f"Preparing Stage 2 native-crop cache in {args.stage2_cache_dir} ...")
        prepare_stage2_cache(train_samples + val_samples, args.data_dir, args.stage2_cache_dir, patch_size=640)

    train_ds = Stage2FineDataset(
        train_samples,
        args.data_dir,
        crop_size=STAGE2_CROP_SIZE,
        max_jitter_px=110.0,
        augment=True,
        cache_dir=args.stage2_cache_dir if not args.smoke_test else None,
        seed=args.seed,
    )
    val_ds = Stage2FineDataset(
        val_samples,
        args.data_dir,
        crop_size=STAGE2_CROP_SIZE,
        max_jitter_px=0.0,
        augment=False,
        cache_dir=args.stage2_cache_dir if not args.smoke_test else None,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_s2,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_s2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = Stage2FineModel(num_classes=len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_s2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_s2))

    focal_loss_fn = GaussianFocalLoss()
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    use_amp = _should_use_amp(device, args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1e9
    no_improve = 0
    best_path = args.weights_dir / "stage2_best.pt"
    history = []
    grad_accum = max(1, args.grad_accum)

    for epoch in range(1, args.epochs_s2 + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"[Stage 2] Epoch {epoch}/{args.epochs_s2}", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar, 1):
            images = batch["image"].to(device, non_blocking=True)
            heatmaps = batch["heatmap"].to(device, non_blocking=True)
            xy_crop = batch["xy_crop"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                hm_logits, cls_logits = model(images)
                loc_loss, hm_l, coord_l = compute_localization_loss(
                    hm_logits,
                    heatmaps,
                    xy_crop,
                    stride=STAGE2_HM_STRIDE,
                    focal_loss_fn=focal_loss_fn,
                    coord_weight=0.05,
                )
                if (labels != IGNORE_INDEX).any():
                    cls_loss = ce_loss_fn(cls_logits, labels)
                else:
                    cls_loss = cls_logits.sum() * 0.0
                loss = loc_loss + 1.5 * cls_loss

            scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", loc=f"{loc_loss.item():.4f}", cls=f"{cls_loss.item():.4f}")

            if args.smoke_test and steps >= 2:
                break

        scheduler.step()
        avg_loss = running_loss / max(1, steps)

        if args.smoke_test:
            save_checkpoint(best_path, model, optimizer, epoch=epoch, metrics={"train_loss": avg_loss})
            check_model = Stage2FineModel(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
            load_checkpoint(best_path, check_model, map_location=device)
            print(f"[Smoke Test Stage 2] loss={avg_loss:.4f} | checkpoint save/load OK -> {best_path}")
            return best_path

        val_metrics = evaluate_stage2_oracle(model, val_loader, device)
        val_score = 0.5 * (val_metrics["pck_10px"] / 100.0) + 0.5 * val_metrics["macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "pck_10px": val_metrics["pck_10px"],
                "pck_25px": val_metrics["pck_25px"],
                "median_err_px": val_metrics["median_err_px"],
                "macro_f1": val_metrics["macro_f1"],
                "accuracy": val_metrics["accuracy"],
            }
        )

        print(
            f"[Stage 2] Epoch {epoch:02d} | loss={avg_loss:.4f} | "
            f"PCK@10={val_metrics['pck_10px']:.1f}% | med_err={val_metrics['median_err_px']:.2f}px | "
            f"Macro-F1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if val_score > best_score:
            best_score = val_score
            no_improve = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                metrics={k: v for k, v in val_metrics.items() if k != "report"},
                config={"stage": 2, "crop_size": STAGE2_CROP_SIZE, "hm_stride": STAGE2_HM_STRIDE},
            )
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[Stage 2] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    (args.outputs_dir / "train_history_stage2.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Two-Stage Coarse-to-Fine Aerial GCP Pose Estimation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/train_dataset"))
    parser.add_argument("--annotations", type=Path, default=Path("data/train_dataset/gcp_marks.json"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.cache_stage1"))
    parser.add_argument("--stage2-cache-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    parser.add_argument("--val-fold", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs-s1", type=int, default=35)
    parser.add_argument("--epochs-s2", type=int, default=40)
    parser.add_argument("--batch-s1", type=int, default=8)
    parser.add_argument("--batch-s2", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-s1", type=float, default=3e-4)
    parser.add_argument("--lr-s2", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Force FP32 even on CUDA.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet weights (offline/smoke tests).")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-batch verification and checkpoint test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = load_annotations(args.annotations)
    train_samples, val_samples = build_group_splits(
        samples,
        n_splits=args.n_splits,
        val_fold=args.val_fold,
        seed=args.seed,
    )
    print(
        f"Split summary -> train: {len(train_samples)} ({len({s.project for s in train_samples})} projects), "
        f"val: {len(val_samples)} ({len({s.project for s in val_samples})} projects)"
    )

    if args.smoke_test:
        args.epochs_s1 = 1
        args.epochs_s2 = 1
        args.batch_s1 = 2
        args.batch_s2 = 2
        args.num_workers = 0

    if args.stage in ("1", "both"):
        train_stage1(train_samples, val_samples, args, device)
    if args.stage in ("2", "both"):
        train_stage2(train_samples, val_samples, args, device)


if __name__ == "__main__":
    main()

