"""Evaluation metrics (PCK@10/25/50px, Macro F1-score) and end-to-end two-stage validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.dataset import (
    CLASS_NAMES,
    GCPSample,
    IDX_TO_CLASS,
    IGNORE_INDEX,
    STAGE1_HM_STRIDE,
    STAGE1_SIZE,
    STAGE2_CROP_SIZE,
    STAGE2_HM_STRIDE,
    build_group_splits,
    crop_to_orig,
    extract_native_crop,
    letterbox_to_orig,
    load_annotations,
    normalize_image,
    read_rgb_image,
    resize_and_pad,
)
from src.model import (
    Stage1CoarseModel,
    Stage2FineModel,
    decode_heatmap_peak,
    load_checkpoint,
)


def compute_pck(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    thresholds: tuple[float, ...] = (10.0, 25.0, 50.0),
) -> dict[str, float]:
    """Compute Percentage of Correct Keypoints (in %) and distance stats in original pixel space."""
    dists = np.linalg.norm(pred_xy - gt_xy, axis=1)
    metrics: dict[str, float] = {
        "mean_err_px": float(np.mean(dists)),
        "median_err_px": float(np.median(dists)),
    }
    for thr in thresholds:
        key = f"pck_{int(thr)}px"
        metrics[key] = float(np.mean(dists <= thr) * 100.0)
    return metrics


def compute_classification_metrics(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
) -> dict[str, Any]:
    """Compute Macro F1-score and per-class stats, ignoring unlabeled (IGNORE_INDEX) entries."""
    valid_mask = gt_labels != IGNORE_INDEX
    y_true = gt_labels[valid_mask]
    y_pred = pred_labels[valid_mask]

    if len(y_true) == 0:
        return {"macro_f1": 0.0, "accuracy": 0.0, "num_labeled": 0}

    macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    acc = float(np.mean(y_true == y_pred) * 100.0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=list(CLASS_NAMES),
        zero_division=0,
    )
    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "num_labeled": int(len(y_true)),
        "per_class_f1": {CLASS_NAMES[i]: float(per_class[i]) for i in range(len(CLASS_NAMES))},
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


@torch.no_grad()
def evaluate_stage1(
    model: Stage1CoarseModel,
    loader: DataLoader,
    device: torch.device,
    hm_stride: int = STAGE1_HM_STRIDE,
) -> dict[str, float]:
    """Evaluate Stage 1 coarse model in original image coordinates."""
    model.eval()
    all_pred_orig, all_gt_orig = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        hm_logits = model(images)
        pred_in = decode_heatmap_peak(hm_logits, stride=hm_stride).cpu().numpy()
        gt_orig = batch["xy_orig"].numpy()

        metas = batch["meta"]
        for i in range(len(pred_in)):
            meta_i = {k: float(metas[k][i]) for k in metas}
            px, py = letterbox_to_orig(float(pred_in[i, 0]), float(pred_in[i, 1]), meta_i)
            all_pred_orig.append([px, py])
        all_gt_orig.append(gt_orig)

    pred_arr = np.array(all_pred_orig, dtype=np.float32)
    gt_arr = np.concatenate(all_gt_orig, axis=0)
    return compute_pck(pred_arr, gt_arr)


@torch.no_grad()
def evaluate_stage2_oracle(
    model: Stage2FineModel,
    loader: DataLoader,
    device: torch.device,
    hm_stride: int = STAGE2_HM_STRIDE,
) -> dict[str, Any]:
    """Evaluate Stage 2 model on ground-truth-centered crops (diagnostic upper bound)."""
    model.eval()
    all_pred_orig, all_gt_orig = [], []
    all_pred_cls, all_gt_cls = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        hm_logits, cls_logits = model(images)
        pred_crop = decode_heatmap_peak(hm_logits, stride=hm_stride).cpu().numpy()
        pred_cls = cls_logits.argmax(dim=-1).cpu().numpy()

        gt_orig = batch["xy_orig"].numpy()
        gt_cls = batch["label"].numpy()
        metas = batch["crop_meta"]

        for i in range(len(pred_crop)):
            meta_i = {k: float(metas[k][i]) for k in metas}
            px, py = crop_to_orig(float(pred_crop[i, 0]), float(pred_crop[i, 1]), meta_i)
            all_pred_orig.append([px, py])

        all_gt_orig.append(gt_orig)
        all_pred_cls.append(pred_cls)
        all_gt_cls.append(gt_cls)

    pck = compute_pck(np.array(all_pred_orig, dtype=np.float32), np.concatenate(all_gt_orig, axis=0))
    cls_metrics = compute_classification_metrics(
        np.concatenate(all_pred_cls, axis=0),
        np.concatenate(all_gt_cls, axis=0),
    )
    return {**pck, **cls_metrics}


@torch.no_grad()
def evaluate_two_stage_e2e(
    stage1_model: Stage1CoarseModel,
    stage2_model: Stage2FineModel,
    samples: list[GCPSample],
    data_root: str | Path,
    device: torch.device,
    stage1_size: tuple[int, int] = STAGE1_SIZE,
    crop_size: int = STAGE2_CROP_SIZE,
    vis_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run full end-to-end two-stage inference on validation samples and report original-pixel metrics."""
    stage1_model.eval()
    stage2_model.eval()
    data_root = Path(data_root)

    s1_preds, s2_preds, gt_xy = [], [], []
    pred_labels, gt_labels = [], []
    vis_records = []

    for s in samples:
        orig_img = read_rgb_image(data_root / s.rel_path)

        lb_img, lb_meta = resize_and_pad(orig_img, target_size=stage1_size)
        lb_tensor = normalize_image(lb_img).unsqueeze(0).to(device)
        hm1 = stage1_model(lb_tensor)
        xy1_in = decode_heatmap_peak(hm1, stride=STAGE1_HM_STRIDE)[0].cpu().numpy()
        cx_coarse, cy_coarse = letterbox_to_orig(float(xy1_in[0]), float(xy1_in[1]), lb_meta)

        crop_img, crop_meta = extract_native_crop(orig_img, cx_coarse, cy_coarse, crop_size=crop_size)
        crop_tensor = normalize_image(crop_img).unsqueeze(0).to(device)
        hm2, cls_logits = stage2_model(crop_tensor)
        xy2_crop = decode_heatmap_peak(hm2, stride=STAGE2_HM_STRIDE)[0].cpu().numpy()
        x_final, y_final = crop_to_orig(float(xy2_crop[0]), float(xy2_crop[1]), crop_meta)
        pred_cls = int(cls_logits.argmax(dim=-1).item())

        s1_preds.append([cx_coarse, cy_coarse])
        s2_preds.append([x_final, y_final])
        gt_xy.append([s.x, s.y])
        pred_labels.append(pred_cls)
        gt_labels.append(s.label)

        if vis_path is not None and len(vis_records) < 6:
            err = float(np.hypot(x_final - s.x, y_final - s.y))
            vis_records.append((crop_img, crop_meta, s, x_final, y_final, pred_cls, err))

    s1_arr = np.array(s1_preds, dtype=np.float32)
    s2_arr = np.array(s2_preds, dtype=np.float32)
    gt_arr = np.array(gt_xy, dtype=np.float32)

    s1_pck = compute_pck(s1_arr, gt_arr)
    s2_pck = compute_pck(s2_arr, gt_arr)
    cls_metrics = compute_classification_metrics(
        np.array(pred_labels, dtype=np.int64),
        np.array(gt_labels, dtype=np.int64),
    )

    if vis_path is not None and vis_records:
        _plot_val_samples(vis_records, Path(vis_path))

    return {
        "stage1_coarse": s1_pck,
        "end_to_end": {**s2_pck, **cls_metrics},
    }


def _plot_val_samples(vis_records: list[tuple], out_path: Path) -> None:
    """Save visual comparison of ground-truth vs predicted center & class on validation crops."""
    n = len(vis_records)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes_flat = axes.flatten()
    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx >= n:
            ax.axis("off")
            continue
        crop_img, crop_meta, s, x_pred, y_pred, pred_cls, err = vis_records[idx]
        ax.imshow(crop_img)
        gt_xc = s.x - crop_meta["x1"]
        gt_yc = s.y - crop_meta["y1"]
        pr_xc = x_pred - crop_meta["x1"]
        pr_yc = y_pred - crop_meta["y1"]
        ax.scatter([gt_xc], [gt_yc], s=110, facecolors="none", edgecolors="lime", linewidths=2, label="GT")
        ax.scatter([pr_xc], [pr_yc], s=60, c="red", marker="x", linewidths=2, label="Pred")
        gt_name = s.shape_str or "None"
        pred_name = IDX_TO_CLASS[pred_cls]
        ax.set_title(f"GT: {gt_name} | Pred: {pred_name}\nErr: {err:.1f}px", fontsize=9)
        ax.axis("off")
    axes_flat[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Two-Stage GCP Pose Estimation pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/train_dataset"))
    parser.add_argument("--annotations", type=Path, default=Path("data/train_dataset/gcp_marks.json"))
    parser.add_argument("--stage1-weights", type=Path, default=Path("weights/stage1_best.pt"))
    parser.add_argument("--stage2-weights", type=Path, default=Path("weights/stage2_best.pt"))
    parser.add_argument("--val-fold", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--vis-output", type=Path, default=Path("outputs/val_predictions.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_annotations(args.annotations)
    _, val_samples = build_group_splits(samples, n_splits=args.n_splits, val_fold=args.val_fold)

    s1_model = Stage1CoarseModel(pretrained=False).to(device)
    s2_model = Stage2FineModel(pretrained=False).to(device)
    load_checkpoint(args.stage1_weights, s1_model, map_location=device)
    load_checkpoint(args.stage2_weights, s2_model, map_location=device)

    results = evaluate_two_stage_e2e(
        s1_model,
        s2_model,
        val_samples,
        args.data_dir,
        device=device,
        vis_path=args.vis_output,
    )

    print("=== Stage 1 Coarse Validation (Original Pixels) ===")
    for k, v in results["stage1_coarse"].items():
        print(f"  {k}: {v:.2f}")

    print("\n=== End-to-End Two-Stage Validation ===")
    e2e = results["end_to_end"]
    for k in ("pck_10px", "pck_25px", "pck_50px", "mean_err_px", "median_err_px", "macro_f1", "accuracy"):
        print(f"  {k}: {e2e[k]:.4f}" if "f1" in k else f"  {k}: {e2e[k]:.2f}")
    print("\nClassification Report:")
    print(e2e["report"])


if __name__ == "__main__":
    main()
