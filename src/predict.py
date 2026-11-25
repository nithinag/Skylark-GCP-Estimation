"""Two-stage test-set inference generating predictions.json matching the required format."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.dataset import (
    CLASS_NAMES,
    IDX_TO_CLASS,
    STAGE1_HM_STRIDE,
    STAGE1_SIZE,
    STAGE2_CROP_SIZE,
    STAGE2_HM_STRIDE,
    crop_to_orig,
    extract_native_crop,
    letterbox_to_orig,
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


def list_test_images(test_dir: Path) -> list[Path]:
    """Return sorted relative image paths under test_dir."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(p.relative_to(test_dir) for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


@torch.no_grad()
def predict_dataset(
    test_dir: Path,
    stage1_weights: Path,
    stage2_weights: Path,
    output_json: Path,
    vis_output: Path | None = None,
) -> dict:
    """Run Stage 1 + Stage 2 inference across all test images and validate output JSON."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s1_model = Stage1CoarseModel(pretrained=False).to(device)
    s2_model = Stage2FineModel(pretrained=False).to(device)
    load_checkpoint(stage1_weights, s1_model, map_location=device)
    load_checkpoint(stage2_weights, s2_model, map_location=device)
    s1_model.eval()
    s2_model.eval()

    rel_paths = list_test_images(test_dir)
    if not rel_paths:
        raise FileNotFoundError(f"No images found under {test_dir}")

    predictions: dict[str, dict] = {}
    vis_items = []

    for rel in tqdm(rel_paths, desc="Test Inference"):
        rel_str = rel.as_posix()
        orig_img = read_rgb_image(test_dir / rel)
        orig_h, orig_w = orig_img.shape[:2]

        lb_img, lb_meta = resize_and_pad(orig_img, target_size=STAGE1_SIZE)
        lb_tensor = normalize_image(lb_img).unsqueeze(0).to(device)
        hm1 = s1_model(lb_tensor)
        xy1_in = decode_heatmap_peak(hm1, stride=STAGE1_HM_STRIDE)[0].cpu().numpy()
        cx_coarse, cy_coarse = letterbox_to_orig(float(xy1_in[0]), float(xy1_in[1]), lb_meta)

        crop_img, crop_meta = extract_native_crop(orig_img, cx_coarse, cy_coarse, crop_size=STAGE2_CROP_SIZE)
        crop_tensor = normalize_image(crop_img).unsqueeze(0).to(device)
        hm2, cls_logits = s2_model(crop_tensor)
        xy2_crop = decode_heatmap_peak(hm2, stride=STAGE2_HM_STRIDE)[0].cpu().numpy()
        x_final, y_final = crop_to_orig(float(xy2_crop[0]), float(xy2_crop[1]), crop_meta)

        pred_idx = int(cls_logits.argmax(dim=-1).item())
        pred_shape = IDX_TO_CLASS[pred_idx]

        if not (0.0 <= x_final < orig_w and 0.0 <= y_final < orig_h):
            raise ValueError(f"Predicted coordinate out of bounds for {rel_str}: ({x_final}, {y_final}) vs {orig_w}x{orig_h}")
        if pred_shape not in CLASS_NAMES:
            raise ValueError(f"Invalid predicted shape {pred_shape}")

        predictions[rel_str] = {
            "mark": {"x": round(float(x_final), 2), "y": round(float(y_final), 2)},
            "verified_shape": pred_shape,
        }

        if vis_output is not None and len(vis_items) < 6:
            vis_items.append((crop_img, crop_meta, rel_str, x_final, y_final, pred_shape))

    if len(predictions) != len(rel_paths):
        raise RuntimeError(f"Prediction count mismatch: {len(predictions)} != {len(rel_paths)}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    if vis_output is not None and vis_items:
        _plot_test_samples(vis_items, vis_output)

    counts = Counter(v["verified_shape"] for v in predictions.values())
    print(f"Saved {len(predictions)} predictions to {output_json}")
    print(f"Predicted shape distribution: {dict(counts)}")
    return predictions


def _plot_test_samples(vis_items: list[tuple], out_path: Path) -> None:
    """Plot sample test predictions on native crops for quick visual verification."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(vis_items):
            ax.axis("off")
            continue
        crop_img, crop_meta, rel_str, x_pred, y_pred, shape = vis_items[idx]
        ax.imshow(crop_img)
        ax.scatter([x_pred - crop_meta["x1"]], [y_pred - crop_meta["y1"]], s=100, c="red", marker="+", linewidths=2)
        ax.set_title(f"{shape} ({x_pred:.1f}, {y_pred:.1f})\n{Path(rel_str).name}", fontsize=8.5)
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Two-Stage inference on test_dataset and write predictions.json.")
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_dataset"))
    parser.add_argument("--stage1-weights", type=Path, default=Path("weights/stage1_best.pt"))
    parser.add_argument("--stage2-weights", type=Path, default=Path("weights/stage2_best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.json"))
    parser.add_argument("--vis-output", type=Path, default=Path("outputs/test_predictions_sample.png"))
    args = parser.parse_args()

    predict_dataset(
        test_dir=args.test_dir,
        stage1_weights=args.stage1_weights,
        stage2_weights=args.stage2_weights,
        output_json=args.output,
        vis_output=args.vis_output,
    )


if __name__ == "__main__":
    main()
