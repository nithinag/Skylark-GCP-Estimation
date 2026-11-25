"""Dataset, coordinate transforms, heatmap targets, and GroupKFold splitting for Aerial GCP Pose Estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

CLASS_NAMES: tuple[str, ...] = ("Cross", "Square", "L-Shape")
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
IGNORE_INDEX: int = -1

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

STAGE1_SIZE: tuple[int, int] = (1024, 768)  # (width, height)
STAGE1_HM_STRIDE: int = 2
STAGE2_CROP_SIZE: int = 384
STAGE2_HM_STRIDE: int = 2


@dataclass(frozen=True)
class GCPSample:
    """Single annotated training sample."""

    rel_path: str
    x: float
    y: float
    shape_str: str | None
    label: int
    project: str
    gcp_group: str


def load_annotations(json_path: str | Path) -> list[GCPSample]:
    """Parse gcp_marks.json into typed GCPSample records without inventing missing labels."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw: dict[str, dict[str, Any]] = json.load(f)

    samples: list[GCPSample] = []
    for rel_path, entry in sorted(raw.items()):
        mark = entry["mark"]
        shape_str = entry.get("verified_shape")
        if shape_str is None:
            label = IGNORE_INDEX
        elif shape_str in CLASS_TO_IDX:
            label = CLASS_TO_IDX[shape_str]
        else:
            raise ValueError(f"Unexpected verified_shape '{shape_str}' for {rel_path}")

        parts = rel_path.split("/")
        project = parts[0]
        gcp_group = "/".join(parts[:3])
        samples.append(
            GCPSample(
                rel_path=rel_path,
                x=float(mark["x"]),
                y=float(mark["y"]),
                shape_str=shape_str,
                label=label,
                project=project,
                gcp_group=gcp_group,
            )
        )
    return samples


def build_group_splits(
    samples: list[GCPSample],
    n_splits: int = 5,
    val_fold: int = 3,
    seed: int = 42,
) -> tuple[list[GCPSample], list[GCPSample]]:
    """Split samples by project using StratifiedGroupKFold to prevent data leakage."""
    if not (0 <= val_fold < n_splits):
        raise ValueError(f"val_fold must be in [0, {n_splits - 1}], got {val_fold}")

    groups = [s.project for s in samples]
    strat_y = [s.label if s.label >= 0 else CLASS_TO_IDX["L-Shape"] for s in samples]
    indices = np.arange(len(samples))

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(indices, strat_y, groups))
    train_idx, val_idx = folds[val_fold]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    train_projects = {s.project for s in train_samples}
    val_projects = {s.project for s in val_samples}
    overlap = train_projects & val_projects
    if overlap:
        raise RuntimeError(f"Project leakage detected across splits: {overlap}")

    return train_samples, val_samples


def compute_class_weights(samples: list[GCPSample]) -> torch.Tensor:
    """Compute inverse-frequency balanced class weights from labeled training samples only."""
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for s in samples:
        if s.label >= 0:
            counts[s.label] += 1.0
    if np.any(counts == 0):
        raise ValueError(f"Cannot compute weights with zero-count class: {counts}")
    total = counts.sum()
    weights = total / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB uint8 NumPy array."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resize_and_pad(
    image_rgb: np.ndarray,
    target_size: tuple[int, int] = STAGE1_SIZE,
    pad_value: int = 114,
) -> tuple[np.ndarray, dict[str, float]]:
    """Resize image preserving aspect ratio and center-pad to (target_w, target_h)."""
    orig_h, orig_w = image_rgb.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    meta = {
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "target_w": float(target_w),
        "target_h": float(target_h),
    }
    return canvas, meta


def orig_to_letterbox(x: float, y: float, meta: dict[str, float]) -> tuple[float, float]:
    """Map original image (x, y) into letterboxed Stage 1 coordinates."""
    return x * meta["scale"] + meta["pad_x"], y * meta["scale"] + meta["pad_y"]


def letterbox_to_orig(x_in: float, y_in: float, meta: dict[str, float]) -> tuple[float, float]:
    """Map letterboxed Stage 1 coordinates back into original image (x, y)."""
    x_orig = (x_in - meta["pad_x"]) / meta["scale"]
    y_orig = (y_in - meta["pad_y"]) / meta["scale"]
    x_orig = float(np.clip(x_orig, 0.0, meta["orig_w"] - 1e-3))
    y_orig = float(np.clip(y_orig, 0.0, meta["orig_h"] - 1e-3))
    return x_orig, y_orig


def extract_native_crop(
    image_rgb: np.ndarray,
    center_x: float,
    center_y: float,
    crop_size: int = STAGE2_CROP_SIZE,
    pad_value: int = 114,
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract a native-resolution square crop around (center_x, center_y), padding borders if needed."""
    orig_h, orig_w = image_rgb.shape[:2]
    half = crop_size // 2
    x1 = int(round(center_x)) - half
    y1 = int(round(center_y)) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    canvas = np.full((crop_size, crop_size, 3), pad_value, dtype=np.uint8)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(orig_w, x2)
    src_y2 = min(orig_h, y2)

    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image_rgb[src_y1:src_y2, src_x1:src_x2]

    meta = {
        "x1": float(x1),
        "y1": float(y1),
        "crop_size": float(crop_size),
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
    }
    return canvas, meta


def orig_to_crop(x: float, y: float, crop_meta: dict[str, float]) -> tuple[float, float]:
    """Map original image (x, y) into native crop coordinates."""
    return x - crop_meta["x1"], y - crop_meta["y1"]


def crop_to_orig(xc: float, yc: float, crop_meta: dict[str, float]) -> tuple[float, float]:
    """Map native crop coordinates back into original image (x, y)."""
    x_orig = float(np.clip(xc + crop_meta["x1"], 0.0, crop_meta["orig_w"] - 1e-3))
    y_orig = float(np.clip(yc + crop_meta["y1"], 0.0, crop_meta["orig_h"] - 1e-3))
    return x_orig, y_orig


def make_gaussian_heatmap(
    height: int,
    width: int,
    cx: float,
    cy: float,
    sigma: float = 3.0,
) -> np.ndarray:
    """Create a float32 2D Gaussian heatmap (1, H, W) centered at sub-pixel (cx, cy)."""
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    dist_sq = (xs - np.float32(cx)) ** 2 + (ys - np.float32(cy)) ** 2
    hm = np.exp(-dist_sq / (2.0 * np.float32(sigma) ** 2))
    return np.expand_dims(hm.astype(np.float32), axis=0)


def normalize_image(image_rgb: np.ndarray) -> torch.Tensor:
    """Convert uint8 RGB (H, W, 3) to normalized float32 CHW tensor."""
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy())


def _apply_photometric_aug(
    img: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply lighting, color, and mild blur augmentations common in drone imagery."""
    if rng.random() < 0.7:
        alpha = float(rng.uniform(0.82, 1.18))
        beta = float(rng.uniform(-18.0, 18.0))
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-6.0, 6.0)) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.15), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.88, 1.12), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.6)
    return img


def _apply_geometric_aug(
    img: np.ndarray,
    x: float,
    y: float,
    rng: np.random.Generator,
    allow_rot90: bool,
    max_angle: float,
    scale_range: tuple[float, float],
    shift_frac: float,
    border_margin: float = 16.0,
) -> tuple[np.ndarray, float, float]:
    """Apply coordinate-exact flips, optional 90-degree rotations, and small affine jitter."""
    h, w = img.shape[:2]

    if allow_rot90 and h == w:
        k = int(rng.integers(0, 4))
        if k > 0:
            img = np.ascontiguousarray(np.rot90(img, k=k))
            for _ in range(k):
                x, y = y, (w - 1.0) - x

    if rng.random() < 0.5:
        img = np.ascontiguousarray(np.fliplr(img))
        x = (w - 1.0) - x

    if rng.random() < 0.5:
        img = np.ascontiguousarray(np.flipud(img))
        y = (h - 1.0) - y

    if rng.random() < 0.6:
        angle = float(rng.uniform(-max_angle, max_angle))
        scale = float(rng.uniform(scale_range[0], scale_range[1]))
        dx = float(rng.uniform(-shift_frac, shift_frac)) * w
        dy = float(rng.uniform(-shift_frac, shift_frac)) * h

        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        mat[0, 2] += dx
        mat[1, 2] += dy

        nx = float(mat[0, 0] * x + mat[0, 1] * y + mat[0, 2])
        ny = float(mat[1, 0] * x + mat[1, 1] * y + mat[1, 2])

        if border_margin <= nx < (w - border_margin) and border_margin <= ny < (h - border_margin):
            img = cv2.warpAffine(
                img,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114),
            )
            x, y = nx, ny

    return img, x, y


def prepare_stage1_cache(
    samples: list[GCPSample],
    data_root: str | Path,
    cache_dir: str | Path,
    target_size: tuple[int, int] = STAGE1_SIZE,
) -> None:
    """Pre-resize and pad full images into cache_dir so Stage 1 epochs skip 4K JPEG decoding."""
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    for s in samples:
        out_path = cache_dir / s.rel_path
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = read_rgb_image(data_root / s.rel_path)
        padded, _ = resize_and_pad(img, target_size=target_size)
        cv2.imwrite(str(out_path), cv2.cvtColor(padded, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def prepare_stage2_cache(
    samples: list[GCPSample],
    data_root: str | Path,
    cache_dir: str | Path,
    patch_size: int = 640,
) -> None:
    """Pre-crop native-resolution patches around GT centers so Stage 2 skips 4K JPEG decoding."""
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    for s in samples:
        out_path = cache_dir / s.rel_path
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = read_rgb_image(data_root / s.rel_path)
        patch, _ = extract_native_crop(img, center_x=s.x, center_y=s.y, crop_size=patch_size)
        cv2.imwrite(str(out_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])


class Stage1CoarseDataset(Dataset):
    """Dataset for Stage 1 coarse heatmap localization over full letterboxed aerial images."""

    def __init__(
        self,
        samples: list[GCPSample],
        data_root: str | Path,
        target_size: tuple[int, int] = STAGE1_SIZE,
        hm_stride: int = STAGE1_HM_STRIDE,
        sigma: float = 4.0,
        augment: bool = False,
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.hm_stride = hm_stride
        self.sigma = sigma
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        orig_path = self.data_root / sample.rel_path

        if self.cache_dir is not None and (self.cache_dir / sample.rel_path).exists():
            img = read_rgb_image(self.cache_dir / sample.rel_path)
            with Image.open(orig_path) as im:
                orig_w, orig_h = im.size
            target_w, target_h = self.target_size
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            meta = {
                "orig_w": float(orig_w),
                "orig_h": float(orig_h),
                "scale": float(scale),
                "pad_x": float((target_w - new_w) // 2),
                "pad_y": float((target_h - new_h) // 2),
                "target_w": float(target_w),
                "target_h": float(target_h),
            }
        else:
            orig_img = read_rgb_image(orig_path)
            img, meta = resize_and_pad(orig_img, target_size=self.target_size)

        x_in, y_in = orig_to_letterbox(sample.x, sample.y, meta)

        if self.augment:
            img, x_in, y_in = _apply_geometric_aug(
                img,
                x_in,
                y_in,
                self.rng,
                allow_rot90=False,
                max_angle=12.0,
                scale_range=(0.92, 1.08),
                shift_frac=0.04,
                border_margin=16.0,
            )
            img = _apply_photometric_aug(img, self.rng)

        hm_w = self.target_size[0] // self.hm_stride
        hm_h = self.target_size[1] // self.hm_stride
        cx_hm = x_in / self.hm_stride
        cy_hm = y_in / self.hm_stride
        heatmap = make_gaussian_heatmap(hm_h, hm_w, cx_hm, cy_hm, sigma=self.sigma)

        return {
            "image": normalize_image(img),
            "heatmap": torch.from_numpy(heatmap),
            "xy_in": torch.tensor([x_in, y_in], dtype=torch.float32),
            "xy_orig": torch.tensor([sample.x, sample.y], dtype=torch.float32),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "meta": meta,
            "rel_path": sample.rel_path,
        }


class Stage2FineDataset(Dataset):
    """Dataset for Stage 2 native-resolution crop refinement and 3-class shape classification."""

    def __init__(
        self,
        samples: list[GCPSample],
        data_root: str | Path,
        crop_size: int = STAGE2_CROP_SIZE,
        hm_stride: int = STAGE2_HM_STRIDE,
        sigma: float = 3.5,
        max_jitter_px: float = 110.0,
        augment: bool = False,
        cache_dir: str | Path | None = None,
        cache_patch_size: int = 640,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.hm_stride = hm_stride
        self.sigma = sigma
        self.max_jitter_px = max_jitter_px
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_patch_size = cache_patch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        if self.augment and self.max_jitter_px > 0:
            jitter_x = float(self.rng.uniform(-self.max_jitter_px, self.max_jitter_px))
            jitter_y = float(self.rng.uniform(-self.max_jitter_px, self.max_jitter_px))
        else:
            jitter_x, jitter_y = 0.0, 0.0

        cached_path = (self.cache_dir / sample.rel_path) if self.cache_dir is not None else None
        if cached_path is not None and cached_path.exists():
            patch_img = read_rgb_image(cached_path)
            with Image.open(self.data_root / sample.rel_path) as im:
                orig_w, orig_h = im.size
            # Reconstruct exact original-image origin (px1, py1) of the cached patch
            px1 = int(round(sample.x)) - self.cache_patch_size // 2
            py1 = int(round(sample.y)) - self.cache_patch_size // 2
            cx_in_patch = (sample.x + jitter_x) - px1
            cy_in_patch = (sample.y + jitter_y) - py1
            crop_img, local_meta = extract_native_crop(
                patch_img,
                center_x=cx_in_patch,
                center_y=cy_in_patch,
                crop_size=self.crop_size,
            )
            crop_meta = {
                "x1": float(px1 + local_meta["x1"]),
                "y1": float(py1 + local_meta["y1"]),
                "crop_size": float(self.crop_size),
                "orig_w": float(orig_w),
                "orig_h": float(orig_h),
            }
        else:
            orig_img = read_rgb_image(self.data_root / sample.rel_path)
            crop_img, crop_meta = extract_native_crop(
                orig_img,
                center_x=sample.x + jitter_x,
                center_y=sample.y + jitter_y,
                crop_size=self.crop_size,
            )

        xc, yc = orig_to_crop(sample.x, sample.y, crop_meta)

        if self.augment:
            crop_img, xc, yc = _apply_geometric_aug(
                crop_img,
                xc,
                yc,
                self.rng,
                allow_rot90=True,
                max_angle=25.0,
                scale_range=(0.90, 1.10),
                shift_frac=0.03,
                border_margin=20.0,
            )
            crop_img = _apply_photometric_aug(crop_img, self.rng)

        hm_size = self.crop_size // self.hm_stride
        cx_hm = xc / self.hm_stride
        cy_hm = yc / self.hm_stride
        heatmap = make_gaussian_heatmap(hm_size, hm_size, cx_hm, cy_hm, sigma=self.sigma)

        return {
            "image": normalize_image(crop_img),
            "heatmap": torch.from_numpy(heatmap),
            "xy_crop": torch.tensor([xc, yc], dtype=torch.float32),
            "xy_orig": torch.tensor([sample.x, sample.y], dtype=torch.float32),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "crop_meta": crop_meta,
            "rel_path": sample.rel_path,
        }
