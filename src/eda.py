"""Exploratory Data Analysis (EDA) for the Aerial GCP Pose Estimation dataset."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import ExifTags, Image


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train_dataset"
TEST_DIR = DATA_DIR / "test_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

ORIENTATION_TAG = next(
    (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
)


def find_annotations_file() -> Path:
    """Locate gcp_marks.json inside train_dataset or data/."""
    for candidate in (
        TRAIN_DIR / "gcp_marks.json",
        DATA_DIR / "gcp_marks.json",
        TRAIN_DIR / "curated_gcp_marks.json",
        DATA_DIR / "curated_gcp_marks.json",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find gcp_marks.json in data/ or data/train_dataset/.")


def list_images(root: Path) -> set[str]:
    """Return normalized relative paths of all image files under root."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    paths: set[str] = set()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.add(p.relative_to(root).as_posix())
    return paths


def inspect_images(root: Path, rel_paths: list[str]) -> dict:
    """Inspect image dimensions, formats, color modes, EXIF orientation, and readability."""
    sizes: Counter[tuple[int, int]] = Counter()
    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    exif_orientations: Counter[int | None] = Counter()
    corrupt: list[tuple[str, str]] = []
    per_image_size: dict[str, tuple[int, int]] = {}

    for rel in rel_paths:
        full = root / rel
        try:
            with Image.open(full) as im:
                im.verify()
            with Image.open(full) as im:
                sizes[im.size] += 1
                formats[str(im.format)] += 1
                modes[im.mode] += 1
                per_image_size[rel] = im.size
                orient = None
                if ORIENTATION_TAG is not None:
                    exif = im.getexif()
                    orient = exif.get(ORIENTATION_TAG) if exif else None
                exif_orientations[orient] += 1
        except Exception as exc:  # noqa: BLE001
            corrupt.append((rel, str(exc)))

    return {
        "sizes": dict(sizes),
        "formats": dict(formats),
        "modes": dict(modes),
        "exif_orientations": dict(exif_orientations),
        "corrupt": corrupt,
        "per_image_size": per_image_size,
    }


def plot_samples(
    annotations: dict,
    out_path: Path,
    samples: list[tuple[str, str]],
    title: str,
    crop_half: int = 180,
) -> None:
    """Plot full image + zoomed crop centered on ground-truth (x, y) for each sample."""
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.6 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (rel_path, label_tag) in enumerate(samples):
        entry = annotations[rel_path]
        x, y = entry["mark"]["x"], entry["mark"]["y"]
        shape = entry.get("verified_shape", "MISSING")

        with Image.open(TRAIN_DIR / rel_path) as im:
            arr = np.asarray(im)
        h, w = arr.shape[:2]

        ax_full = axes[row_idx, 0]
        ax_full.imshow(arr)
        ax_full.scatter([x], [y], s=140, facecolors="none", edgecolors="lime", linewidths=2.2)
        ax_full.scatter([x], [y], s=18, c="red", marker="+")
        ax_full.set_title(
            f"{label_tag} | shape={shape}\n{rel_path} ({w}x{h})",
            fontsize=8.5,
        )
        ax_full.axis("off")

        x1 = max(0, int(round(x)) - crop_half)
        y1 = max(0, int(round(y)) - crop_half)
        x2 = min(w, int(round(x)) + crop_half)
        y2 = min(h, int(round(y)) + crop_half)
        crop = arr[y1:y2, x1:x2]

        ax_crop = axes[row_idx, 1]
        ax_crop.imshow(crop)
        ax_crop.scatter([x - x1], [y - y1], s=180, facecolors="none", edgecolors="lime", linewidths=2)
        ax_crop.scatter([x - x1], [y - y1], s=40, c="red", marker="+")
        ax_crop.set_title(f"Zoomed {x2-x1}x{y2-y1}px | center=({x:.1f}, {y:.1f})", fontsize=8.5)
        ax_crop.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(annotations: dict, per_image_size: dict[str, tuple[int, int]], out_path: Path) -> None:
    """Plot class counts, normalized (x, y) distribution, and project distribution."""
    shape_counts = Counter(v.get("verified_shape") or "Missing/None" for v in annotations.values())
    project_counts = Counter(k.split("/")[0] for k in annotations)

    norm_x, norm_y, colors = [], [], []
    color_map = {"L-Shape": "#1f77b4", "Square": "#2ca02c", "Cross": "#d62728"}
    for k, v in annotations.items():
        if k in per_image_size:
            w, h = per_image_size[k]
            norm_x.append(v["mark"]["x"] / w)
            norm_y.append(v["mark"]["y"] / h)
            colors.append(color_map.get(v.get("verified_shape"), "#7f7f7f"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    labels = list(shape_counts.keys())
    vals = [shape_counts[l] for l in labels]
    bars = axes[0].bar(labels, vals, color=[color_map.get(l, "#7f7f7f") for l in labels])
    axes[0].set_title("Class Distribution (train_dataset)")
    axes[0].set_ylabel("Image Count")
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 8, str(val), ha="center", fontsize=9)

    axes[1].scatter(norm_x, norm_y, c=colors, alpha=0.45, s=14)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(1, 0)
    axes[1].set_title("Normalized Marker Center Locations (x/W, y/H)")
    axes[1].set_xlabel("x / W")
    axes[1].set_ylabel("y / H")

    proj_items = project_counts.most_common()
    axes[2].barh([p[0] for p in proj_items][::-1], [p[1] for p in proj_items][::-1], color="#4c72b0")
    axes[2].set_title("Images per Project (train_dataset)")
    axes[2].set_xlabel("Image Count")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ann_path = find_annotations_file()
    print(f"Annotations file: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    train_disk = list_images(TRAIN_DIR)
    test_disk = list_images(TEST_DIR)
    ann_keys = set(annotations.keys())

    missing_on_disk = sorted(ann_keys - train_disk)
    unannotated_on_disk = sorted(train_disk - ann_keys)

    print(f"\n=== FILE & PATH VERIFICATION ===")
    print(f"Annotation entries:        {len(annotations)}")
    print(f"Train images on disk:      {len(train_disk)}")
    print(f"Test images on disk:       {len(test_disk)}")
    print(f"Matched train annotations: {len(ann_keys & train_disk)}")
    print(f"Missing images on disk:    {len(missing_on_disk)} -> {missing_on_disk[:5]}")
    print(f"Unannotated train images:  {len(unannotated_on_disk)} -> {unannotated_on_disk[:5]}")

    train_img_info = inspect_images(TRAIN_DIR, sorted(train_disk))
    test_img_info = inspect_images(TEST_DIR, sorted(test_disk))

    print(f"\n=== IMAGE PROPERTIES ===")
    print(f"Train sizes:   {train_img_info['sizes']}")
    print(f"Train formats: {train_img_info['formats']}, modes: {train_img_info['modes']}")
    print(f"Train EXIF orientation: {train_img_info['exif_orientations']}")
    print(f"Train corrupt: {len(train_img_info['corrupt'])}")
    print(f"Test sizes:    {test_img_info['sizes']}")
    print(f"Test formats:  {test_img_info['formats']}, modes: {test_img_info['modes']}")
    print(f"Test EXIF orientation:  {test_img_info['exif_orientations']}")
    print(f"Test corrupt:  {len(test_img_info['corrupt'])}")

    shapes = Counter()
    invalid_coords = []
    missing_shape_keys = []
    edge_margins = []
    gcp_shapes: dict[str, set] = defaultdict(set)
    basename_counts = Counter(Path(k).name for k in annotations)

    for rel, entry in annotations.items():
        shape = entry.get("verified_shape")
        shapes[shape] += 1
        if shape is None:
            missing_shape_keys.append(rel)

        gcp_id = "/".join(rel.split("/")[:3])
        gcp_shapes[gcp_id].add(shape)

        x = entry["mark"]["x"]
        y = entry["mark"]["y"]
        if rel in train_img_info["per_image_size"]:
            w, h = train_img_info["per_image_size"][rel]
            if not (0 <= x < w and 0 <= y < h) or np.isnan(x) or np.isnan(y):
                invalid_coords.append((rel, x, y, w, h))
            edge_margins.append(min(x, y, w - x, h - y))

    mixed_gcp_labels = {g: s for g, s in gcp_shapes.items() if len(s) > 1}
    dup_basenames = {b: c for b, c in basename_counts.items() if c > 1}

    print(f"\n=== CLASS & DATA QUALITY CHECKS ===")
    print(f"Class distribution:        {dict(shapes)}")
    print(f"Missing verified_shape:    {len(missing_shape_keys)}")
    for m in missing_shape_keys:
        print(f"  - {m}: {annotations[m]}")
    print(f"Invalid/OOB coordinates:   {len(invalid_coords)} -> {invalid_coords[:5]}")
    print(f"Min/Median/Max border dist:{min(edge_margins):.1f} / {np.median(edge_margins):.1f} / {max(edge_margins):.1f} px")
    print(f"Unique GCP groups (3-lvl): {len(gcp_shapes)}")
    print(f"GCP groups with >1 shape:  {len(mixed_gcp_labels)}")
    for g, s in list(mixed_gcp_labels.items())[:10]:
        print(f"  - {g}: {s}")
    print(f"Duplicate filenames across folders: {len(dup_basenames)}")

    train_projects = {p.split("/")[0] for p in train_disk}
    test_projects = {p.split("/")[0] for p in test_disk}
    print(f"\nTrain projects ({len(train_projects)}), Test projects ({len(test_projects)}), Overlap ({len(train_projects & test_projects)}):")
    print(f"  Overlap: {sorted(train_projects & test_projects)}")

    plot_distributions(annotations, train_img_info["per_image_size"], OUTPUT_DIR / "eda_distributions.png")

    sample_list: list[tuple[str, str]] = []
    for target_shape in ["Cross", "Square", "L-Shape"]:
        matches = [k for k, v in annotations.items() if v.get("verified_shape") == target_shape]
        picked = []
        seen_proj = set()
        for m in matches:
            proj = m.split("/")[0]
            if proj not in seen_proj:
                picked.append(m)
                seen_proj.add(proj)
            if len(picked) == 2:
                break
        for p in picked:
            sample_list.append((p, target_shape))

    plot_samples(
        annotations,
        OUTPUT_DIR / "eda_samples.png",
        sample_list,
        "Ground-Truth Samples: Full Aerial Image & 360x360px Marker Zoom",
    )

    if missing_shape_keys:
        plot_samples(
            annotations,
            OUTPUT_DIR / "eda_missing_shape_samples.png",
            [(p, "None") for p in missing_shape_keys],
            "Samples Missing 'verified_shape' in gcp_marks.json",
        )

    print(f"\nSaved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
