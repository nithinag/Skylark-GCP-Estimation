# Aerial GCP Pose Estimation — Skylark Drones Assessment

A two-stage, coarse-to-fine PyTorch computer vision pipeline for **Ground Control Point (GCP)** center localization and shape classification (`Cross`, `Square`, `L-Shape`) in high-resolution aerial drone imagery.

---

## 1. Project Structure

```text
skylark-gcp-estimation/
├── data/
│   ├── train_dataset/          # 1,000 high-res training images + gcp_marks.json
│   └── test_dataset/           # 300 unannotated test images
├── notebooks/
│   └── train_colab.ipynb       # End-to-end Google Colab GPU training & inference notebook
├── outputs/
│   ├── eda_distributions.png   # Class, spatial, and project distributions
│   ├── eda_samples.png         # Sample aerial frames + 360x360 marker crops
│   ├── eda_missing_shape_samples.png
│   └── predictions.json        # Generated after test inference
├── src/
│   ├── eda.py                  # Dataset audit, coordinate/EXIF checks, and visualizations
│   ├── dataset.py              # GroupKFold splits, aspect-preserving resize, crops, heatmaps
│   ├── model.py                # ResNet-34 UNet coarse/fine models, Soft-Argmax, Focal loss
│   ├── train.py                # Stage 1 & Stage 2 training loop with AMP & checkpointing
│   ├── evaluate.py             # End-to-end PCK@10/25/50px & Macro F1 validation
│   └── predict.py              # Two-stage test inference producing predictions.json
├── weights/                    # Saved stage1_best.pt and stage2_best.pt checkpoints
├── requirements.txt
└── README.md
```

---

## 2. Dataset & EDA Findings

Running `python -m src.eda` across all 1,000 training images and 300 test images revealed critical real-world characteristics:

- **Image Resolutions & Aspect Ratios:**
  - Train (`1,000` JPEGs): `642` images at `4096×2730` (3:2) and `358` images at `4096×3068` (4:3).
  - Test (`300` JPEGs): `178` images at `4096×2730`, `120` images at `4096×3068`, and `2` images at `4000×3000` (4:3).
  - All annotations in `data/train_dataset/gcp_marks.json` are already in full 4K pixel space (`x ∈ [66.96, 3937.06]`, `y ∈ [35.04, 2914.11]`).
- **Tiny Targets in 4K Frames:** GCP markers are typically `~30–80 px` wide inside `~12.5 MP` images and appear anywhere in the frame (as close as `35 px` to image borders).
- **Class Imbalance & Missing Labels:**
  - `L-Shape`: `491` (note: annotation strings use `"L-Shape"`, not `"L-Shaped"`)
  - `Square`: `328`
  - `Cross`: `177`
  - Missing `verified_shape`: `4` entries contain only `{"mark": {"x", "y"}}`. These 4 samples are retained for localization training and ignored in classification loss (`ignore_index=-1`).
- **Project-Level Leakage Risk:** Training images span 11 projects (with up to 12 drone photos of the same `gcp_id`), whereas test images span 73 projects (`64` unseen). Validation uses `StratifiedGroupKFold(n_splits=5)` grouped by `project_name` so no site or marker leaks across train/val.

---

## 3. Architecture: Two-Stage Coarse-to-Fine Pipeline

Directly downsampling a `4096×3068` image to `512×384` shrinks a `32 px` marker to `~4×4 px`, destroying shape detail and making `PCK@10px` (in 4K pixels) unreliable. To solve this cleanly:

1. **Stage 1 — Coarse Localization (`Stage1CoarseModel`)**
   - Preserves aspect ratio via letterboxing (`resize_and_pad`) to `1024×768`.
   - `ResNet-34` encoder + UNet decoder predicts a stride-2 (`512×384`) 2D Gaussian heatmap.
   - Local peak + `5×5` Soft-Argmax refinement maps the coarse `(x, y)` back to original 4K coordinates.
2. **Stage 2 — Fine Localization & Classification (`Stage2FineModel`)**
   - Extracts a **`384×384` native-resolution crop** directly from the original 4K image centered at Stage 1's predicted `(x, y)` (with border padding when near image edges).
   - Multi-task `ResNet-34` UNet predicts:
     - **Localization Head:** Stride-2 (`192×192`) 2D Gaussian heatmap -> sub-pixel center mapped back to 4K pixels.
     - **Classification Head:** Global avg + max pooled bottleneck features -> 3 logits (`Cross`, `Square`, `L-Shape`).
   - Trained with realistic crop-position jitter (`±110 px`) so Stage 2 recovers cleanly from imperfect Stage 1 predictions.

---

## 4. Training Strategy

- **Losses:**
  - Localization: Penalty-reduced `GaussianFocalLoss` + `0.05 * SmoothL1Loss` on differentiable 2D Soft-Argmax coordinates.
  - Classification: `CrossEntropyLoss` with inverse-frequency class weights computed strictly on the training fold and `ignore_index=-1`.
- **Aerial Augmentations (coordinate-exact):**
  - Stage 1: Horizontal/vertical flips, small affine rotation (`±12°`), scale (`0.92–1.08`), shift (`±4%`), and brightness/contrast/HSV/blur jitter.
  - Stage 2: Random `±110 px` crop offset jitter, 90° rotations (`0°, 90°, 180°, 270°`), flips, fine affine rotation (`±25°`), scale (`0.90–1.10`), and photometric jitter.

---

## 5. Installation & Quick Verification

```bash
pip install -r requirements.txt
```

Run Exploratory Data Analysis:
```bash
python -m src.eda
```

Run lightweight CPU/GPU smoke test (2 batches per stage + checkpoint save/load):
```bash
python -m src.train --smoke-test --no-pretrained
```

---

## 6. Training on Google Colab (Recommended)

Open [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) in Google Colab with a **T4 GPU** runtime and run cells **1–12** in order. Google Drive mounting is optional—if `drive.mount` fails, the notebook automatically downloads the dataset from the shared Google Drive link via `gdown` into `/content/skylark-gcp-estimation/data/` and bundles trained weights and predictions for direct browser download. Or execute the CLI commands directly:

```bash
# Stage 1: Coarse Localization (caches 1024x768 images for fast epochs)
python -m src.train --stage 1 --epochs-s1 35 --batch-s1 8 --lr-s1 3e-4 --cache-dir data/.cache_stage1

# Stage 2: Fine Localization + Shape Classification
python -m src.train --stage 2 --epochs-s2 40 --batch-s2 16 --lr-s2 3e-4
```

Trained checkpoints are saved to:
- `weights/stage1_best.pt`
- `weights/stage2_best.pt`

---

## 7. Model Weights

Place the trained checkpoints (generated via Google Colab training or downloaded from your Google Drive backup) into `weights/`:
- `weights/stage1_best.pt`
- `weights/stage2_best.pt`

---

## 8. Evaluation & Test Inference

Evaluate end-to-end on the held-out `GroupKFold` validation split (reports `PCK@10px`, `PCK@25px`, `PCK@50px`, mean/median pixel error in original coordinates, and `Macro F1-Score`):

```bash
python -m src.evaluate --val-fold 3 --n-splits 5 --vis-output outputs/val_predictions.png
```

Run end-to-end two-stage inference on all 300 test images to produce `outputs/predictions.json`:

```bash
python -m src.predict \
  --test-dir data/test_dataset \
  --stage1-weights weights/stage1_best.pt \
  --stage2-weights weights/stage2_best.pt \
  --output outputs/predictions.json
```

Output format matches `gcp_marks.json`:
```json
{
  "project/survey/gcp_id/image.JPG": {
    "mark": {"x": 1024.5, "y": 850.2},
    "verified_shape": "L-Shape"
  }
}
```

