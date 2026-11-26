# Skylark Drones AI Engineer Assessment — Submission Guide

- **GitHub Repository:** https://github.com/nithinag/Skylark-GCP-Estimation
- **Project:** Aerial GCP Pose Estimation (Keypoint Localization & 3-Class Shape Classification)

---

## 1. DoSelect Submission Files

| Deliverable | File Path | Size / Status |
|---|---|---|
| GitHub Repository | `https://github.com/nithinag/Skylark-GCP-Estimation` | Source code, Colab notebook, EDA, evaluation & submission artifacts |
| Decision Log | `submission/Decision_Log.pdf` | Full 15-section engineering report with measured results & figures |
| Predictions JSON | `submission/submission_predictions.json` (also at `outputs/submission_predictions.json`) | `300 / 300` test predictions, `51.81 KB` (under 10 MB limit) |

*(Note: Raw dataset images under `data/` and large `.pt` checkpoints under `weights/` are excluded from Git via `.gitignore`.)*

---

## 2. Verified Validation Results (`StratifiedGroupKFold`, `val_fold=3`, `n=197` Unseen-Project Images)

| Pipeline Stage | PCK@10px | PCK@25px | PCK@50px | Median Error | Mean Error | Macro F1 |
|---|---|---|---|---|---|---|
| Stage 1 Coarse (`stage1_best.pt`, Epoch 3) | 90.86% | 90.86% | 90.86% | 2.87 px | 162.08 px | — |
| Stage 2 Oracle (`stage2_best.pt`, Epoch 22) | 99.49% | 99.49% | 99.49% | 0.99 px | 1.97 px | 0.6130 |
| **End-to-End Two-Stage** | **90.86%** | **90.86%** | **90.86%** | **1.04 px** | **160.09 px** | **0.6082** |

- **Per-Class End-to-End F1:** `Cross: 0.4000`, `Square: 0.9104`, `L-Shape: 0.5143` (overall accuracy `61.42%`).

---

## 3. JSON Schema & Class Name Verification

- Format matches `data/train_dataset/gcp_marks.json`:
  ```json
  {
    "<project>/<survey>/<gcp_id>/<image>.JPG": {
      "mark": {"x": 2566.4, "y": 568.3},
      "verified_shape": "L-Shape"
    }
  }
  ```
- **Schema Note:** The assignment PDF prose mentions `"L-Shaped"`, whereas all 491 ground-truth entries in `gcp_marks.json` use **`"L-Shape"`** (`"L-Shaped"` appears 0 times). `submission_predictions.json` preserves `"L-Shape"`, `"Square"`, and `"Cross"` from the training annotations.
- **Test Set Distribution (`300` images):** `L-Shape: 143`, `Cross: 96`, `Square: 61`.

---

## 4. Trained Weights & Reproducibility

Trained checkpoints (`weights/stage1_best.pt` — 280.8 MB, `weights/stage2_best.pt` — 286.8 MB) exceed GitHub's 100 MB file limit; host them via Google Drive / cloud link or reproduce from source:

```bash
pip install -r requirements.txt
python -m src.train --stage both --val-fold 3
python -m src.evaluate --val-fold 3 --vis-output outputs/val_predictions.png
python -m src.predict --test-dir data/test_dataset --output outputs/submission_predictions.json
```
