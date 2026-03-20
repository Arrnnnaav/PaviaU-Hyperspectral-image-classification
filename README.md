# Hyperspectral Image Classification — Pavia University

Benchmarking **SVM**, **Random Forest**, **MLP**, and a **3D CNN (HybridSN)** on the Pavia University hyperspectral dataset. Everything runs in a single notebook — no intermediate files saved, no model checkpoints, no saved images.

---

## Dataset

**Pavia University** — captured by the ROSIS sensor over the University of Pavia, Italy.

| Property | Value |
|---|---|
| Image size | 610 × 340 pixels |
| Spectral bands | 103 |
| Labeled pixels | 42,776 |
| Classes | 9 land-cover types |
| Background (class 0) | excluded from training |

| Class | Name | Pixels |
|---|---|---|
| 1 | Asphalt | 6,631 |
| 2 | Meadows | 18,649 |
| 3 | Gravel | 2,099 |
| 4 | Trees | 3,064 |
| 5 | Painted metal sheets | 1,345 |
| 6 | Bare soil | 5,029 |
| 7 | Bitumen | 1,330 |
| 8 | Self-blocking bricks | 3,682 |
| 9 | Shadows | 947 |

Download both files from the [UPV/EHU Hyperspectral Repository](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and place them alongside the notebook:

```
PaviaU.mat
PaviaU_gt.mat
PaviaU_clean.ipynb
```

---

## Installation

```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn seaborn
```

Python 3.9+ recommended. A CUDA GPU is optional but significantly speeds up the 3D CNN (50 epochs on CPU takes ~15–20 min; on GPU ~2–3 min).

---

## Notebook structure

The notebook is 10 code cells end to end, no branching, no saves:

```
Cell 1  — Imports + constants
Cell 2  — Load data, EDA (false colour, spectral signatures)
Cell 3  — Preprocessing (MinMaxScaler, remove background, 80/20 stratified split)
Cell 4  — PCA (scree plot, apply 103→30, no leakage)
Cell 5  — Train SVM, Random Forest, MLP
Cell 6  — Evaluate classical models (OA/AA/Kappa, confusion matrix, bar chart)
Cell 7  — 3D CNN patch extraction + HSIDataset + DataLoaders
Cell 8  — HybridSN model definition
Cell 9  — Train 3D CNN (50 epochs, StepLR scheduler)
Cell 10 — Evaluate + full prediction maps + error maps + zoom comparison
```

Run top to bottom — every variable flows directly into the next cell.

---

## Pipeline

```
PaviaU.mat
    │
    ▼
MinMaxScaler (per pixel, fit on all)
    │
    ├──────────────────────────────────────────┐
    │  Classical branch                         │  3D CNN branch
    ▼                                           ▼
Remove class 0 → 80/20 stratified split    Reflect-pad image (HALF=3)
    │                                           │
    ▼                                           ▼
PCA  103→30  (fit on train only)           Extract 7×7×103 patches
    │                                       (42,776 labeled pixels)
    ├── SVM (RBF, C=10)                         │
    ├── Random Forest (100 trees, balanced)     ▼
    └── MLP (256→128→64, Adam)             HybridSN
                                           3D conv → 2D conv → FC
    │                                           │
    └──────────────┬─────────────────────────── ┘
                   ▼
        OA / AA / Kappa
        Prediction maps · Error maps · Zoom comparison
```

---

## Models

### Classical models (PCA features)

All three take the 30-component PCA vectors as input — no spatial context.

| Model | Key settings |
|---|---|
| SVM | RBF kernel, C=10, `gamma='scale'` |
| Random Forest | 100 trees, `class_weight='balanced'`, all CPU cores |
| MLP | 256→128→64, ReLU, Adam, early stopping (patience=15) |

`class_weight='balanced'` in Random Forest is important — Meadows has 18,649 pixels but Shadows has only 947. Without reweighting, rare classes get crushed.

---

### 3D CNN — HybridSN

Based on [Roy et al. 2020](https://ieeexplore.ieee.org/document/8736016) — one of the most cited HSI architectures.

**Input shape:** `(B, 1, 103, 7, 7)` — one channel, 103 spectral bands as depth, 7×7 spatial patch.

```
Conv3d(1→8,   k=(7,3,3))  BatchNorm  ReLU   → (B, 8,  97, 5, 5)
Conv3d(8→16,  k=(5,3,3))  BatchNorm  ReLU   → (B, 16, 93, 3, 3)
Conv3d(16→32, k=(3,1,1))  BatchNorm  ReLU   → (B, 32, 91, 3, 3)
reshape → (B, 32×91, 3, 3)
Conv2d(2912→64, k=3, pad=1)  BatchNorm  ReLU  → (B, 64, 3, 3)
Flatten
Linear(576→256)  ReLU  Dropout(0.4)
Linear(256→128)  ReLU  Dropout(0.4)
Linear(128→9)
```

**Training:** Adam (`lr=1e-3`, `weight_decay=1e-4`), StepLR (halve every 15 epochs), 50 epochs, batch size 128.

**Patch extraction:** The image is reflect-padded by `HALF=3` rows/cols so every labeled pixel — including those at the border — can produce a full 7×7 patch. Reflect padding is used instead of zero-padding because zeros create artificial dark borders that confuse the spectral-spatial convolutions.

**Full-map inference** is batched (`batch_size=512`) with `torch.cuda.empty_cache()` after each batch — the full 207,400-pixel map would otherwise OOM a 4 GB GPU.

---

## Evaluation metrics

Every result is reported with the three standard HSI metrics:

| Metric | Formula | What it measures |
|---|---|---|
| OA (Overall Accuracy) | correct / total | Global pixel accuracy |
| AA (Average Accuracy) | mean of per-class OA | Fairness across imbalanced classes |
| κ (Kappa) | OA adjusted for chance | Agreement beyond random guessing |

---

## Typical results

Performance varies with random seed. Indicative ranges on Pavia University with the settings in this notebook:

| Model | OA | AA | Kappa |
|---|---|---|---|
| SVM (RBF) | 94.52% | 92.44% | 0.9270 |
| Random Forest | 91.61% | 87.35% | 0.8866 |
| MLP | 95.77% | 94.12% | 0.9438 |
| 3D CNN (HybridSN) | 99.9% | 99.89% | 0.9999 |

The 3D CNN leads on most classes, particularly on spectrally similar ones (Asphalt vs Bitumen, Gravel vs Bare Soil) where spatial context from the 7×7 patch breaks the tie.

---

## Visualisations produced

The final cell generates all plots inline — nothing is saved to disk:

- False colour composite (bands 60/30/10 → RGB) + ground truth label map
- Mean spectral signatures per class with ±1σ bands
- PCA scree plot + cumulative variance curve
- Confusion matrix (best classical model)
- OA/AA bar chart comparing all three classical models
- Training loss curve + validation accuracy curve (3D CNN)
- 4-model prediction map grid side by side
- Error maps for all 4 models (white = correct, red = wrong)
- 150×150 zoomed region comparing false colour / ground truth / 3D CNN prediction

---

## Key design decisions

**PCA fitted on train only.** If PCA were fitted on train + test combined, test-set variance would leak into the projection axes — the model would effectively have seen test data during preprocessing. Here `pca.fit_transform(X_train)` and `pca.transform(X_test)` are kept strictly separate.

**3D CNN uses the raw scaled cube, not PCA.** PCA destroys spatial structure by collapsing the spectral dimension into orthogonal components. The 3D CNN's entire value is learning joint spectral-spatial features — feeding it PCA output throws that away.

**Reflect padding over zero padding.** Zero-padding creates an artificial constant border that the 3D convolutions learn to associate with "edge pixel", degrading accuracy near the image boundary. Reflect padding mirrors real spectral values instead.

**No files saved.** The original notebook scattered `np.save`, `joblib.dump`, `torch.save`, and `plt.savefig` across 38 cells, then reloaded them in later cells. This version keeps everything in memory — faster iteration, no stale file mismatches, easier to run in one shot.

---

## Citation

```
M. Graña et al., "Hyperspectral Remote Sensing Scenes",
GIC, University of the Basque Country.
http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

S. K. Roy et al., "HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy
for Hyperspectral Image Classification," IEEE Geoscience and Remote
Sensing Letters, vol. 17, no. 2, pp. 277–281, Feb. 2020.
```
