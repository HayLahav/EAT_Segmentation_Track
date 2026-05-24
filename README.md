# EAT Segmentation — Membrane Morphodynamics in Label-Free Microscopy (TBC)

---

## Overview

The **Epithelial-to-Amoeboid Transition (EAT)** is a phenotypic switch in cancer cells linked to aggressive, adhesion-independent metastatic motility. Its earliest cellular reporters — membrane **Ruffles** (actin-driven lamellipodia) and **Blebs** (pressure-driven spherical protrusions) — are systematically missed by existing tools because they occupy only 3–5% of the cell boundary and differ from background by as little as 5–15 intensity units in a 16-bit phase-contrast image.

This project detects EAT morphological signatures from **label-free phase-contrast microscopy** alone — no fluorescent probes, no manual annotation of ruffles or blebs. It combines bio-inspired membrane enhancement with deep segmentation and diffusion-based ruffle prediction, evaluated on the [LIVECell](https://github.com/sartorius-research/LIVECell) dataset on EAT-relevant cancer cell lines (MCF7, SkBr3, SKOV3).

---

## Results (MCF7 Validation Set)

| Version | Architecture | Mean IoU | Mean F1 | Boundary F1 |
|---------|-------------|----------|---------|-------------|
| V1 | ConvNeXt-Tiny + UNet++ | 0.842 | 0.913 | 0.703 |
| V2 | PE-Diffusion (ViT-B/16 + Diffusion) | 0.366 | 0.525 | 0.367 |
| **V3** | **Frozen Hybrid + Ruffle Diffusion Head** | **0.846** | **0.915** | **0.716** |

V3 is the final architecture: the ConvNeXt backbone from V1 is frozen and used to condition a lightweight diffusion head trained solely on ruffle prediction — separating concerns and eliminating the conflicting gradients that degraded V2.

---

## Architecture

### V1 — Hybrid ConvNeXt-Tiny + UNet++

```
Input (B, 1, H, W)  ← phase-contrast grayscale
        │
  ConvNeXt-Tiny Encoder (ImageNet pretrained, grayscale-adapted)
        │  enc1: (B,  96, H/4,  W/4)
        │  enc2: (B, 192, H/8,  W/8)
        │  enc3: (B, 384, H/16, W/16)
        │  enc4: (B, 768, H/32, W/32)
        │
  UNet++ Dense Decoder (4 scales, 9 dense nodes)
        │  x13: (B, 64, H/4, W/4) ← decoder feature map
        │  upsample → (B, 64, H, W)
        │
  ┌─────┴──────┐
  Mask Head   Distance Head
  sigmoid      regression (watershed)
```

The 64-ch decoder output `x13` is exposed via `return_feats=True` for use in V3.

### V3 — Combined Model (Frozen Hybrid + Ruffle Diffusion Head)

```
Input (B, 1, H, W)
        │
  [Frozen HybridSegmentationModel] ──→ mask  (B, 1, H, W)
        │                          └──→ feats (B, 64, H, W)
        │
  cat[x_t (1ch), feats (64ch), mask (1ch)] = 66ch
        │
  [RuffleDiffusionHead — DiffusionUNet]
    cosine schedule (T=1000), DDIM sampling (50 steps)
        │
  predicted ruffle map (B, 1, H, W)
```

Only the diffusion head (~7M parameters) is trained. The backbone (~30M parameters) is permanently frozen.

---

## Research Progression

```
experiments/
  v1_hybrid_convnext_unetpp/   ← ConvNeXt baseline (IoU=0.842)
  v2_pe_diffusion_standalone/  ← ViT-B/16 diffusion attempt (IoU=0.366)
  v3_combined_hybrid_ruffle/   ← Final architecture (IoU=0.846)
```

Each experiment folder contains a `README.md` with architecture details, results, and lessons learned — documenting the full research progression.

---

## Project Structure

```
EAT_Segmentation/
├── notebooks/
│   ├── EAT_Segmentation_Colab.ipynb           # V1 hybrid training (original)
│   ├── EAT_Segmentation_PE_Diffusion.ipynb    # V2 PE-diffusion training
│   └── EAT_Segmentation_Combined_Model.ipynb  # V3 combined model training ← use this
├── src/
│   ├── segmentation/
│   │   ├── hybrid_model.py        # ConvNeXt-Tiny + UNet++ (V1)
│   │   ├── fss.py                 # Fuzzy Spatial Segmentation post-processing
│   │   └── instance_postprocess.py
│   ├── pe_diffusion/
│   │   ├── model.py               # DiffusionUNet, cosine schedule, DDIM sampler
│   │   └── combined_model.py      # V3 CombinedRuffleSegmentation
│   ├── data/
│   │   └── livecell_loader.py     # LIVECell COCO dataset loader
│   ├── features/
│   │   └── extractor.py           # Ruffle + Bleb morphodynamic features
│   ├── evaluation/
│   │   └── acdc_benchmark.py      # IoU, F1, Boundary F1 vs Cell_ACDC
│   ├── training/
│   │   └── train_hybrid.py        # Training loop with interrupt-safe resume
│   └── pipeline.py                # End-to-end: frame → per-cell feature records
├── checkpoints/                   # Saved model weights (not tracked in git)
├── results/                       # Metrics JSON, visualisations, prediction CSVs
├── experiments/                   # Versioned experiment logs with READMEs
├── scripts/
│   └── evaluate.py
├── docs/
│   └── paper.md                   # Full technical paper
└── requirements.txt
```

---

## Notebooks

All training runs on **Google Colab** (T4 GPU, 16 GB VRAM) with Google Drive checkpointing.

### V3 Combined Model — recommended entry point

Open `notebooks/EAT_Segmentation_Combined_Model.ipynb` in Colab and run cells top to bottom:

| Step | Description |
|------|-------------|
| 1–2 | Mount Drive, set up paths |
| 3 | Install dependencies (`scikit-image`, `pycocotools`, `tifffile`) |
| 4–5 | Pseudo-ruffle GT generator (Frangi ridge filter) + dataset |
| 6–7 | VRAM check, initialise `CombinedRuffleSegmentation` |
| 8 | Train diffusion head — 40 epochs, atomic saves, auto-resume |
| 9 | Load checkpoint (if Colab restarted after training) |
| 10–12 | Training curves, visualisation, quantitative evaluation |

Checkpoints are saved atomically to prevent corruption on Colab disconnect:
write to `/tmp` → copy to Drive with `.new` suffix → `os.replace()`.

---

## Installation (local)

```bash
git clone https://github.com/<your-username>/EAT_Segmentation.git
cd EAT_Segmentation
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA recommended.

```
torch>=2.0.0
torchvision>=0.15.0
scipy>=1.10.0
scikit-image>=0.21.0
numpy>=1.24.0
pandas>=2.0.0
tifffile>=2023.1.1
pycocotools>=2.0.6
```

---

## Dataset

This project uses [LIVECell](https://github.com/sartorius-research/LIVECell) (Edlund et al., *Nature Methods* 2021) — the largest publicly available label-free live cell segmentation dataset (1.6M annotated cells, 8 cancer cell lines).

Expected Drive structure:
```
EAT_Segmentation/
  LIVECell/
    images/livecell_train_val_images/
    annotations/
      LIVECell_single_cell_train.json
      LIVECell_single_cell_val.json
```

---

## Morphodynamic Feature Output

For each detected cell per frame the pipeline outputs a 10-feature record:

| Feature | Description |
|---------|-------------|
| Area | Cell instance size (pixels) |
| Sphericity | 4π·Area/Perimeter² — 1.0=amoeboid, <0.5=mesenchymal |
| x_Pos, y_Pos | Cell centroid coordinates |
| Ruffle_Density | Mean ruffle map activation |
| Ruffle_Directionality | Spatial heterogeneity of ruffle activity |
| Protrusion_Length | Effective protrusion extent (px) |
| Boundary_Uncertainty_Score | Mean binary entropy of the boundary map |
| Bleb_Count | Number of connected bleb regions |
| Bleb_Area_Fraction | Bleb pixels / boundary pixels |

Records are written to `results/predictions_{cell_type}.csv`.

---

## Key Design Decisions

**Pseudo-ruffle GT without manual annotation.** Ruffle ground truth is generated automatically using the Frangi ridge filter applied to the GT mask boundary band — a 6-pixel dilated minus eroded ring. No manual ruffle labelling is required.

**Atomic checkpoint saves.** All checkpoints are written via `/tmp → .new → os.replace()` to survive mid-write Colab disconnects without corruption.

**Separation of concerns in V3.** V2's single diffusion head was asked to simultaneously denoise segmentation masks and ruffle texture — two conflicting objectives. V3 freezes the proven V1 backbone and trains a diffusion head on the single task of ruffle prediction, conditioned on domain-adapted ConvNeXt features.

---


