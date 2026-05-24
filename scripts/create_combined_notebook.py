"""Create EAT_Segmentation_Combined_Model.ipynb as a standalone notebook."""
import json

NB_OUT = r'C:\Users\hayla\OneDrive\מסמכים\projects\EAT_Segmentation\notebooks\EAT_Segmentation_Combined_Model.ipynb'


def to_lines(s):
    s = s.rstrip('\n')
    lines = s.split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]


def md(src, cell_id):
    return {"cell_type": "markdown", "metadata": {}, "source": to_lines(src), "id": cell_id}


def code(src, cell_id):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": to_lines(src), "id": cell_id}


cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md(
    "# EAT Segmentation — Combined Model (Google Colab)\n"
    "\n"
    "**Architecture**\n"
    "- **Stage 1 — Frozen Backbone**: HybridSegmentationModel (ConvNeXt-Tiny + UNet++), pre-trained at IoU=0.842. Provides binary mask + 64-ch decoder features.\n"
    "- **Stage 2 — Ruffle Diffusion Head**: Lightweight DiffusionUNet trained on one task only: predict pseudo-ruffle maps conditioned on ConvNeXt features + hybrid mask.\n"
    "\n"
    "**Diffusion design**\n"
    "- x₀ = `pseudo_ruffle_GT` (1 channel, Frangi ridge filter)\n"
    "- Denoiser input: `cat[xₜ (1ch), ConvNeXt_feats (64ch), hybrid_mask (1ch)]` = 66ch\n"
    "- Schedule: cosine beta (Nichol & Dhariwal 2021, T=1000)\n"
    "- Inference: DDIM 50 steps\n"
    "\n"
    "**Why this is better than V2 PE-Diffusion:**\n"
    "- Segmentation handled by proven ConvNeXt backbone (no conflicting gradients)\n"
    "- Diffusion focuses on one task: ruffle prediction\n"
    "- ~3× fewer trainable parameters → faster convergence",
    "cm-title"
))

# ── Step 1 — Mount Drive ──────────────────────────────────────────────────────
cells.append(md("## Step 1 — Mount Google Drive", "cm-s1-md"))
cells.append(code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')",
    "cm-s1-code"
))

# ── Step 2 — Setup ────────────────────────────────────────────────────────────
cells.append(md("## Step 2 — Setup Project", "cm-s2-md"))
cells.append(code(
    "import sys, os\n"
    "\n"
    "PROJECT_DIR = '/content/drive/MyDrive/EAT_Segmentation'\n"
    "CELL_TYPE   = 'MCF7'\n"
    "\n"
    "# Sync project files from Drive\n"
    "os.makedirs('/content/EAT_Segmentation', exist_ok=True)\n"
    "!cp -r {PROJECT_DIR}/src /content/EAT_Segmentation/\n"
    "sys.path.insert(0, '/content/EAT_Segmentation')\n"
    "\n"
    "TRAIN_IMGS = f'{PROJECT_DIR}/LIVECell/images/livecell_train_val_images'\n"
    "TRAIN_ANN  = f'{PROJECT_DIR}/LIVECell/annotations/LIVECell_single_cell_train.json'\n"
    "VAL_ANN    = f'{PROJECT_DIR}/LIVECell/annotations/LIVECell_single_cell_val.json'\n"
    "\n"
    "CKPT_DIR    = f'{PROJECT_DIR}/checkpoints'\n"
    "RESULTS     = f'{PROJECT_DIR}/results'\n"
    "os.makedirs(CKPT_DIR, exist_ok=True)\n"
    "os.makedirs(RESULTS,  exist_ok=True)\n"
    "\n"
    "# Hybrid backbone checkpoint (frozen — not retrained here)\n"
    "HYBRID_CKPT = f'{CKPT_DIR}/hybrid_seg_{CELL_TYPE.lower()}.pt'\n"
    "\n"
    "device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'\n"
    "print(f'Device: {device}  |  Project: {PROJECT_DIR}')\n"
    "print(f'Hybrid checkpoint: {HYBRID_CKPT}')",
    "cm-s2-code"
))

# ── Step 3 — Install ─────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Install Dependencies", "cm-s3-md"))
cells.append(code(
    "# scikit-image for Frangi ridge filter\n"
    "!pip install -q scikit-image\n"
    "\n"
    "# Data loading utilities\n"
    "!pip install -q pycocotools tifffile imagecodecs\n"
    "print('Dependencies ready.')",
    "cm-s3-code"
))

# ── Step 4 — Pseudo-ruffle ───────────────────────────────────────────────────
cells.append(md("## Step 4 — Pseudo-Ruffle Map Generator", "cm-s4-md"))
cells.append(code(
    "import numpy as np\n"
    "from scipy.ndimage import binary_dilation, binary_erosion\n"
    "from skimage.filters import frangi\n"
    "\n"
    "def make_pseudo_ruffle(image: np.ndarray, gt_mask: np.ndarray,\n"
    "                       band_width: int = 6) -> np.ndarray:\n"
    "    m     = gt_mask.astype(bool)\n"
    "    outer = binary_dilation(m, iterations=band_width)\n"
    "    inner = binary_erosion(m,  iterations=band_width)\n"
    "    band  = (outer & ~inner).astype(np.float32)\n"
    "    response = frangi(image, sigmas=(1, 2, 3), black_ridges=False)\n"
    "    ruffle   = response * band\n"
    "    mx = ruffle.max()\n"
    "    if mx > 0:\n"
    "        ruffle = ruffle / mx\n"
    "    return ruffle.astype(np.float32)\n"
    "\n"
    "\n"
    "# Quick visual check\n"
    "import matplotlib.pyplot as plt\n"
    "import tifffile, json\n"
    "\n"
    "with open(TRAIN_ANN) as f:\n"
    "    coco = json.load(f)\n"
    "\n"
    "ann_by_img = {}\n"
    "for a in coco['annotations']:\n"
    "    ann_by_img.setdefault(a['image_id'], []).append(a)\n"
    "\n"
    "sample = next(i for i in coco['images'] if CELL_TYPE.lower() in i['file_name'].lower())\n"
    "img_path = f\"{TRAIN_IMGS}/{sample['file_name']}\"\n"
    "img_raw  = tifffile.imread(img_path).astype(np.float32)\n"
    "img_norm = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-8)\n"
    "\n"
    "H, W = sample['height'], sample['width']\n"
    "mask = np.zeros((H, W), dtype=np.uint8)\n"
    "try:\n"
    "    from pycocotools import mask as coco_mask\n"
    "    for ann in ann_by_img.get(sample['id'], []):\n"
    "        if ann.get('segmentation'):\n"
    "            rle = coco_mask.frPyObjects(ann['segmentation'], H, W)\n"
    "            mask = np.maximum(mask, coco_mask.decode(coco_mask.merge(rle)))\n"
    "except Exception as e:\n"
    "    print(f'mask build error: {e}')\n"
    "\n"
    "pseudo_ruf = make_pseudo_ruffle(img_norm, mask)\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
    "axes[0].imshow(img_norm,   cmap='gray');    axes[0].set_title('Cell Image')\n"
    "axes[1].imshow(mask,       cmap='hot');     axes[1].set_title('GT Mask')\n"
    "axes[2].imshow(pseudo_ruf, cmap='viridis'); axes[2].set_title('Pseudo-Ruffle Map (Frangi)')\n"
    "for ax in axes: ax.axis('off')\n"
    "plt.tight_layout(); plt.show()\n"
    "print('Ruffle map max:', pseudo_ruf.max(), '  mean in band:', pseudo_ruf[pseudo_ruf > 0].mean())",
    "cm-s4-code"
))

# ── Step 5 — Dataset ─────────────────────────────────────────────────────────
cells.append(md("## Step 5 — Dataset with Ruffle Maps", "cm-s5-md"))
cells.append(code(
    "import torch\n"
    "from torch.utils.data import Dataset\n"
    "\n"
    "class LIVECellRuffleDataset(Dataset):\n"
    "    def __init__(self, image_dir, annotation_file, cell_type=None, augment=False):\n"
    "        from src.data.livecell_loader import LIVECellDataset\n"
    "        self.base = LIVECellDataset(image_dir, annotation_file, cell_type,\n"
    "                                    augment=augment, return_boundary=False)\n"
    "\n"
    "    def __len__(self): return len(self.base)\n"
    "\n"
    "    def __getitem__(self, idx):\n"
    "        img, mask = self.base[idx]\n"
    "        ruffle = make_pseudo_ruffle(img[0], mask.astype(np.uint8))\n"
    "        return (img,\n"
    "                mask.astype(np.float32)[None],\n"
    "                ruffle[None])\n"
    "\n"
    "\n"
    "from torch.utils.data import DataLoader\n"
    "\n"
    "BATCH_SIZE = 4   # ConvNeXt is lighter than ViT; use 4 on T4, 8 on A100\n"
    "\n"
    "train_ds = LIVECellRuffleDataset(TRAIN_IMGS, TRAIN_ANN, CELL_TYPE, augment=True)\n"
    "val_ds   = LIVECellRuffleDataset(TRAIN_IMGS, VAL_ANN,   CELL_TYPE, augment=False)\n"
    "\n"
    "train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,\n"
    "                      num_workers=2, pin_memory=True)\n"
    "val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,\n"
    "                      num_workers=2, pin_memory=True)\n"
    "\n"
    "print(f'Train: {len(train_ds)}  |  Val: {len(val_ds)}')\n"
    "img, msk, ruf = train_ds[0]\n"
    "print(f'img {img.shape}  mask {msk.shape}  ruffle {ruf.shape}')",
    "cm-s5-code"
))

# ── Step 6 — VRAM ────────────────────────────────────────────────────────────
cells.append(md("## Step 6 — VRAM Check", "cm-s6-md"))
cells.append(code(
    "import torch\n"
    "if torch.cuda.is_available():\n"
    "    gb = torch.cuda.get_device_properties(0).total_memory / 1e9\n"
    "    print(f'GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {gb:.1f} GB')\n"
    "    if gb < 8:\n"
    "        print('WARNING: Less than 8 GB — set BATCH_SIZE=2 and USE_AMP_R=True')\n"
    "    else:\n"
    "        print('OK for BATCH_SIZE=4 with AMP')\n"
    "else:\n"
    "    print('No GPU — training will be very slow')",
    "cm-s6-code"
))

# ── Step 7 — Init model ───────────────────────────────────────────────────────
cells.append(md("## Step 7 — Initialise Combined Model", "cm-s7-md"))
cells.append(code(
    "import torch\n"
    "from src.pe_diffusion.combined_model import CombinedRuffleSegmentation\n"
    "\n"
    "model_r = CombinedRuffleSegmentation(T=1000).to(device)\n"
    "model_r.load_hybrid(HYBRID_CKPT, device)\n"
    "\n"
    "total     = sum(p.numel() for p in model_r.parameters())\n"
    "trainable = sum(p.numel() for p in model_r.diff_net.parameters())\n"
    "frozen    = sum(p.numel() for p in model_r.hybrid.parameters())\n"
    "print(f'Total params:              {total:,}')\n"
    "print(f'Trainable (diff_net):      {trainable:,}')\n"
    "print(f'Frozen (hybrid backbone):  {frozen:,}')",
    "cm-s7-code"
))

# ── Step 8 — Train ────────────────────────────────────────────────────────────
cells.append(md("## Step 8 — Train Ruffle Diffusion Head", "cm-s8-md"))
cells.append(code(
    "import os, json as _json, shutil\n"
    "from torch.amp import GradScaler, autocast\n"
    "\n"
    "COMBINED_CKPT   = f'{CKPT_DIR}/combined_ruffle_{CELL_TYPE.lower()}.pt'\n"
    "COMBINED_RESUME = COMBINED_CKPT.replace('.pt', '_resume.pt')\n"
    "COMBINED_HIST   = COMBINED_CKPT.replace('.pt', '_history.json')\n"
    "\n"
    "EPOCHS_R     = 40\n"
    "LR_R         = 3e-4\n"
    "USE_AMP_R    = True\n"
    "SAVE_EVERY_R = 5\n"
    "\n"
    "def _atomic_save(obj, path):\n"
    "    tmp = '/tmp/_ckpt_atomic.pt'\n"
    "    torch.save(obj, tmp)\n"
    "    shutil.copy2(tmp, path + '.new')\n"
    "    os.replace(path + '.new', path)\n"
    "\n"
    "optimizer_r = torch.optim.AdamW(model_r.diff_net.parameters(), lr=LR_R, weight_decay=1e-4)\n"
    "scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_r, T_max=EPOCHS_R, eta_min=1e-6)\n"
    "scaler_r    = GradScaler('cuda', enabled=USE_AMP_R)\n"
    "\n"
    "start_epoch_r = 1\n"
    "train_losses_r, val_losses_r = [], []\n"
    "\n"
    "if os.path.isfile(COMBINED_RESUME):\n"
    "    try:\n"
    "        ckpt = torch.load(COMBINED_RESUME, map_location=device)\n"
    "        model_r.diff_net.load_state_dict(ckpt['diff_net'])\n"
    "        optimizer_r.load_state_dict(ckpt['optimizer'])\n"
    "        scheduler_r.load_state_dict(ckpt['scheduler'])\n"
    "        start_epoch_r  = ckpt['epoch'] + 1\n"
    "        train_losses_r = ckpt.get('train_losses', [])\n"
    "        val_losses_r   = ckpt.get('val_losses', [])\n"
    "        print(f'Resumed from epoch {start_epoch_r - 1}')\n"
    "    except Exception as e:\n"
    "        print(f'Resume corrupted ({e}), starting fresh.')\n"
    "        os.remove(COMBINED_RESUME)\n"
    "\n"
    "if start_epoch_r == 1:\n"
    "    print('Starting combined model training from scratch')\n"
    "\n"
    "print(f'Training epochs {start_epoch_r} to {EPOCHS_R}  |  diff_net only')\n"
    "print('-' * 60)\n"
    "\n"
    "for epoch in range(start_epoch_r, EPOCHS_R + 1):\n"
    "    model_r.diff_net.train()\n"
    "    t_loss = 0.0\n"
    "    for img, _, ruf in train_dl:\n"
    "        img, ruf = img.to(device), ruf.to(device)\n"
    "        optimizer_r.zero_grad()\n"
    "        with autocast('cuda', enabled=USE_AMP_R):\n"
    "            loss, _ = model_r.forward_train(img, ruf)\n"
    "        scaler_r.scale(loss).backward()\n"
    "        scaler_r.unscale_(optimizer_r)\n"
    "        torch.nn.utils.clip_grad_norm_(model_r.diff_net.parameters(), 1.0)\n"
    "        scaler_r.step(optimizer_r); scaler_r.update()\n"
    "        t_loss += loss.item()\n"
    "\n"
    "    model_r.diff_net.eval()\n"
    "    v_loss = 0.0\n"
    "    with torch.no_grad():\n"
    "        for img, _, ruf in val_dl:\n"
    "            img, ruf = img.to(device), ruf.to(device)\n"
    "            with autocast('cuda', enabled=USE_AMP_R):\n"
    "                loss, _ = model_r.forward_train(img, ruf)\n"
    "            v_loss += loss.item()\n"
    "\n"
    "    avg_t = t_loss / len(train_dl)\n"
    "    avg_v = v_loss / len(val_dl)\n"
    "    train_losses_r.append(avg_t); val_losses_r.append(avg_v)\n"
    "    scheduler_r.step()\n"
    "    lr_now = optimizer_r.param_groups[0]['lr']\n"
    "    print(f'Epoch {epoch:03d}/{EPOCHS_R}  train={avg_t:.4f}  val={avg_v:.4f}  lr={lr_now:.2e}')\n"
    "\n"
    "    if epoch % SAVE_EVERY_R == 0 or epoch == EPOCHS_R:\n"
    "        ep_path = COMBINED_CKPT.replace('.pt', f'_ep{epoch:03d}.pt')\n"
    "        _atomic_save(model_r.diff_net.state_dict(), ep_path)\n"
    "        print(f'  -> saved {ep_path}')\n"
    "\n"
    "    _atomic_save({'epoch': epoch,\n"
    "                  'diff_net':   model_r.diff_net.state_dict(),\n"
    "                  'optimizer':  optimizer_r.state_dict(),\n"
    "                  'scheduler':  scheduler_r.state_dict(),\n"
    "                  'train_losses': train_losses_r,\n"
    "                  'val_losses':   val_losses_r}, COMBINED_RESUME)\n"
    "\n"
    "_atomic_save(model_r.diff_net.state_dict(), COMBINED_CKPT)\n"
    "if os.path.isfile(COMBINED_RESUME): os.remove(COMBINED_RESUME)\n"
    "with open(COMBINED_HIST, 'w') as f:\n"
    "    _json.dump({'train_losses': train_losses_r, 'val_losses': val_losses_r}, f)\n"
    "print(f'Done. Checkpoint: {COMBINED_CKPT}')",
    "cm-s8-code"
))

# ── Step 9 — Load checkpoint ──────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — Load Checkpoint (if restarting Colab)\n"
    "\n"
    "Run this cell **instead of Step 8** when Colab restarted after training finished.",
    "cm-s9-md"
))
cells.append(code(
    "import torch\n"
    "from src.pe_diffusion.combined_model import CombinedRuffleSegmentation\n"
    "\n"
    "COMBINED_CKPT = f'{CKPT_DIR}/combined_ruffle_{CELL_TYPE.lower()}.pt'\n"
    "COMBINED_HIST = COMBINED_CKPT.replace('.pt', '_history.json')\n"
    "\n"
    "model_r = CombinedRuffleSegmentation(T=1000).to(device)\n"
    "model_r.load_hybrid(HYBRID_CKPT, device)\n"
    "\n"
    "diff_state = torch.load(COMBINED_CKPT, map_location=device)\n"
    "model_r.diff_net.load_state_dict(diff_state)\n"
    "model_r.eval()\n"
    "print(f'Loaded diff_net weights from {COMBINED_CKPT}')\n"
    "\n"
    "import json as _json\n"
    "with open(COMBINED_HIST) as f:\n"
    "    hist = _json.load(f)\n"
    "train_losses_r = hist['train_losses']\n"
    "val_losses_r   = hist['val_losses']\n"
    "print(f'History: {len(train_losses_r)} epochs  |  final train={train_losses_r[-1]:.4f}  val={val_losses_r[-1]:.4f}')",
    "cm-s9-code"
))

# ── Step 10 — Training curves ─────────────────────────────────────────────────
cells.append(md("## Step 10 — Training Curves", "cm-s10-md"))
cells.append(code(
    "import matplotlib.pyplot as plt\n"
    "\n"
    "plt.figure(figsize=(9, 4))\n"
    "plt.plot(train_losses_r, label='Train')\n"
    "plt.plot(val_losses_r,   label='Val')\n"
    "plt.xlabel('Epoch')\n"
    "plt.ylabel('Diffusion MSE Loss')\n"
    "plt.title(f'{CELL_TYPE} — Combined Model Training Curves')\n"
    "plt.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(f'{RESULTS}/training_curves_combined_{CELL_TYPE.lower()}.png', dpi=150)\n"
    "plt.show()\n"
    "print(f'Final  train={train_losses_r[-1]:.4f}  val={val_losses_r[-1]:.4f}')",
    "cm-s10-code"
))

# ── Step 11 — Visualisation ───────────────────────────────────────────────────
cells.append(md("## Step 11 — Inference Visualisation", "cm-s11-md"))
cells.append(code(
    "import random, torch, numpy as np, matplotlib.pyplot as plt\n"
    "\n"
    "model_r.diff_net.eval()\n"
    "N_SHOW = 4\n"
    "idxs   = random.sample(range(len(val_ds)), N_SHOW)\n"
    "\n"
    "fig, axes = plt.subplots(N_SHOW, 5, figsize=(22, 4 * N_SHOW))\n"
    "titles = ['Input', 'GT Mask', 'Hybrid Mask\\n(frozen, IoU=0.842)',\n"
    "          'Ruffle GT\\n(Frangi)', 'Predicted Ruffle\\n(combined diffusion)']\n"
    "for col, t in enumerate(titles): axes[0, col].set_title(t, fontsize=10)\n"
    "\n"
    "for row, idx in enumerate(idxs):\n"
    "    img_t, msk_t, ruf_t = val_ds[idx]\n"
    "    tensor = torch.from_numpy(img_t).unsqueeze(0).to(device)\n"
    "    with torch.no_grad():\n"
    "        mask_pred, ruffle_pred = model_r.sample(tensor, n_steps=50)\n"
    "    axes[row, 0].imshow(img_t[0],                    cmap='gray')\n"
    "    axes[row, 1].imshow(msk_t[0],                    cmap='hot')\n"
    "    axes[row, 2].imshow(mask_pred[0, 0].cpu() > 0.5, cmap='hot')\n"
    "    axes[row, 3].imshow(ruf_t[0],                    cmap='viridis')\n"
    "    axes[row, 4].imshow(ruffle_pred[0, 0].cpu(),     cmap='viridis')\n"
    "    for ax in axes[row]: ax.axis('off')\n"
    "\n"
    "plt.suptitle(f'{CELL_TYPE} — Combined Model Predictions', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig(f'{RESULTS}/combined_vis_{CELL_TYPE.lower()}.png', dpi=150)\n"
    "plt.show()",
    "cm-s11-code"
))

# ── Step 12 — Evaluation ─────────────────────────────────────────────────────
cells.append(md("## Step 12 — Quantitative Evaluation", "cm-s12-md"))
cells.append(code(
    "import numpy as np, torch\n"
    "from torch.utils.data import DataLoader\n"
    "from src.evaluation.acdc_benchmark import evaluate_segmentation\n"
    "\n"
    "eval_dl = DataLoader(val_ds, batch_size=1, shuffle=False)\n"
    "preds, gts = [], []\n"
    "\n"
    "model_r.diff_net.eval()\n"
    "with torch.no_grad():\n"
    "    for i, (img_t, msk_t, _) in enumerate(eval_dl):\n"
    "        tensor = img_t.to(device)\n"
    "        mask_pred, _ = model_r.sample(tensor, n_steps=50)\n"
    "        pred = (mask_pred[0, 0].cpu().numpy() > 0.5).astype(np.int32)\n"
    "        gt   = msk_t[0, 0].numpy().astype(np.int32)\n"
    "        preds.append(pred); gts.append(gt)\n"
    "        if (i + 1) % 20 == 0:\n"
    "            print(f'{i+1}/{len(eval_dl)} evaluated...')\n"
    "\n"
    "metrics_r = evaluate_segmentation(preds, gts, verbose=True)\n"
    "\n"
    "import json as _json\n"
    "with open(f'{RESULTS}/metrics_combined_{CELL_TYPE.lower()}.json', 'w') as f:\n"
    "    _json.dump(metrics_r, f, indent=2)\n"
    "print(f'Saved -> {RESULTS}/metrics_combined_{CELL_TYPE.lower()}.json')\n"
    "print()\n"
    "print('=== Comparison ===')\n"
    "print(f'V1 Hybrid model  IoU={0.842:.3f}  F1={0.913:.3f}')\n"
    "print(f'V2 PE-Diffusion  IoU={0.366:.3f}  F1={0.525:.3f}')\n"
    "print(f'V3 Combined      IoU={metrics_r[\"mean_iou\"]:.3f}  F1={metrics_r[\"mean_f1\"]:.3f}')",
    "cm-s12-code"
))

# ── Summary ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## Summary\n"
    "\n"
    "| Version | Architecture | Mean IoU | Mean F1 |\n"
    "|---------|--------------|----------|---------|\n"
    "| V1 | ConvNeXt-Tiny + UNet++ | 0.842 | 0.913 |\n"
    "| V2 | PE-Diffusion (ViT-B/16) | 0.366 | 0.525 |\n"
    "| V3 | Combined (frozen hybrid + ruffle diffusion) | TBD | TBD |\n"
    "\n"
    "| Output File | Location |\n"
    "|-------------|----------|\n"
    "| Checkpoint | `checkpoints/combined_ruffle_mcf7.pt` |\n"
    "| Metrics | `results/metrics_combined_mcf7.json` |\n"
    "| Visualisation | `results/combined_vis_mcf7.png` |\n"
    "| Training curves | `results/training_curves_combined_mcf7.png` |",
    "cm-summary"
))

# ── Build and write notebook ──────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

with open(NB_OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Written: {NB_OUT}')
print(f'Total cells: {len(cells)}')

with open(NB_OUT, encoding='utf-8') as f:
    json.load(f)
print('JSON valid.')
