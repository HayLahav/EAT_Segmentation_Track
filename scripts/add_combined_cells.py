"""Insert combined model cells into the PE Diffusion notebook."""
import json

NB_PATH = r'C:\Users\hayla\OneDrive\מסמכים\projects\EAT_Segmentation\notebooks\EAT_Segmentation_PE_Diffusion.ipynb'

with open(NB_PATH, encoding='utf-8-sig') as f:
    nb = json.load(f)

# ── New cells to insert before the Summary cell ──────────────────────────────

MD_COMBINED = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Part 2 — Combined Model: Frozen Hybrid Segmentation + Ruffle Diffusion Head\n",
        "\n",
        "Uses the already-trained `hybrid_seg_mcf7.pt` (IoU=0.842) frozen as the segmentation backbone.\n",
        "Only the lightweight `RuffleDiffusionHead` is trained — conditioned on ConvNeXt decoder features + predicted mask.\n",
        "\n",
        "**Why this is better than the standalone PE-Diffusion model:**\n",
        "- No conflicting segmentation vs ruffle gradients\n",
        "- ConvNeXt features are domain-adapted and encode membrane detail\n",
        "- ~3× fewer trainable parameters → faster convergence"
    ],
    "id": "combined-md-intro"
}

TRAIN_COMBINED = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, json as _json, shutil\n",
        "import torch\n",
        "from torch.amp import GradScaler, autocast\n",
        "from src.pe_diffusion.combined_model import CombinedRuffleSegmentation\n",
        "\n",
        "COMBINED_CKPT   = f'{CKPT_DIR}/combined_ruffle_{CELL_TYPE.lower()}.pt'\n",
        "COMBINED_RESUME = COMBINED_CKPT.replace('.pt', '_resume.pt')\n",
        "COMBINED_HIST   = COMBINED_CKPT.replace('.pt', '_history.json')\n",
        "\n",
        "EPOCHS_R     = 40\n",
        "LR_R         = 3e-4\n",
        "USE_AMP_R    = True\n",
        "SAVE_EVERY_R = 5\n",
        "\n",
        "def _atomic_save(obj, path):\n",
        "    tmp = '/tmp/_ckpt_atomic.pt'\n",
        "    torch.save(obj, tmp)\n",
        "    shutil.copy2(tmp, path + '.new')\n",
        "    os.replace(path + '.new', path)\n",
        "\n",
        "# Build model — hybrid weights are loaded and frozen inside load_hybrid()\n",
        "model_r = CombinedRuffleSegmentation(T=1000).to(device)\n",
        "model_r.load_hybrid(CHECKPOINT, device)   # CHECKPOINT = hybrid_seg_mcf7.pt\n",
        "\n",
        "# Only train the diffusion head\n",
        "optimizer_r = torch.optim.AdamW(model_r.diff_net.parameters(), lr=LR_R, weight_decay=1e-4)\n",
        "scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_r, T_max=EPOCHS_R, eta_min=1e-6)\n",
        "scaler_r    = GradScaler('cuda', enabled=USE_AMP_R)\n",
        "\n",
        "start_epoch_r = 1\n",
        "train_losses_r, val_losses_r = [], []\n",
        "\n",
        "if os.path.isfile(COMBINED_RESUME):\n",
        "    try:\n",
        "        ckpt = torch.load(COMBINED_RESUME, map_location=device)\n",
        "        model_r.diff_net.load_state_dict(ckpt['diff_net'])\n",
        "        optimizer_r.load_state_dict(ckpt['optimizer'])\n",
        "        scheduler_r.load_state_dict(ckpt['scheduler'])\n",
        "        start_epoch_r  = ckpt['epoch'] + 1\n",
        "        train_losses_r = ckpt.get('train_losses', [])\n",
        "        val_losses_r   = ckpt.get('val_losses', [])\n",
        "        print(f'Resumed from epoch {start_epoch_r - 1}')\n",
        "    except Exception as e:\n",
        "        print(f'Resume corrupted ({e}), starting fresh.')\n",
        "        os.remove(COMBINED_RESUME)\n",
        "\n",
        "if start_epoch_r == 1:\n",
        "    print('Starting combined model training from scratch')\n",
        "\n",
        "print(f'Training epochs {start_epoch_r} to {EPOCHS_R}  |  diff_net only')\n",
        "print('-' * 60)\n",
        "\n",
        "for epoch in range(start_epoch_r, EPOCHS_R + 1):\n",
        "    model_r.diff_net.train()\n",
        "    t_loss = 0.0\n",
        "    for img, _, ruf in train_dl:\n",
        "        img, ruf = img.to(device), ruf.to(device)\n",
        "        optimizer_r.zero_grad()\n",
        "        with autocast('cuda', enabled=USE_AMP_R):\n",
        "            loss, _ = model_r.forward_train(img, ruf)\n",
        "        scaler_r.scale(loss).backward()\n",
        "        scaler_r.unscale_(optimizer_r)\n",
        "        torch.nn.utils.clip_grad_norm_(model_r.diff_net.parameters(), 1.0)\n",
        "        scaler_r.step(optimizer_r); scaler_r.update()\n",
        "        t_loss += loss.item()\n",
        "\n",
        "    model_r.diff_net.eval()\n",
        "    v_loss = 0.0\n",
        "    with torch.no_grad():\n",
        "        for img, _, ruf in val_dl:\n",
        "            img, ruf = img.to(device), ruf.to(device)\n",
        "            with autocast('cuda', enabled=USE_AMP_R):\n",
        "                loss, _ = model_r.forward_train(img, ruf)\n",
        "            v_loss += loss.item()\n",
        "\n",
        "    avg_t = t_loss / len(train_dl)\n",
        "    avg_v = v_loss / len(val_dl)\n",
        "    train_losses_r.append(avg_t); val_losses_r.append(avg_v)\n",
        "    scheduler_r.step()\n",
        "    lr_now = optimizer_r.param_groups[0]['lr']\n",
        "    print(f'Epoch {epoch:03d}/{EPOCHS_R}  train={avg_t:.4f}  val={avg_v:.4f}  lr={lr_now:.2e}')\n",
        "\n",
        "    if epoch % SAVE_EVERY_R == 0 or epoch == EPOCHS_R:\n",
        "        ep_path = COMBINED_CKPT.replace('.pt', f'_ep{epoch:03d}.pt')\n",
        "        _atomic_save(model_r.diff_net.state_dict(), ep_path)\n",
        "        print(f'  -> saved {ep_path}')\n",
        "\n",
        "    _atomic_save({'epoch': epoch,\n",
        "                  'diff_net':   model_r.diff_net.state_dict(),\n",
        "                  'optimizer':  optimizer_r.state_dict(),\n",
        "                  'scheduler':  scheduler_r.state_dict(),\n",
        "                  'train_losses': train_losses_r,\n",
        "                  'val_losses':   val_losses_r}, COMBINED_RESUME)\n",
        "\n",
        "_atomic_save(model_r.diff_net.state_dict(), COMBINED_CKPT)\n",
        "if os.path.isfile(COMBINED_RESUME): os.remove(COMBINED_RESUME)\n",
        "with open(COMBINED_HIST, 'w') as f:\n",
        "    _json.dump({'train_losses': train_losses_r, 'val_losses': val_losses_r}, f)\n",
        "print(f'Done. Checkpoint: {COMBINED_CKPT}')"
    ],
    "id": "combined-train"
}

VIS_COMBINED = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import random, torch, numpy as np, matplotlib.pyplot as plt\n",
        "\n",
        "model_r.diff_net.eval()\n",
        "N_SHOW = 4\n",
        "idxs   = random.sample(range(len(val_ds)), N_SHOW)\n",
        "\n",
        "fig, axes = plt.subplots(N_SHOW, 5, figsize=(22, 4 * N_SHOW))\n",
        "titles = ['Input', 'GT Mask', 'Hybrid Mask\\n(frozen, IoU=0.842)', 'Ruffle GT\\n(Frangi)', 'Predicted Ruffle\\n(combined diffusion)']\n",
        "for col, t in enumerate(titles): axes[0, col].set_title(t, fontsize=10)\n",
        "\n",
        "for row, idx in enumerate(idxs):\n",
        "    img_t, msk_t, ruf_t = val_ds[idx]\n",
        "    tensor = torch.from_numpy(img_t).unsqueeze(0).to(device)\n",
        "    with torch.no_grad():\n",
        "        mask_pred, ruffle_pred = model_r.sample(tensor, n_steps=50)\n",
        "    axes[row, 0].imshow(img_t[0],                    cmap='gray')\n",
        "    axes[row, 1].imshow(msk_t[0],                    cmap='hot')\n",
        "    axes[row, 2].imshow(mask_pred[0, 0].cpu() > 0.5, cmap='hot')\n",
        "    axes[row, 3].imshow(ruf_t[0],                    cmap='viridis')\n",
        "    axes[row, 4].imshow(ruffle_pred[0, 0].cpu(),     cmap='viridis')\n",
        "    for ax in axes[row]: ax.axis('off')\n",
        "\n",
        "plt.suptitle(f'{CELL_TYPE} — Combined Model Predictions', fontsize=13)\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{RESULTS}/combined_vis_{CELL_TYPE.lower()}.png', dpi=150)\n",
        "plt.show()"
    ],
    "id": "combined-vis"
}

EVAL_COMBINED = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import numpy as np, torch\n",
        "from torch.utils.data import DataLoader\n",
        "from src.evaluation.acdc_benchmark import evaluate_segmentation\n",
        "\n",
        "eval_dl = DataLoader(val_ds, batch_size=1, shuffle=False)\n",
        "preds, gts = [], []\n",
        "\n",
        "model_r.diff_net.eval()\n",
        "with torch.no_grad():\n",
        "    for i, (img_t, msk_t, _) in enumerate(eval_dl):\n",
        "        tensor = img_t.to(device)\n",
        "        mask_pred, _ = model_r.sample(tensor, n_steps=50)\n",
        "        pred = (mask_pred[0, 0].cpu().numpy() > 0.5).astype(np.int32)\n",
        "        gt   = msk_t[0, 0].numpy().astype(np.int32)\n",
        "        preds.append(pred); gts.append(gt)\n",
        "        if (i + 1) % 20 == 0:\n",
        "            print(f'{i+1}/{len(eval_dl)} evaluated...')\n",
        "\n",
        "metrics_r = evaluate_segmentation(preds, gts, verbose=True)\n",
        "\n",
        "import json as _json\n",
        "with open(f'{RESULTS}/metrics_combined_{CELL_TYPE.lower()}.json', 'w') as f:\n",
        "    _json.dump(metrics_r, f, indent=2)\n",
        "print(f'Saved -> {RESULTS}/metrics_combined_{CELL_TYPE.lower()}.json')\n",
        "print()\n",
        "print('=== Comparison ===')\n",
        "print(f'Hybrid model  IoU={0.842:.3f}  F1={0.913:.3f}')\n",
        "print(f'Combined      IoU={metrics_r[\"mean_iou\"]:.3f}  F1={metrics_r[\"mean_f1\"]:.3f}')"
    ],
    "id": "combined-eval"
}

# Insert before the summary cell
ids = [c.get('id', '') for c in nb['cells']]
idx = next(i for i, cid in enumerate(ids) if cid == 'ua8e1YJc6T4n')

new_cells = [MD_COMBINED, TRAIN_COMBINED, VIS_COMBINED, EVAL_COMBINED]
for offset, cell in enumerate(new_cells):
    nb['cells'].insert(idx + offset, cell)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Total cells: {len(nb['cells'])}")

# Validate JSON
with open(NB_PATH, encoding='utf-8') as f:
    json.load(f)
print("JSON valid.")
