"""
Training script for HybridSegmentationModel on LIVECell.

EAT-relevant cell types: MCF7, SkBr3, SKOV3

Usage:
    python -m src.training.train_hybrid \\
        --livecell_dir  /path/to/LIVECell/images \\
        --train_ann     /path/to/LIVECell_single_cell_train.json \\
        --val_ann       /path/to/LIVECell_single_cell_val.json \\
        --cell_type     MCF7 --epochs 50 \\
        --output        checkpoints/hybrid_seg_mcf7.pt

Resume after Colab interruption:
    Re-run the same command. If checkpoints/<name>_resume.pt exists it is
    automatically loaded and training continues from the saved epoch.

Usage (Google Colab — see notebooks/EAT_Segmentation_Colab.ipynb):
    Run the training cell in the notebook.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.livecell_loader import LIVECellDataset
from src.segmentation.hybrid_model import HybridSegmentationModel
from src.training.losses import dice_loss


def _resume_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return base + "_resume.pt"


def _save_resume(path, epoch, model, optimizer, scheduler, best_val,
                 train_losses, val_losses, epochs_no_improve=0):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss":        best_val,
        "train_losses":         train_losses,
        "val_losses":           val_losses,
        "epochs_no_improve":    epochs_no_improve,
    }, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Cell type  : {args.cell_type}")

    train_ds = LIVECellDataset(args.livecell_dir, args.train_ann, args.cell_type,
                               augment=True,  return_boundary=False)
    val_ds   = LIVECellDataset(args.livecell_dir, args.val_ann,   args.cell_type,
                               augment=False, return_boundary=False)
    print(f"Train : {len(train_ds)} images  |  Val : {len(val_ds)} images")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    model = HybridSegmentationModel(
        pretrained=True,
        use_checkpoint=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    resume_pt         = _resume_path(args.output)
    best_val          = float("inf")
    start_epoch       = 1
    train_losses      = []
    val_losses        = []
    epochs_no_improve = 0

    if os.path.isfile(resume_pt):
        print(f"Resuming from {resume_pt} …")
        ckpt = torch.load(resume_pt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val          = ckpt["best_val_loss"]
        start_epoch       = ckpt["epoch"] + 1
        train_losses      = ckpt.get("train_losses", [])
        val_losses        = ckpt.get("val_losses",   [])
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        print(f"  Resumed at epoch {start_epoch}  (best val so far: {best_val:.4f})")
    else:
        print("No resume checkpoint found — starting fresh.")

    print(f"\nTraining | Augmentation: ON (train) | Scheduler: ReduceLROnPlateau(patience=3)")
    print(f"Early stopping patience: {args.patience}")
    print("─" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_dl:
            imgs  = imgs.to(device)
            masks = masks.to(device).unsqueeze(1).float()
            out   = model(imgs)
            pred  = out["mask"]
            loss  = F.binary_cross_entropy(pred, masks) + dice_loss(pred, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs  = imgs.to(device)
                masks = masks.to(device).unsqueeze(1).float()
                out   = model(imgs)
                pred  = out["mask"]
                val_loss += (F.binary_cross_entropy(pred, masks) + dice_loss(pred, masks)).item()

        avg_t = train_loss / len(train_dl)
        avg_v = val_loss   / len(val_dl)
        train_losses.append(avg_t)
        val_losses.append(avg_v)
        scheduler.step(avg_v)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{args.epochs}  train={avg_t:.4f}  val={avg_v:.4f}  lr={current_lr:.2e}")

        if avg_v < best_val:
            best_val          = avg_v
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.output)
            print(f"  → best model saved: {args.output}")
        else:
            epochs_no_improve += 1

        _save_resume(resume_pt, epoch, model, optimizer, scheduler,
                     best_val, train_losses, val_losses,
                     epochs_no_improve=epochs_no_improve)

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no val improvement for {args.patience} epochs.")
            break

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")

    if os.path.isfile(resume_pt):
        os.remove(resume_pt)
        print(f"Resume checkpoint removed: {resume_pt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train EAT Hybrid Segmentation Model on LIVECell")
    p.add_argument("--livecell_dir", required=True)
    p.add_argument("--train_ann",    required=True)
    p.add_argument("--val_ann",      required=True)
    p.add_argument("--cell_type",    default="MCF7",
                   help="MCF7 | SkBr3 | SKOV3 | None (all)")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--output",       default="checkpoints/hybrid_seg.pt")
    p.add_argument("--patience",     type=int,   default=7,
                   help="Early stopping patience (0 = disabled)")
    train(p.parse_args())
