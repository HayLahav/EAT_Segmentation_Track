"""
Evaluate trained model on LIVECell test set.

Writes two output files:
    results/metrics.json        — mean IoU, F1, Boundary F1
    results/predictions.csv     — per-cell features for all test frames

Usage:
    python scripts/evaluate.py \\
        --livecell_dir  /path/to/LIVECell/images/livecell_test_images \\
        --test_ann      /path/to/LIVECell_single_cell_test.json \\
        --checkpoint    checkpoints/hybrid_seg_mcf7.pt \\
        --cell_type     MCF7 \\
        --output_dir    results/
"""
import argparse
import os
import sys

# Allow running from project root without installing package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.livecell_loader import LIVECellDataset
from src.pipeline import EATSegmentationPipeline
from src.evaluation.acdc_benchmark import evaluate_segmentation
from src.output.results_writer import write_cell_csv, write_metrics_json


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Cell type: {args.cell_type}")

    pipeline = EATSegmentationPipeline(args.checkpoint, device=device)
    test_ds  = LIVECellDataset(args.livecell_dir, args.test_ann, args.cell_type)
    loader   = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test set: {len(test_ds)} images")

    all_records, preds, gts = [], [], []

    for frame_idx, (img, mask) in enumerate(loader):
        frame_np = img[0, 0].numpy()
        records  = pipeline.process_frame(frame_np, frame_idx=frame_idx)
        all_records.extend(records)

        # Build binary pred mask for evaluation
        tensor = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0).to(
            torch.device(device)
        )
        with torch.no_grad():
            out = pipeline.model(tensor)
        pred_mask = (out["mask"][0, 0].cpu().numpy() > 0.5).astype(np.int32)
        gt_mask   = mask[0].numpy().astype(np.int32)
        preds.append(pred_mask)
        gts.append(gt_mask)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{len(test_ds)} frames")

    print("\nEvaluation results:")
    metrics = evaluate_segmentation(preds, gts, verbose=True)

    write_metrics_json(metrics, os.path.join(args.output_dir, "metrics.json"))
    write_cell_csv(all_records, os.path.join(args.output_dir, "predictions.csv"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--livecell_dir", required=True)
    p.add_argument("--test_ann",     required=True)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--cell_type",    default="MCF7")
    p.add_argument("--output_dir",   default="results/")
    run(p.parse_args())
