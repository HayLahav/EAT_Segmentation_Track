"""
Benchmark evaluator — comparison against Cell_ACDC outputs.
Reference: https://github.com/SchmollerLab/Cell_ACDC

Metrics:
  IoU (Jaccard)    — overall overlap
  F1  (Dice)       — harmonic precision/recall
  BF1 (Boundary F1) — edge accuracy, key for Ruffle/Bleb membrane regions

LIVECell baselines (from literature):
  Das & Zun 2025 semi-unsupervised: IoU=0.43, F1=0.60
  Cellpose / Cell_ACDC:             IoU≈0.55–0.65
  Our target:                       IoU≥0.70, F1≥0.80, BF1≥0.65
"""
import numpy as np
from scipy.ndimage import binary_dilation


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Jaccard index. Inputs: 2D int/bool arrays."""
    p, g = pred > 0, gt > 0
    return float((p & g).sum() / ((p | g).sum() + 1e-8))


def compute_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice F1 coefficient."""
    p, g = pred > 0, gt > 0
    tp   = (p & g).sum()
    return float(2 * tp / (2 * tp + (p & ~g).sum() + (~p & g).sum() + 1e-8))


def compute_boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 3) -> float:
    """
    Boundary F1 — measures how accurately cell edges are predicted.
    Most sensitive to Ruffle/Bleb membrane irregularities.
    tolerance: pixel distance within which boundary pixels count as matching.
    """
    def _boundary(mask):
        return (binary_dilation(mask > 0).astype(int) - (mask > 0).astype(int))

    pb, gb    = _boundary(pred), _boundary(gt)
    precision = (pb & binary_dilation(gb > 0, iterations=tolerance)).sum() / (pb.sum() + 1e-8)
    recall    = (gb & binary_dilation(pb > 0, iterations=tolerance)).sum() / (gb.sum() + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def evaluate_segmentation(
    predictions: list,
    ground_truths: list,
    verbose: bool = False,
) -> dict:
    """
    Evaluate predicted masks against LIVECell ground truth masks.

    Args:
        predictions:   list of (H, W) int/bool arrays
        ground_truths: list of (H, W) int/bool arrays
    Returns:
        dict with mean_iou, mean_f1, mean_boundary_f1, per_frame list
    """
    assert len(predictions) == len(ground_truths), "predictions and ground_truths must have equal length"
    ious, f1s, bf1s = [], [], []
    for pred, gt in zip(predictions, ground_truths):
        ious.append(compute_iou(pred, gt))
        f1s.append(compute_f1(pred, gt))
        bf1s.append(compute_boundary_f1(pred, gt))

    result = {
        "mean_iou":         float(np.mean(ious)),
        "mean_f1":          float(np.mean(f1s)),
        "mean_boundary_f1": float(np.mean(bf1s)),
        "per_frame": [
            {"iou": i, "f1": f, "boundary_f1": b}
            for i, f, b in zip(ious, f1s, bf1s)
        ],
    }
    if verbose:
        print(f"  IoU: {result['mean_iou']:.4f} | F1: {result['mean_f1']:.4f} | "
              f"BF1: {result['mean_boundary_f1']:.4f}")
    return result
