"""
Fuzzy Spatial Segmentation (FSS)
Based on: Das & Zun 2025 — Semi-Unsupervised Microscopy Segmentation with
Fuzzy Logic and Spatial Statistics for Cross-Domain Analysis Using a GUI.

Applied here as a post-processing step on the trained model's probability map.
Only uncertain boundary pixels (the fuzzy 'gray' class) are modified;
confident predictions (dark / bright classes) are left unchanged.

Original Das & Zun used fuzzy membership on raw intensity images from an
untrained pipeline. Here the input is already a calibrated probability map
from a trained deep network, so the membership thresholds are defined in
probability space rather than intensity space.
"""
import numpy as np
from scipy.ndimage import uniform_filter


def morans_i(x: np.ndarray, window: int = 5) -> float:
    """
    Global Moran's I spatial autocorrelation statistic.
    Positive  → spatial clustering  (active membrane / cell zones)
    Negative  → spatial dispersion  (random noise)
    Zero      → no spatial structure

    Args:
        x:      2D float array (H, W)
        window: neighbourhood size for spatial weights
    """
    z         = x - x.mean()
    n         = x.size
    local_avg = uniform_filter(z, size=window)
    num       = (z * local_avg).sum()
    denom     = (z ** 2).sum() + 1e-8
    # Weight sum: each pixel has approximately window² neighbours
    W         = float(n * (window ** 2 - 1))
    return float((n / W) * (num / denom)) if W > 0 else 0.0


def _local_morans_field(prob_map: np.ndarray, window: int) -> np.ndarray:
    """
    Per-pixel local Moran's I-style field over the probability map.
    Positive → pixel's neighbourhood clusters toward cell.
    Negative → pixel's neighbourhood clusters toward background.
    """
    z         = prob_map.astype(np.float64) - prob_map.mean()
    local_avg = uniform_filter(z, size=window)
    return local_avg


class FuzzySpatialSegmentation:
    """
    Refines the model's cell probability map using fuzzy membership + Moran's I.

    Three-class fuzzy model (Das & Zun eq. 4–6), applied in probability space:
      Dark  class: p < uncertain_lo   → confident background, left unchanged
      Gray  class: uncertain_lo ≤ p ≤ uncertain_hi
                   → uncertain boundary pixels; Moran's I field pushes them
                     toward the dominant local class (cell or background)
      Bright class: p > uncertain_hi  → confident cell body, left unchanged

    The gray class targets Ruffle/Bleb boundary pixels — the model outputs
    p ≈ 0.35–0.65 precisely where membrane contrast is lowest. Spatial
    autocorrelation resolves the ambiguity: a boundary pixel surrounded by
    confirmed cell pixels is pushed above 0.5 (included); one surrounded by
    background is pushed below 0.5 (excluded).

    Args:
        spatial_window: neighbourhood radius for local Moran's I field (pixels)
        moran_weight:   correction strength applied to gray-class pixels
        uncertain_lo:   lower bound of fuzzy gray class in probability space
        uncertain_hi:   upper bound of fuzzy gray class in probability space
    """
    def __init__(self, spatial_window: int = 7, moran_weight: float = 0.5,
                 uncertain_lo: float = 0.35, uncertain_hi: float = 0.65):
        self.spatial_window = spatial_window
        self.moran_weight   = moran_weight
        self.uncertain_lo   = uncertain_lo
        self.uncertain_hi   = uncertain_hi

    def refine(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Args:
            prob_map: (H, W) float32 ∈ [0, 1] — model's cell probability map
        Returns:
            refined:  (H, W) float32 ∈ [0, 1] — spatially coherent boundary map
        """
        # Gray class: pixels the model is genuinely uncertain about
        uncertain = (prob_map >= self.uncertain_lo) & (prob_map <= self.uncertain_hi)

        # Local Moran's I field: positive = cell cluster, negative = background
        moran_field = _local_morans_field(prob_map, self.spatial_window)

        # Push uncertain pixels toward the dominant class of their neighbourhood;
        # confident pixels (dark / bright classes) are never modified
        refined = prob_map.copy().astype(np.float64)
        refined[uncertain] = np.clip(
            prob_map[uncertain] + self.moran_weight * moran_field[uncertain],
            0.0, 1.0,
        )
        return refined.astype(np.float32)
