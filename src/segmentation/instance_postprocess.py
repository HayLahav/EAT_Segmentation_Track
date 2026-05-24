"""
Instance segmentation post-processing.
Converts semantic mask + distance transform → per-cell instance IDs via watershed.
Extracts morphological features: Area, x_Pos, y_Pos, Sphericity.
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage import label, center_of_mass


def masks_to_instances(
    mask: np.ndarray,
    distance: np.ndarray,
    threshold: float = 0.5,
    min_cell_area: int = 50,
) -> np.ndarray:
    """
    Semantic mask + distance transform → integer instance map.

    Args:
        mask          (H, W) float32 ∈ [0,1]  from model sigmoid output
        distance      (H, W) float32           from model distance head
        threshold     float                    binarization threshold
        min_cell_area int                      discard instances smaller than this (pixels)

    Returns:
        instance_map (H, W) int32 — 0 = background, 1..N = cell instance IDs
    """
    binary      = mask > threshold
    smooth_dist = ndimage.gaussian_filter(distance, sigma=2.0)
    local_max   = (smooth_dist == ndimage.maximum_filter(smooth_dist, size=10)) & binary
    markers, n  = label(local_max)

    if n == 0:
        labeled, _ = label(binary)
        return labeled.astype(np.int32)

    try:
        from skimage.segmentation import watershed
        instance_map = watershed(-smooth_dist, markers, mask=binary)
    except ImportError:
        instance_map, _ = label(binary)

    # Remove very small instances (noise)
    for cell_id in np.unique(instance_map):
        if cell_id > 0 and (instance_map == cell_id).sum() < min_cell_area:
            instance_map[instance_map == cell_id] = 0

    return instance_map.astype(np.int32)


def extract_morphology(instance_map: np.ndarray) -> dict:
    """
    Per-cell morphological features from an integer instance map.

    Returns:
        {cell_id: {"Area": int, "x_Pos": float, "y_Pos": float, "Sphericity": float}}

    Sphericity (2D circularity proxy):
        = 4π·Area / Perimeter²   →  1.0 = perfect circle (Bleb)
        Low values indicate elongated/irregular shapes (Ruffle / pseudopod)
    """
    features = {}
    for cell_id in np.unique(instance_map):
        if cell_id == 0:
            continue
        region    = instance_map == cell_id
        area      = int(region.sum())
        cy, cx    = center_of_mass(region)
        eroded    = ndimage.binary_erosion(region)
        perimeter = int((region.astype(int) - eroded.astype(int)).sum())
        sphericity = min(float(4 * np.pi * area / (perimeter ** 2 + 1e-8)), 1.0)

        features[cell_id] = {
            "Area":       area,
            "x_Pos":      float(cx),
            "y_Pos":      float(cy),
            "Sphericity": sphericity,
        }
    return features
