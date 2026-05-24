import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.filters import frangi
from skimage.morphology import binary_dilation, binary_erosion, disk


def _membrane_band(mask_binary: np.ndarray, width: int = 4) -> np.ndarray:
    outer = binary_dilation(mask_binary, disk(width))
    inner = binary_erosion(mask_binary, disk(width))
    return outer & ~inner


def _touching_pixels(cell_mask: np.ndarray, instance_map: np.ndarray) -> np.ndarray:
    """True where cell_mask borders a different cell label (not background)."""
    cell_id = instance_map[cell_mask][0] if cell_mask.any() else 0
    dilated = binary_dilation(cell_mask, disk(2))
    neighbor_region = dilated & ~cell_mask
    neighbor_labels = instance_map[neighbor_region]
    has_neighbor = np.isin(neighbor_labels, [l for l in np.unique(neighbor_labels) if l > 0 and l != cell_id])
    touching = np.zeros_like(cell_mask)
    touching[neighbor_region] = has_neighbor
    return touching.astype(bool)


def _soft_mask(cell_mask: np.ndarray, sharpness: float = 0.3) -> np.ndarray:
    dt = ndimage.distance_transform_edt(cell_mask)
    return 1.0 / (1.0 + np.exp(-sharpness * dt))


def _contour_curvature(contour: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Signed curvature along a 2-D contour (N, 2) in row,col order."""
    n = len(contour)
    if n < 3:
        return np.zeros(n)
    y, x = contour[:, 0].astype(float), contour[:, 1].astype(float)
    # smooth with a uniform kernel
    k = np.ones(smooth) / smooth
    y = np.convolve(y, k, mode="same")
    x = np.convolve(x, k, mode="same")
    dy = np.gradient(y)
    dx = np.gradient(x)
    d2y = np.gradient(dy)
    d2x = np.gradient(dx)
    denom = (dx**2 + dy**2) ** 1.5 + 1e-8
    return (dx * d2y - dy * d2x) / denom


def _frangi_ridge_stats(
    image_crop: np.ndarray,
    soft_mask: np.ndarray,
    free_band: np.ndarray,
) -> tuple:
    """Returns (ridge_density, mean_ridge_length) on the free membrane band."""
    response = frangi(image_crop * soft_mask, sigmas=(1, 2, 3), black_ridges=False)
    ridge_binary = (response > 0.05) & free_band
    labeled, n_cc = ndimage.label(ridge_binary)
    if n_cc == 0:
        return 0.0, 0.0
    lengths = [float((labeled == i).sum() ** 0.5) for i in range(1, n_cc + 1)]
    perimeter = free_band.sum() + 1e-8
    density = ridge_binary.sum() / perimeter
    mean_length = float(np.mean(lengths))
    return float(density), mean_length


def extract_cell_features(
    image: np.ndarray,
    instance_map: np.ndarray,
    cell_id: int,
    dt: float,
    prev_instance_map: np.ndarray = None,
) -> dict:
    """
    Per-cell classical CV feature extraction.

    Args:
        image         (H, W) float32 ∈ [0, 1]  — raw frame
        instance_map  (H, W) int32              — labeled instance map
        cell_id       int                       — which cell to analyse
        dt            float                     — minutes since last frame
        prev_instance_map (H, W) int32          — previous frame instance map

    Returns dict with 9 features (same column names as before for CSV compat):
        Ruffle_Density, Ruffle_Directionality, Protrusion_Length,
        Bleb_Count, Bleb_Area_Fraction, Bleb_Mean_Circularity,
        Bleb_Formation_Rate, Bleb_Lifetime_Mean, Bleb_Perimeter_Fraction,
        Boundary_Uncertainty_Score
    """
    cell_mask = instance_map == cell_id
    if not cell_mask.any():
        return _zero_features()

    band = _membrane_band(cell_mask)
    touching = _touching_pixels(cell_mask, instance_map)
    free_band = band & ~touching

    perimeter = float(free_band.sum()) + 1e-8

    # ── Ruffle features via Frangi on free membrane band ─────────────────────
    rows, cols = np.where(cell_mask)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    pad = 8
    sr = slice(max(r0 - pad, 0), min(r1 + pad, image.shape[0]))
    sc = slice(max(c0 - pad, 0), min(c1 + pad, image.shape[1]))

    crop_img   = image[sr, sc]
    crop_soft  = _soft_mask(cell_mask[sr, sc])
    crop_band  = free_band[sr, sc]

    ruffle_density, mean_ridge_len = _frangi_ridge_stats(crop_img, crop_soft, crop_band)

    # Directionality: std of ridge orientation angles in the free band
    response_full = frangi(crop_img * crop_soft, sigmas=(1, 2, 3), black_ridges=False)
    ridge_pixels = response_full[crop_band]
    ruffle_dir = float(ridge_pixels.std()) if len(ridge_pixels) > 1 else 0.0

    # ── Bleb features via contour curvature ──────────────────────────────────
    contours = _get_contours(cell_mask)
    n_blebs = 0
    circularities = []
    bleb_area_total = 0
    bleb_on_boundary = 0

    for contour in contours:
        if len(contour) < 8:
            continue
        curv = _contour_curvature(contour)
        # positive curvature = outward bump = bleb candidate
        peaks, props = find_peaks(
            curv, height=0.02, distance=5, prominence=0.01
        )
        n_blebs += len(peaks)

        # Approximate each peak as a small arc and measure its circularity
        for pk in peaks:
            arc_len = 6
            arc = contour[max(0, pk - arc_len): pk + arc_len + 1]
            if len(arc) < 3:
                continue
            area = arc_len * 2
            perim_arc = arc_len * 2 + 1e-8
            c = min(4 * np.pi * area / (perim_arc**2), 1.0)
            circularities.append(c)

        bleb_area_total += len(peaks) * 25  # rough px² per bleb arc

    bleb_area_frac = bleb_area_total / (cell_mask.sum() + 1e-8)
    bleb_circularity = float(np.mean(circularities)) if circularities else 0.0

    # Bleb_Perimeter_Fraction: fraction of free-band perimeter at curvature peaks
    bleb_perimeter_frac = float(n_blebs * 12 / perimeter) if perimeter > 0 else 0.0

    # Formation rate: new blebs compared to previous frame
    if prev_instance_map is not None:
        prev_mask = prev_instance_map == cell_id
        prev_contours = _get_contours(prev_mask)
        prev_n_blebs = 0
        for c2 in prev_contours:
            if len(c2) < 8:
                continue
            cv2 = _contour_curvature(c2)
            pks2, _ = find_peaks(cv2, height=0.02, distance=5, prominence=0.01)
            prev_n_blebs += len(pks2)
        new_blebs = max(0, n_blebs - prev_n_blebs)
        formation_rate = new_blebs / (dt / 60.0 + 1e-8)
    else:
        formation_rate = 0.0

    bleb_lifetime = float(dt / max(n_blebs, 1))

    # Boundary uncertainty: fuzzy entropy of the distance transform boundary
    dt_map = ndimage.distance_transform_edt(cell_mask)
    boundary_prob = 1.0 / (1.0 + np.exp(dt_map[band] - 3.0))
    boundary_uncertainty = float((boundary_prob * (1 - boundary_prob)).mean()) if band.any() else 0.0

    return {
        "Ruffle_Density":          ruffle_density,
        "Ruffle_Directionality":   ruffle_dir,
        "Protrusion_Length":       mean_ridge_len,
        "Bleb_Count":              int(n_blebs),
        "Bleb_Area_Fraction":      float(bleb_area_frac),
        "Bleb_Mean_Circularity":   bleb_circularity,
        "Bleb_Formation_Rate":     formation_rate,
        "Bleb_Lifetime_Mean":      bleb_lifetime,
        "Bleb_Perimeter_Fraction": bleb_perimeter_frac,
        "Boundary_Uncertainty_Score": boundary_uncertainty,
    }


def _zero_features() -> dict:
    return {
        "Ruffle_Density": 0.0, "Ruffle_Directionality": 0.0,
        "Protrusion_Length": 0.0, "Bleb_Count": 0,
        "Bleb_Area_Fraction": 0.0, "Bleb_Mean_Circularity": 0.0,
        "Bleb_Formation_Rate": 0.0, "Bleb_Lifetime_Mean": 0.0,
        "Bleb_Perimeter_Fraction": 0.0, "Boundary_Uncertainty_Score": 0.0,
    }


def _get_contours(mask: np.ndarray) -> list:
    """Return list of (N,2) row-col contour arrays from a binary mask."""
    from skimage.measure import find_contours
    return find_contours(mask.astype(float), level=0.5)
