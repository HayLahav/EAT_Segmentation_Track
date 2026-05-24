import os
import json
import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import Dataset


class LIVECellDataset(Dataset):
    """
    LIVECell dataset loader (Edlund et al., Nature Methods 2021).
    COCO-format annotations. Phase-contrast, 8 cancer cell lines.

    Download (run once in Colab):
        !aws s3 sync s3://livecell-dataset/LIVECell_dataset_2021/ /content/LIVECell --no-sign-request

    EAT-relevant cell types: MCF7, SkBr3, SKOV3

    Args:
        image_dir:        path to LIVECell images folder
        annotation_file:  path to COCO JSON annotation file
                          (LIVECell_single_cell_train.json / _val.json / _test.json)
        cell_type:        optional filter string, e.g. "MCF7". None = all 8 types.
        augment:          apply random flips, rotations, and brightness jitter (train only)
        return_boundary:  if True, return a 3-tuple (img, mask, boundary) where
                          boundary is a float32 map of cell edge pixels (4px wide)
    """
    def __init__(self, image_dir: str, annotation_file: str, cell_type: str = None,
                 augment: bool = False, return_boundary: bool = False):
        with open(annotation_file) as f:
            coco = json.load(f)

        ann_by_image = {}
        for ann in coco["annotations"]:
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        self.augment          = augment
        self.return_boundary  = return_boundary
        self.samples = []
        for img_info in coco["images"]:
            if cell_type and cell_type.lower() not in img_info["file_name"].lower():
                continue
            self.samples.append({
                "path":        os.path.join(image_dir, img_info["file_name"]),
                "height":      img_info["height"],
                "width":       img_info["width"],
                "annotations": ann_by_image.get(img_info["id"], []),
                "file_name":   img_info["file_name"],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = tifffile.imread(s["path"]).astype(np.float32)
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img[np.newaxis]   # (1, H, W)

        # Build binary cell mask from COCO segmentation polygons
        mask = np.zeros((s["height"], s["width"]), dtype=np.uint8)
        for ann in s["annotations"]:
            if ann.get("segmentation"):
                try:
                    from pycocotools import mask as coco_mask
                    rle = coco_mask.frPyObjects(ann["segmentation"],
                                                s["height"], s["width"])
                    m   = coco_mask.decode(coco_mask.merge(rle))
                    mask = np.maximum(mask, m)
                except Exception:
                    pass

        if self.augment:
            img, mask = self._augment(img, mask)

        if self.return_boundary:
            boundary = self._make_boundary(mask)
            return img.astype(np.float32), mask.astype(np.int64), boundary

        return img.astype(np.float32), mask.astype(np.int64)

    @staticmethod
    def _make_boundary(mask: np.ndarray) -> np.ndarray:
        """4px-wide boundary ring: dilation(mask, 2px) XOR erosion(mask, 2px)."""
        m        = mask.astype(bool)
        outer    = binary_dilation(m, iterations=2)
        inner    = binary_erosion(m,  iterations=2)
        boundary = outer & ~inner
        return boundary.astype(np.float32)

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        # img: (1, H, W) float32,  mask: (H, W) uint8
        # Horizontal flip
        if np.random.rand() > 0.5:
            img  = img[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        # Vertical flip
        if np.random.rand() > 0.5:
            img  = img[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
        # 180-degree rotation only — 90/270 swap H and W, breaking batching
        if np.random.rand() > 0.5:
            img  = np.rot90(img,  2, axes=(1, 2)).copy()
            mask = np.rot90(mask, 2, axes=(0, 1)).copy()
        # Brightness jitter
        img = np.clip(img + np.random.uniform(-0.1, 0.1), 0.0, 1.0)
        # Contrast jitter — scale around mean (more relevant for phase-contrast)
        mean = img.mean()
        img  = np.clip((img - mean) * np.random.uniform(0.8, 1.2) + mean, 0.0, 1.0)
        # Gaussian noise — helps diffusion model learn to denoise
        img  = np.clip(img + np.random.normal(0, 0.015, img.shape), 0.0, 1.0).astype(np.float32)
        return img, mask
