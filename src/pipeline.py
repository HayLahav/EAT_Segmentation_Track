"""
EAT Segmentation Pipeline — end-to-end inference on a single LIVECell frame.

Input:  (H, W) float32 numpy array, normalized to [0, 1]
Output: list of cell record dicts, one per detected instance

Each record contains:
    cell_id, frame, Area, Sphericity, x_Pos, y_Pos,
    Ruffle_Density, Ruffle_Directionality, Protrusion_Length,
    Bleb_Count, Bleb_Area_Fraction, Bleb_Mean_Circularity,
    Bleb_Formation_Rate, Bleb_Lifetime_Mean, Bleb_Perimeter_Fraction,
    Boundary_Uncertainty_Score
"""
import numpy as np
import torch

from src.segmentation.hybrid_model import HybridSegmentationModel
from src.segmentation.fss import FuzzySpatialSegmentation
from src.segmentation.instance_postprocess import masks_to_instances, extract_morphology
from src.features.extractor import extract_cell_features


class EATSegmentationPipeline:
    def __init__(self, model_path: str = None, device: str = "cpu",
                 use_fss: bool = True, use_viaevca: bool = False):
        self.device = torch.device(device)
        self.use_fss = use_fss
        self.model = HybridSegmentationModel(
            pretrained=False, use_viaevca=False
        ).to(self.device)
        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        self.model.eval()
        self.fss = FuzzySpatialSegmentation()
        self._prev_instance_map = None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        dt_minutes: float = 10.0,
    ) -> list:
        """
        Args:
            frame       (H, W) float32 ∈ [0, 1]
            frame_idx   integer frame number (used as column in output CSV)
            dt_minutes  time since previous frame in minutes

        Returns list of cell record dicts.
        """
        H, W = frame.shape

        # Pad to multiple of 32 (ResNet requirement)
        ph = ((H + 31) // 32) * 32
        pw = ((W + 31) // 32) * 32
        padded = np.zeros((ph, pw), dtype=np.float32)
        padded[:H, :W] = frame

        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        mask_np = out["mask"][0, 0].cpu().numpy()[:H, :W]
        dist_np = out["distance"][0, 0].cpu().numpy()[:H, :W]

        refined = self.fss.refine(mask_np) if self.use_fss else (mask_np > 0.5).astype("float32")
        instance_map = masks_to_instances(refined, dist_np)
        morph = extract_morphology(instance_map)

        records = []
        for cid, cell_morph in morph.items():
            feats = extract_cell_features(
                image=frame,
                instance_map=instance_map,
                cell_id=int(cid),
                dt=dt_minutes,
                prev_instance_map=self._prev_instance_map,
            )
            records.append({
                "cell_id": int(cid),
                "frame": frame_idx,
                **cell_morph,
                **feats,
            })

        self._prev_instance_map = instance_map
        return records
