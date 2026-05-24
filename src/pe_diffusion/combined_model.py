"""
Combined Ruffle Segmentation
=============================
Architecture
------------
Stage 1 — HybridSegmentationModel (frozen, already trained):
    ConvNeXt-Tiny encoder + UNet++ decoder
    → mask   (B, 1, H, W)   binary segmentation  (IoU=0.842 on MCF7)
    → feats  (B, 64, H, W)  rich multi-scale decoder feature map

Stage 2 — RuffleDiffusionHead (trainable):
    Lightweight DDPM denoiser conditioned on hybrid features + predicted mask.
    Input:  cat[x_t (1ch), feats (64ch), mask (1ch)] = 66ch  +  timestep t
    Output: predicted noise ε (1ch) — ruffle channel only

Training objective
------------------
  x_0  = pseudo_ruffle_GT  (B,1,H,W) in [-1,1]
  x_t  = sqrt(ab_t)*x_0 + sqrt(1-ab_t)*eps,   eps ~ N(0,I)
  loss = MSE(eps_pred, eps)          # pure ruffle denoising objective

Why this is better than PEDiffusionSegmentation
------------------------------------------------
  - Segmentation handled by the proven hybrid model (no conflicting gradients)
  - Diffusion head focuses on one task: ruffle prediction
  - ConvNeXt features encode domain-adapted boundary detail
  - Predicted mask gives diffusion a spatial prior (where the boundary band is)
  - ~3x fewer trainable parameters than the full PE-diffusion model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pe_diffusion.model import DiffusionUNet, DiffusionSchedule
from src.segmentation.hybrid_model import HybridSegmentationModel


class CombinedRuffleSegmentation(nn.Module):
    """
    Frozen HybridSegmentationModel + trainable RuffleDiffusionHead.

    Usage
    -----
    # Build and load hybrid weights
    model = CombinedRuffleSegmentation(T=1000).to(device)
    model.load_hybrid(HYBRID_CHECKPOINT, device)

    # Training (only diff_net parameters update)
    optimizer = torch.optim.AdamW(model.diff_net.parameters(), lr=5e-5)
    loss, mask = model.forward_train(img, pseudo_ruffle)

    # Inference
    mask, ruffle = model.sample(img, n_steps=50)
    """

    def __init__(self, T: int = 1000):
        super().__init__()
        self.hybrid   = HybridSegmentationModel(pretrained=False, use_checkpoint=False)
        # 1ch noisy ruffle + 64ch decoder feats + 1ch mask = 66ch input, 1ch output
        self.diff_net = DiffusionUNet(in_ch=1 + 64 + 1, out_ch=1)
        self.schedule = DiffusionSchedule(T=T)

    def load_hybrid(self, path: str, device) -> None:
        """Load pre-trained hybrid weights and permanently freeze the encoder+decoder."""
        state = torch.load(path, map_location=device)
        # hybrid_seg checkpoints may store full model state or just state_dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = self.hybrid.load_state_dict(state, strict=False)
        if unexpected:
            print(f"  (ignored {len(unexpected)} keys from older checkpoint: viaevca/refinement modules)")
        if missing:
            raise RuntimeError(f"Missing keys in hybrid checkpoint: {missing}")
        for p in self.hybrid.parameters():
            p.requires_grad_(False)
        self.hybrid.eval()
        print(f"Hybrid model loaded and frozen from {path}")

    @torch.no_grad()
    def _hybrid_feats(self, img: torch.Tensor):
        """Run frozen hybrid model → (mask, decoder_feats)."""
        out, feats = self.hybrid(img, return_feats=True)
        return out["mask"], feats   # (B,1,H,W), (B,64,H,W)

    def forward_train(self, img: torch.Tensor, pseudo_ruffle: torch.Tensor):
        """
        Args:
            img           (B, 1, H, W)  grayscale cell image
            pseudo_ruffle (B, 1, H, W)  Frangi pseudo-ruffle GT in [0, 1]
        Returns:
            loss   scalar
            mask   (B, 1, H, W) hybrid segmentation (for logging)
        """
        mask, feats = self._hybrid_feats(img)

        x0       = pseudo_ruffle * 2 - 1                            # scale to [-1, 1]
        B        = x0.shape[0]
        t        = torch.randint(0, self.schedule.T, (B,), device=img.device)
        xt, eps  = self.schedule.q_sample(x0, t)

        eps_pred = self.diff_net(torch.cat([xt, feats, mask], dim=1), t)
        loss     = F.mse_loss(eps_pred, eps)
        return loss, mask

    @torch.no_grad()
    def sample(self, img: torch.Tensor, n_steps: int = 50):
        """
        Args:
            img     (B, 1, H, W)  grayscale cell image
            n_steps int           DDIM reverse steps (50 recommended)
        Returns:
            mask   (B, 1, H, W)  binary segmentation from hybrid model
            ruffle (B, 1, H, W)  predicted ruffle map in [0, 1]
        """
        mask, feats = self._hybrid_feats(img)
        x_T = torch.randn(img.shape[0], 1, img.shape[2], img.shape[3],
                          device=img.device)

        def model_fn(xt, t):
            return self.diff_net(torch.cat([xt, feats, mask], dim=1), t)

        x0     = self.schedule.ddim_sample(model_fn, x_T, n_steps=n_steps)
        ruffle = torch.sigmoid(x0 * 3)
        return mask, ruffle
