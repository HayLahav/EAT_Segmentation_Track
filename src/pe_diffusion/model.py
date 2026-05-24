"""
PE-UNet++ Diffusion Segmentation Model
=======================================
Architecture
------------
1. ViTEncoder        — timm ViT-B/16 (grayscale-adapted, dynamic image size).
                       Extracts spatial tokens from 4 intermediate blocks,
                       all at H/16 × W/16, 768 channels.

2. ViTUNetPPDecoder  — Bridges ViT (single-scale) to UNet++ (multi-scale).
                       Projects + resamples 4 ViT block outputs into 4 encoder
                       levels at H/4, H/8, H/16, H/32 — then runs the exact
                       same UNet++ dense node structure as HybridSegmentationModel.
                       Outputs: coarse_mask (B,1,H,W) + cond_feats (B,64,H,W).

3. DiffusionUNet     — Lightweight DDPM denoiser conditioned on cond_feats.
                       Input:  cat[x_t (2ch), cond_feats (64ch)] + timestep t
                       Output: predicted noise ε (2ch)
                               ch0 = segmentation noise
                               ch1 = ruffle-map noise

Training objective
------------------
  x_0  = cat[GT_mask, pseudo_ruffle]  (B,2,H,W) in [-1,1]
  x_t  = sqrt(ab_t)*x_0 + sqrt(1-ab_t)*eps,   eps ~ N(0,I)
  loss = 2*MSE(eps_pred[:,0], eps[:,0])   (mask, weighted 2x)
       +   MSE(eps_pred[:,1], eps[:,1])   (ruffle)
       + BCE(coarse, GT_mask) + Dice(coarse, GT_mask)

Inference (DDIM, 50 steps)
--------------------------
  x_T ~ N(0,I)  ->  DDIM reverse conditioned on ViT+UNet++ features
  mask   = sigmoid(x_0[:,0] * 3)
  ruffle = sigmoid(x_0[:,1] * 3)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 0.  PE weight loader  (call after model.__init__, before training)
# ─────────────────────────────────────────────────────────────────────────────

def load_pe_weights(encoder: "ViTEncoder",
                    pe_repo_path: str = "/content/perception_models",
                    model_name: str = "PE-Core-B16-224") -> None:
    """
    Load PE-Core-B weights from facebookresearch/perception_models into
    our ViTEncoder, then re-apply the grayscale patch-embed adaptation.

    Call this right after instantiating PEDiffusionSegmentation:

        model = PEDiffusionSegmentation(pretrained=False)
        load_pe_weights(model.encoder, '/content/perception_models')

    The function tries several import patterns that match Meta's repo
    conventions.  If none work, it prints the top-level directory listing
    so you can adapt the import manually.
    """
    import sys, os
    sys.path.insert(0, pe_repo_path)

    # core.vision_encoder.pe is the correct module path in facebookresearch/perception_models
    from core.vision_encoder.pe import VisionTransformer   # noqa
    pe_vit = VisionTransformer.from_config(model_name, pretrained=True)

    # ── Copy weights to our encoder ──────────────────────────────────────────
    # PE and timm ViT-B share the same architecture; keys may differ slightly.
    src = pe_vit.state_dict()
    dst = encoder.state_dict()

    # Build a best-effort key mapping: strip common prefixes/suffixes
    def _norm(k):
        for prefix in ("model.", "encoder.", "backbone.", "trunk."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        return k

    src_norm = {_norm(k): v for k, v in src.items()}
    mapped, skipped = {}, []
    for dst_key in dst:
        norm = _norm(dst_key)
        if norm in src_norm and src_norm[norm].shape == dst[dst_key].shape:
            mapped[dst_key] = src_norm[norm]
        else:
            skipped.append(dst_key)

    encoder.load_state_dict({**dst, **mapped}, strict=False)
    print(f"PE weights loaded: {len(mapped)} tensors copied, {len(skipped)} skipped.")

    # Re-apply grayscale adaptation on patch embedding
    pe_conv = encoder.patch_embed.proj
    grey    = torch.nn.Conv2d(1, 768, kernel_size=16, stride=16, bias=False)
    grey.weight.data.copy_(pe_conv.weight.data.mean(dim=1, keepdim=True))
    encoder.patch_embed.proj = grey.to(next(encoder.parameters()).device)
    print("Grayscale patch-embed re-applied.")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ViT Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """
    ViT-B/16 backbone (grayscale-adapted, dynamic image size).
    Extracts spatial feature maps from blocks [2, 5, 8, 11] — all at H/16.

    Default: loads timm ImageNet weights (good baseline).
    Better:  call load_pe_weights(encoder) after init to swap in PE-Core-B
             weights from facebookresearch/perception_models.

    freeze_blocks: freeze the first N transformer blocks (default 8 of 12).
    """
    EXTRACT = [2, 5, 8, 11]

    def __init__(self, pretrained: bool = True, freeze_blocks: int = 8):
        super().__init__()
        assert _TIMM_AVAILABLE, "pip install timm"
        vit = timm.create_model(
            "vit_base_patch16_224.augreg_in21k",
            pretrained=pretrained,
            dynamic_img_size=True,
            num_classes=0,
        )
        # Adapt patch embedding from 3-ch RGB to 1-ch grayscale
        old_proj = vit.patch_embed.proj
        new_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16, bias=False)
        new_proj.weight.data.copy_(old_proj.weight.data.mean(dim=1, keepdim=True))
        vit.patch_embed.proj = new_proj

        self.patch_embed = vit.patch_embed
        self.cls_token   = vit.cls_token
        self.pos_embed   = vit.pos_embed
        self.pos_drop    = nn.Dropout(p=0.0)
        self.blocks      = vit.blocks
        self.embed_dim   = 768
        self.patch_size  = 16

        for i, blk in enumerate(self.blocks):
            if i < freeze_blocks:
                for p in blk.parameters():
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        tokens = self.patch_embed(x)
        # newer timm with dynamic_img_size returns (B, H', W', C) — flatten to (B, N, C)
        if tokens.dim() == 4:
            tokens = tokens.reshape(B, -1, self.embed_dim)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self._interp_pos_embed(B, h, w, x.device)
        tokens = self.pos_drop(tokens)

        feats = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i in self.EXTRACT:
                sp = tokens[:, 1:].permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
                feats.append(sp)   # (B, 768, H/16, W/16)
        return feats               # [block2, block5, block8, block11]

    def _interp_pos_embed(self, B, h, w, device):
        pe  = self.pos_embed[0, 1:]                       # (N_orig, 768)
        N   = pe.shape[0]
        h0  = w0 = int(N ** 0.5)
        pe  = pe.reshape(1, h0, w0, 768).permute(0, 3, 1, 2)
        pe  = F.interpolate(pe.to(device), size=(h, w),
                            mode="bicubic", align_corners=False)
        pe  = pe.flatten(2).permute(0, 2, 1)              # (1, h*w, 768)
        cls = self.pos_embed[:, :1].to(device)
        return torch.cat([cls, pe], dim=1).expand(B, -1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  UNet++ decoder primitives
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class _UpProject(nn.Module):
    """Bilinear upsample then 1x1 project."""
    def __init__(self, ic, oc):
        super().__init__()
        self.proj = nn.Conv2d(ic, oc, 1, bias=False)
    def forward(self, x, size=None):
        kw = {"size": size, "mode": "bilinear", "align_corners": False} if size \
             else {"scale_factor": 2.0, "mode": "bilinear", "align_corners": False}
        return self.proj(F.interpolate(x, **kw))


class _DenseNode(nn.Module):
    """One UNet++ dense node: cat inputs -> ConvBnRelu."""
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = _ConvBnRelu(ic, oc)
    def forward(self, tensors):
        return self.conv(torch.cat(tensors, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ViT -> UNet++ Decoder
# ─────────────────────────────────────────────────────────────────────────────

class ViTUNetPPDecoder(nn.Module):
    """
    Bridges ViT single-scale features to a multi-scale UNet++ decoder.

    All 4 ViT block outputs are at H/16. They are projected and resampled
    to create 4 encoder levels matching the ConvNeXt skip channel sizes:

        e1   96ch  H/4    <- ViT block 2  + upsample x4
        e2  192ch  H/8    <- ViT block 5  + upsample x2
        e3  384ch  H/16   <- ViT block 8  (no resample)
        e4  768ch  H/32   <- ViT block 11 + avg-pool x2

    Then runs the identical UNet++ dense node layout as HybridSegmentationModel:
        Scale H/16: x31
        Scale H/8:  x21, x22
        Scale H/4:  x11, x12, x13   <- final decoder features (64ch)

    Outputs:
        coarse_mask  (B, 1, H, W)   sigmoid segmentation probability
        cond_feats   (B, 64, H, W)  dense feature map for diffusion conditioning
    """
    def __init__(self):
        super().__init__()
        # ViT -> encoder level projections
        self.to_e1 = nn.Conv2d(768,  96, 1, bias=False)
        self.to_e2 = nn.Conv2d(768, 192, 1, bias=False)
        self.to_e3 = nn.Conv2d(768, 384, 1, bias=False)
        self.to_e4 = nn.Conv2d(768, 768, 1, bias=False)

        # UNet++ upsampling projections (same as HybridSegmentationModel)
        self.up_e4  = _UpProject(768, 256)   # H/32 -> H/16
        self.up_e3  = _UpProject(384, 128)   # H/16 -> H/8
        self.up_x31 = _UpProject(256, 128)   # H/16 -> H/8
        self.up_e2  = _UpProject(192,  64)   # H/8  -> H/4
        self.up_x21 = _UpProject(128,  64)   # H/8  -> H/4
        self.up_x22 = _UpProject(128,  64)   # H/8  -> H/4

        # UNet++ dense nodes (identical channel math to HybridSegmentationModel)
        self.x31 = _DenseNode(384 + 256,            256)
        self.x21 = _DenseNode(192 + 128,            128)
        self.x22 = _DenseNode(192 + 128 + 128,      128)
        self.x11 = _DenseNode( 96 + 64,              64)
        self.x12 = _DenseNode( 96 + 64 + 64,         64)
        self.x13 = _DenseNode( 96 + 64 + 64 + 64,    64)

        self.coarse_head = nn.Conv2d(64, 1, 1)

    def forward(self, feats, target_size):
        """
        feats:       list of 4 (B, 768, H/16, W/16) from ViTEncoder
        target_size: (H, W) original image resolution
        """
        e1 = F.interpolate(self.to_e1(feats[0]),
                           scale_factor=4.0, mode="bilinear", align_corners=False)
        e2 = F.interpolate(self.to_e2(feats[1]),
                           scale_factor=2.0, mode="bilinear", align_corners=False)
        e3 = self.to_e3(feats[2])
        e4 = F.avg_pool2d(self.to_e4(feats[3]), kernel_size=2)

        # UNet++ decoder
        x31 = self.x31([e3,      self.up_e4(e4,   e3.shape[2:])])
        x21 = self.x21([e2,      self.up_e3(e3,   e2.shape[2:])])
        x22 = self.x22([e2, x21, self.up_x31(x31, e2.shape[2:])])
        x11 = self.x11([e1,           self.up_e2(e2,   e1.shape[2:])])
        x12 = self.x12([e1, x11,      self.up_x21(x21, e1.shape[2:])])
        x13 = self.x13([e1, x11, x12, self.up_x22(x22, e1.shape[2:])])

        d = F.interpolate(x13, size=target_size, mode="bilinear", align_corners=False)
        return self.coarse_head(d), d   # coarse logits, cond_feats


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Diffusion Schedule
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionSchedule(nn.Module):
    """Cosine beta schedule (Nichol & Dhariwal 2021) with q_sample and DDIM reverse.

    Keeps signal meaningful longer than the linear schedule — better for fine
    spatial details like thin boundary/ruffle lines.
    """

    def __init__(self, T: int = 1000, s: float = 0.008):
        super().__init__()
        self.T = T
        steps = torch.arange(T + 1, dtype=torch.float64)
        f     = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
        ab    = (f / f[0]).clamp(min=1e-5).float()   # alpha_bar_t, shape (T+1,)
        ab    = ab[1:]                                 # drop t=0 entry → shape (T,)
        self.register_buffer("sqrt_ab",    ab.sqrt())
        self.register_buffer("sqrt_1m_ab", (1.0 - ab).sqrt())
        self.register_buffer("alpha_bar",  ab)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s  = self.sqrt_ab[t].view(-1, 1, 1, 1)
        s1 = self.sqrt_1m_ab[t].view(-1, 1, 1, 1)
        return s * x0 + s1 * noise, noise

    def ddim_sample(self, model_fn, x_T, n_steps: int = 20):
        B, device = x_T.shape[0], x_T.device
        idx = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=device)
        x   = x_T
        for i, t_val in enumerate(idx):
            t      = t_val.long().expand(B)
            eps    = model_fn(x, t)
            ab_t   = self.alpha_bar[t].view(-1, 1, 1, 1)
            x0_hat = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
            if i < len(idx) - 1:
                ab_p = self.alpha_bar[idx[i + 1].long()].view(-1, 1, 1, 1)
                x    = ab_p.sqrt() * x0_hat + (1 - ab_p).sqrt() * eps
            else:
                x    = x0_hat
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Diffusion U-Net
# ─────────────────────────────────────────────────────────────────────────────

class _AdaGN(nn.Module):
    def __init__(self, ch, t_dim):
        super().__init__()
        self.gn  = nn.GroupNorm(8, ch)
        self.lin = nn.Linear(t_dim, 2 * ch)
    def forward(self, x, te):
        sc, sh = self.lin(te).chunk(2, dim=1)
        return self.gn(x) * (1 + sc[:, :, None, None]) + sh[:, :, None, None]


class _DBlock(nn.Module):
    def __init__(self, ic, oc, t_dim, downsample=False):
        super().__init__()
        self.c1    = nn.Conv2d(ic, oc, 3, stride=2 if downsample else 1, padding=1)
        self.c2    = nn.Conv2d(oc, oc, 3, padding=1)
        self.adagn = _AdaGN(oc, t_dim)
        self.skip  = nn.Conv2d(ic, oc, 1) if (ic != oc or downsample) else nn.Identity()
        self.ds    = downsample
    def forward(self, x, te):
        h = F.silu(self.adagn(self.c1(x), te))
        h = self.c2(h)
        s = self.skip(F.avg_pool2d(x, 2) if self.ds else x)
        return F.silu(h + s)


class DiffusionUNet(nn.Module):
    """
    Lightweight UNet denoiser.
    Input : cat[x_t (2ch), cond_feats (64ch)] = 66ch  +  timestep t
    Output: predicted noise eps (out_ch channels)
    """
    T_DIM = 256

    def __init__(self, in_ch: int = 66, out_ch: int = 2):
        super().__init__()
        T = self.T_DIM
        self.t_mlp = nn.Sequential(nn.Linear(T, T * 4), nn.SiLU(), nn.Linear(T * 4, T))
        self.enc0  = _DBlock(in_ch,  64, T)
        self.enc1  = _DBlock(64,  128, T, downsample=True)
        self.enc2  = _DBlock(128, 256, T, downsample=True)
        self.bot   = _DBlock(256, 256, T)
        self.dec2  = _DBlock(256 + 256, 128, T)
        self.dec1  = _DBlock(128 + 128,  64, T)
        self.dec0  = _DBlock( 64 +  64,  64, T)
        self.out   = nn.Conv2d(64, out_ch, 1)

    @staticmethod
    def _sinusoidal(t, dim):
        half  = dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / (half - 1))
        args  = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, x, t):
        te = self.t_mlp(self._sinusoidal(t, self.T_DIM))
        e0 = self.enc0(x,  te)
        e1 = self.enc1(e0, te)
        e2 = self.enc2(e1, te)
        b  = self.bot(e2,  te)
        d  = self.dec2(torch.cat([F.interpolate(b, e2.shape[2:], mode="nearest"), e2], 1), te)
        d  = self.dec1(torch.cat([F.interpolate(d, e1.shape[2:], mode="nearest"), e1], 1), te)
        d  = self.dec0(torch.cat([F.interpolate(d, e0.shape[2:], mode="nearest"), e0], 1), te)
        return self.out(d)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Combined Model
# ─────────────────────────────────────────────────────────────────────────────

class PEDiffusionSegmentation(nn.Module):
    """
    Full pipeline: ViTEncoder -> ViTUNetPPDecoder -> DiffusionUNet.

    Training:
        loss, coarse_mask = model.forward_train(img, gt_mask, pseudo_ruffle)

    Inference:
        mask, ruffle = model.sample(img, n_steps=20)
    """
    def __init__(self, T: int = 1000, pretrained: bool = True):
        super().__init__()
        self.encoder  = ViTEncoder(pretrained=pretrained, freeze_blocks=6)
        self.decoder  = ViTUNetPPDecoder()
        self.diff_net = DiffusionUNet(in_ch=2 + 64 + 1)  # +1 for coarse mask conditioning
        self.schedule = DiffusionSchedule(T=T)

    def _encode(self, img: torch.Tensor):
        H, W = img.shape[2:]
        ph   = ((H + 15) // 16) * 16
        pw   = ((W + 15) // 16) * 16
        pad  = F.pad(img, [0, pw - W, 0, ph - H])
        feats          = self.encoder(pad)
        coarse, cond   = self.decoder(feats, (H, W))
        return coarse, cond

    def forward_train(self, img, gt_mask, pseudo_ruffle):
        coarse, cond = self._encode(img)
        x0           = torch.cat([gt_mask * 2 - 1, pseudo_ruffle * 2 - 1], dim=1)
        B            = x0.shape[0]
        t            = torch.randint(0, self.schedule.T, (B,), device=img.device)
        xt, noise    = self.schedule.q_sample(x0, t)
        coarse_prob  = torch.sigmoid(coarse).detach()
        eps_pred     = self.diff_net(torch.cat([xt, cond, coarse_prob], dim=1), t)
        diff_loss    = (2.0 * F.mse_loss(eps_pred[:, 0:1], noise[:, 0:1])   # mask channel weighted 2x
                        +     F.mse_loss(eps_pred[:, 1:2], noise[:, 1:2]))   # ruffle channel
        coarse_loss  = F.binary_cross_entropy_with_logits(coarse, gt_mask)
        p            = torch.sigmoid(coarse)
        dice         = 1 - (2 * (p * gt_mask).sum() + 1) / (p.sum() + gt_mask.sum() + 1)
        loss         = diff_loss + coarse_loss + dice
        return loss, torch.sigmoid(coarse)

    @torch.no_grad()
    def sample(self, img: torch.Tensor, n_steps: int = 50):
        coarse_logits, cond = self._encode(img)
        coarse = torch.sigmoid(coarse_logits)
        x_T = torch.randn(img.shape[0], 2, img.shape[2], img.shape[3], device=img.device)

        def model_fn(xt, t):
            return self.diff_net(torch.cat([xt, cond, coarse], dim=1), t)

        x0     = self.schedule.ddim_sample(model_fn, x_T, n_steps=n_steps)
        mask   = torch.sigmoid(x0[:, 0:1] * 3)
        ruffle = torch.sigmoid(x0[:, 1:2] * 3)
        return mask, ruffle
