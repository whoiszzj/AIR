from __future__ import annotations
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_sum
from fused_ssim import fused_ssim
import numpy as np
import matplotlib.pyplot as plt


def wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint
    class _CheckpointingWrapper(module.__class__):
        _restore_cls = module.__class__
        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)
        
    module.__class__ = _CheckpointingWrapper
    return module


def unwrap_module_with_gradient_checkpointing(module: nn.Module):
    module.__class__ = module.__class__._restore_cls


def wrap_dinov3_attention_with_sdpa(module: nn.Module):
    assert torch.__version__ >= '2.0', "SDPA requires PyTorch 2.0 or later"
    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)      # (B, H, N, C // H)

            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C) 

            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    module.__class__ = _AttentionWrapper
    return module

def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

# Render 2D gaussians into an image using gsplat.
def render(xy, scaling, rotation, color, H, W):
    # Ensure gsplat always receives float32 tensors (mixed precision may produce float16).
    xy = xy.to(dtype=torch.float32)
    scaling = scaling.to(dtype=torch.float32)
    rotation = rotation.to(dtype=torch.float32)
    color = color.to(dtype=torch.float32)
    tile_bounds = (
        (W + 16 - 1) // 16,
        (H + 16 - 1) // 16,
        1,
    )
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
        xy, scaling, rotation, H, W, tile_bounds
    )
    opacity = torch.ones_like(rotation)
    out_img, out_alpha = rasterize_gaussians_sum(
        xys, depths, radii, conics, num_tiles_hit,
        color, opacity, H, W, 16, 16, return_alpha=True
    )
    # out_img = torch.clamp(out_img, 0, 1)  # [H, W, 3]
    out_img = out_img.view(H, W, 3).permute(2, 0, 1).contiguous()
    return {"image": out_img, "alpha": out_alpha}

@torch.no_grad()
def _adam_update(p, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    p.addcdiv_(m_hat, (v_hat.sqrt() + eps), value=-lr)


# Optimize pseudo gaussians to match a target image with SSIM and L1 loss.
def render_pseudo(
    gaussian_raw, patch_center,
    target_img,                   # [3, H, W] or [H, W, 3]
    H, W,
    patch_size: int,
    mask: torch.Tensor = None,  #[L,]
    step: int = 10,
    lr: float = 1e-3,
):
    # Ensure optimization and fused ops are executed in float32 under mixed precision.
    gaussian_raw_u = gaussian_raw.detach().clone().to(torch.float32)
    patch_center = patch_center.to(torch.float32)
    target_img = target_img.to(torch.float32)

    m_ = torch.zeros_like(gaussian_raw_u) 
    v_ = torch.zeros_like(gaussian_raw_u)

    first_img = None
    first_alpha = None
    last_img = None

    with torch.enable_grad():
        for t in range(1, step + 1):
            gaussian_raw_u.requires_grad_(True)
            
            xy = torch.tanh(gaussian_raw_u[:, 0:2]) * patch_size + patch_center
            xy = pixel_to_xy(xy, H, W)
            scaling = torch.sigmoid(gaussian_raw_u[:, 2:4]) * patch_size + 0.5
            rotation = torch.sigmoid(gaussian_raw_u[:, 4:5]) * 2 * torch.pi
            color = torch.tanh(gaussian_raw_u[:, 5:8])
            
            out = render(xy, scaling, rotation, color, H, W)  # dict: {"image": [3,H,W]}
            img = out["image"]
            if first_img is None:
                first_img = img.detach()
                first_alpha = out["alpha"].detach()

            loss = 0.7 * F.l1_loss(img, target_img) + 0.3 * (1.0 - fused_ssim(img.unsqueeze(0), target_img.unsqueeze(0))) 

            grads = torch.autograd.grad(loss, [gaussian_raw_u])
            g_ = grads[0].detach()
            if mask is not None:
                g_ = g_ * mask.unsqueeze(1)
            _adam_update(gaussian_raw_u, g_, m_, v_, t, lr)

            gaussian_raw_u = gaussian_raw_u.detach()
            last_img = img.detach()

    pseudo_gaussian = gaussian_raw_u

    return {
        "first_img": first_img,
        "first_alpha": first_alpha,
        "last_image": last_img,  
        "pseudo_gaussian": pseudo_gaussian,
    }

def pixel_to_xy(pixel, H, W):
    x = pixel[..., 0] / (max(W - 1, 1)) * 2 - 1
    y = pixel[..., 1] / (max(H - 1, 1)) * 2 - 1
    return torch.stack([x, y], dim=-1)


def xy_to_pixel(xy, H, W):
    x = (xy[..., 0] + 1) / 2 * (W - 1)
    y = (xy[..., 1] + 1) / 2 * (H - 1)
    return torch.stack([x, y], dim=-1)

def mse_from_psnr(psnr, max_val=1.0):
    """
    Compute MSE from PSNR.
    
    Parameters:
        psnr (float): Peak Signal-to-Noise Ratio in dB.
        max_val (float): Maximum pixel value. 
                         Use 255 for 8-bit images, 1.0 for normalized images.

    Returns:
        float: Mean Squared Error (MSE)
    """
    return (max_val ** 2) / (10 ** (psnr / 10.0))
