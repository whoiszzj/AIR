from typing import *
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ConvNeXtUnet, DINOv3Encoder, ConvStack, PatchEmbedding
from .utils import normalized_view_plane_uv


class GaussianHeadViT(nn.Module):
    encoder: DINOv3Encoder
    patch_embedding: PatchEmbedding
    neck: ConvStack
    gaussian_feat_decoder: ConvStack

    def __init__(
        self,
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        gaussian_feat_decoder: Dict[str, Any],
        patch_embedding: Dict[str, Any],
    ):
        super(GaussianHeadViT, self).__init__()
        self.encoder = DINOv3Encoder(**encoder)
        self.neck = ConvStack(**neck)
        self.gaussian_feat_decoder = ConvStack(**gaussian_feat_decoder)
        self.patch_embedding = PatchEmbedding(**patch_embedding)
        self.patch_size = patch_embedding["patch_size"]
        self.feat_dim = patch_embedding["out_dim"]

        self.feat_layer_norm = nn.LayerNorm(self.feat_dim)
        self.gaussian_linear = nn.Linear(self.feat_dim, 8)

    def init_weights(self):
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.neck.enable_gradient_checkpointing()
        self.gaussian_feat_decoder.enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()

    def get_feat(self, x):
        # image: [B, 3, H, W]
        # num_tokens: int
        batch_size, _, img_h, img_w = x.shape
        device, dtype = x.device, x.dtype

        base_w = img_w // 16
        base_h = img_h // 16

        # Backbones encoding
        features, cls_token = self.encoder(x, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2 ** level,
                height=base_h * 2 ** level,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        features = self.neck(features)
        # Heads decoding
        gaussian_feat = self.gaussian_feat_decoder(features)[-1]
        # Resize
        gaussian_feat = F.interpolate(
            gaussian_feat,
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        return gaussian_feat, cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, cls_token = self.get_feat(x)

        # Patchify the final feature map.
        patched_feats, patched_centers, (patched_h, patched_w) = self.patch_embedding(feat)
        # patched_feats: [B, C, ph, pw]
        # patched_centers: [B, 2, ph, pw]
        L = patched_h * patched_w
        # flatten
        patched_feats = patched_feats.flatten(2).permute(0, 2, 1)  # [B, L, C]
        patched_feats = self.feat_layer_norm(patched_feats)
        patched_centers = patched_centers.flatten(2).permute(0, 2, 1)  # [B, L, 2]
        # Convert features to gaussian parameters.
        gaussians = self.gaussian_linear(patched_feats)  # [B, L, 8]
        return gaussians, patched_centers, cls_token


class GaussianHeadConv(nn.Module):
    encoder: ConvNeXtUnet
    patch_embedding: PatchEmbedding

    def __init__(
        self,
        encoder: Dict[str, Any],
        patch_embedding: Dict[str, Any],
    ):
        super(GaussianHeadConv, self).__init__()
        self.encoder = ConvNeXtUnet(**encoder)
        self.patch_embedding = PatchEmbedding(**patch_embedding)
        self.patch_size = patch_embedding["patch_size"]
        self.feat_dim = patch_embedding["out_dim"]

        self.feat_layer_norm = nn.LayerNorm(self.feat_dim)
        self.gaussian_linear = nn.Linear(self.feat_dim, 8)

    def init_weights(self):
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()

    def get_feat(self, x):
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.get_feat(x)

        # Patchify the final feature map.
        patched_feats, patched_centers, (patched_h, patched_w) = self.patch_embedding(feat)
        # patched_feats: [B, C, ph, pw]
        # patched_centers: [B, 2, ph, pw]
        L = patched_h * patched_w
        # flatten
        patched_feats = patched_feats.flatten(2).permute(0, 2, 1)  # [B, L, C]
        patched_feats = self.feat_layer_norm(patched_feats)
        patched_centers = patched_centers.flatten(2).permute(0, 2, 1)  # [B, L, 2]
        # Convert features to gaussian parameters.
        gaussians = self.gaussian_linear(patched_feats)  # [B, L, 8]
        return gaussians, patched_centers