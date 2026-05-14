from typing import *
from numbers import Number
import importlib
import itertools
import functools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3.models.vision_transformer import DinoVisionTransformer as DINOv3VisionTransformer
from .convnext_unet.encoder import convnext_large, convnext_base
from .convnext_unet.decoder import UnetDecoder
from .utils import wrap_dinov3_attention_with_sdpa, wrap_module_with_gradient_checkpointing
class ResidualConvBlock(nn.Module):  
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = None, 
        hidden_channels: int = None, 
        kernel_size: int = 3, 
        padding_mode: str = 'replicate', 
        activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu', 
        in_norm: Literal['group_norm', 'layer_norm', 'instance_norm', 'none'] = 'layer_norm',
        hidden_norm: Literal['group_norm', 'layer_norm', 'instance_norm'] = 'group_norm',
    ):  
        super(ResidualConvBlock, self).__init__()  
        if out_channels is None:  
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation =='relu':
            activation_cls = nn.ReLU
        elif activation == 'leaky_relu':
            activation_cls = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        elif activation =='silu':
            activation_cls = nn.SiLU
        elif activation == 'elu':
            activation_cls = nn.ELU
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.GroupNorm(in_channels // 32, in_channels) if in_norm == 'group_norm' else \
                nn.GroupNorm(1, in_channels) if in_norm == 'layer_norm' else \
                nn.InstanceNorm2d(in_channels) if in_norm == 'instance_norm' else \
                nn.Identity(),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32, hidden_channels) if hidden_norm == 'group_norm' else \
                nn.GroupNorm(1, hidden_channels) if hidden_norm == 'layer_norm' else \
                nn.InstanceNorm2d(hidden_channels) if hidden_norm == 'instance_norm' else\
                nn.Identity(),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode)
        )
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()  
  
    def forward(self, x):  
        skip = self.skip_connection(x)  
        x = self.layers(x)
        x = x + skip
        return x  


class DINOv3Encoder(nn.Module):
    "Wrapped DINOv3 encoder supporting gradient checkpointing. Input is RGB image in range [0, 1]."
    backbone: DINOv3VisionTransformer
    image_mean: torch.Tensor
    image_std: torch.Tensor
    dim_features: int

    def __init__(self, backbone: str, intermediate_layers: Union[int, List[int]], dim_out: int, **deprecated_kwargs):
        super(DINOv3Encoder, self).__init__()

        self.intermediate_layers = intermediate_layers

        # Load the backbone
        self.hub_loader = getattr(importlib.import_module(".dinov3.hub.backbones", __package__), backbone)
        self.backbone_name = backbone
        self.backbone = self.hub_loader(pretrained=False)

        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features = intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers)

        self.output_projections = nn.ModuleList([
            nn.Conv2d(in_channels=self.dim_features, out_channels=dim_out, kernel_size=1, stride=1, padding=0,) 
                for _ in range(self.num_features)
        ])

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.backbone.onnx_compatible_mode = value

    # Extract a usable state_dict from a checkpoint object that may wrap weights in nested dicts.
    @staticmethod
    def _unwrap_checkpoint_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint_obj, dict):
            for key in ("state_dict", "model", "backbone", "student", "teacher", "net"):
                if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                    checkpoint_obj = checkpoint_obj[key]
                    break
        if not isinstance(checkpoint_obj, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_obj)}")
        return checkpoint_obj

    # Normalize checkpoint keys to better match the backbone module by stripping common wrappers/prefixes.
    def _normalize_backbone_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        def _strip_known_prefixes(k: str) -> str:
            # Iteratively strip common wrappers (e.g., DDP 'module.') and model container prefixes.
            prefixes = ("module.", "backbone.", "model.", "encoder.", "student.", "teacher.", "net.")
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if k.startswith(p):
                        k = k[len(p):]
                        changed = True
            return k

        return {_strip_known_prefixes(k): v for k, v in state_dict.items()}

    # Initialize backbone weights from the local DINOv3 checkpoint shipped in this repository.
    def init_weights(self):
        ckpt_path = Path(__file__).resolve().parents[1] / "checkpoints" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        checkpoint_obj = torch.load(ckpt_path, map_location="cpu")
        pretrained_state_dict = self._unwrap_checkpoint_state_dict(checkpoint_obj)
        pretrained_state_dict = self._normalize_backbone_state_dict_keys(pretrained_state_dict)

        missing_keys, unexpected_keys = self.backbone.load_state_dict(pretrained_state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print(
                f"[DINOv3Encoder] Loaded backbone from {ckpt_path}. "
                f"Missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}"
            )

    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            wrap_dinov3_attention_with_sdpa(self.backbone.blocks[i].attn)

    def forward(self, image: torch.Tensor, token_rows: Union[int, torch.LongTensor], token_cols: Union[int, torch.LongTensor], return_class_token: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        image_16 = F.interpolate(image, (token_rows * 16, token_cols * 16), mode="bilinear", align_corners=False, antialias=not self.onnx_compatible_mode)
        image_16 = (image_16 - self.image_mean) / self.image_std

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers(image_16, n=self.intermediate_layers, return_class_token=True)
    
        # Project features to the desired dimensionality
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous())
                for proj, (feat, clstoken) in zip(self.output_projections, features)
        ], dim=1).sum(dim=1)                    

        if return_class_token:
            return x, features[-1][1]
        else:
            return x


class ConvNeXtUnet(nn.Module):
    def __init__(
            self, backbone, dim_out,
            pretrained=True, in_channels=3, bilinear=False, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = dim_out
        self.bilinear = bilinear
        
        ENCODER_DIMS = {
            'convnext_base': [128, 256, 512, 1024],
            'convnext_large': [192, 384, 768, 1536],
        }

        self.dims = ENCODER_DIMS[backbone]

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.dims[0] // 2, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.dims[0] // 2),
            nn.ReLU(inplace=True),
        )

        if backbone == 'convnext_base':
            encoder_model = convnext_base
        elif backbone == 'convnext_large':
            encoder_model = convnext_large
        else:
            raise ValueError(f'Unsupported encoder: {backbone}')

        self.convnext_encoder = encoder_model(pretrained, **kwargs)

        self.unet_decoder = UnetDecoder(
            out_channels=self.out_channels, dims=self.dims,
            in_channels=self.dims[0] // 2, bilinear=self.bilinear
        )

    def init_weights(self):
        pass
    
    def enable_gradient_checkpointing(self):
        self.convnext_encoder.enable_gradient_checkpointing()
        self.unet_decoder.enable_gradient_checkpointing()
        wrap_module_with_gradient_checkpointing(self.in_conv)

    def forward(self, x):
        x = self.in_conv(x)
        x, features = self.convnext_encoder(x)
        x = self.unet_decoder(x, features)
        return x



class Resampler(nn.Sequential):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        type_: Literal['pixel_shuffle', 'nearest', 'bilinear', 'conv_transpose', 'pixel_unshuffle', 'avg_pool', 'max_pool'],
        scale_factor: int = 2, 
    ):
        if type_ == 'pixel_shuffle':
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
            for i in range(1, scale_factor ** 2):
                self[0].weight.data[i::scale_factor ** 2] = self[0].weight.data[0::scale_factor ** 2]
                self[0].bias.data[i::scale_factor ** 2] = self[0].bias.data[0::scale_factor ** 2]
        elif type_ in ['nearest', 'bilinear']:
            nn.Sequential.__init__(self,
                nn.Upsample(scale_factor=scale_factor, mode=type_, align_corners=False if type_ == 'bilinear' else None),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
        elif type_ == 'conv_transpose':
            nn.Sequential.__init__(self,
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
            self[0].weight.data[:] = self[0].weight.data[:, :, :1, :1]
        elif type_ == 'pixel_unshuffle':
            nn.Sequential.__init__(self,
                nn.PixelUnshuffle(scale_factor),
                nn.Conv2d(in_channels * (scale_factor ** 2), out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
        elif type_ == 'avg_pool': 
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        elif type_ == 'max_pool':
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        else:
            raise ValueError(f'Unsupported resampler type: {type_}')

class MLP(nn.Sequential):
    def __init__(self, dims: Sequence[int]):
        nn.Sequential.__init__(self,
            *itertools.chain(*[
                (nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True))
                    for dim_in, dim_out in zip(dims[:-2], dims[1:-1])
            ]),
            nn.Linear(dims[-2], dims[-1]),
        )

class ConvStack(nn.Module):
    def __init__(self, 
        dim_in: List[Optional[int]],
        dim_res_blocks: List[int],
        dim_out: List[Optional[int]],
        resamplers: Union[Literal['pixel_shuffle', 'nearest', 'bilinear', 'conv_transpose', 'pixel_unshuffle', 'avg_pool', 'max_pool'], List],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_in_norm: Literal['layer_norm', 'group_norm' , 'instance_norm', 'none'] = 'layer_norm',
        res_block_hidden_norm: Literal['layer_norm', 'group_norm' , 'instance_norm', 'none'] = 'group_norm',
        activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu',
    ):
        super().__init__()
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(dim_in_, dim_res_block_, kernel_size=1, stride=1, padding=0) if dim_in_ is not None else nn.Identity() 
                for dim_in_, dim_res_block_ in zip(dim_in if isinstance(dim_in, Sequence) else itertools.repeat(dim_in), dim_res_blocks)
        ])
        self.resamplers = nn.ModuleList([
            Resampler(dim_prev, dim_succ, scale_factor=2, type_=resampler) 
                for i, (dim_prev, dim_succ, resampler) in enumerate(zip(
                    dim_res_blocks[:-1], 
                    dim_res_blocks[1:], 
                    resamplers if isinstance(resamplers, Sequence) else itertools.repeat(resamplers)
                ))
        ])
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                *(
                    ResidualConvBlock(
                        dim_res_block_, dim_res_block_, dim_times_res_block_hidden * dim_res_block_, 
                        activation=activation, in_norm=res_block_in_norm, hidden_norm=res_block_hidden_norm
                    ) for _ in range(num_res_blocks[i] if isinstance(num_res_blocks, list) else num_res_blocks)
                )
            ) for i, dim_res_block_ in enumerate(dim_res_blocks)
        ])
        self.output_blocks = nn.ModuleList([
            nn.Conv2d(dim_res_block_, dim_out_, kernel_size=1, stride=1, padding=0) if dim_out_ is not None else nn.Identity() 
                for dim_out_, dim_res_block_ in zip(dim_out if isinstance(dim_out, Sequence) else itertools.repeat(dim_out), dim_res_blocks)
        ])

    def enable_gradient_checkpointing(self):
        for i in range(len(self.resamplers)):
            self.resamplers[i] = wrap_module_with_gradient_checkpointing(self.resamplers[i])
        for i in range(len(self.res_blocks)):
            for j in range(len(self.res_blocks[i])):
                self.res_blocks[i][j] = wrap_module_with_gradient_checkpointing(self.res_blocks[i][j])

    def forward(self, in_features: List[torch.Tensor]):
        out_features = []
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            if i == 0:
                x = feature
            elif feature is not None:
                x = x + feature
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features
    
class PadPatchify(nn.Module):
    def __init__(self, feat_dim_in: int, feat_dim_out: int, patch_size: int, bias: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            feat_dim_in, feat_dim_out,
            kernel_size=patch_size, stride=patch_size,
            bias=bias
        )

    def enable_gradient_checkpointing(self):
        wrap_module_with_gradient_checkpointing(self.conv)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # Dynamically pad on the right/bottom so H/W become multiples of patch_size
        B, C, H, W = feat.shape
        pad_h = ( (H + self.patch_size - 1) // self.patch_size ) * self.patch_size - H
        pad_w = ( (W + self.patch_size - 1) // self.patch_size ) * self.patch_size - W
        if pad_h > 0 or pad_w > 0:
            feat = F.pad(feat, (0, pad_w, 0, pad_h), mode='replicate')
        return self.conv(feat)
    
    def pool_error(self, error_map: torch.Tensor):
        B, _, H, W = error_map.shape
        pad_h = ( (H + self.patch_size - 1) // self.patch_size ) * self.patch_size - H
        pad_w = ( (W + self.patch_size - 1) // self.patch_size ) * self.patch_size - W
        error_map_padded = error_map
        if pad_h > 0 or pad_w > 0:
            error_map_padded = F.pad(error_map_padded, (0, pad_w, 0, pad_h), mode='replicate')
        error_map_padded = F.avg_pool2d(error_map_padded, kernel_size=self.patch_size, stride=self.patch_size)  # [B,3,ph,pw]
        error_map_padded = error_map_padded.mean(dim=1)  # [B,ph,pw]
        return error_map_padded

class PatchEmbedding(nn.Module):
    def __init__(self, 
        in_dim: int,
        out_dim: int,
        patch_size: int
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.patchify = PadPatchify(in_dim, out_dim, patch_size)

    def enable_gradient_checkpointing(self):
        self.patchify.enable_gradient_checkpointing()

    def forward(self, feat: torch.Tensor):
        # Convert feature map into non-overlapping patches; keep map shape [B, C, pH, pW].
        B, C, H, W = feat.shape
        device = feat.device
        dtype = feat.dtype

        patches = self.patchify(feat)  # [B, C, pH, pW]
        B, C, patched_h, patched_w = patches.shape

        # Compute patch center coordinates in original (unpadded) pixel space without flattening
        rows = torch.arange(patched_h, device=device, dtype=dtype)
        cols = torch.arange(patched_w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')  # [pH, pW]
        cy = grid_y * self.patch_size + (self.patch_size - 1) / 2.0
        cx = grid_x * self.patch_size + (self.patch_size - 1) / 2.0
        cy = cy.clamp(max=max(H - 1, 0))  # [pH, pW]
        cx = cx.clamp(max=max(W - 1, 0))  # [pH, pW]
        centers = torch.stack([cx, cy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, pH, pW]

        return patches, centers, (patched_h, patched_w)
        