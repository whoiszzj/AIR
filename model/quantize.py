from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch
from torch import nn
import torch.nn.functional as F
import constriction
import numpy as np
import math
from .utils import pixel_to_xy
from typing import Union, Sequence
from .compress import compress_matrix_flatten_categorical, decompress_matrix_flatten_categorical


class UniformQuantizer(nn.Module):
    def __init__(
        self,
        signed: bool = False,
        bits: Union[int, Sequence[int]] = 8,
        num_channels: int = 1,
        entropy_type: str = "ans",
        loss_type: str = "l1",
        weight: float = 0.001,
        eps: float = 1e-8,
        ste: bool = True,
        raw_scale_init: list[float] = None,
        beta_init: list[float] = None,
        offset_net_in_channels: int = 768,
    ):
        super().__init__()

        assert entropy_type in ["none", "ans"]
        assert loss_type in ["none", "l1", "mse"]

        self.signed = bool(signed)
        self.num_channels = int(num_channels)

        # ---- bits: int or per-channel list ----
        if isinstance(bits, int):
            bits_list = [int(bits)] * self.num_channels
        else:
            bits_list = [int(b) for b in bits]
            assert len(bits_list) == self.num_channels, (
                f"len(bits) must equal num_channels ({self.num_channels}), "
                f"got {len(bits_list)}"
            )

        for b in bits_list:
            assert b > 0, f"bits must be positive, got {b}"

        # store per-channel bits as buffer for device moves / state_dict
        self.register_buffer("bits_per_channel", torch.tensor(bits_list, dtype=torch.int32))

        # ---- per-channel qmin/qmax ----
        if self.signed:
            qmin = -(2 ** (self.bits_per_channel - 1))
            qmax = (2 ** (self.bits_per_channel - 1)) - 1
        else:
            qmin = torch.zeros_like(self.bits_per_channel)
            qmax = (2 ** self.bits_per_channel) - 1

        self.register_buffer("qmin_per_channel", qmin.to(torch.float32))
        self.register_buffer("qmax_per_channel", qmax.to(torch.float32))

        self.entropy_type = entropy_type
        self.loss_type = loss_type

        # Use raw_scale param and map to positive via softplus
        # self.raw_scale = nn.Parameter(torch.zeros(self.num_channels))  # softplus(0) ~ 0.693
        # self.beta = nn.Parameter(torch.zeros(self.num_channels))
        if raw_scale_init is None:
            self.raw_scale = torch.zeros(self.num_channels)
        else:
            self.raw_scale = torch.tensor(raw_scale_init, dtype=torch.float32)
        if beta_init is None:
            self.beta = torch.zeros(self.num_channels)
        else:
            self.beta = torch.tensor(beta_init, dtype=torch.float32)

        # Convert to shared parameters. These values were learnable at first,
        # then frozen after stabilization so only the offset strategy is trained.
        # self.raw_scale = nn.Parameter(self.raw_scale)
        # self.beta = nn.Parameter(self.beta)

        self.offset_net = nn.Linear(offset_net_in_channels, self.num_channels * 2)  # cls_token -> offset

        self.weight = float(weight)
        self.eps = float(eps)
        self.ste = bool(ste)

    def _effective_entropy_type(self) -> str:
        return "ans" if (not self.training) else self.entropy_type

    def _get_scale(self):
        return F.softplus(self.raw_scale) + self.eps

    def _as_last_channel(self, x, channel_dim: int):
        # returns x_last (channel at last dim), and an inverse function to restore
        if channel_dim == -1:
            return x, (lambda y: y)
        x_t = x.transpose(channel_dim, -1)
        return x_t, (lambda y: y.transpose(channel_dim, -1))

    @torch.no_grad()
    def _init_data(self, tensor, channel_dim: int = -1):
        """
        Initialize beta (min) and raw_scale (step) per channel.
        Works for per-channel bits.
        """
        x, inv = self._as_last_channel(tensor, channel_dim)
        x2 = x.reshape(-1, x.shape[-1])  # [N, C]
        assert x2.shape[-1] == self.num_channels, (
            f"tensor channel size {x2.shape[-1]} != num_channels {self.num_channels}"
        )

        t_min = x2.min(dim=0).values
        t_max = x2.max(dim=0).values

        # per-channel levels
        levels = (self.qmax_per_channel - self.qmin_per_channel).to(tensor.device, tensor.dtype)
        levels = torch.clamp(levels, min=1.0)  # safety

        step = (t_max - t_min) / levels
        step = torch.clamp(step, min=self.eps)

        self.beta.data = t_min.to(self.beta.device, self.beta.dtype)

        # inverse softplus: raw = log(expm1(step))
        # clamp to avoid log(0) if step is extremely small
        step_fp = step.to(self.raw_scale.device, self.raw_scale.dtype)
        self.raw_scale.data = torch.log(torch.expm1(step_fp).clamp_min(self.eps))

    def forward(self, x, cls_token, channel_dim: int = -1):
        """
        channel_dim: where the channel dimension is in x.
        Default assumes last dim is channel.
        """
        x_last, inv = self._as_last_channel(x, channel_dim)
        assert x_last.shape[-1] == self.num_channels, (
            f"x channel size {x_last.shape[-1]} != num_channels {self.num_channels}"
        )

        # scale = self._get_scale().to(x_last.device) # [C]
        # beta = self.beta.to(x_last.device)            # [C]

        offset = self.offset_net(cls_token)  # cls_token: [1, C] -> offset: [1, channel_num * 2]
        offset = offset.reshape(1, self.num_channels, 2)
        offset = 0.5 * torch.tanh(offset)
        scale = F.softplus(self.raw_scale.to(offset.device) + offset[0, :, 0]) + self.eps
        beta = self.beta.to(offset.device) + offset[0, :, 1]

        # reshape for broadcast: [..., C]
        qmin = self.qmin_per_channel.to(x_last.device, x_last.dtype)
        qmax = self.qmax_per_channel.to(x_last.device, x_last.dtype)

        x_q = (x_last - beta) / scale
        x_q = torch.max(torch.min(x_q, qmax), qmin)  # per-channel clamp
        x_q_round = torch.round(x_q)

        if self.ste and self.training:
            x_q = x_q + (x_q_round - x_q).detach()
        else:
            x_q = x_q_round

        x_hat_last = x_q * scale + beta
        x_hat = inv(x_hat_last)

        loss = self._distortion_loss(x_hat, x)

        bits = 0
        if not self.training:
            bits = self.size(x_q_round, channel_dim=-1)  # x_q_round is already last-channel

        return x_hat, loss * self.weight, bits

    def _distortion_loss(self, x_hat, x):
        if self.loss_type == "l1":
            return F.l1_loss(x_hat, x)
        elif self.loss_type == "mse":
            return F.mse_loss(x_hat, x)
        return 0.0

    def size(self, quant, channel_dim: int = -1):
        """
        quant: quantized integer-like tensor (can be float but near integers).
        For et="none": exact raw bit count using per-channel bit widths.
        For et="ans": keep your categorical compressor (unchanged).
        """
        et = self._effective_entropy_type()
        if et == "none":
            q_last, _ = self._as_last_channel(quant, channel_dim)
            assert q_last.shape[-1] == self.num_channels
            n_per_channel = q_last.numel() // self.num_channels
            total_bits_per_vector = int(self.bits_per_channel.sum().item())
            return int(n_per_channel * total_bits_per_vector)

        # ANS path (your existing implementation)
        quant_np = quant.to(torch.int32).cpu().detach().numpy()
        summary, table = compress_matrix_flatten_categorical(quant_np, return_table=True)
        return int(summary["compressed_bits"])


def rgb_to_ycrcb(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb: [N, 3] float tensor (any range is ok; linear transform)
    returns: [N, 3] with channels [Y, Cr, Cb] (BT.601 full-range, no offsets)
    """
    assert rgb.ndim == 2 and rgb.shape[1] == 3, f"Expected [N,3], got {rgb.shape}"
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b

    return torch.stack([y, cr, cb], dim=1)


def ycrcb_to_rgb(ycrcb: torch.Tensor) -> torch.Tensor:
    """
    ycrcb: [N, 3] float tensor with channels [Y, Cr, Cb]
    returns: [N, 3] rgb (BT.601 full-range, no offsets)
    """
    assert ycrcb.ndim == 2 and ycrcb.shape[1] == 3, f"Expected [N,3], got {ycrcb.shape}"
    y, cr, cb = ycrcb[:, 0], ycrcb[:, 1], ycrcb[:, 2]

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    return torch.stack([r, g, b], dim=1)


class FeedForwardGaussianCodec(nn.Module):
    def __init__(
        self,
        patch_size,
        bits_xy=8,
        bits_scale=7,  # 7 bit * 2
        bits_rot=7,  # 7 bit
    ):
        super().__init__()
        self.patch_size = float(patch_size)

        # raw_scale_init and beta_init are base statistics estimated from many
        # samples. Training an offset on top of them improves quantization.
        self.xy_quantizer = UniformQuantizer(
            signed=False,
            bits=bits_xy,
            num_channels=2,
            weight=0.01,
            raw_scale_init=[-3.0427, -3.0022],
            beta_init=[-6.7962, -6.0476],
        )
        self.scaling_quantizer = UniformQuantizer(
            signed=False,
            bits=bits_scale,
            num_channels=2,
            weight=0.01,
            raw_scale_init=[-2.9553, -3.2385],
            beta_init=[0.4933, 0.5032],
        )
        self.rotation_quantizer = UniformQuantizer(
            signed=False,
            bits=bits_rot,
            num_channels=1,
            weight=0.01,
            raw_scale_init=[-3.3192],
            beta_init=[1.0155],
        )
        self.color_quantizer = UniformQuantizer(
            signed=False,
            bits=[8, 4, 4],
            num_channels=3,
            weight=0.01,
            raw_scale_init=[-5.3675, -4.0653, -4.4090],
            beta_init=[-0.2550, -0.0688, -0.1273],
        )

        self._inited = False

    @torch.no_grad()
    def maybe_init_quant(self, xy, scaling, rotation, color):
        if not self._inited:
            self.xy_quantizer._init_data(xy)
            self.scaling_quantizer._init_data(scaling)
            self.rotation_quantizer._init_data(rotation)
            self.color_quantizer._init_data(color)
            self._inited = True

    def forward(self, raw_u, cls_token, patch_center, H, W):
        # ---- activation ----
        # Quantize the xy offset directly.
        xy_offset = torch.tanh(raw_u[:, 0:2]) * self.patch_size
        scaling = torch.sigmoid(raw_u[:, 2:4]) * self.patch_size + 0.5
        rotation = torch.sigmoid(raw_u[:, 4:5]) * (2.0 * math.pi)
        color = torch.tanh(raw_u[:, 5:8])
        color = rgb_to_ycrcb(color)

        # if self.training:
        #     self.maybe_init_quant(xy_offset.detach(),scaling.detach(), rotation.detach(), color.detach())

        xy_offset_hat, l_p, p_bit = self.xy_quantizer(xy_offset, cls_token)
        scaling_hat, l_s, s_bit = self.scaling_quantizer(scaling, cls_token)
        rotation_hat, l_r, r_bit = self.rotation_quantizer(rotation, cls_token)
        color_hat, l_c, c_bit = self.color_quantizer(color, cls_token)
        color_hat = ycrcb_to_rgb(color_hat)

        xy_hat = xy_offset_hat + patch_center
        xy_hat = pixel_to_xy(xy_hat, H, W)

        xy = xy_offset + patch_center
        xy = pixel_to_xy(xy, H, W)

        vq_loss = l_p + l_s + l_r + l_c
        return {
            "xy": xy_hat,  # Pixel coordinates converted to [-1, 1].
            "scaling": scaling_hat,
            "rotation": rotation_hat,
            "color": color_hat,
            "vq_loss": vq_loss,
            "unit_bit": [p_bit, s_bit, r_bit, c_bit],
        }
