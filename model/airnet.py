from typing import *
from pathlib import Path
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pixel_to_xy, render, render_pseudo, mse_from_psnr
from .gaussian_head import GaussianHeadViT
from .quantize import FeedForwardGaussianCodec


class AIRNet(nn.Module):
    def __init__(
        self,
        head_num: int,
        gaussian_head_config: List[Dict[str, Any]],
        quantize: bool = False,
        pod: bool = False,
        init_pretrained: bool = True,
    ):
        super(AIRNet, self).__init__()

        self.head_num = head_num
        self.gaussian_heads = nn.ModuleList([GaussianHeadViT(**gaussian_head_config) for _ in range(head_num)])
        self.quantize = bool(quantize)
        self.pod = pod

        patch_embedding_config = gaussian_head_config["patch_embedding"]

        if init_pretrained:
            self.init_weights()
        self.patch_size = patch_embedding_config["patch_size"]
        self.feat_dim = patch_embedding_config["out_dim"]
        self.mse_threshold = mse_from_psnr(35.0)
        self.ssim_threshold = 1 - 0.95
        # print("MSE threshold is set to", self.mse_threshold)

        if self.quantize:
            self.feed_forward_gaussian_codec = FeedForwardGaussianCodec(patch_size=self.patch_size)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, IO[bytes]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        **hf_kwargs,
    ) -> "AIRNet":
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `compiled`
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint_path = pretrained_model_name_or_path
        else:
            raise ValueError(f"Invalid checkpoint path: {pretrained_model_name_or_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        model_config = checkpoint["model_config"]
        if model_kwargs is not None:
            model_config.update(model_kwargs)

        init_config = {
            "head_num": model_config["head_num"],
            "gaussian_head_config": model_config.get("gaussian_head_config"),
            "quantize": bool(model_config.get("quantize", False)),
            "init_pretrained": False,
        }

        model = cls(**init_config)
        model.load_state_dict(checkpoint["model"], strict=False)

        return model

    def init_weights(self):
        for gaussian_head in self.gaussian_heads:
            gaussian_head.init_weights()

    def enable_gradient_checkpointing(self):
        for gaussian_head in self.gaussian_heads:
            gaussian_head.enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        for gaussian_head in self.gaussian_heads:
            gaussian_head.enable_pytorch_native_sdpa()

    # Compute patch-level MSE error map between rendered image and ground truth.
    def _mse_error(self, gt_image: torch.Tensor, render_image: torch.Tensor):
        error_map = F.mse_loss(render_image, gt_image, reduction="none")
        B, _, H, W = error_map.shape
        pad_h = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size - H
        pad_w = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size - W
        error_map_padded = error_map
        if pad_h > 0 or pad_w > 0:
            error_map_padded = F.pad(error_map, (0, pad_w, 0, pad_h), mode="replicate")
        error_map_padded = F.avg_pool2d(
            error_map_padded,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # [B, 3, ph, pw]
        error_map_padded = error_map_padded.mean(dim=1)  # [B, ph, pw]
        return error_map_padded

    def _ssim_error(self, gt_image: torch.Tensor, render_image: torch.Tensor, eps: float = 1e-8):
        """
        Compute patch-level (1 - SSIM) and return a loss map shaped [B, ph, pw].

        Assumes gt_image and render_image are in [0, self.max_val].
        """
        assert gt_image.shape == render_image.shape, "gt and render must have same shape"
        B, C, H, W = gt_image.shape

        # 1. Pad to a multiple of patch_size.
        pad_h = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size - H
        pad_w = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size - W

        if pad_h > 0 or pad_w > 0:
            gt_image = F.pad(gt_image, (0, pad_w, 0, pad_h), mode="replicate")
            render_image = F.pad(render_image, (0, pad_w, 0, pad_h), mode="replicate")

        _, _, Hp, Wp = gt_image.shape
        ph = Hp // self.patch_size
        pw = Wp // self.patch_size

        # 2. Flatten each non-overlapping patch with unfold.
        #    unfold output: [B, C * (patch_size * patch_size), ph * pw]
        patch_area = self.patch_size * self.patch_size
        gt_patches = F.unfold(
            gt_image,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # [B, C*P, N]
        rd_patches = F.unfold(
            render_image,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # [B, C*P, N]

        B, CP, N = gt_patches.shape
        assert CP == C * patch_area

        # 3. Reshape to [B, N, C, P], where P is the number of pixels per patch.
        gt_patches = gt_patches.view(B, C, patch_area, N).permute(0, 3, 1, 2)  # [B, N, C, P]
        rd_patches = rd_patches.view(B, C, patch_area, N).permute(0, 3, 1, 2)  # [B, N, C, P]

        # 4. Compute SSIM statistics over P inside each patch.
        #    mu_x, mu_y, sigma_x^2, sigma_y^2, sigma_xy
        mu_x = gt_patches.mean(dim=3, keepdim=True)  # [B, N, C, 1]
        mu_y = rd_patches.mean(dim=3, keepdim=True)  # [B, N, C, 1]

        sigma_x = (gt_patches - mu_x).pow(2).mean(dim=3, keepdim=True)  # [B, N, C, 1]
        sigma_y = (rd_patches - mu_y).pow(2).mean(dim=3, keepdim=True)  # [B, N, C, 1]
        sigma_xy = ((gt_patches - mu_x) * (rd_patches - mu_y)).mean(
            dim=3,
            keepdim=True,
        )  # [B, N, C, 1]

        # 5. SSIM constants.
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2) + eps

        ssim_patch = num / den  # [B,N,C,1]

        # 6. Average over channels to get one scalar per patch.
        ssim_patch = ssim_patch.mean(dim=2).squeeze(-1)  # [B,N]

        # 7. Reshape back to [B, ph, pw].
        ssim_map = ssim_patch.view(B, ph, pw)  # SSIM for each patch.

        # 8. Use 1 - SSIM as the loss map.
        ssim_loss_map = 1.0 - ssim_map  # [B, ph, pw]

        return ssim_loss_map


    # Quantize gaussians and render them into images.
    def _quant_gaussian_to_image(
        self,
        gaussians: List[torch.Tensor],
        cls_tokens: torch.Tensor,
        patch_centers: List[torch.Tensor],
        H: int,
        W: int,
    ) -> Dict[str, Any]:
        forward_images: List[torch.Tensor] = []
        forward_alphas: List[torch.Tensor] = []
        xys: List[torch.Tensor] = []
        vq_losses: List[torch.Tensor] = []
        unit_bits: List[Any] = []

        for i in range(len(gaussians)):
            quant_results = self.feed_forward_gaussian_codec(
                gaussians[i],
                cls_tokens[i : i + 1],
                patch_centers[i],
                H,
                W,
            )
            xy = quant_results["xy"]
            scale = quant_results["scaling"]
            rotation = quant_results["rotation"]
            color = quant_results["color"]

            render_results = render(xy, scale, rotation, color, H, W)
            forward_images.append(render_results["image"])
            forward_alphas.append(render_results["alpha"])

            xys.append(xy)
            vq_losses.append(quant_results["vq_loss"])
            unit_bits.append(quant_results.get("unit_bit", None))

        forward_images_t = torch.stack(forward_images, dim=0)
        forward_alphas_t = torch.stack(forward_alphas, dim=0)
        vq_losses_t = torch.stack(vq_losses, dim=0)

        if len(unit_bits) > 0 and unit_bits[0] is not None:
            unit_bits_t = torch.tensor(unit_bits, device=forward_images_t.device, dtype=torch.long)
        else:
            unit_bits_t = None

        return {
            "image": forward_images_t,
            "alpha": forward_alphas_t,
            "xys": xys,
            "vq_loss": vq_losses_t,
            "unit_bit": unit_bits_t,
        }

    # Render raw/pseudo/quantized gaussians into images and losses.
    def _gaussian_to_image(
        self,
        gaussians: List[torch.Tensor],
        patch_centers: List[torch.Tensor],
        H,
        W,
        gt_image: List[torch.Tensor] = None,
        render_mode: str = "image",
        masks: List[torch.Tensor] = None,
        step: int = 10,
    ):
        forward_images = []
        forward_alphas = []
        last_images = []
        pseudo_losses = []
        xys = []

        for i in range(len(gaussians)):
            cur_gaussians = gaussians[i]
            cur_patch_centers = patch_centers[i]

            if render_mode == "pod":
                assert gt_image is not None
                # Generate pseudo labels for training.
                pseudo_results = render_pseudo(
                    cur_gaussians,
                    cur_patch_centers,
                    gt_image[i],
                    H,
                    W,
                    self.patch_size,
                    masks[i] if masks is not None else None,
                    step,
                )

                if masks is None:
                    target = torch.sigmoid(pseudo_results["pseudo_gaussian"])
                    pred = torch.sigmoid(cur_gaussians)
                    pseudo_loss = F.smooth_l1_loss(pred, target, reduction="none")
                else:
                    target = torch.sigmoid(pseudo_results["pseudo_gaussian"])
                    pred = torch.sigmoid(cur_gaussians)
                    diff = F.smooth_l1_loss(pred, target, reduction="none")

                    m = masks[i].to(diff.dtype)
                    if m.dim() < diff.dim():
                        m = m.unsqueeze(-1)
                    denom = m.sum() * diff.shape[-1]
                    if denom.item() > 0:
                        pseudo_loss = (diff * m).sum() / denom
                    else:
                        pseudo_loss = diff.mean()

                forward_images.append(pseudo_results["first_img"])
                forward_alphas.append(pseudo_results["first_alpha"])
                last_images.append(pseudo_results["last_image"])
                pseudo_losses.append(pseudo_loss)
            elif render_mode == "image":
                xy = torch.tanh(cur_gaussians[:, 0:2]) * self.patch_size + cur_patch_centers
                xy = pixel_to_xy(xy, H, W)
                scale = torch.sigmoid(cur_gaussians[:, 2:4]) * self.patch_size + 0.5
                rotation = torch.sigmoid(cur_gaussians[:, 4:5]) * 2 * torch.pi
                color = torch.tanh(cur_gaussians[:, 5:8])
                render_results = render(xy, scale, rotation, color, H, W)
                forward_images.append(render_results["image"])
                forward_alphas.append(render_results["alpha"])
                xys.append(xy)
            else:
                raise ValueError(f"Invalid render mode: {render_mode}")

        forward_images = torch.stack(forward_images, dim=0)
        forward_alphas = torch.stack(forward_alphas, dim=0)
        if render_mode == "pod":
            last_images = torch.stack(last_images, dim=0)
            pseudo_losses = torch.stack(pseudo_losses, dim=0)
        else:
            last_images = None
            pseudo_losses = None

        return {
            "image": forward_images,  # tensor
            "alpha": forward_alphas,  # tensor
            "last_image": last_images,  # none or tensor
            "pseudo_loss": pseudo_losses,  # none or tensor
            "xys": xys,  # list used only for visualization stage
        }

    # Select by boolean mask; returns per-batch lists for each tensor [B, L, ...]
    def _gaussian_select(self, mask: torch.Tensor, *tensors: torch.Tensor):
        """Select multiple tensors along L using a boolean mask and return lists per batch.

        Args:
            mask: [B, L] boolean mask
            tensors: each tensor is [B, L, ...]
        Returns:
            tuple of lists; each list has length B and elements with shape [L_b, ...]
        """
        if len(tensors) == 0:
            raise ValueError("_gaussian_select requires at least one tensor to select")

        first = tensors[0]
        B, L = first.shape[:2]

        results: List[List[torch.Tensor]] = []
        for t in tensors:
            assert t.shape[:2] == (B, L) and t.dim() >= 3, "All tensors must be [B, L, ...]"
            per_batch: List[torch.Tensor] = []
            for b in range(B):
                mb = mask[b]
                if mb.any():
                    per_batch.append(t[b][mb])
                else:
                    # Fallback to keep all tokens if none selected
                    per_batch.append(t[b])
            results.append(per_batch)
        return tuple(results) if len(results) > 1 else results[0]

    def forward(self, gt_image: torch.Tensor, stage: int) -> Dict[str, torch.Tensor]:
        B = gt_image.shape[0]
        ori_H, ori_W = gt_image.shape[-2], gt_image.shape[-1]
        render_img = torch.zeros_like(gt_image)

        gaussians_all = None
        centers_all = None

        stage_results = []
        gaussian_nums = [0] * B
        gaussian_nums_per_stage = []
        time_cost_per_stage = []
        L = 1

        for s in range(0, stage + 1):
            start_time = time.time()
            # Use the error maps to decide where new gaussians should be added.
            mse_error = self._mse_error(gt_image, render_img.detach())  # [B, 3, H, W]
            ssim_error = self._ssim_error(gt_image, render_img.detach())  # [B, pH, pW]
            mask = torch.logical_or(
                mse_error > self.mse_threshold,
                ssim_error > self.ssim_threshold,
            )  # [B, pH, pW]
            # mask = torch.ones_like(ssim_error, dtype=torch.bool) # ablation 1: all activated
            # mask = mse_error > self.mse_threshold  # ablation 2: only mse activated
            # mask = ssim_error > self.ssim_threshold  # ablation 3: only ssim activated
            mask = mask.flatten(1)  # [B, L]

            # Use the last layer cls token to predict quantization parameters.
            gaussians, patched_centers, cls_token = self.gaussian_heads[s](gt_image - render_img)
            L = gaussians.shape[1]  # Number of candidate gaussians.

            # Select new gaussians by mask:
            # list([L', 8]), list([L', 2]); len(list) = B.
            selected_gaussians, selected_centers = self._gaussian_select(mask, gaussians, patched_centers)

            gradient_masks = []
            # Initialize or append accumulated gaussians and centers.
            if gaussians_all is None:
                gaussians_all = selected_gaussians
                centers_all = selected_centers
                for i in range(B):
                    if self.pod and self.training:
                        gradient_masks.append(torch.ones_like(gaussians_all[i][:, 0], dtype=torch.bool))
                    gaussian_nums[i] = gaussians_all[i].shape[0]
                gaussian_nums_per_stage.append(np.mean(gaussian_nums).item())
            else:
                for i in range(len(selected_gaussians)):
                    ori_len = gaussians_all[i].shape[0]
                    gaussians_all[i] = torch.cat([gaussians_all[i], selected_gaussians[i]], dim=0)
                    centers_all[i] = torch.cat([centers_all[i], selected_centers[i]], dim=0)
                    if self.pod and self.training:
                        gradient_mask = torch.zeros_like(gaussians_all[i][:, 0], dtype=torch.bool)
                        gradient_mask[ori_len:] = True
                        gradient_masks.append(gradient_mask)
                    gaussian_nums[i] = gaussians_all[i].shape[0]
                gaussian_nums_per_stage.append(np.mean(gaussian_nums).item())

            if self.pod and self.training:
                render_results = self._gaussian_to_image(
                    gaussians_all,
                    centers_all,
                    ori_H,
                    ori_W,
                    gt_image,
                    render_mode="pod",
                    masks=gradient_masks,
                    step=20 * (s + 1),
                )
            else:
                render_results = self._gaussian_to_image(
                    gaussians_all,
                    centers_all,
                    ori_H,
                    ori_W,
                    gt_image,
                    render_mode="image",
                )

            render_results["mse_error"] = mse_error
            render_results["ssim_error"] = ssim_error
            render_results["gaussians"] = selected_gaussians
            stage_results.append(render_results)
            render_img = render_results["image"]
            end_time = time.time()
            time_cost_per_stage.append(end_time - start_time)

        if self.quantize:
            quant_render_results = self._quant_gaussian_to_image(
                gaussians_all,
                cls_token,
                centers_all,
                ori_H,
                ori_W,
            )
            quant_image = quant_render_results["image"]
            quant_loss = quant_render_results["vq_loss"]
            quant_unit_bit = quant_render_results.get("unit_bit", None)
            if quant_unit_bit is not None:
                quant_unit_bit += L * 3
        else:
            quant_image = None
            quant_loss = None
            quant_unit_bit = None

        gaussian_num = torch.tensor(
            [gaussian_nums[b] for b in range(B)],
            dtype=torch.float32,
            device=gt_image.device,
        )
        router_usage = torch.tensor(
            [gaussian_nums[b] / (L * (stage + 1)) for b in range(B)],
            dtype=torch.float32,
            device=gt_image.device,
        )
        if quant_unit_bit is not None:
            quant_total_bits = quant_unit_bit.to(torch.float32).sum(dim=1)
            quant_bpp = quant_total_bits / float(ori_H * ori_W)
        else:
            quant_total_bits = None
            quant_bpp = None
        
        return {
            "stage_results": stage_results,
            "gaussian_num": gaussian_num,
            "gaussian_nums_per_stage": gaussian_nums_per_stage, # list[B][stage]
            "time_cost_per_stage": time_cost_per_stage, # list[stage]
            "router_usage": router_usage,
            "xys": stage_results[-1]["xys"],
            "quant_image": quant_image,
            "quant_loss": quant_loss,
            "quant_unit_bit": quant_unit_bit,
            "quant_total_bits": quant_total_bits,
            "quant_bpp": quant_bpp,
        }