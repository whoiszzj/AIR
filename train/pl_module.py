import os
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributed as dist
from fused_ssim import fused_ssim

from model.airnet import AIRNet
from train.utils import build_optimizer, build_lr_scheduler, compute_psnr

class AIRNetLightningModule(pl.LightningModule):
    """LightningModule wrapping AIRNet for training with logging and custom checkpoints."""

    def __init__(
        self,
        config: Dict[str, Any],
        workspace: str,
        enable_gradient_checkpointing: bool = True,
        pod: bool = False,
        log_every: int = 10,
        vis_every: int = 200,
        num_vis_images: int = 32,
        seed: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        num_iterations: int = 0,
        batch_size_forward: int = 2
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])  # keep CLI params in ckpt
        self.config = config
        self.workspace = workspace
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.pod = pod
        self.log_every = int(log_every)
        self.vis_every = int(vis_every)
        self.num_vis_images = int(num_vis_images)
        self.seed_value = seed
        self.checkpoint_path = checkpoint_path
        self.batch_size_forward = int(batch_size_forward)
        
        self.stage_time_stones = self.config.get("stage_time_stones", None)
        if self.stage_time_stones is None:
            # Enable all stages during refinement.
            self.max_stage = self.config["model"]["head_num"] - 1
            self.cur_stage = self.max_stage
        else:
            self.max_stage = len(self.stage_time_stones)
            self.cur_stage = 0
        print(f"================================================")
        print(f"Stage information:")
        print(f"Stage time stones: {self.stage_time_stones}")
        print(f"Max stage: {self.max_stage}")
        print(f"Cur stage: {self.cur_stage}")
        print(f"================================================")
        
        # model
        self.model = AIRNet(**self.config["model"], pod=self.pod)  # type: ignore[index]
        
        self.head_num = self.config["model"]["head_num"]
        assert self.head_num == self.max_stage + 1, f"head_num must be equal to max_stage + 1, but got {self.head_num} != {self.max_stage + 1}"
        
        if self.enable_gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        # convert to sync batch norm
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # resume buffers
        self._resume_optimizer_state: Optional[Dict[str, Any]] = None
        self._resume_group_lrs: Optional[List[float]] = None
        self._resume_lr_state: Optional[Dict[str, Any]] = None

        # optional router anneal steps
        self.total_num_steps: int = int(num_iterations) if num_iterations is not None else 0

        # visualization cache
        self._vis_batches: Optional[List[Dict[str, Any]]] = None

        # debug flags
        self._did_dump_batch_info: bool = False

    def setup(self, stage: Optional[str] = None) -> None:
        """Apply legacy checkpoint states when trainer is ready."""
        # apply legacy checkpoint now (weights/optimizer/scheduler) if provided
        if self.checkpoint_path is not None:
            self._load_legacy_checkpoint(self.checkpoint_path)

    def update_stage(self) -> None:
        """Advance stage based on current epoch instead of global step."""
        if self.cur_stage >= self.max_stage:
            return
        current_epoch = int(self.current_epoch)
        threshold_epoch = int(self.stage_time_stones[self.cur_stage])
        if current_epoch >= threshold_epoch:
            self.cur_stage += 1
            if self.trainer.is_global_zero:
                print(f"Stage {self.cur_stage} started (epoch-based)")
            # Initialize this stage with the previous stage parameters.
            self.model.gaussian_heads[self.cur_stage].load_state_dict(
                self.model.gaussian_heads[self.cur_stage - 1].state_dict()
            )
            if self.trainer.is_global_zero:
                print(f"Stage {self.cur_stage} encoder parameters loaded")
                
    def on_train_start(self) -> None:
        """Rank-0: prepare visualization batches, save GT once, and run initial visualization."""
        if self.trainer is not None and self.trainer.is_global_zero and self.vis_every > 0:
            try:
                self.prepare_vis_batches(self.config["valid_data"], 1)  # type: ignore[index]
            except Exception:
                raise Exception("Visualization preparation failed")
            try:
                # run an initial visualization before any training step (epoch=0, step=0)
                self.visualize_epoch_step(epoch=0, epoch_step=0, batch_size_forward=1)
            except Exception:
                raise Exception("Initial visualization failed")

    def configure_optimizers(self):
        """Build optimizer and scheduler from config; step scheduler every batch."""
        optimizer = build_optimizer(self.model, self.config["optimizer"])  # type: ignore[index]
        lr_scheduler = build_lr_scheduler(optimizer, self.config["lr_scheduler"])  # type: ignore[index]
        # keep a handle for state restore
        self._lr_scheduler_obj = lr_scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _broadcast_stage(self) -> None:
        """Broadcast current stage from rank-0 to all ranks to keep DDP processes consistent."""
        if self.trainer is None or self.trainer.world_size <= 1:
            return
        try:
            stage_tensor = torch.tensor([int(self.cur_stage)], device=self.device, dtype=torch.long)
            if hasattr(self.trainer.strategy, "broadcast"):
                stage_tensor = self.trainer.strategy.broadcast(stage_tensor)
            elif dist.is_available() and dist.is_initialized():
                dist.broadcast(stage_tensor, src=0)
            self.cur_stage = int(stage_tensor.item())
        except Exception:
            # Keep going even if broadcast fails; better to not crash
            pass


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Single training step computing loss and logging metrics."""
        image: torch.Tensor = batch["image"].to(self.device)
        outputs = self.model(image, stage=self.cur_stage)

        loss_stage = {}
        loss = 0
        for i, stage_outputs in enumerate(outputs["stage_results"]):
            pseudo_loss = stage_outputs["pseudo_loss"]
            # Train with pseudo labels first, then fine-tune with rendered images.
            # Starting directly from rendered images can create unstable gradients.
            if pseudo_loss is not None:
                loss_stage[f"loss/pseudo_{i}"] = pseudo_loss.mean()
                loss += 100 * pseudo_loss.mean()  # Amplify the pseudo-label signal.
            else:
                render_image = stage_outputs["image"]
                s_loss = 0.7 * F.l1_loss(render_image, image, reduction='mean') + 0.3 * (1.0 - fused_ssim(render_image, image, reduction='mean'))
                loss_stage[f"loss/rgb_{i}"] = s_loss
                loss += s_loss

        quant_image = outputs["quant_image"]
        quant_loss = outputs["quant_loss"]
        if quant_image is not None and quant_loss is not None:
            s_loss = 0.7 * F.l1_loss(quant_image, image, reduction='mean') + 0.3 * (1.0 - fused_ssim(quant_image, image, reduction='mean'))
            loss_stage[f"loss/quant_rgb"] = s_loss
            loss_stage[f"loss/vq"] = quant_loss.mean()
            loss += (s_loss + quant_loss.mean())
        
        
        with torch.no_grad():
            # metrics for debug
            should_log = (self.global_step % self.log_every == 0)
            if should_log:

                self.log("train/G-Res", outputs['gaussian_num'].mean()/(image.shape[2]*image.shape[3]), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
                self.log("train/RUsage", outputs['router_usage'].mean(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
                
                self.log("loss/total", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
                if quant_loss is not None:
                    self.log("loss/vq", quant_loss.mean(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
                
                for k, v in loss_stage.items():
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
                psnr_stage = {}
                for i, stage_outputs in enumerate(outputs["stage_results"]):
                    psnr_stage[f"train/psnr_{i}"] = compute_psnr(stage_outputs["image"], image).mean()
                    pseudo_loss = stage_outputs["pseudo_loss"]
                    if pseudo_loss is not None:
                        psnr_stage[f"train/psnr_pseudo_{i}"] = compute_psnr(outputs["stage_results"][i]["last_image"], image).mean()
                if quant_image is not None:
                    psnr_stage[f"train/psnr_vq"] = compute_psnr(quant_image, image).mean()
                        
                for k, v in psnr_stage.items():
                    if "pseudo" in k:
                        self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
                    else:
                        self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss
    
    def on_train_epoch_start(self) -> None:
        """Log learning rate at the beginning of each epoch."""
        # log lr
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
    
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Epoch-based stage sync; run epoch-local-step visualization on rank-0."""
        if self.trainer is None:
            return
             
        # Rank-0 only: epoch-local visualization (e.g., every 1000 steps within current epoch)
        if self.trainer.is_global_zero:
            try:
                epoch_step = int(batch_idx) + 1
                if self.vis_every > 0 and (epoch_step % self.vis_every == 0 or epoch_step == 1):
                    self.visualize_epoch_step(epoch=int(self.current_epoch), epoch_step=epoch_step, batch_size_forward=1)
            except Exception:
                raise Exception("Visualization failed")

    def on_train_epoch_end(self) -> None:
        """Epoch-based stage update/broadcast on all ranks; checkpoint saving on rank-0."""
        if self.trainer is None:
            return
        
        # Advance stage and broadcast once per epoch (all ranks execute)
        self.update_stage()

        # Rank-0 only: save checkpoint using epoch index as identifier
        if not self.trainer.is_global_zero:
            return
        try:
            epoch_index = int(self.current_epoch)
            self._save_legacy_checkpoints(epoch_index)
        except Exception:
            raise Exception("Checkpoint saving failed")

    # -------- Validation --------
    # Purpose: Reset per-epoch validation accumulators to enable a single distributed sync at epoch end.
    def _reset_val_accumulators(self) -> None:
        """Reset validation accumulators on the correct device (called once per validation epoch)."""
        num_stages = int(self.cur_stage) + 1
        device = self.device
        # Stage PSNR: accumulate sum/count per stage.
        self._val_psnr_sum = torch.zeros(num_stages, device=device, dtype=torch.float32)
        self._val_psnr_cnt = torch.zeros(num_stages, device=device, dtype=torch.float32)
        # Misc scalars: [psnr_q, quant_rgb_loss, vq, G-Res, RUsage]
        self._val_misc_sum = torch.zeros(5, device=device, dtype=torch.float32)
        self._val_misc_cnt = torch.zeros(5, device=device, dtype=torch.float32)

    # Purpose: Ensure accumulators are initialized before validation batches run.
    def on_validation_epoch_start(self) -> None:
        """Initialize validation accumulators for a single all_reduce in epoch end."""
        self._reset_val_accumulators()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Accumulate validation metrics locally; reduce once at epoch end to avoid DDP drift."""
        if os.environ.get("IGI_DEBUG_BATCHINFO", "0") == "2":
            self._maybe_dump_batch_info(batch, batch_idx)
        
        image: torch.Tensor = batch["image"].to(self.device)
        outputs = self.model(image, stage=self.cur_stage)

        # Lazily init accumulators in case hooks are skipped (e.g., sanity check quirks).
        if not hasattr(self, "_val_psnr_sum"):
            self._reset_val_accumulators()

        with torch.no_grad():
            # Stage PSNR
            stage_results = outputs.get("stage_results", [])
            for i, stage_outputs in enumerate(stage_results):
                if i >= int(self.cur_stage) + 1:
                    break
                psnr_val = compute_psnr(stage_outputs["image"], image).mean()
                self._val_psnr_sum[i] += psnr_val.detach().to(torch.float32)
                self._val_psnr_cnt[i] += 1.0

            # Quantized branch (optional)
            quant_image = outputs.get("quant_image", None)
            quant_loss = outputs.get("quant_loss", None)
            if quant_image is not None:
                psnr_q = compute_psnr(quant_image, image).mean()
                self._val_misc_sum[0] += psnr_q.detach().to(torch.float32)
                self._val_misc_cnt[0] += 1.0

                quant_rgb_loss = 0.7 * F.l1_loss(quant_image, image, reduction="mean") + 0.3 * (
                    1.0 - fused_ssim(quant_image, image, reduction="mean")
                )
                self._val_misc_sum[1] += quant_rgb_loss.detach().to(torch.float32)
                self._val_misc_cnt[1] += 1.0

            if quant_loss is not None:
                vq_val = quant_loss.mean()
                self._val_misc_sum[2] += vq_val.detach().to(torch.float32)
                self._val_misc_cnt[2] += 1.0

            # Global stats
            g_res = outputs["gaussian_num"].mean() / (image.shape[2] * image.shape[3])
            r_usage = outputs["router_usage"].mean()
            self._val_misc_sum[3] += g_res.detach().to(torch.float32)
            self._val_misc_cnt[3] += 1.0
            self._val_misc_sum[4] += r_usage.detach().to(torch.float32)
            self._val_misc_cnt[4] += 1.0


    def on_validation_epoch_end(self) -> None:
        """Reduce accumulated validation metrics once and log/print on rank-0."""
        if self.trainer is None:
            return
        try:
            # Pack all scalars into one vector so we all_reduce only once.
            vec = torch.cat(
                [self._val_psnr_sum, self._val_psnr_cnt, self._val_misc_sum, self._val_misc_cnt],
                dim=0,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            s = int(self.cur_stage) + 1
            psnr_sum = vec[0:s]
            psnr_cnt = vec[s : 2 * s].clamp_min(1.0)
            misc_sum = vec[2 * s : 2 * s + 5]
            misc_cnt = vec[2 * s + 5 : 2 * s + 10]

            psnr_avg = psnr_sum / psnr_cnt
            misc_avg = misc_sum / misc_cnt.clamp_min(1.0)

            if self.trainer.is_global_zero:
                # Log only on rank-0 to avoid duplicated logger entries.
                for i in range(s):
                    self.log(f"val/psnr_{i}", psnr_avg[i].detach(), on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

                # misc_avg: [psnr_q, quant_rgb_loss, vq, G-Res, RUsage]
                if float(misc_cnt[0].detach().cpu().item()) > 0:
                    self.log("val/psnr_q", misc_avg[0].detach(), on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
                if float(misc_cnt[1].detach().cpu().item()) > 0:
                    self.log("val/quant_rgb_loss", misc_avg[1].detach(), on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
                if float(misc_cnt[2].detach().cpu().item()) > 0:
                    self.log("val/vq", misc_avg[2].detach(), on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
                self.log("val/G-Res", misc_avg[3].detach(), on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
                self.log("val/RUsage", misc_avg[4].detach(), on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

                pairs = [f"val/psnr_{i}={float(psnr_avg[i].detach().cpu().item()):.6f}" for i in range(s)]
                pairs.append(f"val/G-Res={float(misc_avg[3].detach().cpu().item()):.6f}")
                pairs.append(f"val/RUsage={float(misc_avg[4].detach().cpu().item()):.6f}")
                if float(misc_cnt[0].detach().cpu().item()) > 0:
                    pairs.append(f"val/psnr_q={float(misc_avg[0].detach().cpu().item()):.6f}")
                if float(misc_cnt[1].detach().cpu().item()) > 0:
                    pairs.append(f"val/quant_rgb_loss={float(misc_avg[1].detach().cpu().item()):.6f}")
                if float(misc_cnt[2].detach().cpu().item()) > 0:
                    pairs.append(f"val/vq={float(misc_avg[2].detach().cpu().item()):.6f}")
                self.print(f"[Validation][epoch={int(self.current_epoch)}] " + ", ".join(pairs))
        except Exception:
            pass


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Perform optimizer step with closure (Lightning v2-compatible)."""
        try:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
        except Exception:
            # fallback: call closure then step if needed
            if optimizer_closure is not None:
                optimizer_closure()
            optimizer.step()
            optimizer.zero_grad()

    # -------- Visualization helpers --------
    def prepare_vis_batches(self, data_config: Dict[str, Any], batch_size_forward: int) -> None:
        """Collect a fixed set of batches for visualization and save GT images once."""
        if self._vis_batches is not None:
            return
        self._vis_batches = []
        num_vis_images = self.num_vis_images // batch_size_forward * batch_size_forward
        
        from train.dataloader import build_valid_dataloader as _build_valid
        loader = _build_valid(data_config, batch_size_forward)
        # Use DataLoader to gather a fixed number of batches deterministically
        for i, batch in enumerate(loader):
            if i >= num_vis_images // batch_size_forward:
                break
            # Ensure tensors are on CPU for later save; keep raw tensors for model
            self._vis_batches.append(batch)

        # save GT once
        save_dir = Path(self.workspace).joinpath("vis/gt")
        save_dir.mkdir(parents=True, exist_ok=True)
        idx_global = 0
        for i_batch, batch in enumerate(self._vis_batches):
            image = batch["image"].cpu().numpy()
            info: List[Dict[str, Any]] = batch["info"]
            for i_instance in range(image.shape[0]):
                save_sub = save_dir.joinpath(f"{idx_global:04d}")
                save_sub.mkdir(parents=True, exist_ok=True)
                image_i = (image[i_instance].transpose(1, 2, 0) * 255).astype(np.uint8)
                cv2.imwrite(str(save_sub.joinpath("image.jpg")), cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR))
                with save_sub.joinpath("info.json").open("w") as f:
                    json.dump(info[i_instance], f)
                idx_global += 1


    def visualize_epoch_step(self, epoch: int, epoch_step: int, batch_size_forward: int) -> None:
        """Run fixed-setup visualization and save renders/points for (epoch, epoch_step) on rank 0."""
        if self._vis_batches is None:
            return
        save_dir = Path(self.workspace).joinpath(f"vis/epoch_{epoch:04d}", f"step_{epoch_step:06d}")
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        with torch.inference_mode():
            for i_batch, batch in enumerate(self._vis_batches):
                image = batch["image"].to(self.device)
                out = self.model(image, stage=self.cur_stage)

                # New output structure: use stage_results list
                stage_results: List[Dict[str, torch.Tensor]] = out.get("stage_results", [])
                imgs: List[torch.Tensor] = [sr["image"].clamp(0, 1) for sr in stage_results]
                num_stages = len(stage_results)

                # Per-image metrics for each stage (no batch mean; keep [B])
                psnr_per_stage: List[torch.Tensor] = [compute_psnr(imgs[s], image) for s in range(num_stages)]

                quant_image: Optional[torch.Tensor] = out.get("quant_image", None)
                quant_loss: Optional[torch.Tensor] = out.get("quant_loss", None)
                quant_img: Optional[torch.Tensor] = None
                psnr_q: Optional[torch.Tensor] = None
                if quant_image is not None:
                    quant_img = quant_image.clamp(0, 1)
                    psnr_q = compute_psnr(quant_img, image)

                # Scalar tensors for global stats
                gaussians_num = out["gaussian_num"].detach().cpu().numpy()
                router_usage = out["router_usage"].detach().cpu().numpy()

                # xys may be list[Tensor] or None; only used to visualize last-stage points
                xys_list = out.get("xys", None)

                # Convert tensors to numpy for saving
                image_np = image.clamp(0, 1).detach().cpu().numpy()                # [B,3,H,W]
                imgs_np: List[np.ndarray] = [img.detach().cpu().numpy() for img in imgs]
                quant_np: Optional[np.ndarray] = None
                if quant_img is not None:
                    quant_np = quant_img.detach().cpu().numpy()

                # Error maps per stage: [B,1,H,W] -> numpy
                error_maps_np: List[np.ndarray] = [
                    (imgs[s] - image).pow(2).mean(dim=1, keepdim=True).detach().cpu().numpy()
                    for s in range(num_stages)
                ]
                quant_error_np: Optional[np.ndarray] = None
                if quant_img is not None:
                    quant_error_np = (quant_img - image).pow(2).mean(dim=1, keepdim=True).detach().cpu().numpy()

                # Patch-level MSE/SSIM error maps from model outputs (if available)
                mse_error_maps_np: List[Optional[np.ndarray]] = []
                ssim_error_maps_np: List[Optional[np.ndarray]] = []
                for s in range(num_stages):
                    mse_err = stage_results[s].get("mse_error", None)
                    ssim_err = stage_results[s].get("ssim_error", None)
                    if mse_err is None:
                        mse_error_maps_np.append(None)
                    else:
                        mse_error_maps_np.append(mse_err.detach().cpu().numpy())
                    if ssim_err is None:
                        ssim_error_maps_np.append(None)
                    else:
                        ssim_error_maps_np.append(ssim_err.detach().cpu().numpy())

                for i in range(image.shape[0]):
                    idx = i_batch * batch_size_forward + i
                    sub = save_dir.joinpath(f"{idx:04d}")
                    os.makedirs(sub, exist_ok=True)

                    # Assemble info with dynamic stages
                    info_dict: Dict[str, Any] = {
                        "gaussians_num": float(gaussians_num[i]),
                        "G_res": float(gaussians_num[i]) / float(image_np.shape[2] * image_np.shape[3]),
                        "router_usage": float(router_usage[i]),
                        "num_stages": int(num_stages),
                    }
                    for s in range(num_stages):
                        info_dict[f"psnr_{s}"] = float(psnr_per_stage[s][i])
                    if psnr_q is not None:
                        info_dict["psnr_q"] = float(psnr_q[i])
                    if quant_loss is not None:
                        try:
                            if quant_loss.numel() == 1:
                                info_dict["vq"] = float(quant_loss.detach().cpu().item())
                            elif quant_loss.shape[0] == image.shape[0]:
                                info_dict["vq"] = float(quant_loss[i].detach().cpu().item())
                            else:
                                info_dict["vq"] = float(quant_loss.mean().detach().cpu().item())
                        except Exception:
                            pass

                    # Save info.json
                    with open(sub.joinpath("info.json"), "w") as f:
                        json.dump(info_dict, f, indent=4)

                    # Save GT and per-stage renders
                    image_np_i = (image_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    cv2.imwrite(str(sub.joinpath("image.jpg")), cv2.cvtColor(image_np_i, cv2.COLOR_RGB2BGR))
                    for s in range(num_stages):
                        render_np_i = (imgs_np[s][i].transpose(1, 2, 0) * 255).astype(np.uint8)
                        cv2.imwrite(str(sub.joinpath(f"render_stage_{s+1}.jpg")), cv2.cvtColor(render_np_i, cv2.COLOR_RGB2BGR))
                        # Save error heatmap for this stage
                        err_i = error_maps_np[s][i][0]
                        err_norm = (err_i - err_i.min()) / (err_i.max() - err_i.min() + 1e-8)
                        err_uint8 = (err_norm * 255).astype(np.uint8)
                        err_color = cv2.applyColorMap(err_uint8, cv2.COLORMAP_JET)
                        cv2.imwrite(str(sub.joinpath(f"error_stage_{s+1}.png")), err_color)
                        # Save patch-level MSE error heatmap if available
                        if mse_error_maps_np[s] is not None:
                            mse_patch = mse_error_maps_np[s][i]  # [ph, pw]
                            H, W = image_np_i.shape[:2]
                            mse_resized = cv2.resize(mse_patch, (W, H), interpolation=cv2.INTER_NEAREST)
                            mse_norm = (mse_resized - mse_resized.min()) / (mse_resized.max() - mse_resized.min() + 1e-8)
                            mse_uint8 = (mse_norm * 255).astype(np.uint8)
                            mse_color = cv2.applyColorMap(mse_uint8, cv2.COLORMAP_JET)
                            cv2.imwrite(str(sub.joinpath(f"mse_error_stage_{s+1}.png")), mse_color)
                        # Save patch-level SSIM error heatmap if available
                        if ssim_error_maps_np[s] is not None:
                            ssim_patch = ssim_error_maps_np[s][i]  # [ph, pw]
                            H, W = image_np_i.shape[:2]
                            ssim_resized = cv2.resize(ssim_patch, (W, H), interpolation=cv2.INTER_NEAREST)
                            ssim_norm = (ssim_resized - ssim_resized.min()) / (ssim_resized.max() - ssim_resized.min() + 1e-8)
                            ssim_uint8 = (ssim_norm * 255).astype(np.uint8)
                            ssim_color = cv2.applyColorMap(ssim_uint8, cv2.COLORMAP_JET)
                            cv2.imwrite(str(sub.joinpath(f"ssim_error_stage_{s+1}.png")), ssim_color)

                    # Save quantization render/error if available
                    if quant_np is not None:
                        quant_np_i = (quant_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                        cv2.imwrite(str(sub.joinpath("render_quant.jpg")), cv2.cvtColor(quant_np_i, cv2.COLOR_RGB2BGR))
                        if quant_error_np is not None:
                            qerr_i = quant_error_np[i][0]
                            qerr_norm = (qerr_i - qerr_i.min()) / (qerr_i.max() - qerr_i.min() + 1e-8)
                            qerr_uint8 = (qerr_norm * 255).astype(np.uint8)
                            qerr_color = cv2.applyColorMap(qerr_uint8, cv2.COLORMAP_JET)
                            cv2.imwrite(str(sub.joinpath("error_quant.png")), qerr_color)

                    # Overlay last-stage points if available
                    if xys_list is not None:
                        H, W = image_np_i.shape[:2]
                        overlay = image_np_i.copy()
                        xys_i = xys_list[i].detach().cpu().numpy()
                        xs = np.rint((xys_i[:, 0] + 1.0) * 0.5 * (W - 1)).astype(np.int32)
                        ys = np.rint((xys_i[:, 1] + 1.0) * 0.5 * (H - 1)).astype(np.int32)
                        xs = np.clip(xs, 0, W - 1)
                        ys = np.clip(ys, 0, H - 1)
                        overlay[ys, xs] = np.array([255, 0, 0], dtype=np.uint8)
                        cv2.imwrite(str(sub.joinpath("image_points.png")), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        self.model.train()

    def _load_legacy_checkpoint(self, ckpt: str) -> None:
        """Load legacy checkpoints into module/model and stash optimizer/scheduler states."""
        # Load arbitrary checkpoint payloads with full Python object support when needed.
        def _load_dict(path):
            if not path.exists():
                return None
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                raise Exception(f"Failed to load checkpoint from {path}")

        # Load only parameters whose names and shapes match the current model.
        def _load_state_dict_with_missing_keys(state_dict, prefix: str = "model"):
            current_state_dict = self.model.state_dict()
            matched_state_dict = {}
            skipped_shape_keys = []

            for key, value in state_dict.items():
                if key not in current_state_dict:
                    continue
                if current_state_dict[key].shape != value.shape:
                    skipped_shape_keys.append(
                        f"{key}: ckpt={tuple(value.shape)} current={tuple(current_state_dict[key].shape)}"
                    )
                    continue
                matched_state_dict[key] = value

            missing_keys, unexpected_keys = self.model.load_state_dict(matched_state_dict, strict=False)

            if skipped_shape_keys:
                print(f"Skipped {prefix} keys with mismatched shapes:")
                for key in skipped_shape_keys:
                    print(f"  - {key}")

            if missing_keys:
                print(f"Missing {prefix} keys after partial load: {missing_keys}")

            if unexpected_keys:
                print(f"Unexpected {prefix} keys skipped during load: {unexpected_keys}")

        try:
            if ckpt.endswith(".pt"):
                # 1) Main weights: only the model weights are needed.
                data = torch.load(ckpt, map_location="cpu", weights_only=True)
                if "model" in data:
                    _load_state_dict_with_missing_keys(data["model"])

                # 2) sibling files
                base = Path(ckpt)
                stem = base.stem
                if stem.isdigit():
                    i_step = int(stem)
                    opt_path = base.parent.joinpath(f"{i_step:08d}_optimizer.pt")
                    opt = _load_dict(opt_path)
                    if opt is not None:
                        self._resume_optimizer_state = opt.get("optimizer", None)
                        self._resume_lr_state = opt.get("lr_scheduler", None)
                print(f"Loaded checkpoint from {ckpt}")
                        
            elif ckpt == "latest":
                latest_path = Path(self.workspace, "checkpoint", "latest.pt")
                if latest_path.exists():
                    latest = torch.load(latest_path, map_location="cpu", weights_only=True)
                    i_step = int(latest.get("epoch", 0))
                    model_path = latest_path.parent.joinpath(f"{i_step:08d}.pt")
                    opt_path = latest_path.parent.joinpath(f"{i_step:08d}_optimizer.pt")
                    model_dict = _load_dict(model_path)
                    if model_dict is not None and "model" in model_dict:
                        _load_state_dict_with_missing_keys(model_dict["model"])
                    opt = _load_dict(opt_path)
                    if opt is not None:
                        self._resume_optimizer_state = opt.get("optimizer", None)
                        self._resume_lr_state = opt.get("lr_scheduler", None)
                    print(f"Loaded checkpoint from {model_path}")
            else:
                # step index
                i_step = int(ckpt)
                model_path = Path(self.workspace, "checkpoint", f"{i_step:08d}.pt")
                opt_path = Path(self.workspace, "checkpoint", f"{i_step:08d}_optimizer.pt")
                model_dict = _load_dict(model_path)
                if model_dict is not None and "model" in model_dict:
                    _load_state_dict_with_missing_keys(model_dict["model"])
                opt = _load_dict(opt_path)
                if opt is not None:
                    self._resume_optimizer_state = opt.get("optimizer", None)
                    self._resume_lr_state = opt.get("lr_scheduler", None)
        except Exception:
            raise Exception(f"Failed to load checkpoint from {ckpt}")

    def _save_legacy_checkpoints(self, epoch: int) -> None:
        """Save model/optimizer/scheduler and latest meta in the original .pt layout (rank 0)."""
        if self.trainer is None or not self.trainer.is_global_zero:
            return
        try:
            save_root = Path(self.workspace, "checkpoint")
            save_root.mkdir(parents=True, exist_ok=True)

            # model weights
            model_bytes: bytes
            with io.BytesIO() as f:
                torch.save({
                    "model_config": self.config["model"],  # type: ignore[index]
                    "model": self.model.state_dict(),
                    "epoch": self.current_epoch,
                }, f)
                model_bytes = f.getvalue()
            (save_root / f"{epoch:08d}.pt").write_bytes(model_bytes)

            # optimizer + scheduler
            opt = self.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            lr_state = self._lr_scheduler_obj.state_dict() if hasattr(self, "_lr_scheduler_obj") and self._lr_scheduler_obj is not None else {}
            opt_bytes: bytes
            with io.BytesIO() as f:
                torch.save({
                    "model_config": self.config["model"],  # type: ignore[index]
                    "epoch": epoch,
                    "optimizer": opt.state_dict(),
                    "lr_scheduler": lr_state,
                }, f)
                opt_bytes = f.getvalue()
            (save_root / f"{epoch:08d}_optimizer.pt").write_bytes(opt_bytes)

            # latest meta
            latest_bytes: bytes
            with io.BytesIO() as f:
                torch.save({
                    "model_config": self.config["model"],  # type: ignore[index]
                    "epoch": self.current_epoch,
                }, f)
                latest_bytes = f.getvalue()
            (save_root / "latest.pt").write_bytes(latest_bytes)
        except Exception:
            pass