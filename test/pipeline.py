import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from pytorch_msssim import ms_ssim

from model.airnet import AIRNet
from train.utils import compute_psnr

from format import _round_float, _to_float
from io_utils import _prepare_image_tensor
from metrics import (
    _build_stage_info,
    _compute_lpips,
    _extract_quant_metrics,
    _to_float_list,
)
from visualization import _extract_xys, _save_outputs


# Synchronize CUDA before and after timed runs when GPU inference is used.
def _torch_synchronize_if_needed() -> None:
    """Synchronize CUDA device if available to ensure accurate timing."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


# Run timed inference for one image and return the final outputs with timing stats.
def _run_timed_inference(
    model: AIRNet,
    image_tensor: torch.Tensor,
    max_stage: int,
    num_repeats: int = 10,
) -> Dict[str, Any]:
    """Run model inference repeatedly and return outputs from the last timed run."""
    _torch_synchronize_if_needed()
    with torch.inference_mode():
        _ = model(image_tensor, stage=max_stage)

    times_ms: List[float] = []
    outputs: Dict[str, Any] = {}
    for _ in range(num_repeats):
        _torch_synchronize_if_needed()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(image_tensor, stage=max_stage)
        _torch_synchronize_if_needed()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    outputs["timing"] = {
        "inference_time_ms_mean": _round_float(np.mean(times_ms), 3),
        "inference_time_ms_min": _round_float(np.min(times_ms), 3),
    }
    return outputs


# Run inference for one image, save outputs, and return the serialized metrics.
def _infer_single_image(
    model: AIRNet,
    image_path: Path,
    save_dir: Path,
    device: torch.device,
    max_stage: int,
    lpips_metric: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """Process one image and save all visualizations and metrics."""
    image_tensor = _prepare_image_tensor(str(image_path), device)
    outputs = _run_timed_inference(model, image_tensor, max_stage=max_stage)

    stage_results = outputs.get("stage_results", [])
    if len(stage_results) == 0:
        raise RuntimeError("Model returned empty 'stage_results' in inference.")

    imgs_per_stage = [stage_result["image"].clamp(0.0, 1.0) for stage_result in stage_results]
    quant_image = outputs.get("quant_image", None)
    quant_image_np: Optional[np.ndarray] = None
    if isinstance(quant_image, torch.Tensor):
        quant_image_clamped = quant_image.clamp(0.0, 1.0)
        quant_image_np = (quant_image_clamped.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)

    with torch.no_grad():
        psnr_per_stage = [
            float(compute_psnr(img, image_tensor).detach().cpu().numpy()[0])
            for img in imgs_per_stage
        ]
        ssim_per_stage = [
            float(np.asarray(ms_ssim(img, image_tensor, data_range=1.0).detach().cpu().numpy()).reshape(-1)[0])
            for img in imgs_per_stage
        ]
        lpips_per_stage = []
        if lpips_metric is not None:
            for img in imgs_per_stage:
                lpips_value = _compute_lpips(lpips_metric, img, image_tensor)
                if lpips_value is not None:
                    lpips_per_stage.append(float(lpips_value))
        gaussian_num_per_stage = _to_float_list(outputs.get("gaussian_nums_per_stage", []))
        time_cost_per_stage_ms = [
            float(value) * 1000.0 for value in _to_float_list(outputs.get("time_cost_per_stage", []))
        ]
        gaussian_num = _to_float(outputs.get("gaussian_num", None))
        router_usage = _to_float(outputs.get("router_usage", None))
        quant_metrics = _extract_quant_metrics(outputs, image_tensor, lpips_metric=lpips_metric)

    img_np = (image_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    stage_imgs_np = [
        (img.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        for img in imgs_per_stage
    ]

    # Build per-pixel MSE maps to visualize reconstruction error spatially.
    error_maps_color = []
    for img in imgs_per_stage:
        err = (img - image_tensor).pow(2).mean(dim=1, keepdim=True).detach().cpu().numpy()[0, 0]
        err_norm = (err - err.min()) / (err.max() - err.min() + 1e-8)
        err_uint8 = (err_norm * 255).astype(np.uint8)
        error_maps_color.append(cv2.applyColorMap(err_uint8, cv2.COLORMAP_JET))

    # Resize patch-level error maps back to image resolution for visualization.
    mse_error_maps_color = []
    mean_mse_error_per_stage: List[float] = []
    ssim_error_maps_color = []
    height, width = img_np.shape[:2]
    for stage_result in stage_results:
        mse_err = stage_result.get("mse_error", None)
        ssim_err = stage_result.get("ssim_error", None)
        if isinstance(mse_err, torch.Tensor):
            mse_np = mse_err.detach().cpu().numpy()[0]
            mean_mse_error_per_stage.append(float(np.mean(mse_np)))
            mse_resized = cv2.resize(mse_np, (width, height), interpolation=cv2.INTER_NEAREST)
            mse_norm = (mse_resized - mse_resized.min()) / (mse_resized.max() - mse_resized.min() + 1e-8)
            mse_uint8 = (mse_norm * 255).astype(np.uint8)
            mse_error_maps_color.append(cv2.applyColorMap(mse_uint8, cv2.COLORMAP_JET))
        if isinstance(ssim_err, torch.Tensor):
            ssim_np = ssim_err.detach().cpu().numpy()[0]
            ssim_resized = cv2.resize(ssim_np, (width, height), interpolation=cv2.INTER_NEAREST)
            ssim_norm = (ssim_resized - ssim_resized.min()) / (ssim_resized.max() - ssim_resized.min() + 1e-8)
            ssim_uint8 = (ssim_norm * 255).astype(np.uint8)
            ssim_error_maps_color.append(cv2.applyColorMap(ssim_uint8, cv2.COLORMAP_JET))

    mse_error_maps_color_arr = np.array(mse_error_maps_color, dtype=np.uint8) if len(mse_error_maps_color) > 0 else None
    ssim_error_maps_color_arr = np.array(ssim_error_maps_color, dtype=np.uint8) if len(ssim_error_maps_color) > 0 else None

    gnum_val = int(round(float(gaussian_num))) if gaussian_num is not None else 0
    g_res = gnum_val / float(img_np.shape[0] * img_np.shape[1]) if img_np.size > 0 else 0.0
    router_usage_val = float(router_usage) if router_usage is not None else 0.0
    stage_info = _build_stage_info(
        psnr_per_stage,
        ssim_per_stage,
        lpips_per_stage,
        gaussian_num_values=gaussian_num_per_stage,
        time_cost_values_ms=time_cost_per_stage_ms,
        mean_mse_error_values=mean_mse_error_per_stage,
    )

    info = {
        "image_name": image_path.name,
        "resolution": f"{width} * {height}",
        "stage": stage_info,
        "gaussians_num": int(gnum_val),
        "G_res": round(float(g_res), 3),
        "router_usage": round(float(router_usage_val), 3),
    }
    info.update(outputs["timing"])
    if isinstance(quant_metrics, dict) and len(quant_metrics) > 0:
        info.update(quant_metrics)

    _save_outputs(
        save_dir=save_dir,
        image_np=img_np,
        stage_images_np=np.array(stage_imgs_np, dtype=np.uint8),
        error_maps_color=np.array(error_maps_color, dtype=np.uint8),
        mse_error_maps_color=mse_error_maps_color_arr,
        ssim_error_maps_color=ssim_error_maps_color_arr,
        quant_image_np=quant_image_np,
        xys=_extract_xys(outputs),
        info=info,
    )
    return info
