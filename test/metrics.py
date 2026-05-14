import warnings
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pytorch_msssim import ms_ssim

from train.utils import compute_psnr

from format import _round_float, _round_stage_info, _to_float, _write_json

try:
    import lpips
except ImportError:
    lpips = None


# Create an LPIPS metric module when dependencies are available.
def _build_lpips_metric(device: torch.device) -> Optional[torch.nn.Module]:
    """Build the LPIPS metric module or return None when dependency is unavailable."""
    if lpips is None:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'pretrained' is deprecated since 0.13.*",
            category=UserWarning,
            module="torchvision.models._utils",
        )
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13.*",
            category=UserWarning,
            module="torchvision.models._utils",
        )
        metric = lpips.LPIPS(net="vgg")
    metric = metric.to(device)
    metric.eval()
    return metric


# Compute a scalar LPIPS value from normalized [0,1] RGB tensors.
def _compute_lpips(
    metric: Optional[torch.nn.Module],
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Optional[float]:
    """Compute LPIPS on [0,1] tensors by mapping them to [-1,1]."""
    if metric is None:
        return None

    pred_norm = pred.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    target_norm = target.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    lpips_value = metric(pred_norm, target_norm)
    return _to_float(lpips_value)


# Merge per-stage metrics into a compact stage summary structure.
def _build_stage_info(
    psnr_values: List[float],
    ssim_values: List[float],
    lpips_values: Optional[List[float]] = None,
    gaussian_num_values: Optional[List[float]] = None,
    time_cost_values_ms: Optional[List[float]] = None,
    mean_mse_error_values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Build a compact stage summary for serialized outputs."""
    stage_lengths = [len(psnr_values), len(ssim_values)]
    if isinstance(lpips_values, list) and len(lpips_values) > 0:
        stage_lengths.append(len(lpips_values))
    if isinstance(gaussian_num_values, list) and len(gaussian_num_values) > 0:
        stage_lengths.append(len(gaussian_num_values))
    if isinstance(time_cost_values_ms, list) and len(time_cost_values_ms) > 0:
        stage_lengths.append(len(time_cost_values_ms))
    if isinstance(mean_mse_error_values, list) and len(mean_mse_error_values) > 0:
        stage_lengths.append(len(mean_mse_error_values))
    stage_count = min(stage_lengths) if len(stage_lengths) > 0 else 0
    stage_info: Dict[str, Any] = {
        "num": stage_count,
        "psnr": psnr_values[:stage_count],
        "ms-ssim": ssim_values[:stage_count],
    }
    if isinstance(lpips_values, list) and len(lpips_values) > 0:
        stage_info["lpips"] = lpips_values[:stage_count]
    if isinstance(gaussian_num_values, list) and len(gaussian_num_values) > 0:
        stage_info["gaussian_num"] = gaussian_num_values[:stage_count]
    if isinstance(time_cost_values_ms, list) and len(time_cost_values_ms) > 0:
        stage_info["time_cost_ms"] = time_cost_values_ms[:stage_count]
    if isinstance(mean_mse_error_values, list) and len(mean_mse_error_values) > 0:
        stage_info["mean_mse_error"] = mean_mse_error_values[:stage_count]
    return _round_stage_info(stage_info)


# Extract quantization-related metrics from inference outputs when available.
def _extract_quant_metrics(
    outputs: Dict[str, Any],
    image_tensor: torch.Tensor,
    lpips_metric: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """Extract quantization evaluation metrics from model outputs."""
    quant_image = outputs.get("quant_image", None)
    if quant_image is None:
        return {}

    quant_img = quant_image.clamp(0.0, 1.0)
    with torch.no_grad():
        quant_psnr = float(compute_psnr(quant_img, image_tensor).detach().cpu().numpy()[0])
        quant_msssim = float(np.asarray(ms_ssim(quant_img, image_tensor, data_range=1.0).detach().cpu().numpy()).reshape(-1)[0])
        quant_lpips = _compute_lpips(lpips_metric, quant_img, image_tensor)

    unit_bit = outputs.get("quant_unit_bit", None)
    total_bits = outputs.get("quant_total_bits", None)
    bpp = outputs.get("quant_bpp", None)

    quant_bits: Dict[str, Any] = {}
    if isinstance(unit_bit, torch.Tensor) and unit_bit.numel() > 0:
        ub = unit_bit.detach().cpu().to(torch.long).reshape(-1, 4)[0].tolist()
        quant_bits = {
            "pos_bits": int(ub[0]),
            "scale_bits": int(ub[1]),
            "rot_bits": int(ub[2]),
            "color_bits": int(ub[3]),
            "total_bits": int(sum(ub)),
        }
    elif isinstance(total_bits, torch.Tensor) and total_bits.numel() > 0:
        total_bits_val = _to_float(total_bits)
        if total_bits_val is not None:
            quant_bits["total_bits"] = float(total_bits_val)

    quant_bpp = _to_float(bpp)

    bits_per_gaussian: Optional[float] = None
    if "total_bits" in quant_bits:
        gnum = _to_float(outputs.get("gaussian_num", None))
        if gnum is not None and gnum > 0:
            bits_per_gaussian = float(quant_bits["total_bits"]) / gnum

    out: Dict[str, Any] = {
        "quant_psnr": _round_float(quant_psnr, 3),
        "quant_ms-ssim": _round_float(quant_msssim, 5),
    }
    if quant_lpips is not None:
        out["quant_lpips"] = _round_float(quant_lpips, 5)
    if len(quant_bits) > 0:
        out["quant_bits"] = quant_bits
    if quant_bpp is not None:
        out["quant_bpp"] = _round_float(quant_bpp, 3)
    if bits_per_gaussian is not None:
        out["quant_bits_per_gaussian"] = _round_float(bits_per_gaussian, 3)
    return out


# Average nested numeric metrics across multiple images in batch mode.
def _average_nested_metrics(values: List[Any]) -> Optional[Any]:
    """Average nested dictionaries, lists, and scalar metrics."""
    valid_values = [value for value in values if value is not None]
    if len(valid_values) == 0:
        return None

    if all(isinstance(value, Real) and not isinstance(value, bool) for value in valid_values):
        return float(np.mean([float(value) for value in valid_values]))

    if all(isinstance(value, np.generic) for value in valid_values):
        return float(np.mean([float(value.item()) for value in valid_values]))

    if all(isinstance(value, list) for value in valid_values):
        min_len = min(len(value) for value in valid_values)
        result = []
        for idx in range(min_len):
            averaged_item = _average_nested_metrics([value[idx] for value in valid_values])
            if averaged_item is not None:
                result.append(averaged_item)
        return result

    if all(isinstance(value, dict) for value in valid_values):
        result_dict: Dict[str, Any] = {}
        keys = sorted({key for value in valid_values for key in value.keys()})
        for key in keys:
            averaged_item = _average_nested_metrics([value.get(key) for value in valid_values])
            if averaged_item is not None:
                result_dict[key] = averaged_item
        return result_dict

    scalar_values = [_to_float(value) for value in valid_values]
    scalar_values = [value for value in scalar_values if value is not None]
    if len(scalar_values) == len(valid_values):
        return float(np.mean(scalar_values))

    return None


# Convert a stage-wise metric sequence into a float list.
def _to_float_list(values: Any) -> List[float]:
    """Convert a list-like object of scalar values into Python floats."""
    if not isinstance(values, (list, tuple, np.ndarray)):
        return []

    result: List[float] = []
    for value in values:
        scalar = _to_float(value)
        if scalar is not None:
            result.append(float(scalar))
    return result


# Compute per-stage variance for compact stage summaries in batch outputs.
def _compute_stage_variance(infos: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Compute per-stage metric variance across images."""
    stage_list = [info["stage"] for info in infos if isinstance(info.get("stage"), dict)]
    if len(stage_list) == 0:
        return None

    psnr_lists = [stage["psnr"] for stage in stage_list if isinstance(stage.get("psnr"), list) and len(stage["psnr"]) > 0]
    ssim_lists = [stage["ms-ssim"] for stage in stage_list if isinstance(stage.get("ms-ssim"), list) and len(stage["ms-ssim"]) > 0]
    lpips_lists = [stage["lpips"] for stage in stage_list if isinstance(stage.get("lpips"), list) and len(stage["lpips"]) > 0]
    gaussian_num_lists = [
        stage["gaussian_num"]
        for stage in stage_list
        if isinstance(stage.get("gaussian_num"), list) and len(stage["gaussian_num"]) > 0
    ]
    mean_mse_error_lists = [
        stage["mean_mse_error"]
        for stage in stage_list
        if isinstance(stage.get("mean_mse_error"), list) and len(stage["mean_mse_error"]) > 0
    ]
    time_cost_lists = [
        stage["time_cost_ms"]
        for stage in stage_list
        if isinstance(stage.get("time_cost_ms"), list) and len(stage["time_cost_ms"]) > 0
    ]
    if (
        len(psnr_lists) == 0
        and len(ssim_lists) == 0
        and len(lpips_lists) == 0
        and len(gaussian_num_lists) == 0
        and len(mean_mse_error_lists) == 0
        and len(time_cost_lists) == 0
    ):
        return None

    variance_info: Dict[str, List[float]] = {}
    if len(psnr_lists) > 0:
        psnr_stage_count = min(len(values) for values in psnr_lists)
        variance_info["psnr"] = [
            round(float(np.var([float(values[idx]) for values in psnr_lists])), 3)
            for idx in range(psnr_stage_count)
        ]
    if len(ssim_lists) > 0:
        ssim_stage_count = min(len(values) for values in ssim_lists)
        variance_info["ms-ssim"] = [
            round(float(np.var([float(values[idx]) for values in ssim_lists])), 3)
            for idx in range(ssim_stage_count)
        ]
    if len(lpips_lists) > 0:
        lpips_stage_count = min(len(values) for values in lpips_lists)
        variance_info["lpips"] = [
            round(float(np.var([float(values[idx]) for values in lpips_lists])), 5)
            for idx in range(lpips_stage_count)
        ]
    if len(gaussian_num_lists) > 0:
        gaussian_num_stage_count = min(len(values) for values in gaussian_num_lists)
        variance_info["gaussian_num"] = [
            round(float(np.var([float(values[idx]) for values in gaussian_num_lists])), 3)
            for idx in range(gaussian_num_stage_count)
        ]
    if len(mean_mse_error_lists) > 0:
        mean_mse_error_stage_count = min(len(values) for values in mean_mse_error_lists)
        variance_info["mean_mse_error"] = [
            round(float(np.var([float(values[idx]) for values in mean_mse_error_lists])), 8)
            for idx in range(mean_mse_error_stage_count)
        ]
    if len(time_cost_lists) > 0:
        time_cost_stage_count = min(len(values) for values in time_cost_lists)
        variance_info["time_cost_ms"] = [
            round(float(np.var([float(values[idx]) for values in time_cost_lists])), 3)
            for idx in range(time_cost_stage_count)
        ]
    if len(variance_info) == 0:
        return None

    stage_counts = [len(values) for key, values in variance_info.items() if key != "num"]
    variance_info["num"] = min(stage_counts) if len(stage_counts) > 0 else 0
    return _round_stage_info(variance_info)


# Compute the mean image width and height from serialized per-image resolutions.
def _compute_mean_resolution(infos: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Compute average width and height from per-image resolution strings."""
    widths: List[float] = []
    heights: List[float] = []
    for info in infos:
        resolution = info.get("resolution")
        if not isinstance(resolution, str) or "*" not in resolution:
            continue
        width_str, height_str = [part.strip() for part in resolution.split("*", maxsplit=1)]
        try:
            widths.append(float(width_str))
            heights.append(float(height_str))
        except ValueError:
            continue

    if len(widths) == 0 or len(heights) == 0:
        return None
    return {
        "width": _round_float(np.mean(widths), 3),
        "height": _round_float(np.mean(heights), 3),
    }


# Save aggregated batch statistics after processing a directory of images.
def _save_batch_summary(
    batch_root: Path,
    input_dir: Path,
    image_paths: List[Path],
    infos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Save batch-level mean metrics and the source input directory."""
    mean_info = _average_nested_metrics(infos)
    if not isinstance(mean_info, dict):
        mean_info = {}

    if isinstance(mean_info.get("stage"), dict):
        mean_info["stage"] = _round_stage_info(mean_info["stage"])
    for key in (
        "G_res",
        "router_usage",
        "quant_psnr",
        "quant_ms-ssim",
        "quant_lpips",
        "quant_bpp",
        "quant_bits_per_gaussian",
        "inference_time_ms_mean",
        "inference_time_ms_min",
    ):
        rounded_value = _round_float(mean_info.get(key), 3)
        if rounded_value is not None:
            mean_info[key] = rounded_value

    variance_info: Dict[str, Any] = {}
    stage_variance = _compute_stage_variance(infos)
    if stage_variance is not None:
        variance_info["stage"] = stage_variance

    summary = {
        "input_dir": str(input_dir.resolve()),
        "num_images": len(image_paths),
        "mean_info": mean_info,
    }
    mean_resolution = _compute_mean_resolution(infos)
    if mean_resolution is not None:
        summary["mean_resolution"] = mean_resolution
    if len(variance_info) > 0:
        summary["variance_info"] = variance_info
    _write_json(batch_root.joinpath("summary.json"), summary)
    return summary
