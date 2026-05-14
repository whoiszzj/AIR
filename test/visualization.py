from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from torch_kdtree import build_kd_tree

from format import _write_json


# Extract optional XY outputs as a tensor for density-map construction and visualization.
def _extract_xys(outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Extract XY point coordinates from outputs."""
    xys_value = outputs.get("xys", None)
    if xys_value is None:
        return None

    if isinstance(xys_value, (list, tuple)):
        if len(xys_value) == 0:
            return None
        xys_tensor = xys_value[0]
    else:
        xys_tensor = xys_value

    if not isinstance(xys_tensor, torch.Tensor):
        return None
    return xys_tensor.detach()


def _data_normalize_tensor(data: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor with mean/std clipping, matching the training utility."""
    data_mean = data.mean()
    data_std = data.std()
    data_max = torch.minimum(data_mean + 3 * data_std, data.max())
    data_min = torch.maximum(data_mean - 3 * data_std, data.min())
    data = (data - data_min) / (data_max - data_min)
    data = torch.clamp(data, 0, 1)
    return data


def _build_position_density_map(
    xys: Optional[torch.Tensor],
    height: int,
    width: int,
    k: int = 20,
) -> Optional[torch.Tensor]:
    """Build a normalized Gaussian-position density map from XY coordinates in [-1, 1]."""
    if xys is None:
        return None

    xy = xys.detach().reshape(-1, 2).to(dtype=torch.float32)
    if xy.shape[0] == 0:
        return None

    finite_mask = torch.isfinite(xy).all(dim=-1)
    xy = xy[finite_mask]
    if xy.shape[0] == 0:
        return None

    xy = xy.clone()
    xy[..., 0] = (xy[..., 0] + 1.0) * float(width) / 2.0
    xy[..., 1] = (xy[..., 1] + 1.0) * float(height) / 2.0

    kd_tree = build_kd_tree(xy)
    xy_map = torch.meshgrid(
        torch.arange(width, device=xy.device),
        torch.arange(height, device=xy.device),
        indexing="xy",
    )
    xy_map = torch.stack(xy_map, dim=-1).reshape(-1, 2).float()  # [H*W, 2]

    k = min(int(k), xy.shape[0])
    dist, _idx = kd_tree.query(xy_map, nr_nns_searches=k)
    dist = torch.sqrt(dist)
    dist_max = dist.max(dim=-1)[0]  # [H*W, 1]
    density = k / torch.sqrt(dist_max)
    density = _data_normalize_tensor(density)
    density = density.view(height, width)
    return density


# Save reconstructed images, visualizations, and metrics to disk.
def _save_outputs(
    save_dir: Path,
    image_np: np.ndarray,
    stage_images_np: np.ndarray,
    error_maps_color: np.ndarray,
    mse_error_maps_color: Optional[np.ndarray],
    ssim_error_maps_color: Optional[np.ndarray],
    quant_image_np: Optional[np.ndarray],
    xys: Optional[torch.Tensor],
    info: Dict[str, Any],
) -> None:
    """Save images, error visualizations and metrics to disk following training layout."""
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_json(save_dir.joinpath("info.json"), info)

    cv2.imwrite(str(save_dir.joinpath("image.jpg")), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    for idx, img_np in enumerate(stage_images_np):
        cv2.imwrite(
            str(save_dir.joinpath(f"render_stage_{idx + 1}.jpg")),
            cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
        )
    if quant_image_np is not None:
        cv2.imwrite(
            str(save_dir.joinpath("render_quant.jpg")),
            cv2.cvtColor(quant_image_np, cv2.COLOR_RGB2BGR),
        )
    for idx, err_np in enumerate(error_maps_color):
        cv2.imwrite(str(save_dir.joinpath(f"error_stage_{idx + 1}.png")), err_np)

    if mse_error_maps_color is not None:
        for idx, err_np in enumerate(mse_error_maps_color):
            cv2.imwrite(str(save_dir.joinpath(f"mse_error_stage_{idx + 1}.png")), err_np)

    if ssim_error_maps_color is not None:
        for idx, err_np in enumerate(ssim_error_maps_color):
            cv2.imwrite(str(save_dir.joinpath(f"ssim_error_stage_{idx + 1}.png")), err_np)

    if xys is not None:
        height, width = image_np.shape[:2]
        position_density = _build_position_density_map(xys, height, width)
        if position_density is not None:
            position_density_np = position_density.detach().cpu().numpy()
            density_uint8 = (position_density_np * 255.0).clip(0, 255).astype(np.uint8)
            density_heatmap = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)
            np.save(str(save_dir.joinpath("position_density.npy")), position_density_np)
            cv2.imwrite(str(save_dir.joinpath("position_density.png")), density_heatmap)

        xys_np = xys.detach().cpu().numpy()
        overlay = image_np.copy()
        xs = np.rint((xys_np[:, 0] + 1.0) * 0.5 * (width - 1)).astype(np.int32)
        ys = np.rint((xys_np[:, 1] + 1.0) * 0.5 * (height - 1)).astype(np.int32)
        xs = np.clip(xs, 0, width - 1)
        ys = np.clip(ys, 0, height - 1)
        overlay[ys, xs] = np.array([255, 0, 0], dtype=np.uint8)
        cv2.imwrite(str(save_dir.joinpath("image_points.png")), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
