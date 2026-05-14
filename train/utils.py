import fnmatch
import os
from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Union

import cv2
import numpy as np
import sympy
import torch
import torch.nn as nn

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def read_image(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read an image and return a uint8 RGB array of shape (H, W, 3).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    image = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    return image


def get_image_files(
    root_dir: str,
    recursive: bool = True,
    accept_image_format: List[str] = None,
    load_from_file: bool = True,
) -> List[str]:
    """
    Recursively traverse the root directory and collect all image file paths.
    
    Args:
        root_dir: The root directory to start traversal
        recursive: Whether to traverse subdirectories recursively
        
    Returns:
        List of image file paths
    """
    imgs_path_file = Path(root_dir) / "imgs_path.txt"
    if load_from_file and imgs_path_file.exists():
        try:
            with open(imgs_path_file, "r", encoding="utf-8") as f:
                imgs = [line.strip() for line in f if line.strip()]
            print(f"File list loaded from {imgs_path_file}")
            return imgs
        except OSError as e:
            print(f"Warning: Failed to read {imgs_path_file}: {e}")

    # Common image file extensions
    if accept_image_format is None:
        accept_image_format = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".tiff",
            ".tif",
            ".webp",
            ".svg",
        ]
    image_extensions = {format.lower() for format in accept_image_format}

    imgs = []
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Warning: Directory {root_dir} does not exist")
        return imgs

    if recursive:
        # Recursively walk directories using os.walk for better performance on slow disks
        try:
            for dirpath, dirnames, filenames in os.walk(root_path, onerror=lambda e: None):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in image_extensions:
                        imgs.append(str(Path(dirpath) / filename))
        except OSError as e:
            print(f"Warning: Error walking directory {root_dir}: {e}")
    else:
        # Only check files in the root directory using os.scandir (fewer stats on slow disks)
        try:
            with os.scandir(root_path) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            if os.path.splitext(entry.name)[1].lower() in image_extensions:
                                imgs.append(str(Path(entry.path)))
                    except OSError:
                        # Skip entries that cannot be accessed
                        continue
        except OSError as e:
            print(f"Warning: Error scanning directory {root_dir}: {e}")

    # write file list with "imgs_path.txt"
    try:
        with open(imgs_path_file, "w", encoding="utf-8") as f:
            f.write("\n".join(imgs))
        print(f"File list saved to {imgs_path_file}")
    except OSError as e:
        print(f"Warning: Failed to write {imgs_path_file}: {e}")

    return imgs


def any_match(s: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(s, pat) for pat in patterns)


def build_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
    named_param_groups = [
        {
            k: p
            for k, p in model.named_parameters()
            if any_match(k, param_group_config["params"]["include"])
            and not any_match(k, param_group_config["params"].get("exclude", []))
        }
        for param_group_config in optimizer_config["params"]
    ]
    excluded_params = [
        k
        for k, p in model.named_parameters()
        if p.requires_grad and not any(k in named_params for named_params in named_param_groups)
    ]
    assert len(excluded_params) == 0, (
        "The following parameters require grad but are excluded from the optimizer: "
        f"{excluded_params}"
    )
    optimizer_cls = getattr(torch.optim, optimizer_config["type"])
    optimizer = optimizer_cls(
        [
            {
                **param_group_config,
                "params": list(params.values()),
            }
            for param_group_config, params in zip(
                optimizer_config["params"],
                named_param_groups,
            )
        ]
    )
    return optimizer


def parse_lr_lambda(s: str) -> Callable[[int], float]:
    epoch = sympy.symbols("epoch")
    lr_lambda = sympy.sympify(s)
    return sympy.lambdify(epoch, lr_lambda, "math")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any],
) -> torch.optim.lr_scheduler._LRScheduler:
    if scheduler_config["type"] == "SequentialLR":
        child_schedulers = [
            build_lr_scheduler(optimizer, child_scheduler_config)
            for child_scheduler_config in scheduler_config["params"]["schedulers"]
        ]
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=child_schedulers,
            milestones=scheduler_config["params"]["milestones"],
        )
    elif scheduler_config["type"] == "LambdaLR":
        lr_lambda = scheduler_config["params"]["lr_lambda"]
        if isinstance(lr_lambda, str):
            lr_lambda = parse_lr_lambda(lr_lambda)
        elif isinstance(lr_lambda, list):
            lr_lambda = [parse_lr_lambda(l) for l in lr_lambda]
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )
    else:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
        scheduler = scheduler_cls(optimizer, **scheduler_config.get("params", {}))
    return scheduler


def compute_psnr(pred, target):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) for each image in a batch.
    Returns a tensor of PSNR values, one for each image.
    """
    pred = pred.detach().clamp(0, 1).float()
    target = target.detach().clamp(0, 1).float()
    # Compute per-image MSE over all dimensions except batch
    mse = torch.mean((pred - target) ** 2, dim=tuple(range(1, pred.dim())))
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

