import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Resolve user-friendly checkpoint inputs into a concrete checkpoint file.
def _resolve_checkpoint_path(checkpoint: Optional[str]) -> str:
    """Resolve a checkpoint file, directory, or simple step spec into a concrete .pt file path."""
    if checkpoint is None:
        raise ValueError("Checkpoint must be provided.")

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.suffix == ".pt":
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return str(checkpoint_path)

    if checkpoint_path.exists():
        if checkpoint_path.is_dir():
            ckpt_dir = checkpoint_path
            if not (ckpt_dir / "latest.pt").exists() and (ckpt_dir / "checkpoint").is_dir():
                ckpt_dir = ckpt_dir / "checkpoint"
            latest = ckpt_dir / "latest.pt"
            if latest.exists():
                meta = torch.load(latest, map_location="cpu", weights_only=True)
                epoch = int(meta.get("epoch", 0))
                base = ckpt_dir / f"{epoch:08d}.pt"
                if not base.exists():
                    raise FileNotFoundError(f"Model checkpoint not found for epoch {epoch}: {base}")
                return str(base)
            raise FileNotFoundError(f"Checkpoint directory does not contain latest.pt: {ckpt_dir}")
        return str(checkpoint_path)

    ckpt_dir = Path("checkpoint")
    if checkpoint == "latest":
        latest = ckpt_dir / "latest.pt"
        if not latest.exists():
            raise FileNotFoundError(f"Latest checkpoint meta not found: {latest}")
        meta = torch.load(latest, map_location="cpu", weights_only=True)
        epoch = int(meta.get("epoch", 0))
        base = ckpt_dir / f"{epoch:08d}.pt"
        if not base.exists():
            raise FileNotFoundError(f"Model checkpoint not found for epoch {epoch}: {base}")
        return str(base)

    candidates = [ckpt_dir / f"{checkpoint}.pt"]
    if checkpoint.isdigit():
        candidates.insert(0, ckpt_dir / f"{int(checkpoint):08d}.pt")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Checkpoint spec could not be resolved: {checkpoint}")


# Expand the input path into a sorted list of supported image files.
def _list_image_paths(input_path: str) -> List[Path]:
    """Collect supported image files from a file path or a directory path."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {path}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {input_path}")

    image_paths = sorted(
        [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS]
    )
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No supported images found in directory: {input_path}")
    return image_paths


# Load a single RGB image and convert it into a normalized tensor.
def _prepare_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Load a single RGB image and convert to tensor [1,3,H,W] in [0,1]."""
    if isinstance(image_path, (str, os.PathLike)):
        data = Path(image_path).read_bytes()
    else:
        data = image_path.read()
    image = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)
