import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import click
import torch

# Add project root and this directory to Python path so direct script execution works.
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(TEST_DIR, "..")
for path in (PROJECT_ROOT, TEST_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.airnet import AIRNet

from format import _format_json_compact
from io_utils import IMAGE_EXTENSIONS, _list_image_paths, _resolve_checkpoint_path
from metrics import _build_lpips_metric, _save_batch_summary
from pipeline import _infer_single_image


# Run the inference entrypoint for either a single image or a directory of images.
@click.command()
@click.option("--checkpoint", "checkpoint_spec", type=str, required=True, help="Checkpoint .pt path, checkpoint directory, latest, or step")
@click.option("--image", "--input", "input_path", type=str, required=True, help="Path to an input image or an image directory")
@click.option("--output", "output_path", type=str, default="output", help="Directory for saving inference outputs")
def main(
    checkpoint_spec: str,
    input_path: str,
    output_path: str,
) -> None:
    """Run inference for a single image or a directory of images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = _resolve_checkpoint_path(checkpoint_spec)
    model = AIRNet.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()
    lpips_metric = _build_lpips_metric(device)
    if lpips_metric is None:
        click.echo("LPIPS metrics are unavailable: install `lpips` and `torchvision` to enable them.")

    max_stage = max(getattr(model, "head_num", 1) - 1, 0)
    image_paths = _list_image_paths(input_path)
    input_path_obj = Path(input_path)
    is_batch_mode = input_path_obj.is_dir()
    output_root = Path(output_path)

    if not is_batch_mode:
        save_root = output_root / image_paths[0].stem
        info = _infer_single_image(
            model=model,
            image_path=image_paths[0],
            save_dir=save_root,
            device=device,
            max_stage=max_stage,
            lpips_metric=lpips_metric,
        )
        print("Inference info:\n" + _format_json_compact(info))
        return

    batch_root = output_root / input_path_obj.name
    batch_root.mkdir(parents=True, exist_ok=True)
    infos: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            click.echo(f"[{idx}/{len(image_paths)}] Skip non-image file: {image_path.name}")
            continue
        click.echo(f"[{idx}/{len(image_paths)}] Processing {image_path.name}")
        save_dir = batch_root / image_path.stem
        info = _infer_single_image(
            model=model,
            image_path=image_path,
            save_dir=save_dir,
            device=device,
            max_stage=max_stage,
            lpips_metric=lpips_metric,
        )
        infos.append(info)

    summary = _save_batch_summary(
        batch_root=batch_root,
        input_dir=input_path_obj,
        image_paths=image_paths,
        infos=infos,
    )
    print("Batch inference summary:\n" + _format_json_compact(summary))


if __name__ == "__main__":
    main()
