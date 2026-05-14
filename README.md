# AIR: Amortized Image Reconstruction Framework for Self-Supervised Feed-Forward 2D Gaussian Splatting

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Paper](https://img.shields.io/badge/Paper-AIR-b31b1b.svg)](#citation)

This repository contains the official implementation of **AIR: Amortized Image Reconstruction Framework for Self-Supervised Feed-Forward 2D Gaussian Splatting**.

AIR addresses image reconstruction with 2D Gaussian splatting. It predicts a compact Gaussian representation in a single feed-forward pass, avoiding the costly per-image iterative optimization used by previous Gaussian-based image representation methods.

The main contributions are:

- A self-supervised feed-forward framework for amortized 2D Gaussian image reconstruction.
- A stage-wise residual prediction design with Stage Control, which adds Gaussians only to under-reconstructed regions.
- A Predict-Optimize-Distill training strategy that stabilizes Gaussian prediction, followed by finetuning and image-adaptive quantization for compact storage.

For a complete understanding of the method, please refer to our paper: **AIR: Amortized Image Reconstruction Framework for Self-Supervised Feed-Forward 2D Gaussian Splatting**. [TODO: add paper/arXiv link]

## Overview

The pipeline progressively reconstructs an image from an empty canvas. At each stage, AIR predicts Gaussian increments from the current residual, uses Stage Control to keep only necessary primitives, and accumulates the activated Gaussians into the final representation.

<div align="center">
  <img src="./assets/pipeline.png" alt="AIR pipeline" />
</div>
<sub>
<strong>Overview of AIR.</strong> Overview of AIR. AIR amortizes the iterative optimization of GaussianImage through self-supervised stage-wise residual prediction. At each stage, the network predicts Gaussian increments from the current reconstruction residual, while an explicit stage-control mechanism activates additional primitives only in under-reconstructed regions. During POD pretraining, a short-horizon optimizer refines a detached copy of each predicted increment and distills the refined increment back as Gaussian-space supervision. After POD stabilizes the predictor, cross-stage rendering finetuning supervises the accumulated reconstructions directly, promoting stage coordination for the final output.
</sub>

## Quick Start

### Setup

Clone the repository and create the conda environment:

```bash
git clone https://github.com/whoiszzj/AIR.git
cd AIR
conda create -n AIR python=3.12
conda activate AIR
pip install -r requirements.txt
mkdir -p checkpoints data output workspace
```

### Data & Checkpoints

Prepare your image dataset and checkpoints:

```bash
./data
├── kodak
│   ├── kodim01.png
│   ├── kodim02.png
│   └── ...
└── 0844x2.png

./checkpoints
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── ps_5.pt
├── ps_6.pt
├── ps_7.pt
└── ps_14.pt
```

[TODO: Add checkpoint download link and place downloaded `.pt` files under `checkpoints/`.]

## Run a Simple Demo

Run inference on a single image with the default script:

```bash
bash infer.sh
```

The default inference script expects:

```bash
CHECKPOINT=checkpoints/ps_7.pt
IMAGE=data/0844x2.png
OUTPUT=output
```

You can also call the Python entrypoint directly:

```bash
python test/infer.py \
  --checkpoint checkpoints/ps_7.pt \
  --image data/0844x2.png \
  --output output
```

The `--image` argument accepts either a single image file or a directory containing supported image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`).

The output structure for a single image is:

```bash
./output
└── 0844x2
    ├── info.json
    ├── image.jpg
    ├── render_stage_*.jpg
    ├── error_stage_*.png
    ├── mse_error_stage_*.png
    ├── ssim_error_stage_*.png
    ├── image_points.png
    ├── position_density.png
    └── position_density.npy
```

If quantization is enabled in the checkpoint, the output may also include:

```bash
render_quant.jpg
```

The `info.json` file records metrics such as PSNR, MS-SSIM, LPIPS when available, Gaussian counts, timing, and quantization statistics when applicable.
Note that the inference script runs the same image 10 times for timing. For practical use, you can comment out that timing block to improve per-image processing speed. You can also disable unnecessary visualization outputs.

## Test on Your Own Dataset

1. Put images into a directory, for example:

```bash
./data/my_images
├── image_001.png
├── image_002.png
└── image_003.png
```

2. Run batch inference:

```bash
python test/infer.py \
  --checkpoint checkpoints/ps_7.pt \
  --image data/my_images \
  --output output
```

3. Results will be saved as:

```bash
./output
└── my_images
    ├── summary.json
    ├── image_001
    │   ├── info.json
    │   ├── image.jpg
    │   └── render_stage_*.jpg
    ├── image_002
    └── image_003
```

## Training

### Configuration

Training uses JSON config files under `configs/`:

```bash
./configs
├── 0_pod_config.json
├── 1_finetune_config.json
├── 2_finetune_quantize_config.json
├── 3_div2k_refine_config.json
└── 4_patch_size_finetune_config.json
```

Before training, update dataset paths in the selected config file:

```json
{
  "train_data": {
    "datasets": [
      {
        "name": "[TODO: dataset name]",
        "path": "[TODO: path to training images]",
        "weight": 1.0,
        "label_type": "normal"
      }
    ]
  },
  "valid_data": {
    "datasets": [
      {
        "name": "[TODO: validation dataset name]",
        "path": "[TODO: path to validation images]",
        "weight": 1.0,
        "label_type": "normal"
      }
    ]
  }
}
```

### Start Training

Run the provided training script:

```bash
bash train.sh
```

The script currently launches a short POD training/debug run:

```bash
python train/train_airnet_pl.py \
  --config configs/0_pod_config.json \
  --workspace workspace/debug \
  --batch_size_forward 5 \    # Batch size per GPU
  --gradient_accumulation_steps 1 \
  --enable_gradient_checkpointing True \
  --enable_mixed_precision False \
  --pod True \
  --max_epochs 12 \   # Number of epochs
  --num_iterations 10000 \  # Maximum iterations per epoch; if the number of training samples is smaller, use the size of the training set
  --log_every 20 \ # Log to TensorBoard every N steps
  --vis_every 2000 \  # Run visualization every N iterations; images are selected from the val set
  --num_vis_images 32 \ # Number of images per visualization
  --enable_tensorboard True \
  --seed 0
```

After POD pretraining, launch the finetuning stage with `configs/1_finetune_config.json`. The `--checkpoint` argument should point to the checkpoint produced by the previous POD stage:

```bash
python train/train_airnet_pl.py \
  --config configs/1_finetune_config.json \
  --workspace workspace/finetune \
  --checkpoint workspace/pod/checkpoint/latest.pt \
  --batch_size_forward 2 \
  --gradient_accumulation_steps 1 \
  --enable_gradient_checkpointing True \
  --enable_mixed_precision True \
  --pod False \
  --max_epochs 30 \
  --num_iterations 10000 \
  --log_every 20 \
  --vis_every 2000 \
  --num_vis_images 32 \
  --seed 0
```

For multi-node training on a SLURM cluster, we also provide `train_slurm.sh` as a DDP launch template:

```bash
sbatch train_slurm.sh
```

Before submitting the job, update the `TODO` fields in `train_slurm.sh`, including the partition, project path, conda environment, config file, workspace, and checkpoint path.

### Checkpoints and Logs

Training outputs are saved under the selected workspace:

```bash
./workspace/debug
├── config.json
├── checkpoint
│   ├── latest.pt
│   ├── 00000000.pt
│   ├── 00000000_optimizer.pt
│   └── ...
└── tensorboard_logs
    └── ...
```

Launch TensorBoard with:

```bash
tensorboard --logdir workspace/debug/tensorboard_logs
```

## Train Your Own Checkpoints

A typical workflow is:

1. Prepare ImageNet-style or large-scale natural image training data for POD pretraining.
2. Prepare DIV2K or another high-resolution target dataset for refinement.
3. Edit the dataset paths in `configs/*.json`.
4. Start from `configs/0_pod_config.json` with `--pod True`.
5. Fine-tune with `configs/1_finetune_config.json`.
6. Enable quantization-aware training with `configs/2_finetune_quantize_config.json` if compact storage is required.
7. Adapt patch size with `configs/4_patch_size_finetune_config.json` when a different Gaussian density or bitrate operating point is needed.
8. Refine on the target dataset with `configs/3_div2k_refine_config.json`.
9. Use the generated checkpoint with `test/infer.py`.

## Repository Structure

```bash
.
├── configs                 # Training configuration files
├── model                   # AIRNet, Gaussian heads, quantization, rendering utilities
├── train                   # PyTorch Lightning training pipeline
├── test                    # Inference, metrics, visualization, and I/O helpers
├── submodules              # Local dependencies: gsplat and fused-ssim
├── train.sh                # Example training command
├── run_slurm.sh            # SLURM multi-node DDP training template
├── infer.sh                # Example inference command
└── requirements.txt        # Python dependencies
```

## Acknowledgments

This work is supported by the Natural Science Foundation of China under Grant 62302174. The computation is completed in the HPC Platform of Huazhong University of Science and Technology. We also thank Farsee2 Technology Ltd for providing devices to support the validation of our method.

Thanks to the following open-source projects and libraries:

1. [GaussianImage](https://github.com/Xinjie-Q/GaussianImage)
2. [Instant-GI](https://github.com/whoiszzj/Instant-GI.git)
3. [Image-GS](https://github.com/NYU-ICL/image-gs.git)
4. [gsplat](https://github.com/nerfstudio-project/gsplat)
5. [DINOv3](https://github.com/facebookresearch/dinov3)
6. [PyTorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning)
7. [Torch-KDTree](https://github.com/thomgrand/torch_kdtree.git)

## Citation

If you find this work useful or relevant to your research, please cite:

```bibtex
 [TODO: add paper citation]
```

## License

This project is released under the MIT License. [TODO: add the LICENSE file if it is not included in the repository.]

## BUG?

This library is under active development, and there may be bugs or incomplete documentation. If you encounter any issues, please feel free to open an issue or submit a pull request.
