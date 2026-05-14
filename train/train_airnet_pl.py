import os
import json
from pathlib import Path
from typing import Optional

import click
import sys
import os

# Add project root to Python path so that `import train.*` works when running this file directly
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import  LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from train.pbar import CompactTQDMProgressBar
import torch
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
from train.pl_module import AIRNetLightningModule
from train.dataloader import build_train_dataloader, build_valid_dataloader


@click.command()
@click.option('--config', 'config_path', type=str, default='configs/train_config.json')
@click.option('--workspace', type=str, default='workspace/debug', help='Path to the workspace')
@click.option('--checkpoint', 'checkpoint_path', type=str, default=None, help='Path to the legacy checkpoint to load (e.g., latest or step)')
@click.option('--batch_size_forward', type=int, default=2, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in backbone')
@click.option('--enable_mixed_precision', type=bool, default=True, help='Use mixed precision training')
@click.option('--max_epochs', type=int, default=20, help='Number of epochs to train the model')
@click.option('--num_iterations', type=int, default=10000, help='Number of iterations to train the model')
@click.option('--log_every', type=int, default=20, help='Log metrics every n iterations')
@click.option('--vis_every', type=int, default=2000, help='Visualize every n iterations')
@click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize')
@click.option('--pod', type=bool, default=False, help='Predict-Optimization-Distill strategy')
@click.option('--enable_tensorboard', type=bool, default=True, help='Log metrics to TensorBoard')
@click.option('--seed', type=int, default=114514, help='Random seed')
@click.option('--num_sanity_val_steps', type=int, default=2, help='Number of sanity validation steps run before training starts')
@click.option('--num_nodes', type=int, default=1, help='Number of nodes participating in distributed training')
@click.option('--devices', type=int, default=-1, help='GPUs to use: -1 for all visible, 1 for single GPU')
# Run Lightning training with CLI config loading plus overrides for experimentation.
def main(
    config_path: str,
    workspace: str,
    checkpoint_path: Optional[str],
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    enable_gradient_checkpointing: bool,
    enable_mixed_precision: bool,
    max_epochs: int,
    num_iterations: int,
    log_every: int,
    vis_every: int,
    num_vis_images: int,
    pod: bool,
    enable_tensorboard: bool,
    seed: Optional[int],
    num_sanity_val_steps: int,
    num_nodes: int,
    devices: int
):
    # load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # workspace & config dump
    Path(workspace).mkdir(parents=True, exist_ok=True)
    with Path(workspace, 'config.json').open('w') as f:
        json.dump(config, f, indent=4)

    # pl seed
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    # module
    module = AIRNetLightningModule(
        config=config,
        workspace=workspace,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        pod=pod,
        log_every=log_every,
        vis_every=vis_every,
        num_vis_images=num_vis_images,
        seed=seed,
        checkpoint_path=checkpoint_path,
        num_iterations=num_iterations,
        batch_size_forward=batch_size_forward
    )

    train_loader = build_train_dataloader(config['train_data'], batch_size_forward)
    val_loader = build_valid_dataloader(config['valid_data'], 1)

    # logger
    logger = None
    if enable_tensorboard:
        logger = TensorBoardLogger(save_dir=str(Path(workspace, 'tensorboard_logs')), name='')

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Progress bar only on rank 0 to avoid duplicated rendering from multiple processes
    callbacks = [lr_monitor,
                 CompactTQDMProgressBar(
                    keep_keys=("loss/total_step", "lr_step", "train/RUsage_step", "train/psnr_0_step", "train/psnr_1_step", "train/psnr_2_step", "train/psnr_3_step", "train/psnr_vq_step", "loss/vq_step"),
                    format_map={
                        "loss/total_step": "{:.1e}",
                        "lr_step": "{:.2e}",
                        "train/RUsage_step": "{:.2f}",
                        "train/psnr_0_step": "{:.2f}",
                        "train/psnr_1_step": "{:.2f}",
                        "train/psnr_2_step": "{:.2f}",
                        "train/psnr_3_step": "{:.2f}",
                        "train/psnr_vq_step": "{:.2f}",
                        "loss/vq_step": "{:.1e}"
                    },
                )
    ]
    
    # Trainer
    precision = '16-mixed' if enable_mixed_precision else 32

    if torch.cuda.is_available():
        requested_devices = devices if devices > 0 else "auto"
    else:
        requested_devices = "auto"
        
    strategy_cfg = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=False)

    trainer = pl.Trainer(
        default_root_dir=str(Path(workspace)),
        max_epochs=int(max_epochs),
        limit_train_batches=int(num_iterations),
        check_val_every_n_epoch=1,
        accelerator='gpu' if torch.cuda.is_available() else 'auto',
        devices=requested_devices, 
        num_nodes=int(max(1, num_nodes)),  
        strategy=strategy_cfg,
        accumulate_grad_batches=gradient_accumulation_steps,
        precision=precision,
        log_every_n_steps=log_every,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=callbacks,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    # Fit
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()


