from typing import Any, Dict, List, Optional, Tuple
import os
import random
import math

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.v2.functional as TF

# Add project root to Python path when running this file directly
try:
    import sys
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
except Exception:
    pass

from train.utils import get_image_files, read_image


class TrainImageDataset(Dataset):
    # Map-style dataset for training. Returns image and metadata for batch-level collate/augmentations.
    def __init__(self, data_config: Dict[str, Any]):
        super().__init__()
        self.data_config = data_config
        self.image_augmentation = data_config.get('image_augmentation', [])
        self.accept_image_format = data_config.get('accept_image_format', None)

        # Only support fixed-size policy via image_sizes (random pick per batch)
        self.image_sizes = data_config.get('image_sizes', None)

        # Flatten items and record per-dataset sampling weights
        items: List[Dict[str, Any]] = []
        self.dataset_weights: Dict[str, float] = {}
        for dataset in data_config['datasets']:
            name = dataset['name']
            self.dataset_weights[name] = float(dataset.get('weight', 1.0))

            files = get_image_files(dataset['path'], recursive=True, accept_image_format=self.accept_image_format)
            try:
                files = sorted(list(files))
            except Exception:
                pass

            for fp in files:
                items.append({
                    'dataset': name,
                    'file_path': fp,
                    'label_type': dataset.get('label_type', 'unknown'),
                })

        self.items = items

    def __len__(self) -> int:
        # Total number of individual images
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load a single image and return metadata for collate/augmentations
        it = self.items[idx]
        img = read_image(it['file_path'])  # uint8 RGB HWC
        return {
            'image': img,
            'dataset': it['dataset'],
            'file_path': it['file_path'],
            'label_type': it['label_type']
        }


class ValidImageDataset(Dataset):
    # Map-style dataset for validation. Deterministic order, no augmentations.
    def __init__(self, data_config: Dict[str, Any]):
        super().__init__()
        self.data_config = data_config
        self.accept_image_format = data_config.get('accept_image_format', None)
        self.area_range = data_config.get('area_range', [160000, 640000])
        self.aspect_ratio_range = data_config.get('aspect_ratio_range', [0.5, 2.0])

        # Only support fixed-size policy via image_sizes (random pick per batch)
        self.image_sizes = data_config.get('image_sizes', None)

        items: List[Dict[str, Any]] = []
        max_valid_num = int(data_config.get('max_valid_num', -1))
        for dataset in data_config['datasets']:
            name = dataset['name']
            files = get_image_files(dataset['path'], recursive=True, accept_image_format=self.accept_image_format)
            try:
                files = sorted(list(files))
            except Exception:
                pass
            rng = random.Random(0)
            if max_valid_num > 0:
                files = rng.sample(files, int(max_valid_num))

            for fp in files:
                width, height = _decide_batch_hw_fixed(self.area_range, self.aspect_ratio_range, rng)
                items.append({
                    'dataset': name,
                    'file_path': fp,
                    'label_type': dataset.get('label_type', 'unknown'),
                    'width': width,
                    'height': height,
                })

        self.items = items

    def __len__(self) -> int:
        # Total number of individual images
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load a single image and return metadata for collate
        it = self.items[idx]
        img = read_image(it['file_path'])  # uint8 RGB HWC
        return {
            'image': img,
            'dataset': it['dataset'],
            'file_path': it['file_path'],
            'label_type': it['label_type'],
            'width': it['width'],
            'height': it['height']
        }


def _decide_batch_hw_fixed(area_range: List[int], aspect_ratio_range: List[float], rng: random.Random) -> Tuple[int, int]:
    area = random.uniform(*area_range)
    aspect_ratio = random.uniform(*aspect_ratio_range)
    width, height = int((area * aspect_ratio) ** 0.5), int((area / aspect_ratio) ** 0.5)
    return width, height

def _apply_train_augmentations(img: np.ndarray, rng_np: np.random.Generator, rng_py: random.Random, image_augmentation: List[str], target_wh: Tuple[int, int]) -> np.ndarray:
    # Apply training augmentations after resizing to the target size.
    tgt_w, tgt_h = target_wh[0], target_wh[1]
    ori_h, ori_w = img.shape[:2]
    if ori_w * ori_h < tgt_w * tgt_h:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_AREA
    out = cv2.resize(img, (tgt_w, tgt_h), interpolation=interpolation)

    # Flip (python RNG)
    if rng_py.random() < 0.5:
        out = np.flip(out, axis=1).copy()

    # Secondary flip (numpy RNG)
    if bool(rng_np.choice([True, False])):
        out = np.flip(out, axis=1).copy()

    # Color jittering
    if 'jittering' in image_augmentation:
        t = torch.from_numpy(out).permute(2, 0, 1).float() / 255.0
        t = TF.adjust_brightness(t, float(rng_np.uniform(0.7, 1.3)))
        t = TF.adjust_contrast(t, float(rng_np.uniform(0.7, 1.3)))
        t = TF.adjust_saturation(t, float(rng_np.uniform(0.7, 1.3)))
        t = TF.adjust_hue(t, float(rng_np.uniform(-0.1, 0.1)))
        t = TF.adjust_gamma(t, float(rng_np.uniform(0.7, 1.3)))
        out = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    # Shot noise
    if 'shot_noise' in image_augmentation:
        if float(rng_np.uniform()) < 0.5:
            k = np.exp(float(rng_np.uniform(np.log(100), np.log(10000)))) / 255.0
            out = (rng_np.poisson(out * k) / k).clip(0, 255).astype(np.uint8)

    # JPEG loss
    if 'jpeg_loss' in image_augmentation:
        if float(rng_np.uniform()) < 0.5:
            q = int(rng_np.integers(20, 100))
            out = cv2.imdecode(cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)

    # Blurring
    if 'blurring' in image_augmentation:
        if float(rng_np.uniform()) < 0.5:
            ratio = float(rng_np.uniform(0.25, 1.0))
            small = cv2.resize(out, (int(tgt_w * ratio), int(tgt_h * ratio)), interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (tgt_w, tgt_h), interpolation=int(rng_py.choice([cv2.INTER_LINEAR_EXACT, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])))

    return out


def _to_tensor_batch(batch_imgs: List[np.ndarray]) -> torch.Tensor:
    # Convert a list of uint8 RGB HWC images to a float tensor [B,3,H,W] in [0,1].
    arr = [torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1) for img in batch_imgs]
    return torch.stack(arr, dim=0)


class TrainCollate:
    # Purpose: Collate training samples with fixed target size and augmentations (multiprocessing-picklable).
    def __init__(self, data_config: Dict[str, Any]) -> None:
        self.area_range = data_config.get('area_range', [160000, 640000])
        self.aspect_ratio_range = data_config.get('aspect_ratio_range', [0.5, 2.0])
        self.image_augmentation = data_config.get('image_augmentation', [])

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            
        seed = torch.initial_seed() % (2**32)
        rng_py = random.Random(seed)
        rng_np = np.random.default_rng(seed)

        width, height = _decide_batch_hw_fixed(self.area_range, self.aspect_ratio_range, rng_py)  # type: ignore[arg-type]

        imgs: List[np.ndarray] = []
        for item in batch:
            img = item['image']
            img_aug = _apply_train_augmentations(img, rng_np, rng_py, self.image_augmentation, (width, height))
            imgs.append(img_aug)

        images = _to_tensor_batch(imgs)
        label_type = [item['label_type'] for item in batch]
        info = [{'dataset': item['dataset'], 'file_path': item['file_path']} for item in batch]
        return {'image': images, 'label_type': label_type, 'info': info}


def make_train_collate_fn(data_config: Dict[str, Any]):
    # Purpose: Return a picklable collate callable instance for training.
    return TrainCollate(data_config)


class ValidCollate:
    # Purpose: Collate validation samples with fixed target size and no augmentations (multiprocessing-picklable).
    def __init__(self, data_config: Dict[str, Any]) -> None:
        self.image_sizes = data_config.get('image_sizes', None)
        self.use_ori_image_size = self.image_sizes is None or len(self.image_sizes) == 0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) > 1:
            raise ValueError('image_sizes is required for validation data when batch size is greater than 1')
        
        ori_h, ori_w = batch[0]['image'].shape[:2]
        # Deterministic per-worker/epoch choice using torch.initial_seed
        tgt_w, tgt_h = batch[0]['width'], batch[0]['height']
        
        if ori_w * ori_h < tgt_w * tgt_h:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA

        imgs = [cv2.resize(item['image'], (tgt_w, tgt_h), interpolation=interpolation) for item in batch]
        images = _to_tensor_batch(imgs)
        label_type = [item['label_type'] for item in batch]
        info = [{'dataset': item['dataset'], 'file_path': item['file_path']} for item in batch]
        return {'image': images, 'label_type': label_type, 'info': info}


def make_valid_collate_fn(data_config: Dict[str, Any]):
    # Purpose: Return a picklable collate callable instance for validation.
    return ValidCollate(data_config)


def seed_worker(worker_id: int) -> None:
    # Purpose: Seed numpy/random deterministically per worker/process (top-level for pickling).
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


class RankZeroOnlySampler(Sampler[int]):
    # Yield the full index range only on global rank 0; other ranks yield nothing.
    def __init__(self, dataset: Dataset[Any]):
        self.dataset = dataset
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.rank = int(torch.distributed.get_rank())
            else:
                self.rank = 0
        except Exception:
            self.rank = 0

    def __iter__(self):
        if self.rank == 0:
            return iter(range(len(self.dataset)))
        else:
            return iter([])

    def __len__(self) -> int:
        return len(self.dataset) if self.rank == 0 else 0


def build_train_dataloader(data_config: Dict[str, Any], batch_size_forward: int) -> DataLoader:
    # Build a standard training DataLoader with weighted sampling and batch-level size policy.
    ds = TrainImageDataset(data_config)

    num_workers = int(data_config.get('num_workers', 8))
    collate_fn = make_train_collate_fn(data_config)
    _loader_kwargs = dict(
        batch_size=int(batch_size_forward),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        _loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **_loader_kwargs)
    return loader


def build_valid_dataloader(data_config: Dict[str, Any], batch_size_forward: int) -> DataLoader:
    # Build a validation DataLoader that can be sharded across DDP ranks.
    ds = ValidImageDataset(data_config)

    num_workers = int(data_config.get('num_workers', 4))
    collate_fn = make_valid_collate_fn(data_config)

    sampler = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            sampler = DistributedSampler(
                ds,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False,
                drop_last=False,
            )
        except RuntimeError:
            sampler = None

    _loader_kwargs = dict(
        batch_size=int(batch_size_forward),
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        _loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(ds, **_loader_kwargs)
    return loader