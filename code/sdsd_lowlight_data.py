from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sde_lowlight_data import tensor_from_image


SDSD_DEFAULT_ROOT = r"C:\code\low-light\dataset\sdsd"


def sdsd_subset_root(data_root, subset):
    root = Path(data_root)
    if (root / "train").exists() and (root / "test").exists():
        return root

    if subset == "indoor":
        candidates = [
            root / "SDSD_in_event" / "SDSD_in_2_release",
            root / "SDSD_in_2_release",
            root / "indoor",
        ]
    elif subset == "outdoor":
        candidates = [
            root / "SDSD_out_event" / "SDSD_out_2_release",
            root / "SDSD_out_2_release",
            root / "outdoor",
        ]
    else:
        raise ValueError(f"Unsupported SDSD subset: {subset}")

    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "test").exists():
            return candidate
    raise FileNotFoundError(f"SDSD {subset} release directory not found under {root}")


def _list_npz(directory):
    directory = Path(directory)
    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".npz")


def _list_png(directory):
    directory = Path(directory)
    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".png")


def collect_sdsd_pairs(data_root, subset, split, require_gt=True):
    split_root = sdsd_subset_root(data_root, subset) / split
    if not split_root.exists():
        raise FileNotFoundError(f"SDSD split directory not found: {split_root}")

    samples = []
    for scene_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        low_dir = scene_dir / "low"
        normal_dir = scene_dir / "normal"
        if not low_dir.exists():
            continue

        low_events = _list_npz(low_dir)
        normal_images = _list_png(normal_dir) if normal_dir.exists() else []
        if require_gt and not normal_images:
            raise FileNotFoundError(f"Missing normal/ GT directory for {scene_dir}")
        if normal_images and len(low_events) != len(normal_images):
            raise ValueError(
                f"Pair count mismatch in {scene_dir}: "
                f"{len(low_events)} low event files, {len(normal_images)} normal images"
            )

        if normal_images:
            pairs = zip(low_events, normal_images)
        else:
            pairs = ((low_path, None) for low_path in low_events)

        for low_path, normal_path in pairs:
            samples.append(
                {
                    "low": low_path,
                    "normal": normal_path,
                    "scene": scene_dir.name,
                    "low_name": f"{low_path.stem}.png",
                    "event_name": low_path.name,
                    "normal_name": normal_path.name if normal_path is not None else None,
                }
            )

    if not samples:
        raise RuntimeError(f"No SDSD samples found under {split_root}")
    return samples


def event_tensor_from_npz(event_path, height, width, event_bins=5):
    data = np.load(event_path)
    events = data[data.files[0]] if data.files else np.empty((0, 4), dtype=np.float32)
    channels = event_bins * 2
    voxel = np.zeros((channels, height, width), dtype=np.float32)
    if events.size == 0:
        return torch.from_numpy(voxel)

    events = np.asarray(events)
    timestamps = events[:, 0].astype(np.float64)
    xs = np.clip(events[:, 1].astype(np.int64), 0, width - 1)
    ys = np.clip(events[:, 2].astype(np.int64), 0, height - 1)
    polarities = events[:, 3] > 0

    t_min = timestamps.min()
    t_max = timestamps.max()
    if t_max > t_min:
        bins = ((timestamps - t_min) / (t_max - t_min) * event_bins).astype(np.int64)
        bins = np.clip(bins, 0, event_bins - 1)
    else:
        bins = np.zeros_like(xs)

    channel_ids = bins * 2 + polarities.astype(np.int64)
    np.add.at(voxel, (channel_ids, ys, xs), 1.0)

    max_value = voxel.max()
    if max_value > 0:
        voxel = np.log1p(voxel)
        voxel /= voxel.max()
    return torch.from_numpy(voxel)


def _random_crop_tensors(low, normal, crop_size):
    if crop_size is None or crop_size <= 0:
        return low, normal

    height, width = normal.shape[-2:]
    if width < crop_size or height < crop_size:
        return low, normal

    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    return (
        low[:, top : top + crop_size, left : left + crop_size],
        normal[:, top : top + crop_size, left : left + crop_size],
    )


def _augment_tensors(low, normal):
    if random.random() < 0.5:
        low = torch.flip(low, dims=[2])
        normal = torch.flip(normal, dims=[2])
    if random.random() < 0.5:
        low = torch.flip(low, dims=[1])
        normal = torch.flip(normal, dims=[1])
    return low, normal


class SDSDEventDataset(Dataset):
    def __init__(
        self,
        data_root=SDSD_DEFAULT_ROOT,
        subset="indoor",
        split="train",
        crop_size=256,
        augment=False,
        event_bins=5,
        max_samples=None,
    ):
        self.samples = collect_sdsd_pairs(data_root, subset, split, require_gt=True)
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.crop_size = crop_size
        self.augment = augment
        self.event_bins = event_bins

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        normal_image = Image.open(sample["normal"]).convert("RGB")
        normal = tensor_from_image(normal_image)
        height, width = normal.shape[-2:]
        low = event_tensor_from_npz(sample["low"], height, width, self.event_bins)

        low, normal = _random_crop_tensors(low, normal, self.crop_size)
        if self.augment:
            low, normal = _augment_tensors(low, normal)

        return {
            "low": low.contiguous(),
            "normal": normal.contiguous(),
            "scene": sample["scene"],
            "low_name": sample["low_name"],
            "event_name": sample["event_name"],
            "normal_name": sample["normal_name"],
        }
