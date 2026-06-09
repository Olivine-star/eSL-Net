from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SDE_DEFAULT_ROOT = "/media/hzho0442/Project/Code/idea/Low-light/dataset/sde"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sde_subset_root(data_root, subset):
    root = Path(data_root)
    if subset == "indoor":
        return root / "indoor" / "sde_in_release"
    if subset == "outdoor":
        return root / "outdoor" / "sde_out_release"
    raise ValueError(f"Unsupported SDE subset: {subset}")


def list_images(directory):
    directory = Path(directory)
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def tensor_from_image(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def image_from_tensor(tensor):
    tensor = tensor.detach().clamp(0.0, 1.0).cpu()
    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array)


def _random_crop_pair(low_image, normal_image, crop_size):
    if crop_size is None or crop_size <= 0:
        return low_image, normal_image

    width, height = low_image.size
    if width < crop_size or height < crop_size:
        return low_image, normal_image

    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    box = (left, top, left + crop_size, top + crop_size)
    return low_image.crop(box), normal_image.crop(box)


def _augment_pair(low_image, normal_image):
    if random.random() < 0.5:
        low_image = low_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        normal_image = normal_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        low_image = low_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        normal_image = normal_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return low_image, normal_image


def collect_sde_pairs(data_root, subset, split, require_gt=True):
    split_root = sde_subset_root(data_root, subset) / split
    if not split_root.exists():
        raise FileNotFoundError(f"SDE split directory not found: {split_root}")

    samples = []
    for scene_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        low_dir = scene_dir / "low"
        normal_dir = scene_dir / "normal"
        if not low_dir.exists():
            continue

        low_images = list_images(low_dir)
        normal_images = list_images(normal_dir) if normal_dir.exists() else []
        if require_gt and not normal_images:
            raise FileNotFoundError(f"Missing normal/ GT directory for {scene_dir}")
        if normal_images and len(low_images) != len(normal_images):
            raise ValueError(
                f"Pair count mismatch in {scene_dir}: "
                f"{len(low_images)} low images, {len(normal_images)} normal images"
            )

        if normal_images:
            pairs = zip(low_images, normal_images)
        else:
            pairs = ((low_path, None) for low_path in low_images)

        for low_path, normal_path in pairs:
            samples.append(
                {
                    "low": low_path,
                    "normal": normal_path,
                    "scene": scene_dir.name,
                    "low_name": low_path.name,
                }
            )

    if not samples:
        raise RuntimeError(f"No SDE samples found under {split_root}")
    return samples


class SDELowLightDataset(Dataset):
    def __init__(
        self,
        data_root=SDE_DEFAULT_ROOT,
        subset="indoor",
        split="train",
        crop_size=256,
        augment=False,
        max_samples=None,
    ):
        self.samples = collect_sde_pairs(data_root, subset, split, require_gt=True)
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        low_image = Image.open(sample["low"]).convert("RGB")
        normal_image = Image.open(sample["normal"]).convert("RGB")

        low_image, normal_image = _random_crop_pair(
            low_image, normal_image, self.crop_size
        )
        if self.augment:
            low_image, normal_image = _augment_pair(low_image, normal_image)

        return {
            "low": tensor_from_image(low_image),
            "normal": tensor_from_image(normal_image),
            "scene": sample["scene"],
            "low_name": sample["low_name"],
        }
