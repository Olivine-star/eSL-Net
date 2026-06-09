import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from sde_lowlight_data import (
    SDE_DEFAULT_ROOT,
    collect_sde_pairs,
    image_from_tensor,
    tensor_from_image,
)
from sde_lowlight_model import LowLightEnhanceNet
from train_sde_common import ssim_index


def parse_args(default_subset=None):
    parser = argparse.ArgumentParser(description="Run SDE low-light enhancement inference.")
    parser.add_argument("--subset", choices=["indoor", "outdoor"], default=default_subset)
    parser.add_argument("--data-root", default=SDE_DEFAULT_ROOT)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if args.subset is None and args.input_dir is None:
        parser.error("--subset is required when using test_sde_common.py directly")
    if args.checkpoint is None and args.subset is not None:
        args.checkpoint = f"checkpoints/sde_{args.subset}/best.pt"
    if args.output_dir is None:
        name = args.subset if args.subset is not None else "custom"
        args.output_dir = f"results/sde_{name}_{args.split}"
    return args


def load_model(args, device):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train first or pass --checkpoint."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    base_channels = int(saved_args.get("base_channels", args.base_channels))
    residual_blocks = int(saved_args.get("residual_blocks", args.residual_blocks))
    model = LowLightEnhanceNet(base_channels, residual_blocks).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    return model


def collect_custom_inputs(input_dir):
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    return [
        {"low": path, "normal": None, "scene": path.parent.name, "low_name": path.name}
        for path in paths
    ]


def psnr_value(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse <= 0.0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        samples = collect_custom_inputs(args.input_dir)
    else:
        samples = collect_sde_pairs(args.data_root, args.subset, args.split, require_gt=False)
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]

    psnr_sum = 0.0
    ssim_sum = 0.0
    metric_count = 0

    for index, sample in enumerate(samples, start=1):
        low_image = Image.open(sample["low"]).convert("RGB")
        low = tensor_from_image(low_image).unsqueeze(0).to(device)
        pred = model(low).squeeze(0)

        scene_dir = output_dir / sample["scene"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        image_from_tensor(pred).save(scene_dir / sample["low_name"])

        if sample["normal"] is not None:
            normal_image = Image.open(sample["normal"]).convert("RGB")
            normal = tensor_from_image(normal_image).to(device)
            psnr_sum += psnr_value(pred.unsqueeze(0), normal.unsqueeze(0))
            ssim_sum += ssim_index(pred.unsqueeze(0), normal.unsqueeze(0)).item()
            metric_count += 1

        if index % 200 == 0:
            print(f"processed {index}/{len(samples)}")

    print(f"saved {len(samples)} enhanced images to {output_dir}")
    if metric_count:
        print(
            f"metrics on {metric_count} paired images: "
            f"psnr={psnr_sum / metric_count:.3f} ssim={ssim_sum / metric_count:.4f}"
        )


def main(default_subset=None):
    args = parse_args(default_subset)
    run_inference(args)


if __name__ == "__main__":
    main()
