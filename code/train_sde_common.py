import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sde_lowlight_data import SDE_DEFAULT_ROOT, SDELowLightDataset
from sde_lowlight_model import LowLightEnhanceNet


def parse_args(default_subset=None):
    parser = argparse.ArgumentParser(description="Train low-light enhancement on SDE.")
    parser.add_argument("--subset", choices=["indoor", "outdoor"], default=default_subset)
    parser.add_argument("--data-root", default=SDE_DEFAULT_ROOT)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--lambda-tv", type=float, default=0.02)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.subset is None:
        parser.error("--subset is required when using train_sde_common.py directly")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/sde_{args.subset}"
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ssim_index(pred, target):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, 3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred * pred, 3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, stride=1, padding=1) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-8)).clamp(0.0, 1.0).mean()


def total_variation_loss(image):
    dy = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    dx = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
    return dx + dy


def enhancement_loss(pred, target, lambda_ssim, lambda_tv):
    l1 = F.l1_loss(pred, target)
    ssim_loss = 1.0 - ssim_index(pred, target)
    tv = total_variation_loss(pred)
    return l1 + lambda_ssim * ssim_loss + lambda_tv * tv, {
        "l1": l1.detach(),
        "ssim_loss": ssim_loss.detach(),
        "tv": tv.detach(),
    }


def psnr_value(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse <= 0.0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0
    for batch in loader:
        low = batch["low"].to(device, non_blocking=True)
        normal = batch["normal"].to(device, non_blocking=True)
        pred = model(low)
        batch_size = low.size(0)
        psnr_sum += psnr_value(pred, normal) * batch_size
        ssim_sum += ssim_index(pred, normal).item() * batch_size
        count += batch_size
    return {"psnr": psnr_sum / count, "ssim": ssim_sum / count}


def save_checkpoint(path, model, optimizer, scaler, epoch, best_psnr, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_psnr": best_psnr,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def main(default_subset=None):
    args = parse_args(default_subset)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )

    train_set = SDELowLightDataset(
        args.data_root,
        args.subset,
        "train",
        crop_size=args.crop_size,
        augment=True,
        max_samples=args.max_train_samples,
    )
    val_set = SDELowLightDataset(
        args.data_root,
        args.subset,
        "test",
        crop_size=0,
        augment=False,
        max_samples=args.max_val_samples,
    )
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, 1, False, args.num_workers)

    model = LowLightEnhanceNet(args.base_channels, args.residual_blocks).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_psnr = -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scaler, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))

    print(f"subset={args.subset} train={len(train_set)} val={len(val_set)} device={device}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "l1": 0.0, "ssim_loss": 0.0, "tv": 0.0}
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            low = batch["low"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=args.amp and device.type == "cuda"
            ):
                pred = model(low)
                loss, parts = enhancement_loss(
                    pred, normal, args.lambda_ssim, args.lambda_tv
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = low.size(0)
            seen += batch_size
            running["loss"] += loss.detach().item() * batch_size
            for key, value in parts.items():
                running[key] += value.item() * batch_size

            if step % 100 == 0:
                print(
                    f"epoch {epoch:03d} step {step:05d}/{len(train_loader):05d} "
                    f"loss={running['loss'] / seen:.5f}"
                )

        train_msg = " ".join(f"{key}={value / seen:.5f}" for key, value in running.items())
        print(f"epoch {epoch:03d} train {train_msg}")

        metrics = None
        if args.val_every > 0 and epoch % args.val_every == 0:
            metrics = validate(model, val_loader, device)
            print(
                f"epoch {epoch:03d} val psnr={metrics['psnr']:.3f} "
                f"ssim={metrics['ssim']:.4f}"
            )
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_checkpoint(
                    checkpoint_dir / "best.pt", model, optimizer, scaler, epoch, best_psnr, args
                )

        save_checkpoint(
            checkpoint_dir / "latest.pt", model, optimizer, scaler, epoch, best_psnr, args
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_psnr,
                args,
            )

    print(f"done best_psnr={best_psnr:.3f} checkpoint_dir={checkpoint_dir}")


if __name__ == "__main__":
    main()
