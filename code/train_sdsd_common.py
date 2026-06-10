import argparse
import json
from pathlib import Path

import torch

from sde_lowlight_model import LowLightEnhanceNet
from sdsd_lowlight_data import SDSD_DEFAULT_ROOT, SDSDEventDataset
from train_sde_common import (
    enhancement_loss,
    load_checkpoint,
    make_loader,
    save_checkpoint,
    seed_everything,
    validate,
)


def parse_args(default_subset=None):
    parser = argparse.ArgumentParser(description="Train low-light enhancement on SDSD event data.")
    parser.add_argument("--subset", choices=["indoor", "outdoor"], default=default_subset)
    parser.add_argument("--data-root", default=SDSD_DEFAULT_ROOT)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--event-bins", type=int, default=5)
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
        parser.error("--subset is required when using train_sdsd_common.py directly")
    if args.event_bins <= 0:
        parser.error("--event-bins must be positive")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/sdsd_{args.subset}"
    return args


def main(default_subset=None):
    args = parse_args(default_subset)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )

    train_set = SDSDEventDataset(
        args.data_root,
        args.subset,
        "train",
        crop_size=args.crop_size,
        augment=True,
        event_bins=args.event_bins,
        max_samples=args.max_train_samples,
    )
    val_set = SDSDEventDataset(
        args.data_root,
        args.subset,
        "test",
        crop_size=0,
        augment=False,
        event_bins=args.event_bins,
        max_samples=args.max_val_samples,
    )
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, 1, False, args.num_workers)

    model = LowLightEnhanceNet(
        args.base_channels,
        args.residual_blocks,
        in_channels=args.event_bins * 2,
        out_channels=3,
    ).to(device)
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

    print(
        f"subset={args.subset} train={len(train_set)} val={len(val_set)} "
        f"event_bins={args.event_bins} device={device}"
    )
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
