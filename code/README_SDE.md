# SDE Low-Light Training and Inference

These scripts train an RGB low-light enhancement model on paired SDE `low/` and
`normal/` images. Event `.npz` files are not required by this pipeline.

Activate the requested environment first:

```bash
source ~/.bashrc
conda activate light
cd /media/hzho0442/Project/Code/idea/Low-light/eSL-Net/code
```

Train indoor:

```bash
python train_sde_indoor.py --epochs 100 --batch-size 8 --amp
```

Train outdoor:

```bash
python train_sde_outdoor.py --epochs 100 --batch-size 8 --amp
```

Run indoor test inference:

```bash
python test_sde_indoor.py \
  --checkpoint checkpoints/sde_indoor/best.pt \
  --output-dir results/sde_indoor_test
```

Run outdoor test inference:

```bash
python test_sde_outdoor.py \
  --checkpoint checkpoints/sde_outdoor/best.pt \
  --output-dir results/sde_outdoor_test
```

The test scripts save enhanced images under the output directory while preserving
the SDE scene folders. When `normal/` images are available, they also report PSNR
and SSIM.
