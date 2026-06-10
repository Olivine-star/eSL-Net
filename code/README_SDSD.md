# SDSD Event Low-Light Training and Inference

These scripts train and test RGB low-light enhancement from SDSD event `.npz`
inputs. The default dataset root is:

```bash
C:\code\low-light\dataset\sdsd
```

The expected local layout is:

```text
SDSD_in_event/SDSD_in_2_release/{train,test}/pair*/low/*.npz
SDSD_in_event/SDSD_in_2_release/{train,test}/pair*/normal/*.png
SDSD_out_event/SDSD_out_2_release/{train,test}/pair*/low/*.npz
SDSD_out_event/SDSD_out_2_release/{train,test}/pair*/normal/*.png
```

Activate the environment first:

```bash
conda activate light
cd C:\code\low-light\eSL-Net\code
```

Train indoor:

```bash
python train_sdsd_indoor.py --epochs 100 --batch-size 8 --amp
```

Train outdoor:

```bash
python train_sdsd_outdoor.py --epochs 100 --batch-size 8 --amp
```

Run indoor test inference:

```bash
python test_sdsd_indoor.py ^
  --checkpoint checkpoints/sdsd_indoor/best.pt ^
  --output-dir results/sdsd_indoor_test
```

Run outdoor test inference:

```bash
python test_sdsd_outdoor.py ^
  --checkpoint checkpoints/sdsd_outdoor/best.pt ^
  --output-dir results/sdsd_outdoor_test
```

The test scripts save enhanced RGB images under the output directory while
preserving the SDSD `pair*` folders. When `normal/` images are available, they
also report PSNR and SSIM.
