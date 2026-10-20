# Nonlinear Direct Feedback Alignment (NDFA)

PyTorch implementation of **Nonlinear Direct Feedback Alignment**: training deep networks without end-to-end backpropagation, by replacing the fixed random feedback matrix of DFA with a *learned-then-frozen nonlinear feedback network*.

---

## Supported architectures

| `--model` | File | Datasets |
| --- | --- | --- |
| `c2f2` | `networks/conv.py` | CIFAR-10/100, SVHN |
| `resnet` | `networks/resnet.py` | CIFAR-10/100, SVHN, ImageNet-32 |
| `alexnet` | `networks/alexnet.py` | CIFAR-10/100 |
| `vits` | `networks/vits.py` | CIFAR-10/100, ImageNet-32 |

`--train_mode BP` runs the same architectures with ordinary backpropagation as a baseline (`c2f2`, `alexnet`, and the ViT training loop).

---

## Repository layout

```
NDFA/
├── NDFA.sh                 # CIFAR/SVHN sweep for c2f2 + ResNet-32
├── run_alexnet.sh          # AlexNet sweep
├── run_vit.sh              # ViT sweep
├── train.py                # entry point: c2f2, resnet
├── train_alexnet.py        # entry point: alexnet
├── train_vit.py            # entry point: vits (timm Mixup/CutMix, cosine + warmup)
├── networks/
│   ├── augdfa_nets.py      # AuxDFANetwork (local head + frozen feedback path), AlignmentProbe
│   ├── conv.py             # ConvNet2FC2
│   ├── resnet.py           # LocalResNet / resnet32
│   ├── resnet32_imagenet.py# ImageNet variant
│   ├── alexnet.py          # AlexNet with 6 aux heads
│   ├── vits.py             # ViT, injection index computed from depth/dfa_num
│   └── configs.py          # (stage, layer, global index) tables per model
└── utils/
    ├── OutputTrainingHook.py  # gradient capture / injection autograd Functions
    └── others.py              # logging setup, adaptive pooling helper
```

Each run writes `log.txt` and `args.json` into a directory named `<timestamp><--name>_<seed>`.

---

## Setup

Tested with Python 3.8 and a recent PyTorch (2.x). The only structural requirement is CUDA for the training loops — `train.py` and `train_alexnet.py` call `.cuda()` unconditionally.

```bash
pip install torch torchvision timm einops numpy datasets
```

`datasets` (HuggingFace) and `torch.autograd.forward_ad` are imported by `train_vit.py` / `train_alexnet.py` but not actually used — the import is still required, so keep `datasets` installed.

## Data

The data root is **hard-coded** as `../image_datasets/<dataset>` relative to the working directory, so the dataset folder must sit next to the repo:

```
workspace/
├── image_datasets/
│   ├── cifar10/          # auto-downloaded by torchvision
│   ├── cifar100/
│   ├── svhn/
│   └── imagenet/
│       ├── train/<class>/...
│       └── val/<class>/...
└── NDFA/
```

CIFAR-10/100 and SVHN download themselves on first run (`download=True`). ImageNet must be provided as `ImageFolder` directories.

---

## Training

Edit `CUDA_VISIBLE_DEVICES` and `MAX_JOBS` at the top of each script, then:

```bash
bash NDFA.sh          # c2f2 on SVHN/CIFAR-10/CIFAR-100 + ResNet-32 
bash run_alexnet.sh   # AlexNet on CIFAR-10/CIFAR-100
bash run_vit.sh       # ViT 
```

Single runs:

```bash
# ConvNet-2FC2, 100 epochs
python train.py --dataset cifar10 --model c2f2 --cos_lr --epochs 100 \
    --train_mode NDFA --name c2f2_NDFA_cifar10

# ResNet-32, 4 injections, 200 epochs
python train.py --dataset cifar100 --model resnet --cos_lr --dfa_num 4 --epochs 200 \
    --train_mode NDFA --name Res32_NDFA4_cifar100

# AlexNet
python train_alexnet.py --dataset cifar10 --cos_lr --train_mode NDFA --lr 5e-4 \
    --name alex_NDFA_cifar10

# ViT, 12 injections
python train_vit.py --dataset cifar10 --model vits --batch_size 256 --train_mode NDFA \
    --name vit_NDFA12_cifar10
```

### Key arguments

| Flag | Meaning |
| --- | --- |
| `--train_mode` | `NDFA` (default) or `BP` for the backprop baseline |
| `--dfa_num` | Number of injections (ResNet: 4 or 16; ViT: 1…depth) |
| `--hidden_dim` | Width of the aux head (default 128) |
| `--epochs`, `--batch_size`, `--lr`, `--wd` | Standard optimisation knobs |
| `--cos_lr` | Cosine schedule instead of step decay |
| `--seed` | Seed appended to the run directory name (default 1234) |
| `--name` | Experiment tag used in the output directory name |
| `--save_path` | Override the auto-generated output directory |
| `--layers` | `32` (ResNet depth; only the `resnet32` factory exists) |

## Citation

No paper is linked from this repository yet. If you use this code, please reference the corresponding NDFA work here (add the BibTeX once available):

```bibtex
@misc{ndfa2026,
  title  = {Nonlinear Direct Feedback Alignment},
  author = {YanFan321},
  year   = {2026},
  url    = {https://github.com/YanFan321/NDFA}
}
```
