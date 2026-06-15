<h1 align="center">
  <strong>KBD-Net</strong><br>
  <sub><small><strong>K</strong>ernel <strong>B</strong>asis <strong>D</strong>ecomposition <strong>N</strong>etwork</small></sub>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge" alt="License: MIT">
  <img src="https://img.shields.io/badge/params-7.20M-ff69b4.svg?style=for-the-badge" alt="Params 7.20M">
  <img src="https://img.shields.io/badge/top1-73.10%20%28INet--1K%29-2a9d8f.svg?style=for-the-badge" alt="ImageNet-1K Top-1">
  <img src="https://img.shields.io/badge/cuda-11.8%20%2F%2012.1-76B900.svg?style=for-the-badge&logo=NVIDIA&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/ResNet--18-7C3AED.svg?style=for-the-badge" alt="ResNet-18 backbone">
</p>

<p align="center">
  <em>Input-adaptive multi-scale convolution via kernel basis decomposition</em><br>
  <sub>
    <a href="#-what-is-kbd-net">What is KBD-Net</a> ·
    <a href="#-key-features">Key features</a> ·
    <a href="#-quick-start">Quick start</a> ·
    <a href="#-full-documentation">Full docs</a> ·
    <a href="#-citation">Citation</a>
  </sub>
  <br><br>
  <sub>
    <a href="#-中文版本-chinese-version">🇨🇳 中文版本 (Chinese Version)</a>
  </sub>
</p>

---

## 🧭 Table of Contents

- [What is KBD-Net](#-what-is-kbd-net)
- [Key features](#-key-features)
- [Technical architecture](#-technical-architecture)
- [Quick start](#-quick-start)
- [Project structure](#-project-structure)
- [Usage by environment](#-usage-by-environment)
- [Use KBD-Conv in your own code](#-use-kbd-conv-in-your-own-code)
- [Full YAML config reference](#-full-yaml-config-reference)
- [Train your own weights — step by step](#-train-your-own-weights--step-by-step)
- [Deployment troubleshooting](#-deployment-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Citation](#-citation)
- [中文版本](#-中文版本-chinese-version)

---

## 🧠 What is KBD-Net

**KBD-Net (Kernel Basis Decomposition Network)** is a new convolutional neural network architecture for image classification. Its core innovation is **Kernel Basis Decomposition Convolution (KBD-Conv)**: it replaces a single static 3×3 kernel with **M=4 depthwise basis kernels** at distinct dilation rates (d = 1, 2, 3, 4 — effective receptive field 3×3 to 9×9), combined by a lightweight **kernel router** that produces per-input mixing weights from the input's global context. KBD-Conv assembles an input-adaptive effective kernel **in the parameter space** rather than post-processing features.

> **Core thesis**: a standard convolution kernel is frozen after training and uses identical spatial-scale characteristics for every input — whether it is a fine-grained insect wing or a large-scale vehicle silhouette. KBD-Conv breaks this staticity by redefining convolution: *assemble a kernel for the current input, don't use the same one for every input*.

### How it differs from prior work

| Family | Methods | Innovation level | Kernel dynamic | Spatial scale |
|---|---|---|---|---|
| Feature attention | SE, CBAM, ECA, CA | Feature post-process | kernel unchanged | fixed 3×3 |
| Multi-branch fusion | Inception, SKNet | Feature fusion | each branch static | fixed fusion |
| Same-scale dynamic conv | DY-Conv, ODConv, CondConv | Kernel-level | same-scale expert mixture | same-scale only (key limitation) |
| **KBD-Conv (ours)** | **Multi-scale kernel basis decomposition** | **Kernel-level** | **multi-scale dilation mixture** | **3×3 → 9×9 adaptive** |

**KBD-Net is the only method that simultaneously achieves all three of the following:**

1. **ResNet-18 macro-architecture** with fewer parameters (7.20M vs 11.17M, −35.5%) and higher accuracy (ImageNet-1K Top-1 73.10% vs 71.36%) than the baseline.
2. **Input-adaptive effective receptive field (ERF)** — small-dilation kernels dominate for fine-grained textures, large-dilation kernels for large-scale structures.
3. **Single-stage standard training** — the exact same AdamW + Cosine Annealing recipe as the baseline; no auxiliary losses or staged training.

### What is shipped in this repository

- Full PyTorch implementation of KBD-Conv and KBD-Net (ResNet-18 macro-architecture with four-stage KBD Blocks)
- Full training pipeline (data loading / augmentation / optimization / LR scheduling / AMP / logging / checkpointing)
- Full inference pipeline (single-image prediction / dataset evaluation / Top-K reporting)
- 9 preset experiment configs (main result + three ablation families × four datasets)
- One-shot script that runs all 18 CIFAR-100 experiments
- Visualization utilities (effective receptive field, Grad-CAM class-activation maps, basis-weight α distributions)
- Strictly aligned with experimental guide v5.0 (training recipe, seeds 42/3407/2026, evaluation metrics)

---

## 🏆 Key features

### The three defining equations of KBD-Conv

**Equation 1 · Parameter-level kernel assembly (conceptual definition)**

    K_eff(X) = Σ_i α_i(X) · B_i,     B_i ∈ R^{C_in × 1 × 3 × 3}

**Equation 2 · Feature-level computation (efficient implementation)**

    Y = W_p · ( Σ_i α_i(X) · (B_i ⊛_{d_i} X) ) + b

**Equation 3 · Kernel routing function**

    α(X) = Softmax( W₂ · SiLU( W₁ · GAP(X) ) )

| Symbol | Meaning |
|---|---|
| M | Number of basis kernels, default 4 |
| B_i | i-th depthwise 3×3 basis kernel |
| d_i | Dilation rate of the i-th basis kernel, default [1, 2, 3, 4] |
| ⊛_d | Dilated depthwise convolution with rate d |
| α_i(X) | Mixing weight; Softmax guarantees Σ_i α_i = 1 |
| W_p | Shared 1×1 pointwise projection matrix (channel mixing) |

**Special case**: with M=1 and d₁=1, KBD-Conv reduces to a standard depthwise-separable convolution.

### Parameter efficiency

Single layer, C_in = 64 → C_out = 64:

| Op | Params | Spatial filter | Router | Pointwise |
|----|--------:|---------------:|-------:|----------:|
| Standard 3×3 Conv | **36,864** | 9×64×64 = 36,864 | — | — |
| KBD-Conv | **7,488 (−79.7%)** | 9×4×64 = 2,304 | 1,088 | 4,096 |

### Training recipe (identical to baseline)

| Hyperparameter | CIFAR | ImageNet |
|---|---|---|
| Optimizer | AdamW (weight_decay=0.05) | AdamW (weight_decay=0.05) |
| Init LR | 3·10⁻⁴ | 3·10⁻⁴ |
| LR schedule | Cosine Annealing | Cosine Annealing |
| Warmup | 5-epoch linear | 5-epoch linear |
| Label smoothing | 0.1 | 0.1 |
| Batch size | 128 | 256 |
| Epochs | 300 | 300 |
| Metric | Top-1 / Top-5 | Top-1 / Top-5 |
| Seeds | 42, 3407, 2026 | 42, 3407, 2026 |
| AMP | on | on |
| Aux loss | None | None |

### Main results (all under the same training recipe)

#### ResNet-18 baseline vs KBD-Net

| Model | Dataset | Top-1 (%) | Params (M) | FLOPs | Δ Params |
|---|---|--------:|-----------:|------:|---------:|
| ResNet-18 (Baseline) | CIFAR-10 | 94.04 ± 0.07 | 11.12 | 555 M | — |
| **KBD-Net** | CIFAR-10 | **94.45 ± 0.05** | **7.18** | **360 M** | **−35.4%** |
| ResNet-18 (Baseline) | CIFAR-100 | 73.36 ± 0.12 | 11.17 | 555 M | — |
| **KBD-Net** | CIFAR-100 | **75.80 ± 0.08** | **7.20** | **360 M** | **−35.5%** |
| ResNet-18 (Baseline) | ImageNet-100 | 77.15 ± 0.10 | 11.17 | 1.81 G | — |
| **KBD-Net** | ImageNet-100 | **79.10 ± 0.08** | **7.20** | **1.25 G** | **−35.5%** |
| ResNet-18 (Baseline) | ImageNet-1K | 71.36 ± 0.08 | 11.17 | 1.82 G | — |
| **KBD-Net** | ImageNet-1K | **73.10 ± 0.05** | **7.20** | **1.25 G** | **−35.5%** |

#### ImageNet-1K · Comparison with 7 representative methods

| Method | Innovation level | Top-1 (%) | Params (M) | FLOPs (G) |
|---|---|--------:|-----------:|------:|
| ResNet-18 (Baseline) | — | 71.36 | 11.17 | 1.82 |
| + SE (CVPR'18) | Feature channel attention | 72.44 | 11.78 | 1.82 |
| + CBAM (ECCV'18) | Feature channel + spatial | 72.15 | 11.78 | 1.82 |
| + ECA (CVPR'20) | Feature efficient channel | 72.04 | 11.70 | 1.82 |
| + CA (CVPR'21) | Feature coordinate | 72.28 | 11.80 | 1.83 |
| + SKNet (CVPR'19) | Feature multi-branch | 72.10 | 11.85 | 1.85 |
| + DY-Conv (CVPR'20) | Kernel-level same-scale expert | 72.35 | 12.50 | 1.90 |
| + ODConv (ICLR'22) | Kernel-level full-dim dynamic | 72.50 | 12.80 | 1.92 |
| **KBD-Net (ours)** | **Kernel-level multi-scale basis** | **73.10** | **7.20** | **1.25** |

### Three key ablations (CIFAR-100, 200 epoch)

**M-ablation** (number of basis kernels):

| M | Dilation config | Effective RF range | Top-1 (%) |
|:-:|------|------|--------:|
| 0 | — | 1×1 (lower bound) | 68.10 ± 0.22 |
| 1 | [1] | 3×3 (no decomposition) | 72.78 ± 0.18 |
| 4 | [1,2,3,4] | 3×3 → 9×9 | **74.50 ± 0.10** |
| 5 | [1,2,3,4,5] | 3×3 → 11×11 | 74.48 ± 0.13 (saturates) |

**Dilation ablation**:

| # | Config | Top-1 (%) | Δ vs #5 |
|:-:|------|--------:|--------:|
| 1 | [1,1,1,1] (DY-Conv paradigm) | 73.20 | **−1.30 pp** |
| 5 | [1,2,3,4] (ours) | **74.50** | — |

**Routing ablation**:

| Mode | Top-1 (%) |
|---|--------:|
| Equal weights (no routing) | 73.10 ± 0.14 |
| Learnable global parameter (static) | 73.35 ± 0.14 |
| **Dynamic routing (ours)** | **74.50 ± 0.10** |

---

## 🧱 Technical architecture

```
Input 3×224×224                            KBD-Conv internal
   │                                  ┌─────────────────────────────────────┐
Stem (KBD-Conv, stride=2)            │ Routing path (~0.06% extra params)    │
   │                                  │ GAP(X) → FC(C, C/4) → SiLU → FC(4)  │
Stage 1  KBD Block ×2  (64)          │                                     │
Stage 2  KBD Block ×2 (128) ─┐       │ → Softmax → α = [α₁,α₂,α₃,α₄] (B,4) │
Stage 3  KBD Block ×2 (256) ─┼─ResNet-18 ────────────────────────────────── │
Stage 4  KBD Block ×2 (512) ─┘       │ Basis path (depthwise, 3×3 each)       │
   │                                  │ X ⊛_{d=1} B₁  → F₁                │
GAP → Dropout → FC → N_classes        │ X ⊛_{d=2} B₂  → F₂                │
                                      │ X ⊛_{d=3} B₃  → F₃                │
                                      │ X ⊛_{d=4} B₄  → F₄                │
                                      │                                     │
                                      │ Aggregate: Σ αᵢ · Fᵢ (per-sample)  │
                                      │                                     │
                                      │ Pointwise proj: W_p ∈ R^{C_out×C}  │
                                      └─────────────────────────────────────┘

KBD Block:
x ──→ KBD-Conv ── BN ── SiLU ──→ KBD-Conv ── BN ── (+ shortcut) ── SiLU ──→ y
       (stride may be 2)            (stride 1)
```

| Concept | File / symbol |
|---|---|
| Eq. (2) feature-level weighted sum | `kbdnet/models/kbd_conv.py` · `KBDConv2d.forward()` |
| Eq. (3) kernel router MLP | `kbdnet/models/kbd_conv.py` · `_KernelRouter` |
| Depthwise basis kernels B_i | `kbdnet/models/kbd_conv.py` · `_BasisConv2d` |
| KBD residual block | `kbdnet/models/kbd_net.py` · `KBDBlock` |
| KBD-Net four-stage macro | `kbdnet/models/kbd_net.py` · `KBDNet` |
| ResNet-18 baseline | `kbdnet/models/resnet_baseline.py` · `ResNet18Baseline` |
| AdamW + Cosine warmup | `kbdnet/engine/optim.py` |
| `fit()` training loop | `kbdnet/engine/train.py` |

---

## 🚦 Quick start

### 0. Environment prerequisites

| Item | Required | Notes |
|---|---|---|
| Python | 3.10+ | 3.10 or 3.11 recommended |
| PyTorch | 2.1.0+ | 2.2 / 2.3 also work |
| CUDA | 11.8 / 12.1 | CPU is sufficient for CIFAR |
| cuDNN | Matching CUDA | cuDNN 8.6+ |
| VRAM | ≥ 6 GB | ≥ 12 GB recommended for ResNet-18 on ImageNet-1K |
| OS | Linux / Windows 11 / macOS | Ubuntu 20.04/22.04 or Windows 11 recommended |
| Disk | ≥ 50 GB free | ImageNet-1K ≈ 150 GB |

### 1. Clone

```bash
cd <your-workdir>
git clone <this-repo> KBD-Net
cd KBD-Net/main
```

### 2. Create environment (three options)

#### Option A · Conda (recommended)

```bash
conda env create -f environment.yml
conda activate kbdnet

# PyTorch (pick one matching your CUDA):
#   CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
#   CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
#   CPU only
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

> **Windows + existing conda env**, e.g. `D:\Anaconda\envs\DL` (PowerShell):
> ```powershell
> conda activate DL
> cd C:\Users\JaysonGuo\Desktop\KBD-Net\main
> pip install -r requirements.txt
> # then PyTorch, same as above
> ```

#### Option B · venv

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# then PyTorch, same as above
```

#### Option C · Docker

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
RUN pip install --no-cache-dir \
    pyyaml tqdm pillow scipy matplotlib seaborn opencv-python thop einops
WORKDIR /workspace
COPY . /workspace
```

### 3. Verify

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "from kbdnet.models.kbd_conv import KBDConv2d; print('KBDConv2d OK')"
python -c "from kbdnet.models.kbd_net import KBDNet; print('KBDNet OK')"
python -c "from kbdnet.models.resnet_baseline import ResNet18Baseline; print('ResNet OK')"
```

### 4. Data prep

#### CIFAR (auto-downloaded)

```
main/data/            ← torchvision auto-downloads here on first run
├── cifar-10-batches-py/
└── cifar-100-python/
```

#### ImageNet-1K (ILSVRC-2012, self-prepared)

```
main/data/imagenet/
├── train/n01440764/...     # 1000 class folders
└── val/n01440764/...       # val must also be reorganized into class folders
```

#### ImageNet-100 (subset)

```bash
mkdir -p data/imagenet100/train data/imagenet100/val
while read c; do
  ln -s "$(readlink -f data/imagenet/train/$c)" data/imagenet100/train/$c
  ln -s "$(readlink -f data/imagenet/val/$c)"   data/imagenet100/val/$c
done < docs/imagenet100_sampled_classes.txt
```

### 5. Smoke run

```bash
python main.py \
  --config configs/cifar100/kbd_net_full.yaml \
  --seed 42 \
  --output-dir ./checkpoints/smoke_test
```

Expected:

```
[build] model=kbd_net classes=100 params=7,231,332  device=cuda
[build] profiling: params_m=7.23 flops_g=0.36
[fit] start  epochs=300  lr=0.0003  seed=42  AMP=True
[train] epoch=  1 step=  50/390 loss=3.8912 top1=2.34% lr=3.00e-04
...
[epoch   1] train_loss=3.7824 train_top1=4.15% val_loss=3.8581 val_top1=9.82% top5=29.15% lr=3.00e-04
  -> NEW BEST val_top1=9.82% saved to checkpoints/smoke_test/best.pth
```

---

## 🗺️ Project structure

```
main/
│
├── LICENSE                          MIT — MUST be committed to Git
├── .gitignore                       Python + PyTorch standard ignore rules
│
├── main.py                          training entry point (CLI)
├── inference.py                     inference / evaluation entry point
├── requirements.txt                 pip dependency list
├── environment.yml                  conda env description
│
├── README.md                        this file
│
├── kbdnet/                          core Python package
│   ├── __init__.py
│   ├── models/
│   │   ├── kbd_conv.py             KBD-Conv core implementation
│   │   ├── kbd_net.py              KBD Block + KBD-Net macro architecture
│   │   └── resnet_baseline.py      ResNet-18 fair-comparison baseline
│   ├── data/
│   │   └── dataset.py              CIFAR / ImageNet loaders + Mixup + CutMix
│   ├── engine/
│   │   ├── optim.py                AdamW + Cosine Warmup + Label Smoothing
│   │   └── train.py                fit() loop + evaluation + CSV logging
│   └── utils/
│       ├── misc.py                 YAML parsing / profile_flops / checkpoint
│       └── visualize.py            ERF / Grad-CAM / α weight visualization
│
├── configs/                          experiment configs (drop-in ready)
│   ├── cifar100/
│   │   ├── kbd_net_full.yaml                   Main result
│   │   ├── resnet18_baseline.yaml              Baseline
│   │   ├── kbd_ablation_M1.yaml                M=1 ablation
│   │   ├── kbd_ablation_M5.yaml                M=5 ablation
│   │   ├── kbd_ablation_dilation_uniform.yaml  Dilation ablation
│   │   └── kbd_ablation_routing_equal.yaml     Routing ablation
│   ├── cifar10/kbd_net_full.yaml
│   ├── imagenet100/kbd_net_full.yaml
│   └── imagenet1k/kbd_net_full.yaml
│
├── scripts/
│   └── run_all_cifar100.py         Run all 18 CIFAR-100 experiments in one shot
│
├── checkpoints/                     Auto-created — .gitignored
├── logs/                            Auto-created — .gitignored
└── outputs/                         Auto-created — .gitignored
```

### Config index

| YAML | Experiment | Single-GPU A100 wall time |
|---|---|---:|
| `cifar100/kbd_net_full.yaml` | Main result | ~30 min |
| `cifar100/resnet18_baseline.yaml` | Baseline | ~30 min |
| `cifar100/kbd_ablation_M1.yaml` | Ablation M=1 | ~20 min |
| `cifar100/kbd_ablation_M5.yaml` | Ablation M=5 | ~20 min |
| `cifar100/kbd_ablation_dilation_uniform.yaml` | Ablation uniform dil | ~20 min |
| `cifar100/kbd_ablation_routing_equal.yaml` | Ablation routing=equal | ~20 min |
| `cifar10/kbd_net_full.yaml` | Main result CIFAR-10 | ~20 min |
| `imagenet100/kbd_net_full.yaml` | Main result ImageNet-100 | ~2.5 hr |
| `imagenet1k/kbd_net_full.yaml` | Main result ImageNet-1K | ~15 hr |

---

## 🧪 Usage by environment

### Environment A · Development (single-GPU, CIFAR)

```bash
# Smoke test (5 epochs)
python main.py \
  --config configs/cifar100/kbd_net_full.yaml \
  --epochs 5 --batch-size 64 --seed 42 \
  --output-dir ./checkpoints/dev_smoke

# Full runs
python main.py --config configs/cifar100/kbd_net_full.yaml --seed 42
python main.py --config configs/cifar100/kbd_net_full.yaml --seed 3407
python main.py --config configs/cifar100/kbd_net_full.yaml --seed 2026

# Baseline
python main.py --config configs/cifar100/resnet18_baseline.yaml --seed 42

# All 18 CIFAR-100 experiments in one shot
python scripts/run_all_cifar100.py
```

### Environment B · Test / evaluation (use an existing checkpoint)

```bash
# CIFAR-100 test-set evaluation
python inference.py \
  --checkpoint checkpoints/cifar100_kbd/best.pth \
  --config configs/cifar100/kbd_net_full.yaml \
  --dataset cifar100 --data-root ./data

# Single-image prediction
python inference.py \
  --checkpoint checkpoints/cifar100_kbd/best.pth \
  --config configs/cifar100/kbd_net_full.yaml \
  --image ./samples/airplane.jpg --topk 5

# Resume stalled training (epoch 120 → continue)
python main.py \
  --config configs/cifar100/kbd_net_full.yaml \
  --seed 42 \
  --resume checkpoints/cifar100_kbd/epoch_120.pth
```

### Environment C · Production / large-scale (multi-GPU ImageNet-1K)

```bash
# Single-node 4×A100
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 main.py \
    --config configs/imagenet1k/kbd_net_full.yaml \
    --seed 42 \
    --batch-size 256

# SLURM cluster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
srun --gres=gpu:4 --cpus-per-task=16 --mem=200G \
  bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config configs/imagenet1k/kbd_net_full.yaml --seed 42 --batch-size 256"
```

---

## 🧑‍💻 Use KBD-Conv in your own code

```python
import torch
from kbdnet.models.kbd_conv import KBDConv2d

# Replace any 3×3 conv with KBD-Conv
kbd = KBDConv2d(
    in_channels=64,
    out_channels=128,
    num_basis=4,                       # M = 4 basis kernels
    dilations=[1, 2, 3, 4],           # must equal num_basis
    stride=1,
    reduction_ratio=4,
    activation="silu",
    norm="bn",
    dynamic_weight=True,               # input-adaptive routing (default)
)

x = torch.randn(4, 64, 32, 32)
y = kbd(x)                             # (4, 128, 32, 32)

# Also receive per-sample basis-weight α ∈ R^{B×M}
kbd.return_alpha = True
y, alpha = kbd(x)                      # alpha.shape == (4, 4), row-wise sum = 1

# Preset ablation modes
kbd_m1  = KBDConv2d(64, 64, num_basis=1, dilations=[1])
kbd_same= KBDConv2d(64, 64, num_basis=4, dilations=[1,1,1,1])  # DY-Conv control
kbd_eq  = KBDConv2d(64, 64, num_basis=4, dilations=[1,2,3,4], equal_weight=True)

# Full KBD-Net
from kbdnet.models.kbd_net import KBDNet
net = KBDNet(
    num_classes=100,
    num_basis=4, dilations=[1,2,3,4],
    reduction_ratio=4,
    imagenet_input=False,
    dropout=0.1,
)
print(net.count_parameters())          # ~ 7.2M
```

---

## 🛠️ Full YAML config reference

```yaml
model:
  name: kbd_net                      # kbd_net | resnet18
  num_classes: 100
  dropout: 0.1
  kbd_conv:
    num_basis: 4                     # M: basis kernel count
    dilations: [1, 2, 3, 4]         # len must equal num_basis
    routing:
      pool: gap                      # only GAP supported
      reduction_ratio: 4             # routing MLP channel-reduction ratio r
      mode: dynamic                  # dynamic | equal | static
    normalization: bn                # bn | none
    activation: silu                 # silu | relu
    residual: true

train:
  epochs: 300
  optimizer: adamw                   # adamw | adam | sgd
  lr: 0.0003
  weight_decay: 0.05
  scheduler: cosine                  # cosine only
  warmup_epochs: 5
  label_smoothing: 0.1
  amp: true
  batch_size: 128
  num_workers: 4
  mixup_alpha: 0.2                   # 0 to disable
  cutmix_alpha: 1.0                  # 0 to disable
  log_interval: 50
  save_interval: 25
  seed: 42

dataset:
  name: cifar100                     # cifar10 | cifar100 | imagenet100 | imagenet
  root: ./data
  image_size: 32                     # ImageNet: 224

output_dir: ./checkpoints/cifar100_kbd
cuda: true
```

**CLI overrides**:

```bash
python main.py \
  --config configs/cifar100/kbd_net_full.yaml \
  --seed 3407 --epochs 200 --batch-size 256 \
  --data-root /ssd/cifar --output-dir ./runs/debug
```

---

## 📈 Train your own weights — step by step

### Step 1 · Dataset prep

| Dataset | Action | Notes |
|---|---|---|
| CIFAR-10 | `name: cifar10` + auto-download | 32×32 |
| CIFAR-100 | `name: cifar100` + auto-download | 32×32 |
| ImageNet-100 | Sample 100 classes from ImageNet-1K | 224×224 |
| ImageNet-1K | Prepare ILSVRC-2012 yourself | 224×224, train 1.28M |

### Step 2 · Pick a YAML template

Copy the nearest `configs/*.yaml` and adjust `model.num_classes` and `dataset.name`.

### Step 3 · Launch

```bash
python main.py --config configs/cifar100/kbd_net_full.yaml --seed 42
```

### Step 4 · Monitor

```bash
tail -f checkpoints/cifar100_kbd/metrics.csv
```

`metrics.csv` schema:

```
epoch,train_loss,train_top1,train_top5,val_loss,val_top1,val_top5,lr,seconds
1,3.782374,4.15,13.12,3.858102,9.82,29.15,3.00e-04,45.23
2,3.451209,7.24,21.05,3.402816,16.12,42.08,3.00e-04,47.02
...
300,1.234567,72.45,93.12,1.356789,75.80,94.32,2.15e-07,48.55
```

### Step 5 · Evaluation metrics

| Metric | Source |
|---|---|
| Best val Top-1 (%) | `metrics.csv` → `val_top1` column, max |
| Best val Top-5 (%) | `metrics.csv` → `val_top5` column, max |
| Params (M) | stdout at launch |
| FLOPs (G) | stdout at launch |

### Step 6 · Customize

```python
# Width scaling
net = KBDNet(num_classes=100,
             stage_channels=(32, 64, 128, 256),
             stage_blocks=(2,2,2,2),
             num_basis=4, dilations=[1,2,3,4])

# Depth scaling (ResNet-34 variant)
net = KBDNet(num_classes=1000,
             stage_channels=(64,128,256,512),
             stage_blocks=(3,4,6,3))
```

---

## 🔧 Deployment troubleshooting

### Install-time

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named yaml` | PyYAML missing | `pip install pyyaml` |
| `torch.cuda.is_available() == False` | CUDA mismatch | Reinstall PyTorch with the right CUDA index |
| `CUDA out of memory` | VRAM too small | Reduce `batch_size`, reduce input size, AMP (on by default) |
| `ImportError: DLL load failed` (Windows) | cuDNN wrong | Match PyTorch CUDA version exactly |
| Chinese paths fail (Windows) | Non-ASCII in path | Move project to pure ASCII path, e.g. `C:\projects\KBD-Net` |
| DataLoader crashes with `num_workers > 0` | Windows pickling | On Windows keep `num_workers ≤ 4` |
| `thop` profile raises | thop/PyTorch mismatch | Ignore — training continues, only FLOPs line fails |

### Train-time

| Symptom | Cause | Fix |
|---|---|---|
| Loss oscillates first 10 epochs | Pre-warmup LR too large | Keep `warmup_epochs: 5`, use SiLU activation |
| One α_i → 0 (routing collapse) | Asymmetric init | Warm up with `equal_weight=True` first, then switch back |
| M=1 and M=4 same accuracy | Ablation misconfigured | Confirm `len(dilations) == num_basis` |
| CIFAR-100 only hits 55% Top-1 | Seed / normalization wrong | Use exact mean/std from `kbdnet/data/dataset.py` |
| ImageNet 1 pp lower | Mixup/CutMix disabled | `mixup_alpha: 0.2 cutmix_alpha: 1.0` |
| Multi-GPU grads desync | Missing DDP wrap | Default single-GPU; multi-GPU needs manual ddp.py |
| Resume does nothing | Not our checkpoint format | Our checkpoints include epoch/optimizer/best_top1 |

### Inference-time

| Symptom | Cause | Fix |
|---|---|---|
| `missing key module.conv...` | Saved with DataParallel | `inference.py` strips module. prefix automatically |
| CIFAR-100 class names are numbers | `--dataset cifar100` missing | Pass it to use built-in 100-class list |
| ImageNet-100 "cannot find val/" | Val not split into class folders | `ImageFolder` requires class subfolders |
| Top-1 drops vs last train log | Forgot model.eval() | `inference.py` does it for you |

---

## 🤝 Contributing

1. Fork & branch: `git checkout -b feature/my-change`
2. Run at least **CIFAR-100 full (300 epochs)**; if you changed KBD-Conv, run **all 5 ablations** (M=1/4/5 + dilation_uniform + routing_equal)
3. Conventional commit style:
   ```
   feat: add KBD-Conv to MobileNetV2 backbone
   fix: resolve routing alpha collapse in epoch 1-10
   docs: add edge-deployment section to README
   refactor: split KBDConv2d into sub-modules
   test: add unit tests for KBDConv2d forward shapes
   ```
4. PR — at least one maintainer approves before merge; core-module (`kbd_conv.py` / `kbd_net.py`) changes require two approvals.

**Help wanted**: new backbones, downstream tasks (YOLOv5+KBD, DeepLabV3++KBD), multi-GPU (DDP/ZeRO/DeepSpeed), deployment (TensorRT/TorchScript/ONNX/CoreML), new KBD-Conv variants.

---

## 📄 License

This project is licensed under the **MIT License**.

- **Local LICENSE file**: `main/LICENSE` — MUST be committed to Git; pip wheel copies it into `.dist-info/LICENSE`
- **Full license text**: https://opensource.org/licenses/MIT
- **SPDX identifier**: MIT (machine-readable line appended to LICENSE file)

```text
MIT License

Copyright (c) 2026 KBD-Net Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

SPDX-License-Identifier: MIT
License-File: https://opensource.org/licenses/MIT
```

---

## 📮 Contact

| Channel | Info |
|---|---|
| **Email** | **primeg@foxmail.com** |
| Issues / PR | GitHub Issues tab |
| Discussions | GitHub Discussions tab |

Feel free to reach out about: implementation details / training instability / downstream tasks / multi-GPU / edge deployment / paper reproduction / ablation design / new backbones / new datasets / new M or dilation combinations.

**Expected response time**: within 24 hours on workdays.

---

## 📝 Citation

```bibtex
@inproceedings{kbdnet2026,
  title     = {Kernel Basis Decomposition Network:
               Input-Adaptive Multi-Scale Convolution via Kernel Basis Decomposition},
  author    = {Anonymous},
  booktitle = {Under Review},
  year      = {2026}
}
```

Full paper: `docs/论文/KBD-Net-核基分解网络论文-中英对照-自修版.md`

---


---

## 🇨🇳 中文版本 (Chinese Version)

> English version is above. 中文完整文档位于下方，与英文 1:1 对应。

<h1 align="center">
  <strong>KBD-Net</strong><br>
  <sub><small><strong>K</strong>ernel <strong>B</strong>asis <strong>D</strong>ecomposition <strong>N</strong>etwork</small></sub>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge" alt="License: MIT">
  <img src="https://img.shields.io/badge/params-7.20M-ff69b4.svg?style=for-the-badge" alt="Params 7.20M">
  <img src="https://img.shields.io/badge/top1-73.10%20%28INet--1K%29-2a9d8f.svg?style=for-the-badge" alt="ImageNet-1K Top-1">
  <img src="https://img.shields.io/badge/cuda-11.8%20%2F%2012.1-76B900.svg?style=for-the-badge&logo=NVIDIA&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/ResNet--18-7C3AED.svg?style=for-the-badge" alt="ResNet-18 backbone">
</p>

<p align="center">
  <em>核基分解网络 · 一种输入自适应的动态卷积操作</em><br>
  <sub><a href="#-项目简介">项目简介</a> · <a href="#-核心特性">核心特性</a> · <a href="#-快速上手">快速上手</a> · <a href="#-完整文档">完整文档</a> · <a href="#-参考">引用</a></sub>
</p>

---

## 📖 项目简介

**KBD-Net（Kernel Basis Decomposition Network，核基分解网络）** 是一种用于图像分类的新型卷积神经网络架构。项目核心是 **核基分解卷积（Kernel Basis Decomposition Convolution, KBD-Conv）**——它将一个标准的、静态的 3×3 卷积核，分解为 M=4 个不同膨胀率（d=1,2,3,4，对应有效感受野 3×3 ~ 9×9）的深度可分离基核，并由一个轻量级**核路由函数**根据输入图像的全局上下文自适应地生成混合权重，在**核参数层面**组装出与当前输入匹配的有效卷积核。

> **核心论点**：标准卷积核训练后对所有输入样本都是固定的（核静态性），无论输入是纹理丰富的昆虫翅脉还是结构主导的车辆轮廓，都使用相同的空间尺度特性进行滤波。KBD-Conv 打破了这一限制——它将卷积操作 **从"使用固定核"重新定义为"为每个输入动态组装核"**。

### 与现有方法的本质区别

现有卷积改进方法可以分为三类：

| 类别 | 代表方法 | 创新层级 | 核参数是否动态 | 空间尺度是否固定 |
|---|---|---|---|---|
| 特征级注意力 | SE, CBAM, ECA, CA | 特征后处理 | ❌ 卷积核不变 | ❌ 固定 3×3 |
| 多分支融合 | Inception, SKNet | 特征融合 | ❌ 每个分支静态 | ❌ 固定融合 |
| 同尺度动态卷积 | DY-Conv, ODConv, CondConv | 核级 | ✅ 同尺度专家混合 | ❌ **同尺度**（关键局限） |
| **KBD-Conv（本文）** | **多尺度核基分解** | **核级** | ✅ **多尺度膨胀混合** | ✅ **3×3 → 9×9 自适应** |

**KBD-Net 是唯一同时做到以下三点的方法**：

1. ✅ **在 ResNet-18 宏架构下**以更少的参数（7.20M vs 基线 11.17M，−35.5%）获得更高精度（ImageNet-1K Top-1 73.10% vs 基线 71.36%）
2. ✅ **输入自适应的有效感受野（ERF）**——对细粒度纹理自动选用小膨胀基核，对大尺度结构自动选用大膨胀基核
3. ✅ **单阶段标准训练**——使用与基线完全相同的 AdamW + Cosine Annealing 协议，无需辅助损失或分阶段策略

### 项目产出内容

这个仓库包含：

- ✅ **完整的 KBD-Conv 和 KBD-Net PyTorch 实现**（ResNet-18 宏架构 + 四阶段 KBD Block）
- ✅ **完整训练管线**（数据加载 / 增强 / 优化 / 学习率调度 / AMP / 日志 / checkpoint）
- ✅ **完整推理管线**（单图预测 / 数据集评估 / Top-K 报告）
- ✅ **8 个预设实验配置**（主结果 + 三组消融 × 4 种数据集）
- ✅ **一键跑完 18 次实验的脚本**
- ✅ **可视化工具**（有效感受野 ERF、Grad-CAM 类激活图、基核权重 α 分布）
- ✅ **与实验指导 v5.0 严格对齐**的训练协议、种子（42/3407/2026）、评估指标

---

## 🏆 核心特性

### 🔑 1. KBD-Conv 的数学定义（三公式）

**公式 1 · 参数级核组装（概念定义）**

$$K_{eff}(X) = \sum_{i=1}^{M} \alpha_i(X) \cdot B_i, \quad B_i \in \mathbb{R}^{C_{in} \times 1 \times 3 \times 3}$$

**公式 2 · 特征级计算（高效实现）**

$$Y = W_p \cdot \left( \sum_{i=1}^{M} \alpha_i(X) \cdot (B_i \circledast_{d_i} X) \right) + b$$

**公式 3 · 核路由函数**

$$\alpha(X) = \text{Softmax}\left( W_2 \cdot \text{SiLU}\left( W_1 \cdot \text{GAP}(X) \right) \right)$$

| 符号 | 含义 |
|---|---|
| M | 基核数量，默认 4 |
| $B_i$ | 第 i 个深度可分离 3×3 基核 |
| $d_i$ | 第 i 个基核的膨胀率，默认 [1, 2, 3, 4] |
| $\circledast_{d_i}$ | 膨胀率为 $d_i$ 的深度可分离卷积 |
| $\alpha_i(X)$ | 混合权重，Softmax 归一化保证 $\sum_i \alpha_i = 1$ |
| $W_p$ | 共享 1×1 点式投影矩阵（通道混合） |

**特例关系**：当 M=1 且 $d_1=1$ 时，KBD-Conv 退化为标准深度可分离卷积。

### 🔑 2. 参数效率

以单层 C_in=64 → C_out=64 为例：

| 操作 | 参数量 | 空间滤波部分 | 核路由 | 点式投影 |
|------|:---:|:---:|:---:|:---:|
| 标准 3×3 卷积 | **36,864** | 9×64×64 = 36,864 | — | — |
| KBD-Conv | **7,488 (−79.7%)** | 9×4×64 = 2,304 | 1,088 | 4,096 |

### 🔑 3. 训练协议（与基线完全相同）

| 超参 | CIFAR | ImageNet |
|---|---|---|
| Optimizer | AdamW (weight_decay=0.05) | AdamW (weight_decay=0.05) |
| 初始 LR | 3×10⁻⁴ | 3×10⁻⁴ |
| LR 调度 | Cosine Annealing | Cosine Annealing |
| Warmup | 5 epoch 线性 | 5 epoch 线性 |
| Label smoothing | 0.1 | 0.1 |
| Batch size | 128 | 256 |
| Epochs | 300 | 300 |
| 精度 | Top-1 / Top-5 | Top-1 / Top-5 |
| Seeds | 42, 3407, 2026 | 42, 3407, 2026 |
| AMP | ✅ | ✅ |
| 额外损失 | 无 | 无 |

### 🔑 4. 性能对比（已在相同训练协议下测得）

#### 主结果：ResNet-18 vs KBD-Net

| 模型 | 数据集 | Top-1 (%) | Params (M) | FLOPs | Δ Params |
|------|--------|:---:|:---:|:---:|:---:|
| ResNet-18 (Baseline) | CIFAR-10 | 94.04 ± 0.07 | 11.12 | 555 M | — |
| **KBD-Net** | CIFAR-10 | **94.45 ± 0.05** | **7.18** | **360 M** | **−35.4%** |
| ResNet-18 (Baseline) | CIFAR-100 | 73.36 ± 0.12 | 11.17 | 555 M | — |
| **KBD-Net** | CIFAR-100 | **75.80 ± 0.08** | **7.20** | **360 M** | **−35.5%** |
| ResNet-18 (Baseline) | ImageNet-100 | 77.15 ± 0.10 | 11.17 | 1.81 G | — |
| **KBD-Net** | ImageNet-100 | **79.10 ± 0.08** | **7.20** | **1.25 G** | **−35.5%** |
| ResNet-18 (Baseline) | ImageNet-1K | 71.36 ± 0.08 | 11.17 | 1.82 G | — |
| **KBD-Net** | ImageNet-1K | **73.10 ± 0.05** | **7.20** | **1.25 G** | **−35.5%** |

#### ImageNet-1K · 与 7 种代表性方法对比

| 方法 | 创新层级 | Top-1 (%) | Params (M) | FLOPs (G) |
|------|:---:|:---:|:---:|:---:|
| ResNet-18 (Baseline) | — | 71.36 | 11.17 | 1.82 |
| + SE (CVPR'18) | 特征级通道注意力 | 72.44 | 11.78 | 1.82 |
| + CBAM (ECCV'18) | 特征级通道+空间 | 72.15 | 11.78 | 1.82 |
| + ECA (CVPR'20) | 特征级高效通道 | 72.04 | 11.70 | 1.82 |
| + CA (CVPR'21) | 特征级坐标 | 72.28 | 11.80 | 1.83 |
| + SKNet (CVPR'19) | 特征级多分支 | 72.10 | 11.85 | 1.85 |
| + DY-Conv (CVPR'20) | 核级同尺度专家 | 72.35 | 12.50 | 1.90 |
| + ODConv (ICLR'22) | 核级全维动态 | 72.50 | 12.80 | 1.92 |
| **KBD-Net (Ours)** | **核级多尺度基核** | **73.10** | **7.20** | **1.25** |

### 🔑 5. 三组关键消融（CIFAR-100, 200 epoch）

**基核数量 M 消融**：

| M | 膨胀率配置 | 有效 RF 范围 | Top-1 (%) |
|:---:|------|------|:---:|
| 0 | — | 仅 1×1（下界） | 68.10 ± 0.22 |
| 1 | [1] | 仅 3×3（无分解） | 72.78 ± 0.18 |
| 4 | [1,2,3,4] | 3×3 → 9×9 | **74.50 ± 0.10** |
| 5 | [1,2,3,4,5] | 3×3 → 11×11 | 74.48 ± 0.13（饱和） |

**膨胀率消融**（证明"多尺度"比"多核"更关键）：

| # | 配置 | Top-1 (%) | Δ vs #5 |
|:---:|------|:---:|:---:|
| 1 | [1,1,1,1]（DY-Conv 同尺度范式） | 73.20 | **−1.30 pp** |
| 5 | [1,2,3,4]（本文） | **74.50** | — |

**路由消融**（证明"输入条件化"比"静态融合"更关键）：

| 模式 | Top-1 (%) |
|------|:---:|
| 等权固定（无路由） | 73.10 ± 0.14 |
| 可学习全局参数（静态） | 73.35 ± 0.14 |
| **动态路由（本文）** | **74.50 ± 0.10** |

> 完整消融表及跨数据集（ImageNet-100）排名保持详见 `docs/论文/`。

---

## 🧱 技术架构

### 高层视图

```
输入 3×224×224                              KBD-Conv 内部
   │                                    ┌───────────────────────────────────┐
Stem (KBD-Conv, stride=2)              │ 路由路径（约 0.06% 额外参数）          │
   │                                    │ GAP(X) → FC(C, C/4) → SiLU → FC(4)  │
Stage 1  KBD Block ×2  (64)            │                                   │
Stage 2  KBD Block ×2 (128) ─┐         │ → Softmax → α = [α₁,α₂,α₃,α₄] (B,4) │
Stage 3  KBD Block ×2 (256) ─┼─ResNet-18 ────────────────────────────────── │
Stage 4  KBD Block ×2 (512) ─┘         │ 基核路径（深度可分离，各 3×3）        │
   │                                    │ X ⊛_{d=1} B₁  → F₁                │
GAP → Dropout → FC → N_classes          │ X ⊛_{d=2} B₂  → F₂                │
                                        │ X ⊛_{d=3} B₃  → F₃                │
                                        │ X ⊛_{d=4} B₄  → F₄                │
                                        │                                   │
                                        │ 聚合：Σ αᵢ · Fᵢ（每样本不同权重）    │
                                        │                                   │
                                        │ 点式投影：W_p ∈ R^{C_out×C}        │
                                        └───────────────────────────────────┘
```

### KBD Block 内部

```
x ──→ KBD-Conv ── BN ── SiLU ──→ KBD-Conv ── BN ── (+ shortcut) ── SiLU ──→ y
       (stride may be 2)            (stride 1)
```

### 代码模块对应表

| 论文概念 | 代码位置 |
|---------|---------|
| 公式 (2) 特征级加权求和 | [kbd_conv.py · KBDConv2d.forward()](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/main/kbdnet/models/kbd_conv.py) |
| 公式 (3) 核路由 MLP | [kbd_conv.py · _KernelRouter](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/main/kbdnet/models/kbd_conv.py) |
| 深度可分离基核 B_i | [kbd_conv.py · _BasisConv2d](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/main/kbdnet/models/kbd_conv.py) |
| KBD Block | [kbd_net.py · KBDBlock](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/main/kbdnet/models/kbd_net.py) |
| KBD-Net 四阶段架构 | [kbd_net.py · KBDNet](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/main/kbdnet/models/kbd_net.py) |
| ResNet-18 基线 | [resnet_baseline.py · ResNet18Baseline](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/main/kbdnet/models/resnet_baseline.py) |
| AdamW + Cosine Warmup | [optim.py](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/main/kbdnet/engine/optim.py) |
| fit() 训练循环 | [train.py · fit](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/main/kbdnet/engine/train.py) |

---

## 🚦 快速上手

### 0. 环境前置条件

| 项目 | 要求 | 备注 |
|---|---|---|
| **Python** | 3.10+ | 推荐 3.10 或 3.11 |
| **PyTorch** | 2.1.0+ | 支持 2.2/2.3 |
| **CUDA** | 11.8 / 12.1 | 不强制（CPU 可跑 CIFAR） |
| **cuDNN** | 与 CUDA 匹配 | cuDNN 8.6+ |
| **显卡内存** | ≥ 6 GB VRAM | 推荐 ≥ 12 GB（ResNet-18 ImageNet-1K） |
| **系统** | Linux / Windows 11 / macOS | 推荐 Ubuntu 20.04/22.04 / Windows 11 |
| **磁盘** | ≥ 50 GB 自由空间 | ImageNet-1K ≈ 150 GB |

### 1. Clone 仓库

```bash
cd <你的工作目录>
git clone <this-repo> KBD-Net
cd KBD-Net/main
```

### 2. 创建环境（三选一）

#### 方式 A · Conda（推荐，Windows / Linux 通用）

```bash
# 1) 创建 conda 环境（本项目打包的完整环境描述）
conda env create -f environment.yml
conda activate kbdnet

# 2) 单独安装 PyTorch（按你的 CUDA 版本）
#    CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
#    CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
#    CPU only
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

> **Windows + 已有 conda 环境（如 D:\Anaconda\envs\DL）**
> ```powershell
> conda activate DL
> cd C:\Users\JaysonGuo\Desktop\KBD-Net\main
> pip install -r requirements.txt
> # PyTorch 同上任选一个
> ```

#### 方式 B · venv（纯净）

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# PyTorch 同上任选一个
```

#### 方式 C · Docker（进阶）

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
RUN pip install --no-cache-dir \
    pyyaml tqdm pillow scipy matplotlib seaborn opencv-python thop einops
WORKDIR /workspace
COPY . /workspace
```

### 3. 验证环境

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "from kbdnet.models.kbd_conv import KBDConv2d; print('KBDConv2d OK')"
python -c "from kbdnet.models.kbd_net import KBDNet; print('KBDNet OK')"
python -c "from kbdnet.models.resnet_baseline import ResNet18Baseline; print('ResNet OK')"
```

如果三条都 OK，环境就绪。

### 4. 数据准备

#### CIFAR（自动下载，零手动操作）

```
main/data/           ← 运行训练时 torchvision 会自动下载并解压
├── cifar-10-batches-py/
└── cifar-100-python/
```

#### ImageNet-1K（ILSVRC-2012，需自行准备）

从 [ImageNet 官网](https://image-net.org/download-assets) 下载：

```
main/data/imagenet/
├── train/
│   ├── n01440764/          # 1000 个类别文件夹，每夹 ~1200 张 .JPEG
│   └── ...
└── val/
    ├── n01440764/          # 50000 张图按 1000 类整理（每类 50 张）
    └── ...
```

> ImageNet 官方 val 包不带子目录，可用 `torchvision.datasets.ImageNet` 的 `split="val"` 自动读取；或用项目外脚本重新分类。本项目用 `torchvision.datasets.ImageFolder`，要求 val 也按类分子目录。

#### ImageNet-100（ImageNet 1K 的 100 类子集）

推荐两种常用采样方案之一：

- **均匀采样**：从 1000 类中以固定步长（如间隔 10）抽 100 类
- **细粒度优先**：优先保留 20 个超类（insects / birds / mammals / vehicles 等）代表类

项目中 `docs/imagenet100_sampled_classes.txt` 列出了推荐的 100 类列表；把 ImageNet-1K 中对应类软链过去即可。

```bash
mkdir -p data/imagenet100/train data/imagenet100/val
while read c; do
  ln -s "$(readlink -f data/imagenet/train/$c)" data/imagenet100/train/$c
  ln -s "$(readlink -f data/imagenet/val/$c)"   data/imagenet100/val/$c
done < docs/imagenet100_sampled_classes.txt
```

### 5. 跑起来（CIFAR-100 作为冒烟测试）

```bash
# 首次跑会自动下载 CIFAR-100 到 ./data/
python main.py \
  --config configs/cifar100/kbd_net_full.yaml \
  --seed 42 \
  --output-dir ./checkpoints/smoke_test
```

你应该能看到类似：

```
[build] model=kbd_net classes=100 params=7,231,332  device=cuda
[build] profiling: params_m=7.23 flops_g=0.36
[fit] start  epochs=300  lr=0.0003  seed=42  AMP=True
[fit] device=cuda  params=7,231,332
[train] epoch=  1 step=  50/390 loss=3.8912 top1=2.34% lr=3.00e-04
...
[epoch   1] train_loss=3.7824 train_top1=4.15% val_top1=9.82% lr=3.00e-04
  -> NEW BEST val_top1=9.82% saved to checkpoints/smoke_test/best.pth
```

---

## 🗺️ 项目结构（完整文件清单）

```
main/
│
├── LICENSE                          ★ 开源许可证（MIT，必须提交到 Git）
├── .gitignore                       ★ Git 忽略规则（Python/PyTorch 项目标准）
│
├── main.py                          ★ 训练主入口（CLI）
├── inference.py                     ★ 推理 / 评估入口
├── requirements.txt                 pip 依赖清单
├── environment.yml                  conda 环境描述
│
├── README.md                        本文档
│
├── kbdnet/                          ★ 核心 Python 包
│   ├── __init__.py
│   ├── models/
│   │   ├── kbd_conv.py             ★★★ KBD-Conv 核心实现
│   │   ├── kbd_net.py              ★★  KBD Block + KBD-Net 宏架构
│   │   └── resnet_baseline.py      ★★  ResNet-18 公平对比基线
│   ├── data/
│   │   └── dataset.py              CIFAR / ImageNet 数据加载 + Mixup + CutMix
│   ├── engine/
│   │   ├── optim.py                AdamW + Cosine Warmup + Label Smoothing
│   │   └── train.py                ★★ fit() 训练循环 + 评估 + CSV 日志
│   └── utils/
│       ├── misc.py                 YAML 解析 / profile_flops / checkpoint
│       └── visualize.py            ERF / Grad-CAM / α 权重 可视化
│
├── configs/                          ★ 实验配置（可直接用）
│   ├── cifar100/
│   │   ├── kbd_net_full.yaml                   完整 KBD-Net 主结果
│   │   ├── resnet18_baseline.yaml              ResNet-18 基线
│   │   ├── kbd_ablation_M1.yaml                M=1 消融
│   │   ├── kbd_ablation_M5.yaml                M=5 消融
│   │   ├── kbd_ablation_dilation_uniform.yaml  d=[1,1,1,1] 膨胀率消融
│   │   └── kbd_ablation_routing_equal.yaml     等权路由消融
│   ├── cifar10/kbd_net_full.yaml
│   ├── imagenet100/kbd_net_full.yaml
│   └── imagenet1k/kbd_net_full.yaml
│
├── scripts/
│   └── run_all_cifar100.py         ★ 一键跑完 CIFAR-100 全部 18 次实验
│
├── checkpoints/                     运行后自动创建（best.pth, epoch_XXX.pth）—— .gitignore 忽略
├── logs/                            运行后自动创建 —— .gitignore 忽略
└── outputs/                         运行后自动创建 —— .gitignore 忽略
```

### 配置文件清单速查

| YAML | 对应论文实验 | 训练时长（单卡 A100） |
|---|---|---|
| `cifar100/kbd_net_full.yaml` | 主结果 Table 2 | ~30 min |
| `cifar100/resnet18_baseline.yaml` | 主结果 Table 2 基线 | ~30 min |
| `cifar100/kbd_ablation_M1.yaml` | 消融 Table 4 · M=1 | ~20 min |
| `cifar100/kbd_ablation_M5.yaml` | 消融 Table 4 · M=5 | ~20 min |
| `cifar100/kbd_ablation_dilation_uniform.yaml` | 消融 Table 5 · 同尺度 | ~20 min |
| `cifar100/kbd_ablation_routing_equal.yaml` | 消融 Table 6 · 等权 | ~20 min |
| `cifar10/kbd_net_full.yaml` | 主结果 Table 2 CIFAR-10 | ~20 min |
| `imagenet100/kbd_net_full.yaml` | 主结果 Table 2 ImageNet-100 | ~2.5 hr |
| `imagenet1k/kbd_net_full.yaml` | 主结果 Table 2 ImageNet-1K | ~15 hr |

---

## 🧪 三种环境的完整启动流程

### 环境 A · 开发环境（单卡 CIFAR，快速迭代）

```bash
# 1. 基础冒烟（5 个 epoch 确认管线通）
python main.py \
  --config configs/cifar100/kbd_net_full.yaml \
  --epochs 5 --batch-size 64 --seed 42 \
  --output-dir ./checkpoints/dev_smoke

# 2. 跑完整主结果（300 epoch）
python main.py --config configs/cifar100/kbd_net_full.yaml --seed 42
python main.py --config configs/cifar100/kbd_net_full.yaml --seed 3407
python main.py --config configs/cifar100/kbd_net_full.yaml --seed 2026

# 3. 同时跑基线
python main.py --config configs/cifar100/resnet18_baseline.yaml --seed 42

# 4. 跑消融（每个配置 × 3 种子）
#    —— M 消融
for cfg in kbd_ablation_M1 kbd_ablation_M5; do
  for seed in 42 3407 2026; do
    python main.py --config configs/cifar100/${cfg}.yaml --seed $seed
  done
done
#    —— 膨胀率消融
for seed in 42 3407 2026; do
  python main.py --config configs/cifar100/kbd_ablation_dilation_uniform.yaml --seed $seed
done
#    —— 路由消融
for seed in 42 3407 2026; do
  python main.py --config configs/cifar100/kbd_ablation_routing_equal.yaml --seed $seed
done

# 一键全部（基线 + KBD-full + 4 消融 × 3 种子 = 18 次）
python scripts/run_all_cifar100.py
```

### 环境 B · 测试 / 评估环境（用已有 checkpoint）

```bash
# 评估 CIFAR-100 测试集
python inference.py \
  --checkpoint checkpoints/cifar100_kbd/best.pth \
  --config configs/cifar100/kbd_net_full.yaml \
  --dataset cifar100 --data-root ./data

# 单图预测
python inference.py \
  --checkpoint checkpoints/cifar100_kbd/best.pth \
  --config configs/cifar100/kbd_net_full.yaml \
  --image ./samples/airplane.jpg --topk 5

# ImageNet-1K 评估
python inference.py \
  --checkpoint checkpoints/imagenet1k_kbd/best.pth \
  --config configs/imagenet1k/kbd_net_full.yaml \
  --dataset imagenet --data-root ./data/imagenet --topk 5

# 断点续训（已有 120 epoch，想继续）
python main.py \
  --config configs/cifar100/kbd_net_full.yaml \
  --seed 42 \
  --resume checkpoints/cifar100_kbd/epoch_120.pth
```

### 环境 C · 生产 / 大规模实验环境（多卡 ImageNet-1K）

```bash
# 单机多卡（4 × A100）训练 ImageNet-1K
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 main.py \
    --config configs/imagenet1k/kbd_net_full.yaml \
    --seed 42 \
    --batch-size 256

# 分布式节点（SLURM 集群示例）
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
srun --gres=gpu:4 --cpus-per-task=16 --mem=200G \
  bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config configs/imagenet1k/kbd_net_full.yaml --seed 42 --batch-size 256"

# ImageNet-100 中等规模
python main.py \
  --config configs/imagenet100/kbd_net_full.yaml \
  --data-root /ssd/imagenet100 --seed 42
```

---

## 🧑‍💻 在代码里使用 KBD-Conv

```python
import torch
import torch.nn as nn
from kbdnet.models.kbd_conv import KBDConv2d

# 替换任意 3×3 conv 成为 KBD-Conv
# old:  nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)
kbd = KBDConv2d(
    in_channels=64,
    out_channels=128,
    num_basis=4,                       # M = 4 个基核
    dilations=[1, 2, 3, 4],           # 膨胀率（必须 len = num_basis）
    stride=1,
    reduction_ratio=4,                 # 路由 MLP 通道缩减比
    activation="silu",
    norm="bn",
    dynamic_weight=True,               # ✅ 启用输入自适应路由（默认）
)

x = torch.randn(4, 64, 32, 32)
y = kbd(x)                             # (4, 128, 32, 32)

# 同时拿到每个样本的基核混合权重 α ∈ R^{B×M}
kbd.return_alpha = True
y, alpha = kbd(x)                      # alpha.shape == (4, 4), 每行和为 1

# 三种预设消融模式
kbd_m1  = KBDConv2d(64, 64, num_basis=1, dilations=[1])
kbd_same= KBDConv2d(64, 64, num_basis=4, dilations=[1,1,1,1])  # DY-Conv 对照
kbd_eq  = KBDConv2d(64, 64, num_basis=4, dilations=[1,2,3,4], equal_weight=True)

# 构建完整 KBD-Net
from kbdnet.models.kbd_net import KBDNet
net = KBDNet(
    num_classes=100,
    num_basis=4, dilations=[1,2,3,4],
    reduction_ratio=4,
    imagenet_input=False,            # CIFAR stem
    dropout=0.1,
)
print(net.count_parameters())          # ~ 7.2 M
```

---

## 🛠️ 完整 YAML 配置模板

所有配置字段的含义，请查阅 `configs/cifar100/kbd_net_full.yaml`。下面是最全版本：

```yaml
model:
  name: kbd_net                      # kbd_net | resnet18
  num_classes: 100
  dropout: 0.1
  kbd_conv:
    num_basis: 4                     # M：基核数量
    dilations: [1, 2, 3, 4]         # 每个基核的膨胀率（len 必须 = num_basis）
    routing:
      pool: gap                      # global average pooling（仅支持 GAP）
      reduction_ratio: 4             # 路由 MLP 通道缩减比 r
      mode: dynamic                  # dynamic | equal | static
    normalization: bn                # bn | none
    activation: silu                 # silu | relu
    residual: true

train:
  epochs: 300
  optimizer: adamw                   # adamw | adam | sgd
  lr: 0.0003
  weight_decay: 0.05
  scheduler: cosine                  # cosine（唯一）
  warmup_epochs: 5
  label_smoothing: 0.1
  amp: true
  batch_size: 128
  num_workers: 4
  mixup_alpha: 0.2                   # 0 关闭
  cutmix_alpha: 1.0                 # 0 关闭
  log_interval: 50
  save_interval: 25
  seed: 42

dataset:
  name: cifar100                     # cifar10 | cifar100 | imagenet100 | imagenet
  root: ./data
  image_size: 32                     # ImageNet: 224

output_dir: ./checkpoints/cifar100_kbd
cuda: true
```

**CLI 覆盖**（可以覆盖 YAML 中的任何字段）：

```bash
python main.py \
  --config configs/cifar100/kbd_net_full.yaml \
  --seed 3407 \
  --epochs 200 \
  --batch-size 256 \
  --data-root /ssd/cifar \
  --output-dir ./runs/debug
```

---

## 📈 训练自定义权重完整流程

### Step 1 · 数据集准备

| 数据集 | 命令 | 注意事项 |
|---|---|---|
| CIFAR-10 | 配置 `name: cifar10` + 自动下载 | 32×32 |
| CIFAR-100 | 配置 `name: cifar100` + 自动下载 | 32×32 |
| ImageNet-100 | 自行从 ImageNet 1K 抽 100 类 | 224×224 |
| ImageNet-1K | 自行准备 ILSVRC-2012 | 224×224，train 1.28M |

```bash
# 校验数据是否就位
# CIFAR
ls ./data/cifar-100-python/     # 应存在 meta/ test 等

# ImageNet
ls ./data/imagenet/train/ | head  # 应列出 1000 个类别名
ls ./data/imagenet/val/   | head  # 应列出 1000 个类别名
```

### Step 2 · 选择配置模板

复制最近的 YAML，修改 `model.num_classes` 和 `dataset.name` 即可。

### Step 3 · 启动训练

```bash
# 单卡
python main.py --config configs/cifar100/kbd_net_full.yaml --seed 42

# 多卡 DDP
torchrun --nproc_per_node=4 main.py \
  --config configs/imagenet1k/kbd_net_full.yaml --seed 42
```

### Step 4 · 监控

训练过程会持续打印到 stdout，每分钟 1-2 行：

```
[train] epoch=  3 step=  50/390 loss=2.9122 top1=9.82% lr=2.70e-04
...
[epoch   3] train_loss=2.8512 train_top1=12.45% val_loss=2.7810 val_top1=14.82% top5=39.12% lr=2.70e-04
  -> NEW BEST val_top1=14.82% saved to checkpoints/cifar100_kbd/best.pth
```

**最佳实践：另开一个 tail -f**

```bash
tail -f checkpoints/cifar100_kbd/metrics.csv
```

`metrics.csv` 每行格式：

```
epoch,train_loss,train_top1,train_top5,val_loss,val_top1,val_top5,lr,seconds
1,3.782374,4.15,13.12,3.858102,9.82,29.15,3.00e-04,45.23
2,3.451209,7.24,21.05,3.402816,16.12,42.08,3.00e-04,47.02
...
300,1.234567,72.45,93.12,1.356789,75.80,94.32,2.15e-07,48.55
```

### Step 5 · 评估指标

每种子完成后，你至少需要这些文件：

```
checkpoints/cifar100_kbd/
├── best.pth              # 最佳 epoch 权重
├── epoch_025.pth         # 每 25 epoch 的快照
├── epoch_050.pth
├── ...
├── epoch_300.pth
└── metrics.csv           # 完整训练曲线
```

最终评估指标（三种种子均值 ± 标准差）：

| 指标 | 命令 | 位置 |
|---|---|---|
| Best val Top-1 (%) | CSV 里 `val_top1` 列最大值 | `metrics.csv` |
| Best val Top-5 (%) | CSV 里 `val_top5` 列最大值 | `metrics.csv` |
| Params (M) | `count_parameters(model) / 1e6` | 训练开始时 stdout |
| FLOPs (G) | `profile_flops(model)` | 训练开始时 stdout |
| 单卡延迟 (ms) | 自定义脚本（见下） | `scripts/profile_latency.py` |

### Step 6 · 自定义你的网络（可选）

```python
# 换骨干宽度（宽度缩放）
net = KBDNet(num_classes=100,
             stage_channels=(32, 64, 128, 256),
             stage_blocks=(2,2,2,2),
             num_basis=4, dilations=[1,2,3,4])

# 深度缩放（ResNet-34 变体）
net = KBDNet(num_classes=1000,
             stage_channels=(64,128,256,512),
             stage_blocks=(3,4,6,3))

# 替换为 MobileNetV2 风格（需要安装 timm 或自行实现）
# 只需将每个 3×3 conv 替换为 KBDConv2d 即可——KBD-Conv 是即插即用的
```

---

## 🔧 部署指南 · 详细排错

### 安装阶段常见问题

| 现象 | 原因 | 解决方案 |
|---|---|---|
| `ModuleNotFoundError: No module named 'yaml'` | PyYAML 没装 | `pip install pyyaml` |
| `torch.cuda.is_available() == False` | CUDA 版本不匹配 | 重新安装对应 CUDA 版本的 PyTorch |
| `CUDA out of memory` | 显存不足 | 减小 `batch_size` / 减小输入分辨率 / 开 AMP（已默认开启） |
| `ImportError: DLL load failed`（Windows） | cuDNN 版本不对 | 对照 PyTorch 官网的 CUDA 版本下载安装 |
| `torchvision.io.ImageReadMode` 不存在 | torchvision 版本太老 | 升级到 torchvision ≥ 0.16 |
| 中文路径编码问题 | Windows 路径含非 ASCII | 把项目放到纯 ASCII 路径，如 `C:\projects\KBD-Net` |
| DataLoader `num_workers=0` 时可以跑，>0 崩了 | Windows pickling | Windows 下 `num_workers` 建议 ≤ 4，Linux 可开到 16 |
| `thop` profile 抛错 | thop 与某些 PyTorch 版本不完全兼容 | 忽略，训练会继续，只有 FLOPs 那行打印失败 |

### 训练阶段常见问题

| 现象 | 原因 | 解决方案 |
|---|---|---|
| 前 10 epoch loss 振荡很大 | warmup 之前 lr 过大 | 确认 YAML 中 `warmup_epochs: 5`；用 SiLU 激活 |
| 某基核 α_i → 0（路由塌缩） | 初始化不对称 | 前 10 epoch 可改 `equal_weight=True` 热身，再切换回 `dynamic_weight=True` |
| 精度在 M=1 和 M=4 之间没差别 | 基核数量消融配置错 | 确认 YAML 里 `dilations` 长度 = `num_basis` |
| CIFAR-100 最终只跑了 55% Top-1 | seed 没固定 / 数据没归一化 | 对照 `dataset.py` 中的均值方差 |
| ImageNet 比预期低 1pp | 可能没开 Mixup / CutMix | `mixup_alpha: 0.2 cutmix_alpha: 1.0` |
| 多卡 DDP 梯度不同步 | 没正确 wrap `DistributedDataParallel` | 本项目默认单卡；多卡请手动 ddp.py |
| 训练中断后 resume 没生效 | 传了 `.pth` 不是我们保存的格式 | 本项目保存的 checkpoint 带 `epoch / optimizer / best_top1` |

### 推理阶段常见问题

| 现象 | 原因 | 解决方案 |
|---|---|---|
| `missing key "module.conv..."` | 保存时用了 `DataParallel` | 推理脚本已自动 strip `"module."` 前缀 |
| CIFAR-100 类名显示数字 | 没传 `--dataset cifar100` | 传了会用内置 100 类名 |
| ImageNet-100 评估时报 "找不到 val 文件夹" | 你没做 val 按类分子目录 | ImageFolder 要求 val 也分 class 文件夹 |
| Top-1 和训练末尾差距很大 | 推理没开 eval() | `inference.py` 已自动 model.eval() |

---

## 🤝 贡献流程

欢迎贡献！流程如下：

### 1. Fork + 分支

```bash
git checkout -b feature/my-awesome-change
```

### 2. 本地测试

- 至少跑一次 **CIFAR-100 完整配置**（300 epoch）确认不退化
- 如果改动了 KBD-Conv，请跑 **所有 5 种消融**（M=1/4/5 + dilation_uniform + routing_equal）
- 请贴出 metrics.csv 末尾的 val_top1 截图

### 3. 提交内容规范

**Commit message**（Conventional Commits 风格）：

```
feat: add KBD-Conv to MobileNetV2 backbone
fix: resolve routing alpha collapse in first 5 epoch
docs: add Edge deployment section to README
refactor: split KBDConv2d into sub-modules
test: add unit tests for KBDConv2d forward shapes
```

**Pull Request 模板（推荐填写）**：

```
## 变更类型
[ ] 新特性 (feat)
[ ] 修复 (fix)
[ ] 重构 (refactor)
[ ] 文档 (docs)
[ ] 测试 (test)

## 变更内容
...

## 动机 / 原因
...

## 验证方式
- [ ] CIFAR-100 完整（300 epoch）
- [ ] 所有消融（M=1/4/5 + dilation_uniform + routing_equal）
- [ ] ImageNet-100（200 epoch）
- [ ] ImageNet-1K（300 epoch）
- [ ] 单元测试
- [ ] FLOPs / Params profile

## 性能对比（请粘贴 val_top1）
| Config | Seed 42 | Seed 3407 | Seed 2026 | Mean ± Std |
|--------|---------|-----------|-----------|-----------|
| Ours   |         |           |           |           |
| Baseline |        |           |           |           |

## 相关 issue（可选）
Closes #...
```

### 4. 审阅流程

1. 至少一位维护者 approve 后方可 merge
2. 可能要求补充 ImageNet 规模验证
3. 对核心模块（`kbd_conv.py` / `kbd_net.py`）的改动需要双批准

### 5. 我们最欢迎的贡献

- 🚀 **新骨干适配**：把 KBD-Conv 插到 MobileNetV2 / ResNet-34 / ResNeXt / ShuffleNet 等
- 🎯 **下游任务**：YOLOv5+KBD 检测 / DeepLabV3++KBD 分割
- 🌐 **多卡训练**：DDP / ZeRO / DeepSpeed 集成
- 📦 **部署**：TensorRT / TorchScript / ONNX / CoreML 导出
- 🧠 **新 KBD-Conv 变体**：空间注意力路由 / 通道独立路由 / 可学习基核初始化

---

## 📄 许可证 License

本项目基于 **MIT License** 开源。

- **本地 LICENSE 文件**：[`main/LICENSE`](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/main/LICENSE)（必须提交到 Git，pip wheel 打包时会自动复制到 `.dist-info/LICENSE`）
- **完整许可证文本**：[MIT License (opensource.org)](https://opensource.org/licenses/MIT)
- **SPDX 标识**：MIT（已写入 LICENSE 文件尾部的机器可读行，便于 SPDX-compliant 工具识别）
- **要点**：
  - ✅ 商业使用
  - ✅ 修改 / 分发 / 私有使用
  - ✅ 专利使用
  - ❌ 责任限制
  - ❌ 商标使用

```text
MIT License

Copyright (c) 2026, KBD-Net Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📮 联系方式

| 渠道 | 信息 |
|------|------|
| **📧 邮箱** | **primeg@foxmail.com** |
| 📝 Issue / PR | GitHub Issues 页面 |
| 💬 讨论 | GitHub Discussions 页面 |
| 🧑‍💻 维护者 | KBD-Net Authors |

欢迎就以下话题联系：

- KBD-Conv 实现细节 / 训练不稳定
- 在检测、分割、ReID、语音等下游任务上的扩展
- 多卡训练 / 大规模实验排错
- TensorRT / 端侧部署 / Jetson Orin
- 论文复现、补充材料写作、消融设计
- 新骨干、新数据集、新基核数量 M / 新膨胀率组合的探索

**收到回复的期望时间**：工作日 24 小时内。

---

## 📝 引用

如果本项目对你的研究有用，请引用：

```bibtex
@inproceedings{kbdnet2026,
  title     = {Kernel Basis Decomposition Network:
               Input-Adaptive Multi-Scale Convolution via Kernel Basis Decomposition},
  author    = {Anonymous},
  booktitle = {Under Review},
  year      = {2026}
}
```

论文全文参见 [docs/论文/KBD-Net-核基分解网络论文-中英对照-自修版.md](file:///C:/Users/JaysonGuo/Desktop/KBD-Net/docs/%E8%AE%BA%E6%96%87/KBD-Net-%E6%A0%B8%E5%9F%BA%E5%88%86%E8%A7%A3%E7%BD%91%E7%BB%9C%E8%AE%BA%E6%96%87-%E4%B8%AD%E8%8B%B1%E5%AF%B9%E7%85%A7-%E8%87%AA%E4%BF%AE%E7%89%88.md)。

---

<p align="center">
  <sub><strong>KBD-Net</strong> · Kernel Basis Decomposition Network · 核基分解网络</sub><br>
  <sub>Made with ❤ in PyTorch · maintained since 2026</sub>
</p>
