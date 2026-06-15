#!/usr/bin/env python
"""Launch all CIFAR-100 experiments: baseline, KBD-Net, 5 ablations x 3 seeds.

This is the canonical execution plan described in experimental guide v5.0 § 10.
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SEEDS = [42, 3407, 2026]

EXPERIMENTS = [
    ("cifar100/resnet18_baseline.yaml", "baseline"),
    ("cifar100/kbd_net_full.yaml",      "kbd_full"),
    ("cifar100/kbd_ablation_M1.yaml",    "ablation_M1"),
    ("cifar100/kbd_ablation_M5.yaml",    "ablation_M5"),
    ("cifar100/kbd_ablation_dilation_uniform.yaml", "ablation_dilation_uniform"),
    ("cifar100/kbd_ablation_routing_equal.yaml",    "ablation_routing_equal"),
]


def run() -> int:
    rc = 0
    for cfg_rel, tag in EXPERIMENTS:
        for seed in SEEDS:
            cfg = os.path.join(ROOT, "configs", cfg_rel)
            out = os.path.join(ROOT, "checkpoints", f"{tag}_seed{seed}")
            os.makedirs(out, exist_ok=True)
            cmd = [
                sys.executable, os.path.join(ROOT, "main.py"),
                "--config", cfg,
                "--seed", str(seed),
                "--output-dir", out,
            ]
            print(f"\n{'='*72}")
            print(f"RUN: {' '.join(cmd)}")
            print(f"{'='*72}\n")
            r = subprocess.call(cmd, cwd=ROOT)
            if r != 0:
                print(f"[launch] FAIL tag={tag} seed={seed} rc={r}")
                rc = r
    return rc


if __name__ == "__main__":
    sys.exit(run())
