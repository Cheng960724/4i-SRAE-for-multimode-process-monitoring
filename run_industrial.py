# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_industrial import load_industrial_bundle
from experiment import run_experiment
from model import SRAEConfig


def main():
    parser = argparse.ArgumentParser(description="Reproduce the industrial 4i-SRAE multimode experiment.")
    parser.add_argument("--data", default=str(ROOT / "data" / "industrial" / "data_2018_27.dat"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "industrial"))
    parser.add_argument("--fault-start", type=int, default=95372)
    parser.add_argument("--fault-end", type=int, default=97072)
    parser.add_argument("--fault-onset", type=int, default=400)
    parser.add_argument("--normal-start", type=int, default=93372)
    parser.add_argument("--normal-end", type=int, default=95072)
    args = parser.parse_args()

    bundle = load_industrial_bundle(
        args.data,
        fault_range=(args.fault_start, args.fault_end),
        fault_onset=args.fault_onset,
        normal_range=(args.normal_start, args.normal_end),
    )
    config = SRAEConfig(
        timestep=20,
        latent_dim=28,
        batch_size=128,
        epochs=300,
        w_recon=0.60,
        w_intra=0.15,
        w_inv=0.25,
        recon_weights=(0.60, 0.40),
        noise_std=0.0,
    )
    run_experiment(bundle, config, args.output_dir, confidence=0.999)


if __name__ == "__main__":
    main()
