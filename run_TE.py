# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_te import load_te_bundle
from experiment import run_experiment
from model import SRAEConfig


def main():
    parser = argparse.ArgumentParser(description="Reproduce the manuscript's TE 4i-SRAE multimode experiment.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "TE"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "TE"))
    args = parser.parse_args()

    bundle = load_te_bundle(args.data_dir)
    config = SRAEConfig(
        timestep=16,      
        latent_dim=64,
        batch_size=256,
        epochs=1000,
        w_recon=0.60,
        w_intra=0.15,
        w_inv=0.25,
        recon_weights=(0.60, 0.40),
        noise_std=0.0,
    )
    run_experiment(bundle, config, args.output_dir, confidence=0.999)


if __name__ == "__main__":
    main()
