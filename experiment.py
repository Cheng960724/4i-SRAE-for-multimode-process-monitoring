# -*- coding: utf-8 -*-
"""Shared training/evaluation pipeline for TE and industrial case studies."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from model import FourInputSRAE, SRAEConfig
from monitoring import (
    calculate_far_fdr,
    calculate_fdd,
    create_point_labels,
    create_window_labels,
    fit_control_limit,
    mahalanobis_stat,
)
from preprocessing import (
    create_sliding_window,
    extract_features,
    fit_global_robust_scaler,
    make_four_input_windows,
    set_seed,
)


def run_experiment(bundle: dict, config: SRAEConfig, output_dir: str | Path, confidence: float, seed: int = 42):
    """Train 4i-SRAE and evaluate all named tests in a dataset bundle."""
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler, base_scaled, mode_scaled = fit_global_robust_scaler(bundle["base_train"], bundle["mode_train"])
    train_inputs, val_inputs = make_four_input_windows(
        base_scaled, mode_scaled, config.timestep, bundle.get("validation_split", 0.1)
    )

    network = FourInputSRAE(
        w_recon=config.w_recon,
        w_intra=config.w_intra,
        w_inv=config.w_inv,
        recon_weights=config.recon_weights,
        noise_std=config.noise_std,
    )
    model, encoder, history = network.fit(train_inputs, val_inputs, config)

    pooled_train = np.vstack([base_scaled] + [x for x in mode_scaled.values() if x.size > 0])
    train_windows = create_sliding_window(pooled_train, config.timestep)
    train_features = extract_features(encoder, train_windows, method="last")
    md_model = fit_control_limit(train_features, confidence=confidence)

    rows = []
    for test in bundle["tests"]:
        x = scaler.transform(test["data"])
        windows = create_sliding_window(x, config.timestep)
        if len(windows) == 0:
            continue
        features = extract_features(encoder, windows, method="last")
        stats = np.array([mahalanobis_stat(z, md_model) for z in features])
        point_labels = create_point_labels(len(x), test.get("fault_intervals", []))
        window_labels = create_window_labels(point_labels, config.timestep)
        far, fdr = calculate_far_fdr(stats, window_labels, md_model.control_limit)
        fdd = calculate_fdd(
            stats,
            md_model.control_limit,
            test.get("fault_intervals", []),
            config.timestep,
            test.get("sample_interval_minutes", 1.0),
        )
        rows.append(
            {
                "test": test["name"],
                "condition": test.get("condition", ""),
                "test_type": test.get("test_type", ""),
                "FAR_percent": far * 100.0,
                "FDR_percent": fdr * 100.0,
                "FDD_minutes": fdd,
                "control_limit": md_model.control_limit,
                "n_samples": len(x),
                "n_windows": len(windows),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)
    encoder.save(output_dir / "encoder.keras")
    model.save_weights(output_dir / "4i_srae.weights.h5")
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model": asdict(config), "confidence": confidence, "dataset": bundle.get("metadata", {})}, f, ensure_ascii=False, indent=2)
    print(f"Control limit: {md_model.control_limit:.4f}")
    print(f"Results saved to: {output_dir / 'results.csv'}")
