# -*- coding: utf-8 -*-
"""TE dataset loader for the manuscript's multimode experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TE_VARIABLES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
]

PAPER_FAULTS = [1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
MODE_FILES = {
    "rate_plus3": "Normal_SP5_offset_plus3%.xlsx",
    "rate_minus1": "Normal_SP5_offset_minus1%.xlsx",
    "rate_plus1": "Normal_SP5_offset_plus1%.xlsx",
}


def _read_te_xlsx(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    if df.shape[1] > 52:
        df = df.iloc[:, :52]
    if df.shape[1] < 52:
        raise ValueError(f"{path} has only {df.shape[1]} columns; expected at least 52.")
    return df.iloc[:, TE_VARIABLES].to_numpy()


def _find_fault_file(root: Path, fault_id: int, condition: str) -> Path:
    candidates = []
    if condition == "base":
        candidates.extend([root / f"d{fault_id:02d}_te.xlsx", root / f"d{fault_id:02d}.xlsx"])
    else:
        condition_dir = {
            "rate_plus3": "TEP_Data_Rate_plus3",
            "rate_minus1": "TEP_Data_Rate_minus1",
            "rate_plus1": "TEP_Data_Rate_plus1",
        }[condition]
        d = root / condition_dir
        candidates.extend([
            d / f"d{fault_id:02d}_te_{condition}.xlsx",
            d / f"d{fault_id:02d}_te.xlsx",
            d / f"d{fault_id:02d}.xlsx",
        ])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No TE fault file found. Tried: " + ", ".join(str(p) for p in candidates))


def load_te_bundle(
    data_dir: str | Path,
    normal_sources=("Normal_Seed_64", "Normal_Seed_123", "Normal_Seed_1234"),
    trained_modes=("rate_plus3", "rate_minus1"),
    unseen_mode="rate_plus1",
    mode_train_samples=600,
    mode_test_samples=400,
    faults=PAPER_FAULTS,
    fault_onset=160,
):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    base_parts = [_read_te_xlsx(train_dir / f"{name}.xlsx") for name in normal_sources]
    base_train = np.vstack(base_parts)

    mode_train = {}
    tests = []
    for mode, filename in MODE_FILES.items():
        arr = _read_te_xlsx(train_dir / filename)
        if mode in trained_modes:
            if len(arr) < mode_train_samples + mode_test_samples:
                raise ValueError(f"{filename}: need at least {mode_train_samples + mode_test_samples} rows.")
            mode_train[mode] = arr[:mode_train_samples]
            normal_test = arr[-mode_test_samples:]
            tests.append({
                "name": f"{mode}_normal_test",
                "condition": mode,
                "test_type": "normal robustness",
                "data": normal_test,
                "fault_intervals": [],
                "sample_interval_minutes": 3.0,
            })
        else:
            mode_train[mode] = np.array([])
            if mode == unseen_mode:
                normal_test = arr[-mode_test_samples:]
                tests.append({
                    "name": f"{mode}_unseen_normal_test",
                    "condition": mode,
                    "test_type": "unseen normal robustness",
                    "data": normal_test,
                    "fault_intervals": [],
                    "sample_interval_minutes": 3.0,
                })

    for condition in ("base", *trained_modes, unseen_mode):
        for fault_id in faults:
            path = _find_fault_file(test_dir, fault_id, condition)
            arr = _read_te_xlsx(path)
            tests.append({
                "name": f"{condition}_fault_{fault_id:02d}",
                "condition": condition,
                "test_type": "fault",
                "data": arr,
                "fault_intervals": [(fault_onset, len(arr) - 1, fault_id)],
                "sample_interval_minutes": 3.0,
            })

    return {
        "base_train": base_train,
        "mode_train": mode_train,
        "tests": tests,
        "validation_split": 0.1,
        "metadata": {
            "dataset": "Tennessee Eastman",
            "trained_modes": list(trained_modes),
            "unseen_mode": unseen_mode,
            "faults": list(faults),
            "mode_train_samples": mode_train_samples,
            "mode_test_samples": mode_test_samples,
        },
    }
