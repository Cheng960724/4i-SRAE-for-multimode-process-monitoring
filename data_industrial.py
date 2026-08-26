# -*- coding: utf-8 -*-
"""Industrial catalytic reforming heat-exchange dataset loader.

Index ranges default to the uploaded industrial Python script.  See
VERIFY_BEFORE_UPLOAD.md because the manuscript describes the fault test set as
approximately 528 samples, whereas the uploaded script uses [95372:97072].
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

VARIABLES = [
    "LY_P_2211PDI20011.PV",
    "LY_P_2211FI20006A.PV",
    "LY_P_2211FIC20005.PV",
]


def _load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".dat":
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="iso-8859-1")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        return data
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, index_col=0)
    raise ValueError("Industrial data must be a .dat pickle or Excel file.")


def load_industrial_bundle(
    data_path: str | Path,
    base_range=(91800, 93370),
    mode240_range=(30000, 36399),
    mode245_range=(110000, 115000),
    mode250_range=(145000, 150000),
    mode252_range=(181000, 190000),
    train_per_mode=500,
    mode_test_size=1000,
    fault_range=(95372, 97072),
    fault_onset=400,
    normal_range=(93372, 95072),
):
    data = _load_dataframe(data_path)
    missing = [v for v in VARIABLES if v not in data.columns]
    if missing:
        raise KeyError(f"Missing industrial variables: {missing}")

    def take(r):
        return data[VARIABLES].iloc[r[0] : r[1]].to_numpy()

    base_train = take(base_range)
    raw240, raw250 = take(mode240_range), take(mode250_range)
    if len(raw240) < train_per_mode + mode_test_size or len(raw250) < train_per_mode + mode_test_size:
        raise ValueError("240/250 mode ranges are too short for 500 training + 1000 robustness samples.")

    mode_train = {
        "240": raw240[:train_per_mode],
        "250": raw250[:train_per_mode],
    }
    tests = [
        {
            "name": "fault_test",
            "condition": "process fault",
            "test_type": "fault",
            "data": take(fault_range),
            "fault_intervals": [(fault_onset, len(take(fault_range)) - 1, 1)],
            "sample_interval_minutes": 1.0,
        },
        {
            "name": "normal_test",
            "condition": "255",
            "test_type": "normal",
            "data": take(normal_range),
            "fault_intervals": [],
            "sample_interval_minutes": 1.0,
        },
        {
            "name": "240_test",
            "condition": "240",
            "test_type": "trained-mode robustness",
            "data": raw240[train_per_mode : train_per_mode + mode_test_size],
            "fault_intervals": [],
            "sample_interval_minutes": 1.0,
        },
        {
            "name": "250_test",
            "condition": "250",
            "test_type": "trained-mode robustness",
            "data": raw250[train_per_mode : train_per_mode + mode_test_size],
            "fault_intervals": [],
            "sample_interval_minutes": 1.0,
        },
        {
            "name": "Unseen245_test",
            "condition": "245",
            "test_type": "unseen-mode robustness",
            "data": take(mode245_range)[-mode_test_size:],
            "fault_intervals": [],
            "sample_interval_minutes": 1.0,
        },
        {
            "name": "Unseen252_test",
            "condition": "252",
            "test_type": "unseen-mode robustness",
            "data": take(mode252_range)[-mode_test_size:],
            "fault_intervals": [],
            "sample_interval_minutes": 1.0,
        },
    ]
    return {
        "base_train": base_train,
        "mode_train": mode_train,
        "tests": tests,
        "validation_split": 0.1,
        "metadata": {
            "dataset": "Industrial catalytic reforming heat-exchange unit",
            "trained_modes_tph": [240, 250],
            "unseen_modes_tph": [245, 252],
            "base_mode_tph": 255,
            "index_ranges_from_uploaded_script": True,
        },
    }
