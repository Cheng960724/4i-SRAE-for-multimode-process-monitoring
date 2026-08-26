# -*- coding: utf-8 -*-
"""Data preprocessing and four-input construction utilities."""
from __future__ import annotations

import random
from typing import Dict, Iterable, Sequence

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_sliding_window(data: np.ndarray, timestep: int) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Expected a 2-D array [samples, variables].")
    n = len(data) - timestep + 1
    if n <= 0:
        return np.empty((0, timestep, data.shape[1]), dtype=data.dtype)
    return np.stack([data[i : i + timestep] for i in range(n)], axis=0)


def extract_features(encoder, windows: np.ndarray, method: str = "last") -> np.ndarray:
    features = encoder.predict(windows, verbose=0)
    if method == "last":
        return features[:, -1, :]
    if method == "mean":
        return np.mean(features, axis=1)
    if method == "max":
        return np.max(features, axis=1)
    raise ValueError(f"Unknown feature extraction method: {method}")


def fit_global_robust_scaler(base_data: np.ndarray, mode_data: Dict[str, np.ndarray]):
    """Fit one RobustScaler on the pooled training data, as in the manuscript."""
    train_parts = [base_data] + [x for x in mode_data.values() if getattr(x, "size", 0) > 0]
    scaler = RobustScaler().fit(np.vstack(train_parts))
    base_scaled = scaler.transform(base_data)
    mode_scaled = {
        name: scaler.transform(x) if getattr(x, "size", 0) > 0 else np.array([])
        for name, x in mode_data.items()
    }
    return scaler, base_scaled, mode_scaled


def split_half_train_val(data: np.ndarray, validation_split: float = 0.1):
    """Match the original script: split first/second halves, then hold out the front of each half."""
    half = len(data) // 2
    first, second = data[:half], data[half:]
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Not enough samples to split into four-input train/validation streams.")
    n1 = max(1, int(len(first) * validation_split))
    n2 = max(1, int(len(second) * validation_split))
    return first[n1:], first[:n1], second[n2:], second[:n2]


def make_four_input_windows(
    base_scaled: np.ndarray,
    mode_scaled: Dict[str, np.ndarray],
    timestep: int,
    validation_split: float = 0.1,
):
    """Construct Anchor1/Anchor2/Positive1/Positive2 exactly from baseline/multimode halves."""
    active = [x for x in mode_scaled.values() if getattr(x, "size", 0) > 0]
    if not active:
        raise ValueError("4i-SRAE multimode training requires at least one additional training mode.")
    multimode = np.vstack(active)

    ba1_tr, ba1_va, ba2_tr, ba2_va = split_half_train_val(base_scaled, validation_split)
    bp1_tr, bp1_va, bp2_tr, bp2_va = split_half_train_val(multimode, validation_split)

    train = [create_sliding_window(x, timestep) for x in (ba1_tr, ba2_tr, bp1_tr, bp2_tr)]
    val = [create_sliding_window(x, timestep) for x in (ba1_va, ba2_va, bp1_va, bp2_va)]
    n_train = min(len(x) for x in train)
    n_val = min(len(x) for x in val)
    if n_train <= 0 or n_val <= 0:
        raise ValueError("Sliding-window construction produced an empty stream.")
    return [x[:n_train] for x in train], [x[:n_val] for x in val]
