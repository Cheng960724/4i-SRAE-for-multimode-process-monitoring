# -*- coding: utf-8 -*-
"""Monitoring statistics, KDE control limits and FAR/FDR/FDD metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.stats import gaussian_kde


@dataclass
class MahalanobisModel:
    mean: np.ndarray
    std: np.ndarray
    projection: np.ndarray
    inv_eigenvalues: np.ndarray
    control_limit: float


def _fit_pca_mahalanobis(features: np.ndarray):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    if np.any(std == 0):
        raise ValueError("Zero-variance latent dimension encountered; cannot reproduce PCA-Mahalanobis calculation.")
    normalized = (features - mean) / std
    covariance = np.cov(normalized.T)
    p, singular_values, _ = np.linalg.svd(covariance)
    # The uploaded scripts retain all components (cumulative variance reaches 100%).
    projection = p[:, : len(singular_values)]
    inv_eigenvalues = np.diag(1.0 / singular_values)
    return mean, std, projection, inv_eigenvalues


def mahalanobis_stat(z, model: MahalanobisModel) -> float:
    x = ((z - model.mean) / model.std).reshape(1, -1)
    value = x @ model.projection @ model.inv_eigenvalues @ model.projection.T @ x.T
    return float(value[0, 0])


def fit_control_limit(features: np.ndarray, confidence: float = 0.999) -> MahalanobisModel:
    mean, std, projection, inv_eigenvalues = _fit_pca_mahalanobis(features)
    temp = MahalanobisModel(mean, std, projection, inv_eigenvalues, np.nan)
    stats = np.array([mahalanobis_stat(z, temp) for z in features], dtype=float)
    kde = gaussian_kde(stats, bw_method="silverman")

    threshold = None
    # Preserve the original script's numerical search/integration convention.
    upper = max(1000.0, float(np.max(stats) * 2.0 + 1.0))
    for limit in np.arange(0.0, upper, 0.1):
        if quad(lambda x: kde(x), -limit, limit)[0] > confidence:
            threshold = float(limit)
            break
    if threshold is None:
        threshold = float(np.max(stats) * 1.5)
    temp.control_limit = threshold
    return temp


def create_point_labels(data_length: int, fault_intervals: Sequence[tuple[int, int, int]]) -> np.ndarray:
    labels = np.zeros(data_length, dtype=int)
    for start, end, _ in fault_intervals:
        labels[max(0, start) : min(data_length, end + 1)] = 1
    return labels


def create_window_labels(point_labels: np.ndarray, timestep: int) -> np.ndarray:
    n = len(point_labels) - timestep + 1
    if n <= 0:
        return np.empty((0,), dtype=int)
    return np.array([int(np.any(point_labels[i : i + timestep])) for i in range(n)], dtype=int)


def calculate_far_fdr(stats: np.ndarray, labels: np.ndarray, control_limit: float):
    predictions = (stats > control_limit).astype(int)
    normal = np.where(labels == 0)[0]
    fault = np.where(labels == 1)[0]
    far = float(np.mean(predictions[normal])) if len(normal) else 0.0
    fdr = float(np.mean(predictions[fault])) if len(fault) else 0.0
    return far, fdr


def calculate_fdd(
    stats: np.ndarray,
    control_limit: float,
    fault_intervals: Sequence[tuple[int, int, int]],
    timestep: int,
    sample_interval_minutes: float,
):
    if not fault_intervals:
        return None
    onset = max(0, fault_intervals[0][0] - timestep + 1)
    alarms = np.where(stats[onset:] > control_limit)[0]
    if len(alarms) == 0:
        return None
    return float(alarms[0] * sample_interval_minutes)
