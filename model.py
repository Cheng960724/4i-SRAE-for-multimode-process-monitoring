# -*- coding: utf-8 -*-
"""Core four-input Siamese recurrent autoencoder (4i-SRAE).

Refactored from the uploaded experiment scripts.  The model keeps the original
four-input/shared-encoder/shared-decoder structure and the three normalized
loss terms used in the manuscript.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


@dataclass(frozen=True)
class SRAEConfig:
    timestep: int
    latent_dim: int
    batch_size: int
    epochs: int
    w_recon: float = 0.60
    w_intra: float = 0.15
    w_inv: float = 0.25
    recon_weights: Tuple[float, float] = (0.60, 0.40)
    noise_std: float = 0.0
    learning_rate: float = 1e-3
    clipnorm: float = 1.0
    early_stopping_patience: int = 50


class FourInputSRAE:
    """Four-input SRAE with reconstruction, intra-mode and invariance losses."""

    def __init__(
        self,
        w_recon: float = 0.60,
        w_intra: float = 0.15,
        w_inv: float = 0.25,
        recon_weights: Sequence[float] = (0.60, 0.40),
        noise_std: float = 0.0,
    ) -> None:
        total = float(w_recon + w_intra + w_inv)
        if total <= 0:
            raise ValueError("The sum of loss weights must be positive.")
        self.w_recon = w_recon / total
        self.w_intra = w_intra / total
        self.w_inv = w_inv / total

        recon_total = float(sum(recon_weights))
        if recon_total <= 0:
            raise ValueError("The sum of reconstruction weights must be positive.")
        self.recon_weights = tuple(float(x) / recon_total for x in recon_weights)
        self.noise_std = float(noise_std)
        self.encoder: tf.keras.Model | None = None
        self.model: tf.keras.Model | None = None

    @staticmethod
    def _dist(z_seq1: tf.Tensor, z_seq2: tf.Tensor) -> tf.Tensor:
        """Mean squared latent distance using the last LSTM time step."""
        z1 = z_seq1[:, -1, :]
        z2 = z_seq2[:, -1, :]
        return K.mean(K.mean(K.square(z1 - z2), axis=-1))

    def total_loss(self, args):
        (
            anchor1, recon_a1, anchor2, recon_a2,
            positive1, recon_p1, positive2, recon_p2,
            z_a1, z_a2, z_p1, z_p2,
        ) = args

        mse_a1 = K.mean(K.square(anchor1 - recon_a1), axis=[1, 2])
        mse_a2 = K.mean(K.square(anchor2 - recon_a2), axis=[1, 2])
        mse_p1 = K.mean(K.square(positive1 - recon_p1), axis=[1, 2])
        mse_p2 = K.mean(K.square(positive2 - recon_p2), axis=[1, 2])

        l_recon = K.mean(
            self.recon_weights[0] * (mse_a1 + mse_a2) / 2.0
            + self.recon_weights[1] * (mse_p1 + mse_p2) / 2.0
        )

        l_intra = (self._dist(z_a1, z_a2) + self._dist(z_p1, z_p2)) / 2.0
        l_inv = (self._dist(z_a1, z_p1) + self._dist(z_a2, z_p2)) / 2.0
        total = self.w_recon * l_recon + self.w_intra * l_intra + self.w_inv * l_inv
        return total, l_recon, l_intra, l_inv

    def build(self, timestep: int, n_vars: int, latent_dim: int) -> tuple[tf.keras.Model, tf.keras.Model]:
        inp_a1 = tf.keras.layers.Input((timestep, n_vars), name="anchor1_input")
        inp_a2 = tf.keras.layers.Input((timestep, n_vars), name="anchor2_input")
        inp_p1 = tf.keras.layers.Input((timestep, n_vars), name="positive1_input")
        inp_p2 = tf.keras.layers.Input((timestep, n_vars), name="positive2_input")

        noisy = [tf.keras.layers.GaussianNoise(self.noise_std)(x) for x in (inp_a1, inp_a2, inp_p1, inp_p2)]

        encoder_layer = tf.keras.layers.LSTM(
            latent_dim, activation="tanh", return_sequences=True, name="encoder"
        )
        z_a1, z_a2, z_p1, z_p2 = [encoder_layer(x) for x in noisy]

        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(latent_dim, return_sequences=True, name="decoder_lstm"),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_vars)),
            ],
            name="decoder",
        )
        recon_a1, recon_a2, recon_p1, recon_p2 = [decoder(z) for z in (z_a1, z_a2, z_p1, z_p2)]

        outputs = tf.keras.layers.Lambda(self.total_loss, name="loss_outputs")(
            [
                inp_a1, recon_a1, inp_a2, recon_a2,
                inp_p1, recon_p1, inp_p2, recon_p2,
                z_a1, z_a2, z_p1, z_p2,
            ]
        )
        model = tf.keras.Model([inp_a1, inp_a2, inp_p1, inp_p2], outputs, name="4i_SRAE")
        encoder = tf.keras.Model(inp_a1, z_a1, name="4i_SRAE_encoder")
        self.model, self.encoder = model, encoder
        return model, encoder

    def fit(
        self,
        train_inputs: Sequence[np.ndarray],
        val_inputs: Sequence[np.ndarray],
        config: SRAEConfig,
        verbose: int = 1,
    ) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.callbacks.History]:
        if len(train_inputs) != 4 or len(val_inputs) != 4:
            raise ValueError("train_inputs and val_inputs must each contain four arrays.")
        n_samples = min(len(x) for x in train_inputs)
        n_val = min(len(x) for x in val_inputs)
        if n_samples <= 0 or n_val <= 0:
            raise ValueError("Training/validation windows are empty.")

        train_inputs = [x[:n_samples] for x in train_inputs]
        val_inputs = [x[:n_val] for x in val_inputs]
        model, encoder = self.build(config.timestep, train_inputs[0].shape[2], config.latent_dim)

        def dummy_loss(y_true, y_pred):
            return y_pred

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.learning_rate, clipnorm=config.clipnorm
            ),
            loss=[dummy_loss] * 4,
            loss_weights=[1.0, 0.0, 0.0, 0.0],
        )
        targets = [np.zeros(n_samples, dtype=np.float32)] * 4
        val_targets = [np.zeros(n_val, dtype=np.float32)] * 4
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ]
        history = model.fit(
            train_inputs,
            targets,
            validation_data=(val_inputs, val_targets),
            batch_size=config.batch_size,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
        return model, encoder, history
