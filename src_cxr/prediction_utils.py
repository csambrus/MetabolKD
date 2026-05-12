from __future__ import annotations

import numpy as np
import tensorflow as tf


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    *,
    validate_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kozos predikcio-gyujto util.
    Visszaadja: y_true, y_pred, y_prob.
    """
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    for x_batch, y_batch in dataset:
        probs = model.predict(x_batch, verbose=0)
        y_true_all.append(y_batch.numpy())
        y_prob_all.append(probs)

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)

    if validate_finite and not np.all(np.isfinite(y_prob)):
        raise ValueError(
            "[ERROR] A modell predikcioja NaN vagy inf erteket tartalmaz. "
            "Ellenorizd a tanitast, preprocessinget es input skalazast."
        )

    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob
