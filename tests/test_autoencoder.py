import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.autoencoder as autoencoder
from src.autoencoder import FraudAutoencoder, compute_reconstruction_error, train_autoencoder


def make_autoencoder_data(rows=80, features=32):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(rows, features)),
        columns=[f"feature_{i}" for i in range(features)],
    )
    y = pd.Series([0] * 70 + [1] * 10, name="Class")
    return X, y


def test_autoencoder_trains_without_error(tmp_path, monkeypatch):
    monkeypatch.setattr(autoencoder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(autoencoder, "MODELS_DIR", tmp_path / "models")
    X, y = make_autoencoder_data()

    model, scaler = train_autoencoder(X, y, epochs=2, batch_size=16)

    assert isinstance(model, FraudAutoencoder)
    assert hasattr(scaler, "mean_")
    assert len(model.loss_history_) == 2


def test_reconstruction_error_is_non_negative(tmp_path, monkeypatch):
    monkeypatch.setattr(autoencoder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(autoencoder, "MODELS_DIR", tmp_path / "models")
    X, y = make_autoencoder_data()
    model, scaler = train_autoencoder(X, y, epochs=2, batch_size=16)

    errors = compute_reconstruction_error(model, scaler, X)

    assert np.all(errors >= 0)


def test_autoencoder_output_shape_matches_input_shape():
    model = FraudAutoencoder(input_dim=32)
    batch = torch.randn(5, 32)

    output = model(batch)

    assert output.shape == batch.shape
