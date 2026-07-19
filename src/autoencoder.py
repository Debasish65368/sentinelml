from pathlib import Path
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


class FraudAutoencoder(nn.Module):
    """Feedforward autoencoder for reconstruction-based anomaly detection."""

    def __init__(self, input_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


def _configure_mlflow():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("autoencoder_anomaly_detection")


def _as_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def train_autoencoder(X_train, y_train, epochs=50, batch_size=256, lr=0.001):
    """Train an autoencoder only on normal Class=0 training rows."""
    _configure_mlflow()
    torch.manual_seed(42)
    np.random.seed(42)

    normal_mask = y_train.to_numpy() == 0 if hasattr(y_train, "to_numpy") else np.asarray(y_train) == 0
    X_train_normal = X_train.loc[normal_mask] if isinstance(X_train, pd.DataFrame) else X_train[normal_mask]

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(_as_numpy(X_train_normal)).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X_normal_scaled))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_normal_scaled.shape[1]
    model = FraudAutoencoder(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []

    with mlflow.start_run(run_name="pytorch_autoencoder"):
        mlflow.log_param("model_type", "FraudAutoencoder")
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("architecture", f"{input_dim}->16->8->16->{input_dim}")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("normal_training_rows", len(X_normal_scaled))

        model.train()
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for (batch,) in dataloader:
                optimizer.zero_grad()
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            epoch_loss = float(np.mean(epoch_losses))
            loss_history.append(epoch_loss)
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"Epoch {epoch:03d}/{epochs} - train_loss={epoch_loss:.8f}")

        final_loss = loss_history[-1]
        mlflow.log_metric("final_train_loss", final_loss)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        scaler_path = MODELS_DIR / "autoencoder_scaler.joblib"
        model_path = MODELS_DIR / "autoencoder_state_dict.pt"
        loss_curve_path = MODELS_DIR / "autoencoder_loss_curve.csv"
        joblib.dump(scaler, scaler_path)
        torch.save(model.state_dict(), model_path)
        pd.DataFrame({"epoch": range(1, epochs + 1), "train_loss": loss_history}).to_csv(
            loss_curve_path,
            index=False,
        )
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(loss_curve_path)

    model.loss_history_ = loss_history
    model.final_train_loss_ = final_loss
    model.input_dim_ = input_dim

    return model, scaler


def compute_reconstruction_error(model, scaler, X):
    """Return per-row reconstruction MSE using the fitted scaler."""
    X_scaled = scaler.transform(_as_numpy(X)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(X_scaled)
        reconstruction = model(input_tensor).numpy()
    return np.mean((X_scaled - reconstruction) ** 2, axis=1)


def _threshold_metrics(y_true, errors, threshold):
    y_pred = (errors >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, errors),
    }


def tune_threshold(model, scaler, X_val, y_val):
    """Compare validation reconstruction-error thresholding strategies."""
    errors = compute_reconstruction_error(model, scaler, X_val)
    y_true = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)
    normal_errors = errors[y_true == 0]

    strategies = []
    for percentile in [95, 99]:
        threshold = np.percentile(normal_errors, percentile)
        metrics = _threshold_metrics(y_true, errors, threshold)
        metrics["strategy"] = f"normal_p{percentile}"
        strategies.append(metrics)

    candidate_thresholds = np.unique(np.quantile(errors, np.linspace(0, 1, 1000)))
    best_metrics = None
    for threshold in candidate_thresholds:
        metrics = _threshold_metrics(y_true, errors, threshold)
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
    best_metrics["strategy"] = "f1_optimal"
    strategies.append(best_metrics)

    comparison = pd.DataFrame(strategies).set_index("strategy")
    comparison = comparison[["threshold", "precision", "recall", "f1", "pr_auc"]]
    chosen_threshold = float(comparison.loc["f1_optimal", "threshold"])
    chosen_metrics = comparison.loc["f1_optimal"].to_dict()
    chosen_metrics["strategy"] = "f1_optimal"
    chosen_metrics["threshold_comparison"] = comparison
    chosen_metrics["reconstruction_errors"] = errors

    print("=== Autoencoder Threshold Comparison ===")
    print(comparison.to_string(float_format=lambda value: f"{value:.8f}"))
    print(f"Chosen threshold: {chosen_threshold:.8f} (strategy=f1_optimal)")

    return chosen_threshold, chosen_metrics


def compare_autoencoder_vs_xgboost(autoencoder_results, xgboost_model, X_val, y_val):
    """Compare fraud cases caught by autoencoder thresholding vs XGBoost at 0.5."""
    model = autoencoder_results["model"]
    scaler = autoencoder_results["scaler"]
    threshold = autoencoder_results["threshold"]
    errors = compute_reconstruction_error(model, scaler, X_val)
    autoencoder_pred = errors >= threshold
    xgboost_pred = xgboost_model.predict_proba(X_val)[:, 1] >= 0.5
    y_true = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)
    fraud_mask = y_true == 1

    ae_caught = autoencoder_pred & fraud_mask
    xgb_caught = xgboost_pred & fraud_mask

    both = int((ae_caught & xgb_caught).sum())
    only_autoencoder = int((ae_caught & ~xgb_caught).sum())
    only_xgboost = int((~ae_caught & xgb_caught).sum())
    neither = int((fraud_mask & ~ae_caught & ~xgb_caught).sum())

    overlap = {
        "both_catch": both,
        "only_autoencoder_catches": only_autoencoder,
        "only_xgboost_catches": only_xgboost,
        "neither_catches": neither,
        "total_fraud": int(fraud_mask.sum()),
    }

    print("=== Autoencoder vs XGBoost Fraud Catch Overlap ===")
    print(f"Total fraud cases: {overlap['total_fraud']}")
    print(f"Both catch: {both}")
    print(f"Only autoencoder catches: {only_autoencoder}")
    print(f"Only XGBoost catches: {only_xgboost}")
    print(f"Neither catches: {neither}")

    return overlap
