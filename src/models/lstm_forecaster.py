# ============================================================
# src/models/lstm_forecaster.py
# ============================================================
# LSTM-based energy consumption forecasting model.
#
# Architecture overview:
#   Input sequence (lookback × n_features)
#       → LSTM layers (learn temporal dependencies)
#       → Dropout (prevent overfitting)
#       → Fully-connected output layer
#       → Forecast (horizon × 1)
#
# Why LSTM for energy forecasting?
# Energy consumption is a time series with strong temporal
# dependencies — what happened 1h ago (kettle just turned off)
# and 24h ago (same time yesterday) predicts the next hour.
# LSTM's gated memory cells capture these long-range patterns
# better than simple RNNs or non-sequential models.
# ============================================================

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LSTMNetwork(nn.Module):
    """
    PyTorch LSTM network definition.

    Inherits from nn.Module — the base class for all PyTorch
    neural networks. We define the layers in __init__ and the
    forward pass (how data flows through them) in forward().
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 24,
    ):
        super().__init__()

        # Store dimensions for logging / saving
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.forecast_horizon = forecast_horizon

        # ── LSTM layer ────────────────────────────────────────
        # input_size:  number of features at each time step
        # hidden_size: dimensionality of the hidden state vector
        # num_layers:  stacked LSTM depth (2 = one feeds the next)
        # batch_first: expect input as (batch, seq, features)
        # dropout:     applied between LSTM layers (not after last)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # ── Dropout after LSTM ────────────────────────────────
        # Applied to the LSTM output before the dense layer
        # Randomly zeroes some activations during training to
        # prevent the model from relying on any single feature
        self.dropout = nn.Dropout(dropout)

        # ── Fully-connected output layer ──────────────────────
        # Maps the final LSTM hidden state to forecast_horizon
        # output values (one per future time step)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: define how input flows through the network.

        x shape: (batch_size, lookback, n_features)
        output:  (batch_size, forecast_horizon)
        """
        # lstm_out: (batch, seq, hidden_size) — output at every step
        # (h_n, c_n): final hidden state and cell state (not used here)
        lstm_out, _ = self.lstm(x)

        # We only use the output from the LAST time step
        # because it contains information from the entire sequence
        last_step = lstm_out[:, -1, :]  # shape: (batch, hidden_size)

        # Apply dropout to regularise
        out = self.dropout(last_step)

        # Project to forecast horizon: (batch, forecast_horizon)
        return self.fc(out)


class LSTMForecaster:
    """
    High-level wrapper around LSTMNetwork.

    Handles:
      - Model initialisation
      - Training loop with MLflow experiment tracking
      - Saving / loading model weights
      - Inference with latency measurement
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.forecast_horizon = settings.forecast_horizon_hours

        # Instantiate the network and move it to GPU if available
        self.model = LSTMNetwork(
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)

        # ── Loss function ─────────────────────────────────────
        # MSELoss (Mean Squared Error) penalises large prediction
        # errors more heavily — appropriate for continuous values
        self.criterion = nn.MSELoss()

        # ── Optimiser ─────────────────────────────────────────
        # Adam (Adaptive Moment Estimation) adjusts learning rates
        # per-parameter and generally converges faster than SGD
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self._trained = False
        logger.info(
            "lstm_forecaster_initialised",
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            device=str(self.device),
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
    ) -> dict:
        """
        Train the LSTM model and log metrics to MLflow.

        Training loop per epoch:
          1. Split data into batches via DataLoader
          2. Forward pass: compute predictions
          3. Compute MSE loss against true values
          4. Backward pass: compute gradients
          5. Optimiser step: update weights

        MLflow logs every epoch's train/val loss so we can
        compare runs visually in the MLflow UI.
        """
        # Configure MLflow experiment
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        # ── Convert numpy arrays to PyTorch tensors ───────────
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device)
        X_v = torch.FloatTensor(X_val).to(self.device)
        y_v = torch.FloatTensor(y_val).to(self.device)

        # ── DataLoader batches the training data ──────────────
        # shuffle=True randomises order each epoch — prevents the
        # model from memorising the order of training examples
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": []}

        with mlflow.start_run():
            # Log hyperparameters as MLflow params
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_size": self.model.hidden_size,
                "n_layers": self.model.n_layers,
                "n_features": self.model.n_features,
                "forecast_horizon": self.forecast_horizon,
                "device": str(self.device),
            })

            for epoch in range(epochs):
                # ── Training phase ────────────────────────────
                self.model.train()  # enables dropout
                epoch_train_loss = 0.0

                for X_batch, y_batch in loader:
                    # Zero gradients from the previous batch
                    # (PyTorch accumulates them by default)
                    self.optimizer.zero_grad()

                    # Forward pass: compute predictions
                    predictions = self.model(X_batch)

                    # Compute loss between predictions and targets
                    loss = self.criterion(predictions, y_batch)

                    # Backward pass: compute gradients via autograd
                    loss.backward()

                    # Gradient clipping prevents exploding gradients
                    # which are common in deep RNNs
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Update model weights
                    self.optimizer.step()

                    epoch_train_loss += loss.item()

                avg_train_loss = epoch_train_loss / len(loader)

                # ── Validation phase ──────────────────────────
                # torch.no_grad() disables gradient computation
                # for speed and memory efficiency during eval
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_v)
                    val_loss = self.criterion(val_predictions, y_v).item()

                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(val_loss)

                # Log metrics to MLflow for this epoch
                mlflow.log_metrics(
                    {"train_loss": avg_train_loss, "val_loss": val_loss},
                    step=epoch,
                )

                if (epoch + 1) % 5 == 0:
                    logger.info(
                        "training_epoch",
                        epoch=epoch + 1,
                        train_loss=round(avg_train_loss, 6),
                        val_loss=round(val_loss, 6),
                    )

            # Log final summary metrics
            mlflow.log_metric("final_val_loss", history["val_loss"][-1])

        self._trained = True
        logger.info("training_complete", epochs=epochs)
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on scaled input sequences.

        Returns raw scaled predictions — caller must use the
        scaler's inverse_transform_y() to get kWh values.

        X shape: (n_samples, lookback, n_features)
        Returns: (n_samples, forecast_horizon)
        """
        start = time.perf_counter()

        # torch.no_grad() is essential for inference —
        # disables autograd tracking to save memory and speed up
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info("forecast_complete", n_samples=len(X), latency_ms=latency_ms)
        return predictions

    def save(self, path: str | Path) -> None:
        """Save model weights and architecture metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "n_features": self.model.n_features,
                "hidden_size": self.model.hidden_size,
                "n_layers": self.model.n_layers,
                "forecast_horizon": self.model.forecast_horizon,
            },
            path,
        )
        logger.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "LSTMForecaster":
        """Load a previously saved forecaster from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        forecaster = cls(
            n_features=checkpoint["n_features"],
            hidden_size=checkpoint["hidden_size"],
            n_layers=checkpoint["n_layers"],
        )
        forecaster.model.load_state_dict(checkpoint["model_state_dict"])
        forecaster._trained = True
        logger.info("model_loaded", path=str(path))
        return forecaster
