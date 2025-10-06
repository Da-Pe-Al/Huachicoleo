"""Script de línea de comandos para entrenar y evaluar el autoencoder LSTM.

Este módulo replica el flujo del cuaderno `Red_Huachicoleo.ipynb` y permite
realizar el entrenamiento completo desde la terminal, lo cual es útil para
pruebas automatizadas o ejecuciones desatendidas.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


FEATURES = ["flow", "pressure", "pump_rpm", "tank_level", "power"]


@dataclass
class CalibrationResult:
    threshold: float
    best_percentile: float
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationResult:
    classification_report: dict
    confusion_matrix: Iterable[Iterable[int]]
    roc_auc: float
    average_precision: float


@dataclass
class PipelineArtifacts:
    calibration: CalibrationResult
    evaluation: EvaluationResult


def load_dataset(path: Path) -> pd.DataFrame:
    """Carga y ordena el dataset por timestamp."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_windows(
    values: np.ndarray,
    labels: np.ndarray,
    timestamps: pd.Series,
    window: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Genera ventanas deslizantes y etiqueta positiva si contiene anomalías."""
    X, y, ts = [], [], []
    for start in range(0, len(values) - window + 1, step):
        end = start + window
        X.append(values[start:end])
        y.append(int(labels[start:end].max()))
        ts.append(timestamps.iloc[end - 1])
    return np.array(X), np.array(y), pd.Series(ts, name="window_end")


def build_model(timesteps: int, n_features: int) -> tf.keras.Model:
    """Construye el autoencoder LSTM."""
    inputs = tf.keras.layers.Input(shape=(timesteps, n_features))
    encoded = tf.keras.layers.LSTM(
        64, activation="tanh", return_sequences=True, dropout=0.1
    )(inputs)
    encoded = tf.keras.layers.LSTM(32, activation="tanh", dropout=0.1)(encoded)

    latent = tf.keras.layers.RepeatVector(timesteps)(encoded)

    decoded = tf.keras.layers.LSTM(
        32, activation="tanh", return_sequences=True, dropout=0.1
    )(latent)
    decoded = tf.keras.layers.LSTM(
        64, activation="tanh", return_sequences=True, dropout=0.1
    )(decoded)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features)
    )(decoded)

    model = tf.keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse"
    )
    return model


def reconstruction_errors(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    reconstructions = model.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructions), axis=(1, 2))


def calibrate_threshold(
    errors_normals: np.ndarray,
    errors_val: np.ndarray,
    labels_val: np.ndarray,
) -> CalibrationResult:
    candidate_percentiles = np.linspace(80, 99, 40)
    best_threshold = None
    best_f1 = -np.inf
    best_metrics = None
    for percentile in candidate_percentiles:
        threshold = float(np.percentile(errors_normals, percentile))
        preds = (errors_val > threshold).astype(int)
        precision = precision_score(labels_val, preds, zero_division=0)
        recall = recall_score(labels_val, preds, zero_division=0)
        f1 = f1_score(labels_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = (percentile, precision, recall, f1)
    assert best_threshold is not None and best_metrics is not None
    percentile, precision, recall, f1 = best_metrics
    return CalibrationResult(
        threshold=best_threshold,
        best_percentile=float(percentile),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def evaluate_predictions(
    y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> EvaluationResult:
    report = classification_report(
        y_true,
        predictions,
        target_names=["Normal", "Anomalía"],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, predictions)
    roc_auc = roc_auc_score(y_true, scores)
    avg_precision = average_precision_score(y_true, scores)
    return EvaluationResult(
        classification_report=report,
        confusion_matrix=cm.tolist(),
        roc_auc=float(roc_auc),
        average_precision=float(avg_precision),
    )


def temporal_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_samples = len(df)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    return df_train, df_val, df_test


def run_pipeline(
    data_path: Path,
    window: int,
    step: int,
    train_ratio: float,
    val_ratio: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
    patience: int,
    output_path: Path | None,
) -> PipelineArtifacts:
    df = load_dataset(data_path)
    df_train, df_val, df_test = temporal_split(df, train_ratio, val_ratio)

    scaler = StandardScaler()
    train_mask = (df.index < len(df_train)) & (df["label"] == 0)
    scaler.fit(df.loc[train_mask, FEATURES])
    scaled_values = scaler.transform(df[FEATURES])

    label_array = df["label"].values

    def build_split(start: int, end: int):
        values = scaled_values[start:end]
        labels = label_array[start:end]
        timestamps = df.loc[start : end - 1, "timestamp"]
        return generate_windows(values, labels, timestamps, window, step)

    train_end = len(df_train)
    val_end = train_end + len(df_val)
    X_train_all, y_train_all, _ = build_split(0, train_end)
    X_val, y_val, _ = build_split(train_end, val_end)
    X_test, y_test, ts_test = build_split(val_end, len(df))

    X_train = X_train_all[y_train_all == 0]

    model = build_model(X_train.shape[1], X_train.shape[2])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1,
    )

    val_errors = reconstruction_errors(model, X_val)
    val_normal_errors = val_errors[y_val == 0]
    calibration = calibrate_threshold(val_normal_errors, val_errors, y_val)

    test_errors = reconstruction_errors(model, X_test)
    y_pred = (test_errors > calibration.threshold).astype(int)
    evaluation = evaluate_predictions(y_test, y_pred, test_errors)

    artifacts = PipelineArtifacts(calibration=calibration, evaluation=evaluation)

    if output_path:
        output = {
            "calibration": asdict(calibration),
            "evaluation": asdict(evaluation),
            "timestamps": ts_test.astype(str).tolist(),
            "test_errors": test_errors.tolist(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))

    return artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el autoencoder LSTM para detección de anomalías"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("rebombeo_huachicoleo.csv"),
        help="Ruta al archivo CSV con los datos",
    )
    parser.add_argument("--window", type=int, default=30, help="Tamaño de la ventana en minutos")
    parser.add_argument("--step", type=int, default=5, help="Paso entre ventanas en minutos")
    parser.add_argument(
        "--train-ratio", type=float, default=0.6, help="Proporción de muestras para entrenamiento"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Proporción de muestras para validación"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Número máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de lote")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fracción de entrenamiento usada como validación interna",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Número de épocas sin mejora antes de aplicar parada temprana",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ruta donde se guardarán métricas y errores en formato JSON",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    artifacts = run_pipeline(
        data_path=args.data_path,
        window=args.window,
        step=args.step,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        patience=args.patience,
        output_path=args.output,
    )
    print("\nResumen de calibración:")
    print(json.dumps(asdict(artifacts.calibration), indent=2, ensure_ascii=False))
    print("\nMétricas de evaluación:")
    print(json.dumps(asdict(artifacts.evaluation), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
