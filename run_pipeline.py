"""Pipeline reproducible y experimentación para la detección de robo de hidrocarburos."""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


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
codex/review-hydrocarbon-theft-analysis-code-x3whc0
from sklearn.neighbors import LocalOutlierFactor

from sklearn.preprocessing import StandardScaler


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURES = ["flow", "pressure", "pump_rpm", "tank_level", "power"]


@dataclass
class ModelConfig:
    """Configura la arquitectura del autoencoder."""

    encoder_units: Tuple[int, ...] = (128, 64)
    decoder_units: Tuple[int, ...] = (64, 128)
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configura el entrenamiento del autoencoder."""

    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    validation_split: float = 0.1
    patience: int = 10


@dataclass
class CalibrationResult:
    threshold: float
    best_percentile: float
    precision: float
    recall: float
    f1: float


@dataclass
class ThresholdScenario:
    strategy: str
    threshold: float
    precision: float
    recall: float
    f1: float


@dataclass
class AdvancedModelResult:
    name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float


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
    threshold_experiments: List[ThresholdScenario] = field(default_factory=list)
    advanced_results: List[AdvancedModelResult] = field(default_factory=list)


@dataclass
class SweepResult:
    window: int
    step: int
    model: ModelConfig
    training: TrainingConfig
    calibration: CalibrationResult
    evaluation: EvaluationResult

    def to_dict(self) -> dict:
        data = {
            "window": self.window,
            "step": self.step,
            "model": asdict(self.model),
            "training": asdict(self.training),
            "calibration": asdict(self.calibration),
            "evaluation": asdict(self.evaluation),
        }
        return data


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


def build_model(
    timesteps: int, n_features: int, model_config: ModelConfig, learning_rate: float
) -> tf.keras.Model:
    """Construye el autoencoder LSTM con la configuración indicada."""

    inputs = tf.keras.layers.Input(shape=(timesteps, n_features))
    x = inputs
    for i, units in enumerate(model_config.encoder_units):
        return_sequences = i < len(model_config.encoder_units) - 1
        x = tf.keras.layers.LSTM(
            units,
            activation="tanh",
            return_sequences=return_sequences,
            dropout=model_config.dropout,
        )(x)
    latent = tf.keras.layers.RepeatVector(timesteps)(x)
    x = latent
    for units in model_config.decoder_units:
        x = tf.keras.layers.LSTM(
            units,
            activation="tanh",
            return_sequences=True,
            dropout=model_config.dropout,
        )(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(x)

    model = tf.keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
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


def experiment_thresholds(
    errors_normals: np.ndarray, errors_val: np.ndarray, labels_val: np.ndarray
) -> List[ThresholdScenario]:
    """Evalúa múltiples estrategias de umbral y devuelve sus métricas."""

    scenarios: List[ThresholdScenario] = []

    for percentile in [85, 90, 95, 97.5, 99]:
        threshold = float(np.percentile(errors_normals, percentile))
        preds = (errors_val > threshold).astype(int)
        scenarios.append(
            ThresholdScenario(
                strategy=f"percentil_{percentile}",
                threshold=threshold,
                precision=float(precision_score(labels_val, preds, zero_division=0)),
                recall=float(recall_score(labels_val, preds, zero_division=0)),
                f1=float(f1_score(labels_val, preds, zero_division=0)),
            )
        )

    q1, q3 = np.percentile(errors_normals, [25, 75])
    iqr = q3 - q1
    iqr_threshold = float(q3 + 1.5 * iqr)
    iqr_preds = (errors_val > iqr_threshold).astype(int)
    scenarios.append(
        ThresholdScenario(
            strategy="iqr",
            threshold=iqr_threshold,
            precision=float(precision_score(labels_val, iqr_preds, zero_division=0)),
            recall=float(recall_score(labels_val, iqr_preds, zero_division=0)),
            f1=float(f1_score(labels_val, iqr_preds, zero_division=0)),
        )
    )

    median = float(np.median(errors_normals))
    mad = float(np.median(np.abs(errors_normals - median)))
    mad_threshold = float(median + 3 * mad)
    mad_preds = (errors_val > mad_threshold).astype(int)
    scenarios.append(
        ThresholdScenario(
            strategy="mad",
            threshold=mad_threshold,
            precision=float(precision_score(labels_val, mad_preds, zero_division=0)),
            recall=float(recall_score(labels_val, mad_preds, zero_division=0)),
            f1=float(f1_score(labels_val, mad_preds, zero_division=0)),
        )
    )

    return scenarios


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


def evaluate_scores(
    name: str, y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> AdvancedModelResult:
    return AdvancedModelResult(
        name=name,
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        average_precision=float(average_precision_score(y_true, scores)),
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


def prepare_windows(
    df: pd.DataFrame,
    scaler: StandardScaler,
    window: int,
    step: int,
    train_len: int,
    val_len: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray, np.ndarray, pd.Series, np.ndarray, np.ndarray, pd.Series]:
    values = scaler.transform(df[FEATURES])
    labels = df["label"].values
    timestamps = df["timestamp"]

    def build_split(start: int, end: int):
        return generate_windows(
            values[start:end], labels[start:end], timestamps.iloc[start:end], window, step
        )

    X_train_all, y_train_all, ts_train = build_split(0, train_len)
    X_val, y_val, ts_val = build_split(train_len, train_len + val_len)
    X_test, y_test, ts_test = build_split(train_len + val_len, len(df))
    return (
        X_train_all,
        y_train_all,
        ts_train,
        X_val,
        y_val,
        ts_val,
        X_test,
        y_test,
        ts_test,
    )


def run_pipeline(
    data_path: Path,
    window: int,
    step: int,
    train_ratio: float,
    val_ratio: float,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    output_path: Path | None = None,
    analyze_thresholds: bool = True,
    evaluate_advanced: bool = False,
) -> PipelineArtifacts:
    df = load_dataset(data_path)
    df_train, df_val, df_test = temporal_split(df, train_ratio, val_ratio)

    scaler = StandardScaler()
    train_mask = (df.index < len(df_train)) & (df["label"] == 0)
    scaler.fit(df.loc[train_mask, FEATURES])

    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()

    (
        X_train_all,
        y_train_all,
        _,
        X_val,
        y_val,
        _,
        X_test,
        y_test,
        ts_test,
    ) = prepare_windows(
        df,
        scaler,
        window,
        step,
        len(df_train),
        len(df_val),
    )

    X_train = X_train_all[y_train_all == 0]

    model = build_model(
        X_train.shape[1],
        X_train.shape[2],
        model_config,
        training_config.learning_rate,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=training_config.patience, restore_best_weights=True
    )
    model.fit(
        X_train,
        X_train,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        validation_split=training_config.validation_split,
        callbacks=[early_stopping],
        verbose=1,
    )

    val_errors = reconstruction_errors(model, X_val)
    val_normal_errors = val_errors[y_val == 0]
    calibration = calibrate_threshold(val_normal_errors, val_errors, y_val)

    test_errors = reconstruction_errors(model, X_test)
    y_pred = (test_errors > calibration.threshold).astype(int)
    evaluation = evaluate_predictions(y_test, y_pred, test_errors)

    threshold_experiments: List[ThresholdScenario] = []
    if analyze_thresholds:
        threshold_experiments = experiment_thresholds(val_normal_errors, val_errors, y_val)

    advanced_results: List[AdvancedModelResult] = []
    if evaluate_advanced:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        iso = IsolationForest(random_state=SEED, contamination="auto")
        iso.fit(X_train_flat)
        iso_scores = -iso.decision_function(X_test_flat)
        iso_preds = (iso.predict(X_test_flat) == -1).astype(int)
        advanced_results.append(
            evaluate_scores("IsolationForest", y_test, iso_preds, iso_scores)
        )

        lof = LocalOutlierFactor(
            n_neighbors=min(35, len(X_train_flat) - 1), novelty=True, contamination=0.05
        )
        lof.fit(X_train_flat)
        lof_scores = -lof.decision_function(X_test_flat)
        lof_preds = (lof.predict(X_test_flat) == -1).astype(int)
        advanced_results.append(
            evaluate_scores("LocalOutlierFactor", y_test, lof_preds, lof_scores)
        )

    artifacts = PipelineArtifacts(
        calibration=calibration,
        evaluation=evaluation,
        threshold_experiments=threshold_experiments,
        advanced_results=advanced_results,
    )

    if output_path:
        output = {
            "calibration": asdict(calibration),
            "evaluation": asdict(evaluation),
            "threshold_experiments": [asdict(s) for s in threshold_experiments],
            "advanced_results": [asdict(r) for r in advanced_results],
            "timestamps": ts_test.astype(str).tolist(),
            "test_errors": test_errors.tolist(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))

    return artifacts


def parse_units(values: Sequence[str] | None, default: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    if not values:
        return [default]
    parsed = []
    for value in values:
        units = tuple(int(v) for v in value.split(",") if v)
        if not units:
            raise ValueError(f"Configuración de unidades inválida: {value}")
        parsed.append(units)
    return parsed


def run_sweep(
    data_path: Path,
    train_ratio: float,
    val_ratio: float,
    windows: Sequence[int],
    steps: Sequence[int],
    encoder_options: Sequence[Tuple[int, ...]],
    decoder_options: Sequence[Tuple[int, ...]],
    dropouts: Sequence[float],
    learning_rates: Sequence[float],
    batches: Sequence[int],
    epochs_list: Sequence[int],
    patience_list: Sequence[int],
    val_splits: Sequence[float],
) -> List[SweepResult]:
    results: List[SweepResult] = []
    for (
        window,
        step,
        encoder,
        decoder,
        dropout,
        lr,
        batch,
        epochs,
        patience,
        val_split,
    ) in itertools.product(
        windows,
        steps,
        encoder_options,
        decoder_options,
        dropouts,
        learning_rates,
        batches,
        epochs_list,
        patience_list,
        val_splits,
    ):
        model_cfg = ModelConfig(encoder_units=encoder, decoder_units=decoder, dropout=dropout)
        train_cfg = TrainingConfig(
            learning_rate=lr,
            epochs=epochs,
            batch_size=batch,
            patience=patience,
            validation_split=val_split,
        )
        artifacts = run_pipeline(
            data_path=data_path,
            window=window,
            step=step,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            model_config=model_cfg,
            training_config=train_cfg,
            analyze_thresholds=False,
            evaluate_advanced=False,
        )
        results.append(
            SweepResult(
                window=window,
                step=step,
                model=model_cfg,
                training=train_cfg,
                calibration=artifacts.calibration,
                evaluation=artifacts.evaluation,
            )
        )
    return results


def generate_report(report_path: Path, artifacts: PipelineArtifacts, args: argparse.Namespace) -> None:
    """Crea un resumen en Markdown con los resultados clave."""

    report_lines = [
        "# Resultados del modelo de detección de robo de hidrocarburos",
        "",
        "## Configuración",
        f"- Ventana: {args.window}",
        f"- Paso: {args.step}",
        f"- Ratio entrenamiento: {args.train_ratio}",
        f"- Ratio validación: {args.val_ratio}",
        "",
        "## Calibración del umbral",
        f"- Percentil óptimo: {artifacts.calibration.best_percentile:.2f}",
        f"- Umbral seleccionado: {artifacts.calibration.threshold:.6f}",
        f"- Precisión (val): {artifacts.calibration.precision:.3f}",
        f"- Recall (val): {artifacts.calibration.recall:.3f}",
        f"- F1 (val): {artifacts.calibration.f1:.3f}",
        "",
        "## Métricas en prueba",
        f"- F1: {artifacts.evaluation.classification_report['Anomalía']['f1-score']:.3f}",
        f"- Precisión: {artifacts.evaluation.classification_report['Anomalía']['precision']:.3f}",
        f"- Recall: {artifacts.evaluation.classification_report['Anomalía']['recall']:.3f}",
        f"- ROC-AUC: {artifacts.evaluation.roc_auc:.3f}",
        f"- Average Precision: {artifacts.evaluation.average_precision:.3f}",
        "",
    ]

    if artifacts.threshold_experiments:
        report_lines.append("## Comparativa de estrategias de umbral")
        report_lines.append("")
        report_lines.append("| Estrategia | Umbral | Precisión | Recall | F1 |")
        report_lines.append("|------------|--------|-----------|--------|----|")
        for scenario in artifacts.threshold_experiments:
            report_lines.append(
                f"| {scenario.strategy} | {scenario.threshold:.6f} | {scenario.precision:.3f} | {scenario.recall:.3f} | {scenario.f1:.3f} |"
            )
        report_lines.append("")

    if artifacts.advanced_results:
        report_lines.append("## Modelos avanzados de detección")
        report_lines.append("")
        report_lines.append("| Modelo | Precisión | Recall | F1 | ROC-AUC | AP |")
        report_lines.append("|--------|-----------|--------|----|---------|----|")
        for result in artifacts.advanced_results:
            report_lines.append(
                f"| {result.name} | {result.precision:.3f} | {result.recall:.3f} | {result.f1:.3f} | {result.roc_auc:.3f} | {result.average_precision:.3f} |"
            )
        report_lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el autoencoder LSTM para detección de anomalías",
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
    parser.add_argument("--encoder-units", nargs="+", help="Capas del codificador, e.g. 128,64")
    parser.add_argument("--decoder-units", nargs="+", help="Capas del decodificador, e.g. 64,128")
    parser.add_argument("--dropout", type=float, help="Dropout aplicado en cada capa LSTM")
    parser.add_argument("--learning-rate", type=float, help="Tasa de aprendizaje del optimizador")
    parser.add_argument("--epochs", type=int, help="Número máximo de épocas")
    parser.add_argument("--batch-size", type=int, help="Tamaño de lote")
    parser.add_argument(
        "--validation-split",
        type=float,
        help="Fracción de entrenamiento usada como validación interna",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Número de épocas sin mejora antes de aplicar parada temprana",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ruta donde se guardarán métricas y errores en formato JSON",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Ruta de salida para un resumen en Markdown",
    )
    parser.add_argument(
        "--no-threshold-analysis",
        action="store_true",
        help="Desactiva la comparación de estrategias de umbral",
    )
    parser.add_argument(
        "--advanced-models",
        action="store_true",
        help="Evalúa modelos avanzados (IsolationForest, LOF)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Ejecuta una búsqueda de hiperparámetros en lugar de un solo entrenamiento",
    )
    parser.add_argument(
        "--window-grid",
        type=int,
        nargs="+",
        help="Lista de tamaños de ventana para la búsqueda",
    )
    parser.add_argument(
        "--step-grid",
        type=int,
        nargs="+",
        help="Lista de pasos para la búsqueda",
    )
    parser.add_argument(
        "--encoder-grid",
        nargs="+",
        help="Configuraciones de codificador separadas por espacios (cada una separada por comas)",
    )
    parser.add_argument(
        "--decoder-grid",
        nargs="+",
        help="Configuraciones de decodificador separadas por espacios (cada una separada por comas)",
    )
    parser.add_argument(
        "--dropout-grid",
        type=float,
        nargs="+",
        help="Lista de valores de dropout para la búsqueda",
    )
    parser.add_argument(
        "--lr-grid",
        type=float,
        nargs="+",
        help="Lista de tasas de aprendizaje para la búsqueda",
    )
    parser.add_argument(
        "--batch-grid",
        type=int,
        nargs="+",
        help="Lista de tamaños de lote para la búsqueda",
    )
    parser.add_argument(
        "--epochs-grid",
        type=int,
        nargs="+",
        help="Lista de épocas para la búsqueda",
    )
    parser.add_argument(
        "--patience-grid",
        type=int,
        nargs="+",
        help="Lista de valores de paciencia para la búsqueda",
    )
    parser.add_argument(
        "--val-split-grid",
        type=float,
        nargs="+",
        help="Lista de fracciones de validación interna para la búsqueda",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.sweep:
        windows = args.window_grid or [24, 36, 48]
        steps = args.step_grid or [1, 5, 10]
        encoder_options = parse_units(args.encoder_grid, (128, 64))
        decoder_options = parse_units(args.decoder_grid, (64, 128))
        dropouts = args.dropout_grid or [0.1, 0.2]
        learning_rates = args.lr_grid or [1e-3, 5e-4]
        batches = args.batch_grid or [32, 64]
        epochs_list = args.epochs_grid or [60, 100]
        patience_list = args.patience_grid or [5, 10]
        val_splits = args.val_split_grid or [0.1, 0.2]

        results = run_sweep(
            data_path=args.data_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            windows=windows,
            steps=steps,
            encoder_options=encoder_options,
            decoder_options=decoder_options,
            dropouts=dropouts,
            learning_rates=learning_rates,
            batches=batches,
            epochs_list=epochs_list,
            patience_list=patience_list,
            val_splits=val_splits,
        )
        results_sorted = sorted(
            results,
            key=lambda r: r.evaluation.classification_report["Anomalía"]["f1-score"],
            reverse=True,
        )
        payload = [result.to_dict() for result in results_sorted]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2))
        return

    encoder_units = tuple(int(v) for v in args.encoder_units[0].split(",")) if args.encoder_units else None
    decoder_units = tuple(int(v) for v in args.decoder_units[0].split(",")) if args.decoder_units else None

    model_config = ModelConfig(
        encoder_units=encoder_units or ModelConfig().encoder_units,
        decoder_units=decoder_units or ModelConfig().decoder_units,
        dropout=args.dropout if args.dropout is not None else ModelConfig().dropout,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate if args.learning_rate is not None else TrainingConfig().learning_rate,
        epochs=args.epochs if args.epochs is not None else TrainingConfig().epochs,
        batch_size=args.batch_size if args.batch_size is not None else TrainingConfig().batch_size,
        validation_split=args.validation_split if args.validation_split is not None else TrainingConfig().validation_split,
        patience=args.patience if args.patience is not None else TrainingConfig().patience,
    )

    artifacts = run_pipeline(
        data_path=args.data_path,
        window=args.window,
        step=args.step,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        model_config=model_config,
        training_config=training_config,
        output_path=args.output,
        analyze_thresholds=not args.no_threshold_analysis,
        evaluate_advanced=args.advanced_models,
    )

    print("\nResumen de calibración:")
    print(json.dumps(asdict(artifacts.calibration), indent=2, ensure_ascii=False))
    print("\nMétricas de evaluación:")
    print(json.dumps(asdict(artifacts.evaluation), indent=2, ensure_ascii=False))

    if artifacts.threshold_experiments:
        print("\nComparativa de umbrales:")
        print(json.dumps([asdict(s) for s in artifacts.threshold_experiments], indent=2, ensure_ascii=False))

    if artifacts.advanced_results:
        print("\nModelos avanzados:")
        print(json.dumps([asdict(r) for r in artifacts.advanced_results], indent=2, ensure_ascii=False))

    if args.report:
        generate_report(args.report, artifacts, args)


if __name__ == "__main__":
    main()
