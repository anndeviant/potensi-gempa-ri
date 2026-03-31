from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BACKEND_DIR = Path(__file__).resolve().parent
DATA_PATH = BACKEND_DIR / "data" / "hazard" / "grid_df.csv"
MODEL_DIR = BACKEND_DIR / "models" / "hazard"

FEATURE_COLUMNS = [
    "max_mag",
    "avg_mag",
    "avg_depth",
    "gempa_in_radius_50",
    "density",
]


@dataclass
class TrainingResult:
    model_path: Path
    encoder_path: Path
    metadata_path: Path
    train_size: int
    test_size: int
    accuracy: float


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File data hazard tidak ditemukan: {path}")

    df = pd.read_csv(path)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak lengkap: {missing}")

    data = df[FEATURE_COLUMNS].copy()
    data = data.dropna()
    if data.empty:
        raise ValueError("Data hazard kosong setelah dropna")
    return data


def make_hazard_label(df: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    score = df["max_mag"] * 0.4 + df["density"] * 0.3 + df["gempa_in_radius_50"] * 0.3

    q1 = float(score.quantile(0.25))
    q2 = float(score.quantile(0.50))
    q3 = float(score.quantile(0.75))

    bins = [-np.inf, q1, q2, q3, np.inf]
    labels = ["low", "medium", "high", "very_high"]
    hazard = pd.cut(score, bins=bins, labels=labels, include_lowest=True)

    return hazard.astype(str), {"q1": q1, "q2": q2, "q3": q3}


def train_model(data: pd.DataFrame) -> TrainingResult:
    y_label, quantiles = make_hazard_label(data)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_label)
    X = data[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(encoder.classes_),
        random_state=42,
        n_estimators=300,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "hazard_model.joblib"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    metadata_path = MODEL_DIR / "training_metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "data_path": str(DATA_PATH),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": accuracy,
        "classes": encoder.classes_.tolist(),
        "hazard_score_quantiles": quantiles,
        "classification_report": class_report,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainingResult(
        model_path=model_path,
        encoder_path=encoder_path,
        metadata_path=metadata_path,
        train_size=len(X_train),
        test_size=len(X_test),
        accuracy=accuracy,
    )


def main() -> None:
    data = load_training_data(DATA_PATH)
    result = train_model(data)

    print("Hazard model retraining selesai")
    print(f"- Train size: {result.train_size}")
    print(f"- Test size: {result.test_size}")
    print(f"- Accuracy: {result.accuracy:.4f}")
    print(f"- Model: {result.model_path}")
    print(f"- Encoder: {result.encoder_path}")
    print(f"- Metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()
