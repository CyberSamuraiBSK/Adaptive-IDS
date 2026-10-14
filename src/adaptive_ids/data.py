"""CSV loading and leakage-aware preprocessing."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    label_encoder: LabelEncoder
    scaler: StandardScaler
    normal_class_id: int


def load_dataset(
    path: str | Path,
    label_column: str = "Label",
    normal_label: str = "BENIGN",
    test_size: float = 0.30,
    random_state: int = 42,
) -> DatasetSplit:
    """Load a labeled flow CSV and create a stratified train/test split."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See data/README.md for setup instructions."
        )

    frame = pd.read_csv(path, low_memory=False)
    if label_column not in frame.columns:
        raise ValueError(
            f"Expected target column {label_column!r}; found columns: {list(frame.columns)}"
        )

    y_raw = frame[label_column].astype(str).str.strip()
    features = frame.drop(columns=[label_column]).select_dtypes(include=[np.number])
    if features.empty:
        raise ValueError("No numeric feature columns were found in the CSV.")

    features = features.replace([np.inf, -np.inf], np.nan)
    valid = features.notna().all(axis=1) & y_raw.notna()
    features = features.loc[valid].astype(np.float64)
    y_raw = y_raw.loc[valid]

    if len(features) < 10:
        raise ValueError(f"Only {len(features)} valid rows remain; more data is needed.")
    if y_raw.nunique() < 2:
        raise ValueError("At least two label classes are required.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    if normal_label not in encoder.classes_:
        preview = ", ".join(map(str, encoder.classes_[:10]))
        raise ValueError(
            f"Normal label {normal_label!r} not found. Available labels include: {preview}. "
            "Pass the correct value with --normal-label."
        )
    normal_id = int(encoder.transform([normal_label])[0])

    # Stratification can fail when a class contains fewer than two rows.
    counts = pd.Series(y).value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features.to_numpy(),
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(features.columns),
        label_encoder=encoder,
        scaler=scaler,
        normal_class_id=normal_id,
    )
