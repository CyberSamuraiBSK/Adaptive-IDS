import numpy as np
import pandas as pd

from src.adaptive_ids.data import load_dataset


def test_loader_splits_and_scales_training_only(tmp_path):
    rng = np.random.default_rng(7)
    count = 100
    frame = pd.DataFrame({
        "Flow Duration": rng.normal(10, 2, count),
        "Packet Count": rng.normal(20, 5, count),
        "Label": ["BENIGN"] * 50 + ["ATTACK"] * 50,
    })
    path = tmp_path / "flows.csv"
    frame.to_csv(path, index=False)

    split = load_dataset(path, random_state=11)
    assert split.X_train.shape[1] == 2
    assert len(split.X_train) + len(split.X_test) == count
    assert split.normal_class_id == int(split.label_encoder.transform(["BENIGN"])[0])
    assert np.isfinite(split.X_train).all()
    assert np.isfinite(split.X_test).all()
