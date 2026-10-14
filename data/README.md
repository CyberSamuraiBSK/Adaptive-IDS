# Dataset instructions

This project expects a CSV of pre-extracted numeric network-flow features with a target column named `Label`.

1. Obtain CICIDS2017 from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Combine/sample the CSV files as needed for your experiment.
3. Ensure benign traffic is labeled `BENIGN`, or pass the correct value using `--normal-label`.
4. Place the prepared CSV here, or pass its path with `--dataset`.

## ⚠️
The dataset is not included in this repository. 

Preprocessing replaces infinite values with missing values and removes rows with missing feature/label values. The scaler is fitted only on the training split to avoid test-set leakage.
