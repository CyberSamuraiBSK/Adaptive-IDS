# Adaptive-IDS

**A hybrid, adaptive intrusion detection system for known, anomalous, and adversarially perturbed network-flow data.**

Adaptive-IDS explores whether lightweight incremental learning can help an intrusion detection system respond to changing traffic patterns without repeatedly retraining every model from scratch.

The project is based on the idea and prototype in *AI-based Intrusion Detection System: Real-Time Retraining for Adaptive Security against Known and Adversarial Attacks*. It reorganizes the prototype into a reproducible Python project focused on implementation, evaluation, and further development.

## ⚙️How it works

```text
CICIDS2017-style CSV
        |
        v
Cleaning + train-only feature scaling
        |
        +------------------------------+
        |                              |
        v                              v
 SGDClassifier                 Isolation Forest
 Known attack classes          Traffic anomalies
        |                              |
        +---------------+--------------+
                        |
                        v
              Detection / evaluation
                        |
                        v
          Confirmed-label feedback buffer
                        |
                        v
            Background partial retraining
                        |
                        v
                Updated model state
```

### Main components

- **Supervised classifier:** an `SGDClassifier` classifies known traffic/attack labels and supports incremental `partial_fit` updates.
- **Anomaly detector:** an `IsolationForest` identifies flows that differ from its learned baseline. It is fitted on benign training flows by default.
- **Background retraining:** a worker consumes bounded batches of *confirmed, labeled* feedback and updates the classifier without making the caller wait for model training.
- **Evaluation:** reports standard classification metrics for the supervised model and binary benign/anomaly metrics for Isolation Forest.
- **Adversarial experiment:** an optional PGD-style perturbation routine explores model sensitivity in standardized feature space.

## ⚠️Important scope and limitations

This repository is a **research prototype**, not a production network sensor. It currently reads pre-extracted flow features from CSV files; it does not capture packets, extract flows from a live interface, or enforce firewall rules.

Labels are available in benchmark evaluation data. In a real deployment, correct labels are not normally known at prediction time. Therefore, online updates must come from an analyst, incident response workflow, or another trusted labeling source. The optional feedback replay in this project is an experiment that simulates such confirmed feedback; it must not be interpreted as an automatic way to discover ground truth.

Feature-space adversarial perturbations are useful for controlled experiments but may not correspond to valid, realizable network traffic. Results should be reported as feature-space robustness, not proof of resistance to real attacks.

The implementation is intentionally conservative: it does not automatically retrain on every prediction or treat an unverified model prediction as ground truth.

## 🏗️Repository structure

```text
Adaptive-IDS/
├── README.md
├── requirements.txt
├── .gitignore
├── config.json
├── data/
│   └── README.md
├── reports/
│   └── .gitkeep
├── src/
│   └── adaptive_ids/
│       ├── __init__.py
│       ├── data.py
│       ├── detector.py
│       ├── retraining.py
│       ├── attacks.py
│       ├── evaluation.py
│       └── cli.py
└── tests/
    └── test_data.py
```

## Requirements

- Python 3.10 or newer
- pandas
- NumPy
- scikit-learn
- joblib

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Adaptive-IDS.git
cd Adaptive-IDS
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Obtain the dataset

Download CICIDS2017 from the official Canadian Institute for Cybersecurity dataset page:
https://www.unb.ca/cic/datasets/ids-2017.html

Prepare a CSV containing numeric flow features and a target column named `Label`. Place it at `data/cicids2017_combined_sampled.csv`, or supply another path using `--dataset`.

Do not commit the dataset unless you have checked its licensing, size, and redistribution terms. See [`data/README.md`](data/README.md).

## Run the project

From the repository root:

```bash
python -m src.adaptive_ids.cli --dataset data/cicids2017_combined_sampled.csv
```

The command trains both models, evaluates them on a held-out test split, and writes a JSON report to `reports/metrics.json`.

To run a small, controlled feature-space perturbation experiment:

```bash
python -m src.adaptive_ids.cli --dataset data/cicids2017_combined_sampled.csv --pgd
```

To simulate incremental learning from a stream of **confirmed labels** after an initial evaluation:

```bash
python -m src.adaptive_ids.cli --dataset data/cicids2017_combined_sampled.csv --feedback-replay
```

`--feedback-replay` is an offline experiment: test labels are used as if they had become confirmed feedback after prediction. This is not a realistic assumption for arbitrary live traffic, and its metrics must be labeled accordingly.

Use `--help` to see all command-line options.

## Evaluation methodology

For a meaningful experiment, compare at least:

1. **Static baseline:** evaluate the initially trained models without updates.
2. **Feedback replay:** evaluate the initial model, then update only from a bounded stream of confirmed labels and evaluate again on a separate future segment.
3. **Adversarial feature-space evaluation:** compare clean and perturbed test inputs across several perturbation budgets.
4. **Resource impact:** measure prediction latency, retraining duration, queue depth, and memory use.
5. **Robustness and safety:** report per-class precision/recall/F1, false-positive rate, and class imbalance; do not rely on accuracy alone.

Use a time-ordered split where possible to reduce leakage between highly similar flows. For a rigorous adversarial experiment, perturb only features that can meaningfully change in a real flow and validate the generated examples against domain constraints.

## Research idea and contribution

The core idea is to combine supervised known-attack classification with unsupervised anomaly detection, then use asynchronous, incremental updates to reduce the cost of adapting to changing data. The research question is not simply whether retraining improves a score, but **when it improves detection, how quickly it does so, and what latency, false-positive, and stability costs it introduces**.

Potential extensions:
- Adaptive retraining thresholds based on recent error/drift measurements.
- Drift detection and a policy for deciding when retraining is justified.
- Model versioning, rollback, and validation before promoting updated models.
- Analyst feedback integration for verified labels.
- Flow extraction from PCAP files or integration with a live telemetry source.
- Reproducible comparison against static SGD, Isolation Forest, and other baselines.

## 🔬Development status

This repository is a modular research starting point. Results must be generated on the dataset and configuration used for each experiment; numbers reported in the source paper should not be assumed to reproduce automatically with this refactored implementation.

## Safety

Run experiments on offline datasets or a lab network you are authorized to test. This project does not perform packet injection or deploy blocking rules.
