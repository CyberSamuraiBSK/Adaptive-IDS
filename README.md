# Adaptive-IDS

**A hybrid, adaptive intrusion detection system for known, anomalous, and adversarially perturbed network-flow data.**

This Adaptive-IDS investigates the possibility of using lightweight incremental learning for an IDS to adapt to changing traffic patterns without retraining all models from scratch each time.

---

## ⚙️How it works

```text
CICIDS2017-dataset CSV
        |
        v
Preprocessing + train-only feature scaling
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

---

### Main components

- **Supervised classifier:** an `SGDClassifier` classifies known traffic/attack labels and supports incremental `partial_fit` updates.
- **Anomaly detector:** an `IsolationForest` identifies flows that differ from its learned baseline. It is fitted on benign training flows by default.
- **Background retraining:** a worker consumes bounded batches of *confirmed, labeled* feedback and updates the classifier without making the caller wait for model training.
- **Evaluation:** reports standard classification metrics for the supervised model and binary benign/anomaly metrics for Isolation Forest.
- **Adversarial experiment:** an optional PGD-style perturbation routine explores model sensitivity in standardized feature space.

## ⚠️Important scope and limitations

This detection system is a **prototype for research purposes** and not an actual network sensor. It only reads the extracted flow features from CSV files and does not involve any packet capturing or implementation of firewall policies.

The labeling can be done using the benchmarking data for evaluation. In a real-world setting, however, the right labels would not be available at the time of making predictions. This means that online learning updates would have to be done using some sort of an analyst or through an incident response process. Feedback replay used in this project is an experiment in the form of simulated feedback.

Adversarial perturbations on feature space are valuable for experiments under control but do not have anything to do with real attacks. The results must be presented as robustness in feature space and cannot be considered evidence of attack resistance.

This approach is deliberately conservative because it neither retunes for each prediction nor regards the model’s prediction of an unknown situation as being true.

---

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

---

## 📋Requirements

- Python 3.10 or newer
- pandas
- NumPy
- scikit-learn
- joblib

## 📐Setup

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

---

## ✅Run the project

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

---

## 🤖Evaluation methodology

For a meaningful experiment, compare at least:

1. **Static baseline:** evaluate the initially trained models without updates.
2. **Feedback replay:** evaluate the initial model, then update only from a bounded stream of confirmed labels and evaluate again on a separate future segment.
3. **Adversarial feature-space evaluation:** compare clean and perturbed test inputs across several perturbation budgets.
4. **Resource impact:** measure prediction latency, retraining duration, queue depth, and memory use.
5. **Robustness and safety:** report per-class precision/recall/F1, false-positive rate, and class imbalance; do not rely on accuracy alone.

Use a time-ordered split where possible to reduce leakage between highly similar flows. For a rigorous adversarial experiment, perturb only features that can meaningfully change in a real flow and validate the generated examples against domain constraints.

---

## 🧪Research idea and contribution

The fundamental principle here is that of combining supervised attack classifications with unsupervised anomaly detections, followed by asynchronous, incremental updates to lower the costs of adaptation to changing data. However, the question raised in the research is not so much whether retraining will improve the score, but **how and when this is achieved, and at what costs of latency, false positives, and instability**.

Potential extensions:
- Adaptive retraining thresholds based on recent error/drift measurements.
- Drift detection and a policy for deciding when retraining is justified.
- Model versioning, rollback, and validation before promoting updated models.
- Analyst feedback integration for verified labels.
- Flow extraction from PCAP files or integration with a live telemetry source.
- Reproducible comparison against static SGD, Isolation Forest, and other baselines.

---

## 🔬Development status

This repo serves as a modular beginning point for research work. Numbers should be produced using the dataset and setting for each individual experiment, and it cannot be expected that numbers given in the source paper will automatically replicate using this implementation.

---

## 🛟Safety

Experiment using offline data sets or a lab network that you have permission to experiment with. No packet injection or blocking is performed with this project.
