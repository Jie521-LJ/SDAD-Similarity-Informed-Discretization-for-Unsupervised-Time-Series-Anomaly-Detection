# SDAD: Similarity-Informed Discretization for Unsupervised Time Series Anomaly Detection

<p align="center">
  Reproducible implementation and evaluation on the Server Machine Dataset (SMD)
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-required-ee4c2c.svg">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-SMD-green.svg">
</p>

## Overview

This repository contains the reproducible code package for **SDAD**, an unsupervised multivariate time-series anomaly detection pipeline based on a vector-quantized representation backbone.

The provided implementation learns normal temporal patterns from the **training split only** and evaluates anomaly scores on selected machines from the **Server Machine Dataset (SMD)**. The package includes the SDAD pipeline, lightweight deep-learning baselines, traditional anomaly-detection baselines, and scripts for generating the final evaluation materials.


## Included SMD Subsets

The repository includes the SMD files required for the reported reproduction on the following 10 machines:

```text
machine-1-1  machine-1-2  machine-1-3  machine-1-4  machine-1-5
machine-1-6  machine-1-7  machine-2-1  machine-2-2  machine-2-3
```

The data are stored under:

```text
repos_paper/TranAD/data/SMD/
├── train/
├── test/
└── labels/
```

The **training split is used for unsupervised model fitting**. Test labels are used only when computing evaluation metrics and evaluation-time point-adjusted results. See [`DATASET.md`](DATASET.md) for additional details and data examples.

## Repository Structure

```text
.
├── README.md                       # GitHub repository homepage
├── README_REPRODUCE.md             # Minimal reproduction instructions
├── DATASET.md                      # Included SMD data description
├── requirements.txt                # Python dependencies
├── run_reproduce.sh                # Linux reproduction entry point
├── run_reproduce.ps1               # Windows reproduction entry point
│
├── experiments/
│   ├── run_sdad_vq_pipeline.py     # Main SDAD multi-machine pipeline
│   ├── sdad_vq_anomaly.py          # VQ training and anomaly scores
│   ├── sdad_postprocess_eval.py    # Thresholding and post-processing
│   ├── sdad_vq_native_scores.py    # Additional VQ-native scores
│   ├── sdad_vq_fusion.py           # VQ feature-fusion experiments
│   ├── smd_light_deep_baselines.py # MLP-AE, LSTM-AE, Conv-AE
│   ├── smd_baseline_experiment.py  # ZScore, PCA, Isolation Forest
│   └── build_final_materials.py    # Result tables and figures
│
├── repos_pipeline/
│   └── sdad_vq/
│       └── models/                 # VQ-VAE backbone implementation
│
└── repos_paper/
    └── TranAD/data/SMD/            # Selected SMD train/test/label files
```

## Installation

Clone the repository and install the required Python packages:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
python -m pip install -r requirements.txt
```

Core dependencies are:

```text
numpy
pandas
matplotlib
scikit-learn
torch
tabulate
pillow
```

A CUDA-capable GPU is optional. The reproduction scripts automatically use CUDA when it is available through PyTorch and otherwise fall back to CPU.

## Quick Environment Check

Before running the full experiment, verify that the Python environment and included SMD files are available.

### Linux

```bash
bash run_reproduce.sh --check-only
```

If you want to use a specific conda environment:

```bash
CONDA_ENV_NAME=<your_env> bash run_reproduce.sh --check-only
```

You can also explicitly provide a Python executable:

```bash
PYTHON_BIN=/path/to/python bash run_reproduce.sh --check-only
```

### Windows

The provided PowerShell script uses the conda `base` environment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_reproduce.ps1 -CheckOnly
```

## Reproduce the Experiments

### Linux

```bash
python -m pip install -r requirements.txt
bash run_reproduce.sh
```

### Windows

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_reproduce.ps1
```

The full reproduction performs the following steps:

1. Checks Python dependencies.
2. Checks all required SMD data files.
3. Runs the SDAD VQ anomaly-detection pipeline on the 10 selected machines.
4. Runs lightweight deep baselines: **MLP-AE**, **LSTM-AE**, and **Conv-AE**.
5. Runs traditional baselines: **ZScoreMax**, **PCARecon**, and **IsolationForest**.
6. Builds summary tables and figures from the reproduced results.

## Default SDAD Configuration

The main reproduction script invokes `experiments/run_sdad_vq_pipeline.py` with the following principal defaults:

| Setting | Default |
|---|---:|
| Window size | 24 |
| Training iterations | 400 |
| Batch size | 128 |
| Learning rate | 1e-3 |
| VQ codebook size | 64 |
| Code dimension | 32 |
| Top-K dimensions | 3 |
| Quantizer | `orig` |
| Training-score threshold quantile | 0.995 |
| Score smoothing window | 5 |
| Gap filling | 20 |
| Minimum anomaly segment length | 3 |
| Random seed | 123 |

Additional options are available directly from the command-line interface:

```bash
python experiments/run_sdad_vq_pipeline.py --help
```

For example, a custom subset can be run with:

```bash
python experiments/run_sdad_vq_pipeline.py \
  --root . \
  --machines machine-1-1 machine-1-2 machine-1-3 \
  --device cuda
```

## Evaluation Protocol

SDAD is trained without anomaly labels. Each selected SMD training sequence is normalized using statistics derived from the corresponding training split, divided into sliding windows, and used to fit the VQ-VAE backbone.

For the main reproducible pipeline, anomaly thresholds are estimated from the **training anomaly scores** using a `0.995` quantile. The post-processing stage can additionally apply score smoothing, short-gap filling, minimum-segment filtering, and point adjustment.

The generated metric tables include:

- Precision
- Recall
- F1 score
- AUROC
- AUPRC

> **Evaluation note:** point adjustment uses test labels only as an evaluation-time convention. It is not used to train the model or construct the underlying anomaly score. Raw and post-processed predictions are also saved so that the evaluation protocol remains inspectable.

## Baselines

The reproduction package includes two baseline groups.

**Lightweight deep baselines**

- MLP Autoencoder (`MLP-AE`)
- LSTM Autoencoder (`LSTM-AE`)
- Convolutional Autoencoder (`Conv-AE`)

**Traditional baselines**

- Maximum absolute z-score (`ZScoreMax`)
- PCA reconstruction error (`PCARecon`)
- Isolation Forest (`IsolationForest`)

These baselines are executed automatically by the top-level reproduction scripts.

## Outputs

After a complete run, the main results are written to:

```text
experiments/results/
deliverables/patent_materials/
```

The SDAD experiment directories contain artifacts such as:

```text
metrics.csv
train_scores.csv
scores.csv
train_window_scores.csv
window_scores.csv
training_history.csv
stage1_config.json
sdad_vqvae_smd.pt
score_plot.png
```

The multi-machine pipeline also writes aggregate tables, including:

```text
experiments/results/sdad_vq_light_cuda_summary.csv
experiments/results/sdad_vq_light_cuda_primary_table.csv
```

## Reproducibility Notes

- The default random seed is fixed to `123` in the SDAD pipeline.
- CUDA is used automatically when available through PyTorch; otherwise CPU execution is supported.
- The repository already contains all 30 SMD files required for the 10 selected machines: training data, test data, and labels.
- The Linux entry script can use an active virtual environment, a named conda environment, an explicitly supplied Python binary, or a system Python installation.
- A quick `--check-only` mode is provided so the environment and dataset can be validated before launching the full experiments.

## Dataset

This package uses selected machine subsets from the **Server Machine Dataset (SMD)**. Please refer to [`DATASET.md`](DATASET.md) for the exact included files and data layout.

When redistributing or publishing results based on SMD, please follow the original dataset's applicable citation and licensing requirements.

## Citation

If you use this code in academic work, please cite the corresponding SDAD paper. The final BibTeX entry can be added here when the publication metadata are available.

