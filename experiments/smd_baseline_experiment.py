from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler


def load_smd_machine(root: Path, machine: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = root / "repos_paper" / "TranAD" / "data" / "SMD"
    train = np.loadtxt(base / "train" / f"{machine}.txt", delimiter=",")
    test = np.loadtxt(base / "test" / f"{machine}.txt", delimiter=",")
    labels = np.loadtxt(base / "labels" / f"{machine}.txt", delimiter=",").astype(int)
    if len(test) != len(labels):
        raise ValueError(f"test/label length mismatch: {len(test)} != {len(labels)}")
    return train, test, labels


def normalize_scores(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    lo = np.nanmin(score)
    hi = np.nanmax(score)
    if hi <= lo:
        return np.zeros_like(score)
    return (score - lo) / (hi - lo)


def evaluate_score(name: str, score: np.ndarray, y_true: np.ndarray) -> dict[str, float | str | int]:
    k = int(y_true.sum())
    ranked = np.argsort(score)[::-1]
    y_pred = np.zeros_like(y_true, dtype=int)
    y_pred[ranked[:k]] = 1
    threshold = float(score[ranked[k - 1]]) if k > 0 else float("inf")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "method": name,
        "threshold_policy": "exact top-k using true anomaly count",
        "threshold": threshold,
        "predicted_anomalies": int(y_pred.sum()),
        "true_anomalies": int(y_true.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
    }


def run_experiment(root: Path, machine: str, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test, labels = load_smd_machine(root, machine)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    scores: dict[str, np.ndarray] = {}

    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0)
    train_std[train_std == 0] = 1.0
    scores["ZScoreMax"] = np.max(np.abs((test - train_mean) / train_std), axis=1)

    pca = PCA(n_components=0.95, svd_solver="full", random_state=0)
    train_pca = pca.fit_transform(train_scaled)
    _ = train_pca  # Keeps the fit step explicit for readability.
    test_recon = pca.inverse_transform(pca.transform(test_scaled))
    scores["PCARecon"] = np.mean((test_scaled - test_recon) ** 2, axis=1)

    iso = IsolationForest(
        n_estimators=100,
        contamination=float(labels.mean()),
        random_state=0,
        n_jobs=-1,
    )
    iso.fit(train_scaled)
    scores["IsolationForest"] = -iso.score_samples(test_scaled)

    metrics = [evaluate_score(name, score, labels) for name, score in scores.items()]
    metrics_df = pd.DataFrame(metrics).sort_values("f1", ascending=False)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    sample_rows = pd.DataFrame(
        {
            "time_index": np.arange(len(labels)),
            "label": labels,
            **{f"score_{name}": score for name, score in scores.items()},
        }
    )
    sample_rows.to_csv(output_dir / "scores.csv", index=False)

    fig, axes = plt.subplots(len(scores), 1, figsize=(13, 8), sharex=True)
    if len(scores) == 1:
        axes = [axes]
    x = np.arange(len(labels))
    for ax, (name, score) in zip(axes, scores.items()):
        ax.plot(x, normalize_scores(score), linewidth=0.8, label=f"{name} score")
        ax.fill_between(x, 0, labels, color="tab:red", alpha=0.2, step="pre", label="label")
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time index")
    fig.suptitle(f"SMD {machine}: baseline anomaly scores vs labels")
    fig.tight_layout()
    fig.savefig(output_dir / "score_plot.png", dpi=180)
    plt.close(fig)

    report = [
        f"# SMD Baseline Experiment: {machine}",
        "",
        "## Data",
        "",
        f"- Train shape: `{train.shape}`",
        f"- Test shape: `{test.shape}`",
        f"- Label shape: `{labels.shape}`",
        f"- True anomalies: `{int(labels.sum())}`",
        f"- Anomaly ratio: `{labels.mean():.4f}`",
        "",
        "## Methods",
        "",
        "- `ZScoreMax`: standardize each dimension by train statistics, use max absolute z-score.",
        "- `PCARecon`: fit PCA on normalized train data, use reconstruction error as anomaly score.",
        "- `IsolationForest`: fit Isolation Forest on normalized train data, use negative score as anomaly score.",
        "",
        "Threshold policy: predict the same anomaly ratio as the label set, for a controlled first-pass baseline comparison.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Outputs",
        "",
        "- `metrics.csv`: metric table",
        "- `scores.csv`: per-timestamp anomaly scores",
        "- `score_plot.png`: score curves with anomaly labels",
    ]
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")
    return metrics_df.assign(machine=machine)


def run_batch(root: Path, machines: list[str], output_root: Path) -> pd.DataFrame:
    frames = []
    for machine in machines:
        out_dir = output_root / f"smd_baselines_{machine.replace('-', '_')}"
        frames.append(run_experiment(root, machine, out_dir))
    summary = pd.concat(frames, ignore_index=True)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "smd_baseline_multi_machine.csv", index=False)
    print(f"Saved multi-machine baseline summary to: {output_root / 'smd_baseline_multi_machine.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--machines", nargs="+", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.machines:
        run_batch(Path(args.root), args.machines, Path(args.output_dir or "experiments/results"))
    else:
        output_dir = args.output_dir or f"experiments/results/smd_baselines_{args.machine.replace('-', '_')}"
        run_experiment(Path(args.root), args.machine, Path(output_dir))


if __name__ == "__main__":
    main()
