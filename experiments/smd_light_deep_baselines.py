from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from sdad_vq_anomaly import make_windows, normalize_by_train, point_scores_from_windows


class MLPAutoEncoder(nn.Module):
    def __init__(self, window_size: int, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        flat_dim = window_size * input_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.flatten(1))
        return out.view(x.shape[0], self.window_size, self.input_dim)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, window_size: int, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = torch.relu(self.to_latent(hidden[-1]))
        repeated = torch.relu(self.from_latent(latent)).unsqueeze(1).repeat(1, self.window_size, 1)
        decoded, _ = self.decoder(repeated)
        return self.out(decoded)


class ConvAutoEncoder(nn.Module):
    def __init__(self, window_size: int, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        _ = window_size
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x.permute(0, 2, 1))
        out = self.decoder(z)
        return out.permute(0, 2, 1)


def build_model(name: str, window_size: int, input_dim: int, hidden_dim: int, latent_dim: int) -> nn.Module:
    if name == "mlp_ae":
        return MLPAutoEncoder(window_size, input_dim, hidden_dim, latent_dim)
    if name == "lstm_ae":
        return LSTMAutoEncoder(window_size, input_dim, hidden_dim, latent_dim)
    if name == "conv_ae":
        return ConvAutoEncoder(window_size, input_dim, hidden_dim, latent_dim)
    raise ValueError(f"Unknown model: {name}")


def evaluate_scores(method: str, score: np.ndarray, labels: np.ndarray) -> dict[str, float | int | str]:
    k = int(labels.sum())
    pred = np.zeros_like(labels, dtype=int)
    if k > 0:
        pred[np.argsort(score)[::-1][:k]] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    return {
        "method": method,
        "threshold_policy": "exact top-k using true anomaly count",
        "predicted_anomalies": int(pred.sum()),
        "true_anomalies": int(labels.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }


def train_model(
    model: nn.Module,
    train_windows: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    rows = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = mse(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        rows.append({"epoch": epoch, "loss": float(np.mean(losses))})
    return pd.DataFrame(rows)


def reconstruction_timestep_scores(
    model: nn.Module,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            recon = model(batch)
            batches.append(((recon - batch) ** 2).mean(dim=2).detach().cpu().numpy())
    return np.concatenate(batches, axis=0)


def plot_scores(labels: np.ndarray, score_columns: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(score_columns), 1, figsize=(13, 8), sharex=True)
    if len(score_columns) == 1:
        axes = [axes]
    x = np.arange(len(labels))
    for ax, (name, score) in zip(axes, score_columns.items()):
        lo, hi = float(np.min(score)), float(np.max(score))
        scaled = (score - lo) / (hi - lo + 1e-12)
        ax.plot(x, scaled, linewidth=0.8, label=name)
        ax.fill_between(x, 0, labels, color="tab:red", alpha=0.2, step="pre", label="label")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_machine(args: argparse.Namespace, machine: str) -> pd.DataFrame:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.root).resolve()
    smd_dir = root / "repos_paper" / "TranAD" / "data" / "SMD"
    train = np.loadtxt(smd_dir / "train" / f"{machine}.txt", delimiter=",")
    test = np.loadtxt(smd_dir / "test" / f"{machine}.txt", delimiter=",")
    labels = np.loadtxt(smd_dir / "labels" / f"{machine}.txt", delimiter=",").astype(int)

    train_norm, test_norm = normalize_by_train(train, test)
    train_windows, _ = make_windows(train_norm, args.window_size, args.train_stride)
    train_eval_windows, train_eval_starts = make_windows(train_norm, args.window_size, args.train_score_stride)
    test_windows, test_starts = make_windows(test_norm, args.window_size, args.eval_stride)

    out_dir = Path(args.output_dir or "experiments/results/light_deep_baselines") / machine.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    input_dim = train.shape[1]

    train_scores = pd.DataFrame({"time_index": np.arange(len(train_norm))})
    test_scores = pd.DataFrame({"time_index": np.arange(len(labels)), "label": labels})
    metrics_rows = []
    score_columns: dict[str, np.ndarray] = {}

    for model_name in args.models:
        print(f"=== {machine}: {model_name} ===")
        model = build_model(model_name, args.window_size, input_dim, args.hidden_dim, args.latent_dim).to(device)
        history = train_model(model, train_windows, args, device)
        history.to_csv(out_dir / f"{model_name}_history.csv", index=False)

        train_recon = reconstruction_timestep_scores(model, train_eval_windows, args.batch_size, device)
        test_recon = reconstruction_timestep_scores(model, test_windows, args.batch_size, device)
        train_point = point_scores_from_windows(
            train_eval_starts, train_recon, len(train_norm), args.window_size
        )
        test_point = point_scores_from_windows(test_starts, test_recon, len(test_norm), args.window_size)

        column = f"{model_name}_score"
        train_scores[column] = train_point
        test_scores[column] = test_point
        score_columns[column] = test_point
        metrics_rows.append(evaluate_scores(model_name.upper().replace("_", "-"), test_point, labels))

    train_scores.to_csv(out_dir / "train_scores.csv", index=False)
    test_scores.to_csv(out_dir / "scores.csv", index=False)
    metrics = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    plot_scores(labels, score_columns, out_dir / "score_plot.png", f"SMD {machine}: lightweight deep baselines")

    report = [
        f"# Lightweight Deep Baselines on SMD {machine}",
        "",
        "These baselines are unsupervised reconstruction models trained only on the normal training split.",
        "",
        f"- Train shape: `{train.shape}`",
        f"- Test shape: `{test.shape}`",
        f"- True anomaly points: `{int(labels.sum())}`",
        f"- Window size: `{args.window_size}`",
        f"- Epochs: `{args.epochs}`",
        f"- Hidden / latent dimensions: `{args.hidden_dim}` / `{args.latent_dim}`",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
    ]
    (out_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(metrics.to_string(index=False))
    return metrics.assign(machine=machine)


def run(args: argparse.Namespace) -> None:
    frames = [run_machine(args, machine) for machine in args.machines]
    summary = pd.concat(frames, ignore_index=True)
    out_dir = Path(args.output_dir or "experiments/results/light_deep_baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "light_deep_baseline_summary.csv", index=False)
    average = summary.groupby("method")[["precision", "recall", "f1", "auroc", "auprc"]].mean().reset_index()
    average.to_csv(out_dir / "light_deep_baseline_average.csv", index=False)
    print(f"Saved summary to: {out_dir / 'light_deep_baseline_summary.csv'}")
    print(f"Saved average to: {out_dir / 'light_deep_baseline_average.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--machines", nargs="+", default=["machine-1-1", "machine-1-2", "machine-1-3"])
    parser.add_argument("--models", nargs="+", default=["mlp_ae", "lstm_ae", "conv_ae"])
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--train-score-stride", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
