from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def make_windows(array: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(array) - window_size + 1, stride)
    windows = np.stack([array[i : i + window_size] for i in starts]).astype(np.float32)
    return windows, starts


def normalize_by_train(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = train.min(axis=0)
    hi = train.max(axis=0)
    scale = hi - lo
    scale[scale == 0] = 1.0
    return (train - lo) / scale, (test - lo) / scale


def point_scores_from_windows(
    starts: np.ndarray,
    window_scores: np.ndarray,
    series_length: int,
    window_size: int,
) -> np.ndarray:
    score_sum = np.zeros(series_length, dtype=np.float64)
    score_count = np.zeros(series_length, dtype=np.float64)
    for start, per_timestep_score in zip(starts, window_scores):
        end = start + window_size
        score_sum[start:end] += per_timestep_score
        score_count[start:end] += 1.0
    score_count[score_count == 0] = 1.0
    return score_sum / score_count


def robust_train_scale(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    center = np.quantile(reference, 0.50)
    high = np.quantile(reference, 0.95)
    scale = high - center
    if scale <= 1e-12:
        scale = np.std(reference) + 1e-12
    return np.maximum((values - center) / scale, 0.0)


def token_rarity_scores(
    reference_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    nb_code: int,
    alpha: float = 1.0,
) -> np.ndarray:
    counts = np.bincount(reference_tokens.reshape(-1), minlength=nb_code).astype(np.float64)
    probs = (counts + alpha) / (counts.sum() + alpha * nb_code)
    eval_scores = -np.log(probs[eval_tokens]).mean(axis=1)
    return eval_scores


def transition_rarity_scores(
    reference_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    nb_code: int,
    alpha: float = 1.0,
) -> np.ndarray:
    if reference_tokens.shape[1] < 2:
        return np.zeros(eval_tokens.shape[0], dtype=np.float64)

    train_pairs = reference_tokens[:, :-1] * nb_code + reference_tokens[:, 1:]
    eval_pairs = eval_tokens[:, :-1] * nb_code + eval_tokens[:, 1:]
    counts = np.bincount(train_pairs.reshape(-1), minlength=nb_code * nb_code).astype(np.float64)
    probs = (counts + alpha) / (counts.sum() + alpha * nb_code * nb_code)
    eval_scores = -np.log(probs[eval_pairs]).mean(axis=1)
    return eval_scores


def encode_tokens(model: torch.nn.Module, windows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    token_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            token_batches.append(model.encode(batch).cpu().numpy())
    return np.concatenate(token_batches, axis=0).astype(np.int64)


def reconstruction_scores(
    model: torch.nn.Module,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    per_timestep_batches = []
    window_mean_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            recon, _, _ = model(batch)
            per_timestep = ((recon - batch) ** 2).mean(dim=2).cpu().numpy()
            per_timestep_batches.append(per_timestep)
            window_mean_batches.append(per_timestep.mean(axis=1))
    return np.concatenate(per_timestep_batches, axis=0), np.concatenate(window_mean_batches, axis=0)


def reconstruction_error_windows(
    model: torch.nn.Module,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    error_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            recon, _, _ = model(batch)
            error_batches.append(((recon - batch) ** 2).cpu().numpy())
    return np.concatenate(error_batches, axis=0)


def point_dim_scores_from_windows(
    starts: np.ndarray,
    window_errors: np.ndarray,
    series_length: int,
    window_size: int,
) -> np.ndarray:
    feature_dim = window_errors.shape[2]
    score_sum = np.zeros((series_length, feature_dim), dtype=np.float64)
    score_count = np.zeros((series_length, 1), dtype=np.float64)
    for start, per_timestep_error in zip(starts, window_errors):
        end = start + window_size
        score_sum[start:end] += per_timestep_error
        score_count[start:end] += 1.0
    score_count[score_count == 0] = 1.0
    return score_sum / score_count


def dimension_normalized_scores(
    train_point_errors: np.ndarray,
    eval_point_errors: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.quantile(train_point_errors, 0.50, axis=0)
    high = np.quantile(train_point_errors, 0.95, axis=0)
    scale = high - center
    fallback = np.std(train_point_errors, axis=0) + 1e-12
    scale = np.where(scale <= 1e-12, fallback, scale)
    normalized = np.maximum((eval_point_errors - center) / scale, 0.0)
    mean_score = normalized.mean(axis=1)
    k = max(1, min(top_k, normalized.shape[1]))
    topk_score = np.partition(normalized, -k, axis=1)[:, -k:].mean(axis=1)
    return mean_score, topk_score


def evaluate_scores(method: str, score: np.ndarray, labels: np.ndarray) -> dict[str, float | int | str]:
    labels = labels.astype(int)
    k = int(labels.sum())
    ranked = np.argsort(score)[::-1]
    pred = np.zeros_like(labels, dtype=int)
    pred[ranked[:k]] = 1
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


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.root).resolve()
    vq_source_dir = root / "repos_pipeline" / "sdad_vq"
    sys.path.insert(0, str(vq_source_dir))
    from models.vqvae import VQVAE

    smd_dir = root / "repos_paper" / "TranAD" / "data" / "SMD"
    train = np.loadtxt(smd_dir / "train" / f"{args.machine}.txt", delimiter=",")
    test = np.loadtxt(smd_dir / "test" / f"{args.machine}.txt", delimiter=",")
    labels = np.loadtxt(smd_dir / "labels" / f"{args.machine}.txt", delimiter=",").astype(int)

    train_norm, test_norm = normalize_by_train(train, test)
    train_windows, _ = make_windows(train_norm, args.window_size, args.train_stride)
    train_eval_windows, train_eval_starts = make_windows(train_norm, args.window_size, args.train_score_stride)
    test_windows, test_starts = make_windows(test_norm, args.window_size, args.eval_stride)

    device = torch.device(args.device)
    model_args = SimpleNamespace(
        dataname="smd",
        quantizer=args.quantizer,
        beta=args.beta,
        mu=args.mu,
    )
    model = VQVAE(
        model_args,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act,
        norm=args.vq_norm,
    ).to(device)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    model.train()
    history = []
    step = 0
    while step < args.iters:
        for (batch,) in loader:
            batch = batch.to(device)
            recon, commit_loss, perplexity = model(batch)
            recon_loss = mse(recon, batch)
            loss = recon_loss + args.commit * commit_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            history.append(
                {
                    "iter": step,
                    "loss": float(loss.detach().cpu()),
                    "recon_loss": float(recon_loss.detach().cpu()),
                    "commit_loss": float(commit_loss.detach().cpu()),
                    "perplexity": float(perplexity.detach().cpu()),
                }
            )
            if step >= args.iters:
                break

    train_fit_recon_timestep, train_fit_recon_window = reconstruction_scores(model, train_windows, args.batch_size, device)
    train_eval_recon_timestep, train_eval_recon_window = reconstruction_scores(
        model, train_eval_windows, args.batch_size, device
    )
    test_recon_timestep, test_recon_window = reconstruction_scores(model, test_windows, args.batch_size, device)
    train_eval_recon_errors = reconstruction_error_windows(model, train_eval_windows, args.batch_size, device)
    test_recon_errors = reconstruction_error_windows(model, test_windows, args.batch_size, device)

    train_tokens = encode_tokens(model, train_windows, args.batch_size, device)
    train_eval_tokens = encode_tokens(model, train_eval_windows, args.batch_size, device)
    test_tokens = encode_tokens(model, test_windows, args.batch_size, device)
    train_fit_token_rarity = token_rarity_scores(train_tokens, train_tokens, args.nb_code)
    train_eval_token_rarity = token_rarity_scores(train_tokens, train_eval_tokens, args.nb_code)
    test_token_rarity = token_rarity_scores(train_tokens, test_tokens, args.nb_code)
    train_fit_transition_rarity = transition_rarity_scores(train_tokens, train_tokens, args.nb_code)
    train_eval_transition_rarity = transition_rarity_scores(train_tokens, train_eval_tokens, args.nb_code)
    test_transition_rarity = transition_rarity_scores(train_tokens, test_tokens, args.nb_code)

    recon_window_scaled = robust_train_scale(test_recon_window, train_fit_recon_window)
    token_rarity_scaled = robust_train_scale(test_token_rarity, train_fit_token_rarity)
    transition_rarity_scaled = robust_train_scale(test_transition_rarity, train_fit_transition_rarity)
    composite_window = (
        args.recon_weight * recon_window_scaled
        + args.token_weight * token_rarity_scaled
        + args.transition_weight * transition_rarity_scaled
    )
    train_composite_window = (
        args.recon_weight * robust_train_scale(train_eval_recon_window, train_fit_recon_window)
        + args.token_weight * robust_train_scale(train_eval_token_rarity, train_fit_token_rarity)
        + args.transition_weight * robust_train_scale(train_eval_transition_rarity, train_fit_transition_rarity)
    )

    train_recon_point_scores = point_scores_from_windows(
        train_eval_starts, train_eval_recon_timestep, len(train_norm), args.window_size
    )
    train_recon_point_dim_errors = point_dim_scores_from_windows(
        train_eval_starts, train_eval_recon_errors, len(train_norm), args.window_size
    )
    train_token_point_scores = point_scores_from_windows(
        train_eval_starts,
        np.repeat(train_eval_token_rarity[:, None], args.window_size, axis=1),
        len(train_norm),
        args.window_size,
    )
    train_transition_point_scores = point_scores_from_windows(
        train_eval_starts,
        np.repeat(train_eval_transition_rarity[:, None], args.window_size, axis=1),
        len(train_norm),
        args.window_size,
    )
    train_composite_point_scores = point_scores_from_windows(
        train_eval_starts,
        np.repeat(train_composite_window[:, None], args.window_size, axis=1),
        len(train_norm),
        args.window_size,
    )
    recon_point_scores = point_scores_from_windows(
        test_starts, test_recon_timestep, len(test_norm), args.window_size
    )
    recon_point_dim_errors = point_dim_scores_from_windows(
        test_starts, test_recon_errors, len(test_norm), args.window_size
    )
    train_dimnorm_mean_scores, train_dimnorm_topk_scores = dimension_normalized_scores(
        train_recon_point_dim_errors, train_recon_point_dim_errors, args.top_dim_k
    )
    dimnorm_mean_scores, dimnorm_topk_scores = dimension_normalized_scores(
        train_recon_point_dim_errors, recon_point_dim_errors, args.top_dim_k
    )
    token_point_scores = point_scores_from_windows(
        test_starts,
        np.repeat(test_token_rarity[:, None], args.window_size, axis=1),
        len(test_norm),
        args.window_size,
    )
    transition_point_scores = point_scores_from_windows(
        test_starts,
        np.repeat(test_transition_rarity[:, None], args.window_size, axis=1),
        len(test_norm),
        args.window_size,
    )
    composite_point_scores = point_scores_from_windows(
        test_starts,
        np.repeat(composite_window[:, None], args.window_size, axis=1),
        len(test_norm),
        args.window_size,
    )

    metrics_rows = [
        evaluate_scores("SDAD-Composite", composite_point_scores, labels),
        evaluate_scores("SDAD-Reconstruction", recon_point_scores, labels),
        evaluate_scores("SDAD-DimNorm-TopK", dimnorm_topk_scores, labels),
        evaluate_scores("SDAD-DimNorm-Mean", dimnorm_mean_scores, labels),
        evaluate_scores("SDAD-Token-Rarity", token_point_scores, labels),
        evaluate_scores("SDAD-Transition-Rarity", transition_point_scores, labels),
    ]
    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False)
    metrics = metrics_rows[0]

    out_dir = Path(args.output_dir or f"experiments/results/sdad_smd_{args.machine.replace('-', '_')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_config = vars(args).copy()
    stage1_config["sdad_vq_source"] = str(vq_source_dir / "models" / "vqvae.py")
    (out_dir / "stage1_config.json").write_text(json.dumps(stage1_config, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "window_start": train_eval_starts,
            "recon_window": train_eval_recon_window,
            "token_rarity": train_eval_token_rarity,
            "transition_rarity": train_eval_transition_rarity,
            "composite_window": train_composite_window,
        }
    ).to_csv(out_dir / "train_window_scores.csv", index=False)
    pd.DataFrame(
        {
            "window_start": test_starts,
            "recon_window": test_recon_window,
            "token_rarity": test_token_rarity,
            "transition_rarity": test_transition_rarity,
            "composite_window": composite_window,
        }
    ).to_csv(out_dir / "window_scores.csv", index=False)
    pd.DataFrame(
        {
            "time_index": np.arange(len(train_norm)),
            "sdad_composite_score": train_composite_point_scores,
            "sdad_recon_score": train_recon_point_scores,
            "sdad_dimnorm_topk_score": train_dimnorm_topk_scores,
            "sdad_dimnorm_mean_score": train_dimnorm_mean_scores,
            "sdad_token_rarity_score": train_token_point_scores,
            "sdad_transition_rarity_score": train_transition_point_scores,
        }
    ).to_csv(out_dir / "train_scores.csv", index=False)
    pd.DataFrame(
        {
            "time_index": np.arange(len(labels)),
            "label": labels,
            "sdad_composite_score": composite_point_scores,
            "sdad_recon_score": recon_point_scores,
            "sdad_dimnorm_topk_score": dimnorm_topk_scores,
            "sdad_dimnorm_mean_score": dimnorm_mean_scores,
            "sdad_token_rarity_score": token_point_scores,
            "sdad_transition_rarity_score": transition_point_scores,
        }
    ).to_csv(out_dir / "scores.csv", index=False)
    torch.save(model.state_dict(), out_dir / "sdad_vqvae_smd.pt")

    fig, ax = plt.subplots(figsize=(13, 3.5))
    scaled = (composite_point_scores - composite_point_scores.min()) / (
        composite_point_scores.max() - composite_point_scores.min() + 1e-12
    )
    ax.plot(scaled, linewidth=0.8, label="SDAD composite score")
    ax.fill_between(np.arange(len(labels)), 0, labels, color="tab:red", alpha=0.2, step="pre", label="label")
    ax.set_xlabel("time index")
    ax.set_ylabel("normalized score")
    ax.set_title(f"SDAD anomaly score on SMD {args.machine}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "score_plot.png", dpi=180)
    plt.close(fig)

    report = [
        f"# SDAD on SMD {args.machine}",
        "",
        "SDAD denotes Similarity-guided Discrete-code Anomaly Detection.",
        "This experiment reuses a VQ-VAE component as an anomaly detector.",
        "It combines reconstruction error, code-token rarity, and code-transition rarity learned from normal training windows.",
        "",
        "## Data",
        "",
        f"- Train shape: `{train.shape}`",
        f"- Test shape: `{test.shape}`",
        f"- Labels shape: `{labels.shape}`",
        f"- True anomaly points: `{int(labels.sum())}`",
        "",
        "## Configuration",
        "",
        f"- Window size: `{args.window_size}`",
        f"- Train stride: `{args.train_stride}`",
        f"- Train score stride: `{args.train_score_stride}`",
        f"- Eval stride: `{args.eval_stride}`",
        f"- Iterations: `{args.iters}`",
        f"- Quantizer: `{args.quantizer}`",
        f"- Codebook size: `{args.nb_code}`",
        f"- Code dimension: `{args.code_dim}`",
        f"- Commitment weight: `{args.commit}`",
        f"- EMA mu / beta: `{args.mu}` / `{args.beta}`",
        f"- Dimension-normalized top-k: `{args.top_dim_k}`",
        f"- Model width: `{args.width}`",
        f"- Model depth: `{args.depth}`",
        f"- Down/stride: `{args.down_t}` / `{args.stride_t}`",
        f"- Score weights: recon={args.recon_weight}, token={args.token_weight}, transition={args.transition_weight}",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Outputs",
        "",
        "- `metrics.csv`: metric table",
        "- `train_scores.csv`: per-timestep training anomaly scores for train-quantile thresholds",
        "- `scores.csv`: per-timestep anomaly scores",
        "- `train_window_scores.csv`: per-window training score components",
        "- `window_scores.csv`: per-window score components",
        "- `score_plot.png`: score curve and labels",
        "- `training_history.csv`: training loss trace",
    ]
    (out_dir / "README.md").write_text("\n".join(report), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"Saved results to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--train-score-stride", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--commit", type=float, default=0.02)
    parser.add_argument("--nb-code", type=int, default=64)
    parser.add_argument("--code-dim", type=int, default=32)
    parser.add_argument("--top-dim-k", type=int, default=3)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--vq-act", default="relu", choices=["relu", "silu", "gelu"])
    parser.add_argument("--vq-norm", default=None)
    parser.add_argument(
        "--quantizer",
        default="orig",
        choices=["ema", "orig", "ema_reset", "reset", "lfq", "ema_reset_sim"],
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--recon-weight", type=float, default=0.5)
    parser.add_argument("--token-weight", type=float, default=0.3)
    parser.add_argument("--transition-weight", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
