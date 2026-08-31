from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sdad_vq_anomaly import (
    evaluate_scores,
    make_windows,
    normalize_by_train,
    point_scores_from_windows,
    robust_train_scale,
)


def load_stage1_config(result_dir: Path) -> dict:
    config_path = result_dir / "stage1_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing stage1 config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_vqvae(root: Path, result_dir: Path, config: dict, device: torch.device) -> torch.nn.Module:
    vq_source_dir = root / "repos_pipeline" / "sdad_vq"
    sys.path.insert(0, str(vq_source_dir))
    from models.vqvae import VQVAE

    model_args = SimpleNamespace(
        dataname="smd",
        quantizer=config["quantizer"],
        beta=config["beta"],
        mu=config["mu"],
    )
    model = VQVAE(
        model_args,
        nb_code=config["nb_code"],
        code_dim=config["code_dim"],
        down_t=config["down_t"],
        stride_t=config["stride_t"],
        width=config["width"],
        depth=config["depth"],
        dilation_growth_rate=config["dilation_growth_rate"],
        activation=config["vq_act"],
        norm=config["vq_norm"],
    ).to(device)
    model.load_state_dict(torch.load(result_dir / "sdad_vqvae_smd.pt", map_location=device))
    model.eval()
    return model


def get_codebook(model: torch.nn.Module) -> torch.Tensor:
    quantizer = model.vqvae.quantizer
    if hasattr(quantizer, "codebook"):
        return quantizer.codebook
    if hasattr(quantizer, "embedding"):
        return quantizer.embedding.weight
    raise TypeError(f"Unsupported quantizer for codebook extraction: {type(quantizer)!r}")


def latent_vq_distance_and_tokens(
    model: torch.nn.Module,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
    quantizer: str,
) -> tuple[np.ndarray, np.ndarray]:
    codebook = get_codebook(model).detach()
    distance_batches: list[np.ndarray] = []
    token_batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            latent = model.encode2(batch)
            batch_size_now, code_len, code_dim = latent.shape
            flat = latent.reshape(-1, code_dim)

            if quantizer in {"ema_reset_sim", "lfq"}:
                flat_norm = F.normalize(flat, dim=-1)
                codebook_norm = F.normalize(codebook, dim=-1)
                similarity = flat_norm @ codebook_norm.t()
                max_similarity, tokens = similarity.max(dim=-1)
                distance = 1.0 - max_similarity
            else:
                distance_matrix = torch.cdist(flat, codebook, p=2).pow(2) / max(code_dim, 1)
                distance, tokens = distance_matrix.min(dim=-1)

            distance_batches.append(distance.reshape(batch_size_now, code_len).detach().cpu().numpy())
            token_batches.append(tokens.reshape(batch_size_now, code_len).detach().cpu().numpy())

    return np.concatenate(distance_batches, axis=0), np.concatenate(token_batches, axis=0).astype(np.int64)


def expand_code_scores(token_scores: np.ndarray, window_size: int) -> np.ndarray:
    code_len = token_scores.shape[1]
    if code_len == window_size:
        return token_scores
    positions = np.linspace(0, code_len - 1, window_size)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    weight = positions - left
    return (1.0 - weight)[None, :] * token_scores[:, left] + weight[None, :] * token_scores[:, right]


def position_rarity_scores(
    reference_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    nb_code: int,
    alpha: float,
) -> np.ndarray:
    code_len = reference_tokens.shape[1]
    scores = np.zeros_like(eval_tokens, dtype=np.float64)
    for index in range(code_len):
        counts = np.bincount(reference_tokens[:, index], minlength=nb_code).astype(np.float64)
        probs = (counts + alpha) / (counts.sum() + alpha * nb_code)
        scores[:, index] = -np.log(probs[eval_tokens[:, index]])
    return scores


def transition_rarity_scores(
    reference_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    nb_code: int,
    alpha: float,
) -> np.ndarray:
    code_len = reference_tokens.shape[1]
    scores = np.zeros_like(eval_tokens, dtype=np.float64)
    for index in range(code_len):
        train_prev = reference_tokens[:-1, index]
        train_next = reference_tokens[1:, index]
        pair_ids = train_prev * nb_code + train_next
        counts = np.bincount(pair_ids, minlength=nb_code * nb_code).astype(np.float64)
        probs = (counts + alpha) / (counts.sum() + alpha * nb_code * nb_code)

        eval_score = np.full(len(eval_tokens), -np.log(1.0 / nb_code), dtype=np.float64)
        if len(eval_tokens) > 1:
            eval_pairs = eval_tokens[:-1, index] * nb_code + eval_tokens[1:, index]
            eval_score[1:] = -np.log(probs[eval_pairs])
            eval_score[0] = float(np.median(eval_score[1:]))
        scores[:, index] = eval_score
    return scores


def token_point_scores(
    starts: np.ndarray,
    token_scores: np.ndarray,
    series_length: int,
    window_size: int,
) -> np.ndarray:
    return point_scores_from_windows(starts, expand_code_scores(token_scores, window_size), series_length, window_size)


def add_scores(
    score_df: pd.DataFrame,
    distance_point: np.ndarray,
    position_point: np.ndarray,
    transition_point: np.ndarray,
    hybrid: np.ndarray,
) -> pd.DataFrame:
    out = score_df.copy()
    out["sdad_vq_distance_score"] = distance_point
    out["sdad_vq_position_rarity_score"] = position_point
    out["sdad_vq_transition_rarity_score"] = transition_point
    out["sdad_vq_native_hybrid_score"] = hybrid
    return out


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.root).resolve()
    result_dir = Path(args.result_dir or f"experiments/results/sdad_smd_{args.machine.replace('-', '_')}")
    config = load_stage1_config(result_dir)
    device = torch.device(args.device)

    smd_dir = root / "repos_paper" / "TranAD" / "data" / "SMD"
    train = np.loadtxt(smd_dir / "train" / f"{args.machine}.txt", delimiter=",")
    test = np.loadtxt(smd_dir / "test" / f"{args.machine}.txt", delimiter=",")
    labels = np.loadtxt(smd_dir / "labels" / f"{args.machine}.txt", delimiter=",").astype(int)

    train_norm, test_norm = normalize_by_train(train, test)
    train_fit_windows, _ = make_windows(train_norm, config["window_size"], config["train_stride"])
    train_eval_windows, train_eval_starts = make_windows(
        train_norm, config["window_size"], config["train_score_stride"]
    )
    test_windows, test_starts = make_windows(test_norm, config["window_size"], config["eval_stride"])

    model = load_vqvae(root, result_dir, config, device)
    fit_distance, fit_tokens = latent_vq_distance_and_tokens(
        model, train_fit_windows, args.batch_size, device, config["quantizer"]
    )
    train_distance, train_tokens = latent_vq_distance_and_tokens(
        model, train_eval_windows, args.batch_size, device, config["quantizer"]
    )
    test_distance, test_tokens = latent_vq_distance_and_tokens(
        model, test_windows, args.batch_size, device, config["quantizer"]
    )

    train_position = position_rarity_scores(fit_tokens, train_tokens, config["nb_code"], args.alpha)
    test_position = position_rarity_scores(fit_tokens, test_tokens, config["nb_code"], args.alpha)
    train_transition = transition_rarity_scores(fit_tokens, train_tokens, config["nb_code"], args.alpha)
    test_transition = transition_rarity_scores(fit_tokens, test_tokens, config["nb_code"], args.alpha)

    train_distance_point = token_point_scores(train_eval_starts, train_distance, len(train_norm), config["window_size"])
    test_distance_point = token_point_scores(test_starts, test_distance, len(test_norm), config["window_size"])
    train_position_point = token_point_scores(train_eval_starts, train_position, len(train_norm), config["window_size"])
    test_position_point = token_point_scores(test_starts, test_position, len(test_norm), config["window_size"])
    train_transition_point = token_point_scores(
        train_eval_starts, train_transition, len(train_norm), config["window_size"]
    )
    test_transition_point = token_point_scores(test_starts, test_transition, len(test_norm), config["window_size"])

    train_scores_path = result_dir / "train_scores.csv"
    test_scores_path = result_dir / "scores.csv"
    train_scores = pd.read_csv(train_scores_path)
    test_scores = pd.read_csv(test_scores_path)
    if len(train_scores) != len(train_distance_point) or len(test_scores) != len(test_distance_point):
        raise ValueError("VQ-native scores do not align with existing point score files")

    train_dim = train_scores["sdad_dimnorm_topk_score"].to_numpy(dtype=float)
    test_dim = test_scores["sdad_dimnorm_topk_score"].to_numpy(dtype=float)
    train_hybrid = (
        args.dim_weight * robust_train_scale(train_dim, train_dim)
        + args.distance_weight * robust_train_scale(train_distance_point, train_distance_point)
        + args.position_weight * robust_train_scale(train_position_point, train_position_point)
        + args.transition_weight * robust_train_scale(train_transition_point, train_transition_point)
    )
    test_hybrid = (
        args.dim_weight * robust_train_scale(test_dim, train_dim)
        + args.distance_weight * robust_train_scale(test_distance_point, train_distance_point)
        + args.position_weight * robust_train_scale(test_position_point, train_position_point)
        + args.transition_weight * robust_train_scale(test_transition_point, train_transition_point)
    )

    train_scores = add_scores(
        train_scores, train_distance_point, train_position_point, train_transition_point, train_hybrid
    )
    test_scores = add_scores(test_scores, test_distance_point, test_position_point, test_transition_point, test_hybrid)
    train_scores.to_csv(train_scores_path, index=False)
    test_scores.to_csv(test_scores_path, index=False)

    metrics = pd.DataFrame(
        [
            evaluate_scores("SDAD-VQ-Distance", test_distance_point, labels),
            evaluate_scores("SDAD-VQ-Position-Rarity", test_position_point, labels),
            evaluate_scores("SDAD-VQ-Transition-Rarity", test_transition_point, labels),
            evaluate_scores("SDAD-VQ-Native-Hybrid", test_hybrid, labels),
        ]
    ).sort_values("f1", ascending=False)
    metrics.to_csv(result_dir / "vq_native_metrics.csv", index=False)

    used_codes = int(np.unique(fit_tokens).size)
    top_share = float(np.bincount(fit_tokens.reshape(-1), minlength=config["nb_code"]).max() / fit_tokens.size)
    print(metrics.to_string(index=False))
    print(f"Train code usage: {used_codes}/{config['nb_code']}; top code share: {top_share:.4f}")
    print(f"Saved VQ-native scores to: {result_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dim-weight", type=float, default=0.50)
    parser.add_argument("--distance-weight", type=float, default=0.30)
    parser.add_argument("--position-weight", type=float, default=0.15)
    parser.add_argument("--transition-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
