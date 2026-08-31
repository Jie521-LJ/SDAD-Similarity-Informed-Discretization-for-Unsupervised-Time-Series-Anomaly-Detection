from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from sdad_vq_anomaly import encode_tokens, make_windows, normalize_by_train


def entropy_perplexity(tokens: np.ndarray, nb_code: int) -> tuple[int, float, float]:
    flat = tokens.reshape(-1)
    counts = np.bincount(flat, minlength=nb_code).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    used = int((counts > 0).sum())
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    ppl = float(np.exp(entropy))
    top_share = float(counts.max() / max(counts.sum(), 1.0))
    return used, ppl, top_share


def load_model(root: Path, result_dir: Path, config: dict, device: torch.device) -> torch.nn.Module:
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


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    result_dir = Path(args.result_dir or f"experiments/results/sdad_smd_{args.machine.replace('-', '_')}")
    config = json.loads((result_dir / "stage1_config.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)

    smd_dir = root / "repos_paper" / "TranAD" / "data" / "SMD"
    train = np.loadtxt(smd_dir / "train" / f"{args.machine}.txt", delimiter=",")
    test = np.loadtxt(smd_dir / "test" / f"{args.machine}.txt", delimiter=",")
    labels = np.loadtxt(smd_dir / "labels" / f"{args.machine}.txt", delimiter=",").astype(int)
    train_norm, test_norm = normalize_by_train(train, test)
    train_windows, _ = make_windows(train_norm, config["window_size"], args.stride)
    test_windows, _ = make_windows(test_norm, config["window_size"], args.stride)

    model = load_model(root, result_dir, config, device)
    train_tokens = encode_tokens(model, train_windows, args.batch_size, device)
    test_tokens = encode_tokens(model, test_windows, args.batch_size, device)

    train_used, train_ppl, train_top = entropy_perplexity(train_tokens, config["nb_code"])
    test_used, test_ppl, test_top = entropy_perplexity(test_tokens, config["nb_code"])
    adjacent_same = float((train_tokens[1:] == train_tokens[:-1]).all(axis=1).mean()) if len(train_tokens) > 1 else 0.0
    slot_used = [int(np.unique(train_tokens[:, i]).size) for i in range(train_tokens.shape[1])]

    history = pd.read_csv(result_dir / "training_history.csv")
    metrics = pd.read_csv(result_dir / "metrics.csv")
    stage2_path = result_dir / "stage2_gpt_metrics.csv"

    print(f"machine: {args.machine}")
    print(f"config: quantizer={config['quantizer']}, nb_code={config['nb_code']}, code_dim={config['code_dim']}, width={config['width']}, depth={config['depth']}, iters={config['iters']}")
    print(f"train_shape: {train.shape}, test_shape: {test.shape}, anomaly_points: {int(labels.sum())}")
    print(f"code_len: {train_tokens.shape[1]}, train_windows: {len(train_tokens)}, test_windows: {len(test_tokens)}")
    print(f"train_code_usage: {train_used}/{config['nb_code']}, token_ppl={train_ppl:.2f}, top_code_share={train_top:.3f}")
    print(f"test_code_usage: {test_used}/{config['nb_code']}, token_ppl={test_ppl:.2f}, top_code_share={test_top:.3f}")
    print(f"slot_unique_codes: {slot_used}")
    print(f"adjacent_window_exact_same: {adjacent_same:.3f}")
    print("training_tail:")
    print(history.tail(5).to_string(index=False))
    print("stage1_metrics:")
    print(metrics[["method", "f1", "auroc", "auprc"]].to_string(index=False))
    if stage2_path.exists():
        stage2 = pd.read_csv(stage2_path)
        print("stage2_gpt_metrics:")
        print(stage2[["method", "f1", "auroc", "auprc"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
