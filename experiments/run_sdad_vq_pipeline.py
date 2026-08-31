from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from sdad_postprocess_eval import run as run_postprocess
from sdad_vq_fusion import run as run_fusion
from sdad_vq_native_scores import run as run_vq_native
from sdad_vq_anomaly import run as run_stage1


def result_dir_for(machine: str, prefix: str) -> Path:
    return Path("experiments") / "results" / f"{prefix}_{machine.replace('-', '_')}"


def stage1_args(args: argparse.Namespace, machine: str, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        root=args.root,
        machine=machine,
        window_size=args.window_size,
        train_stride=args.train_stride,
        train_score_stride=args.train_score_stride,
        eval_stride=args.eval_stride,
        iters=args.iters,
        batch_size=args.batch_size,
        lr=args.lr,
        commit=args.commit,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        top_dim_k=args.top_dim_k,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        vq_act=args.vq_act,
        vq_norm=args.vq_norm,
        quantizer=args.quantizer,
        beta=args.beta,
        mu=args.mu,
        device=args.device,
        seed=args.seed,
        recon_weight=args.recon_weight,
        token_weight=args.token_weight,
        transition_weight=args.transition_weight,
        output_dir=str(output_dir),
    )


def postprocess_args(result_dir: Path, args: argparse.Namespace, output_prefix: str) -> SimpleNamespace:
    return SimpleNamespace(
        result_dir=str(result_dir),
        score_columns=args.score_columns,
        threshold_policy="train_quantile",
        quantile=args.quantile,
        smoothing_window=args.smoothing_window,
        fill_gap=args.fill_gap,
        min_segment_length=args.min_segment_length,
        include_point_adjusted=True,
        output_prefix=output_prefix,
    )


def add_context(df: pd.DataFrame, machine: str, source: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "source", source)
    out.insert(0, "machine", machine)
    return out


def run(args: argparse.Namespace) -> None:
    summary_frames: list[pd.DataFrame] = []

    for machine in args.machines:
        out_dir = result_dir_for(machine, args.prefix)
        print(f"\n=== {machine}: VQ stage-1 ===")
        if not args.skip_stage1:
            run_stage1(stage1_args(args, machine, out_dir))

        raw_metrics = pd.read_csv(out_dir / "metrics.csv")
        summary_frames.append(add_context(raw_metrics, machine, "raw_topk"))

        if args.with_vq_native:
            print(f"\n=== {machine}: VQ-native scores ===")
            run_vq_native(
                SimpleNamespace(
                    root=args.root,
                    machine=machine,
                    result_dir=str(out_dir),
                    batch_size=args.batch_size,
                    alpha=1.0,
                    dim_weight=0.50,
                    distance_weight=0.30,
                    position_weight=0.15,
                    transition_weight=0.05,
                    device=args.device,
                    seed=args.seed,
                )
            )
            native_metrics = pd.read_csv(out_dir / "vq_native_metrics.csv")
            summary_frames.append(add_context(native_metrics, machine, "vq_native_topk"))

        if args.with_fusion:
            print(f"\n=== {machine}: VQ feature fusion ===")
            run_fusion(
                SimpleNamespace(
                    result_dir=str(out_dir),
                    feature_columns=[
                        "sdad_recon_score",
                        "sdad_dimnorm_topk_score",
                        "sdad_dimnorm_mean_score",
                        "sdad_composite_score",
                        "sdad_vq_distance_score",
                        "sdad_vq_position_rarity_score",
                        "sdad_vq_transition_rarity_score",
                    ],
                    n_estimators=300,
                    max_samples=4096,
                    contamination="auto",
                    pca_variance=0.95,
                    seed=args.seed,
                )
            )
            fusion_metrics = pd.read_csv(out_dir / "vq_fusion_metrics.csv")
            summary_frames.append(add_context(fusion_metrics, machine, "vq_fusion_topk"))

        print(f"\n=== {machine}: train-quantile postprocess ===")
        pp = run_postprocess(postprocess_args(out_dir, args, "postprocess_vq_train_q0p995"))
        summary_frames.append(add_context(pp, machine, "train_quantile_postprocess"))

    out_root = Path("experiments") / "results"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = out_root / f"{args.prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    primary = summary[
        (summary["source"] == "train_quantile_postprocess")
        & (summary["method"] == "SDAD-DimNorm-TopK+PostProcess+PointAdjust")
    ][["machine", "method", "precision", "recall", "f1", "auroc", "auprc", "predicted_anomalies", "true_anomalies"]]
    primary_path = out_root / f"{args.prefix}_primary_table.csv"
    primary.to_csv(primary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved primary table to: {primary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--prefix", default="sdad_vq_light")
    parser.add_argument("--machines", nargs="+", default=["machine-1-1", "machine-1-2", "machine-1-3"])
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--train-score-stride", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=1)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--recon-weight", type=float, default=0.5)
    parser.add_argument("--token-weight", type=float, default=0.3)
    parser.add_argument("--transition-weight", type=float, default=0.2)
    parser.add_argument(
        "--score-columns",
        nargs="+",
        default=[
            "sdad_dimnorm_topk_score",
            "sdad_dimnorm_mean_score",
            "sdad_composite_score",
            "sdad_recon_score",
        ],
    )
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--fill-gap", type=int, default=20)
    parser.add_argument("--min-segment-length", type=int, default=3)
    parser.add_argument("--with-vq-native", action="store_true")
    parser.add_argument("--with-fusion", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
