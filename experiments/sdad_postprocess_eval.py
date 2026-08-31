from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def smooth_score(score: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return score.astype(float)
    return (
        pd.Series(score.astype(float))
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def positive_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.astype(int)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def fill_short_gaps(pred: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return pred.copy()

    out = pred.astype(int).copy()
    segments = positive_segments(out)
    if len(segments) < 2:
        return out

    for (_, previous_end), (next_start, _) in zip(segments[:-1], segments[1:]):
        gap = next_start - previous_end
        if 0 < gap <= max_gap:
            out[previous_end:next_start] = 1
    return out


def remove_short_segments(pred: np.ndarray, min_length: int) -> np.ndarray:
    if min_length <= 1:
        return pred.copy()

    out = pred.astype(int).copy()
    for start, end in positive_segments(out):
        if end - start < min_length:
            out[start:end] = 0
    return out


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    adjusted = pred.astype(int).copy()
    for start, end in positive_segments(labels):
        if adjusted[start:end].any():
            adjusted[start:end] = 1
    return adjusted


def topk_prediction(score: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    pred = np.zeros(len(score), dtype=int)
    if k <= 0:
        return pred, float("inf")

    ranked = np.argsort(score)[::-1]
    selected = ranked[: min(k, len(score))]
    pred[selected] = 1
    threshold = float(score[selected[-1]]) if len(selected) else float("inf")
    return pred, threshold


def quantile_prediction(score: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(score, quantile))
    return (score >= threshold).astype(int), threshold


def evaluate(
    method: str,
    labels: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
    threshold_policy: str,
    threshold: float,
    smoothing_window: int,
    fill_gap: int,
    min_segment_length: int,
    point_adjusted: bool,
) -> dict[str, float | int | str | bool]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    return {
        "method": method,
        "threshold_policy": threshold_policy,
        "threshold": threshold,
        "smoothing_window": smoothing_window,
        "fill_gap": fill_gap,
        "min_segment_length": min_segment_length,
        "point_adjusted": point_adjusted,
        "predicted_anomalies": int(pred.sum()),
        "predicted_segments": len(positive_segments(pred)),
        "true_anomalies": int(labels.sum()),
        "true_segments": len(positive_segments(labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }


def build_initial_prediction(
    score: np.ndarray,
    labels: np.ndarray,
    threshold_policy: str,
    quantile: float,
    train_score: np.ndarray | None = None,
) -> tuple[np.ndarray, float, str]:
    if threshold_policy == "topk":
        pred, threshold = topk_prediction(score, int(labels.sum()))
        return pred, threshold, "exact top-k using true anomaly count"
    if threshold_policy == "quantile":
        pred, threshold = quantile_prediction(score, quantile)
        return pred, threshold, f"test score quantile {quantile:.4f}"
    if threshold_policy == "train_quantile":
        if train_score is None:
            raise ValueError("train_quantile requires train_scores.csv with the same score column")
        threshold = float(np.quantile(train_score, quantile))
        return (score >= threshold).astype(int), threshold, f"train score quantile {quantile:.4f}"
    raise ValueError(f"Unsupported threshold policy: {threshold_policy}")


def run(args: argparse.Namespace) -> pd.DataFrame:
    if not hasattr(args, "method_names"):
        args.method_names = {
            "sdad_recon_score": "SDAD-Reconstruction",
            "sdad_composite_score": "SDAD-Composite",
            "sdad_dimnorm_topk_score": "SDAD-DimNorm-TopK",
            "sdad_dimnorm_mean_score": "SDAD-DimNorm-Mean",
            "sdad_transformer_code_score": "SDAD-Transformer-Code",
            "sdad_hybrid_dimnorm_transformer_score": "SDAD-Hybrid-DimNorm-Transformer",
            "sdad_gpt_code_score": "SDAD-GPT-Code",
            "sdad_hybrid_dimnorm_gpt_score": "SDAD-Hybrid-DimNorm-GPT",
            "sdad_vq_distance_score": "SDAD-VQ-Distance",
            "sdad_vq_position_rarity_score": "SDAD-VQ-Position-Rarity",
            "sdad_vq_transition_rarity_score": "SDAD-VQ-Transition-Rarity",
            "sdad_vq_native_hybrid_score": "SDAD-VQ-Native-Hybrid",
            "sdad_vq_weighted_fusion_score": "SDAD-VQ-Weighted-Fusion",
            "sdad_vq_iforest_fusion_score": "SDAD-VQ-IForest-Fusion",
            "sdad_vq_pca_fusion_score": "SDAD-VQ-PCA-Fusion",
            "mlp_ae_score": "MLP-AE",
            "lstm_ae_score": "LSTM-AE",
            "conv_ae_score": "Conv-AE",
            "sdad_token_rarity_score": "SDAD-Token-Rarity",
            "sdad_transition_rarity_score": "SDAD-Transition-Rarity",
        }

    result_dir = Path(args.result_dir)
    scores_path = result_dir / "scores.csv"
    scores_df = pd.read_csv(scores_path)
    train_scores_df = None
    train_scores_path = result_dir / "train_scores.csv"
    if args.threshold_policy == "train_quantile":
        if not train_scores_path.exists():
            raise FileNotFoundError(f"Missing training scores for train_quantile: {train_scores_path}")
        train_scores_df = pd.read_csv(train_scores_path)

    labels = scores_df["label"].to_numpy(dtype=int)
    rows: list[dict[str, float | int | str | bool]] = []
    output_scores = pd.DataFrame(
        {
            "time_index": scores_df["time_index"].to_numpy(dtype=int),
            "label": labels,
        }
    )

    for column in args.score_columns:
        if column not in scores_df.columns:
            raise ValueError(f"Score column not found in {scores_path}: {column}")

        base_name = args.method_names.get(column, column)
        score = smooth_score(scores_df[column].to_numpy(dtype=float), args.smoothing_window)
        train_score = None
        if train_scores_df is not None:
            if column not in train_scores_df.columns:
                raise ValueError(f"Score column not found in {train_scores_path}: {column}")
            train_score = smooth_score(train_scores_df[column].to_numpy(dtype=float), args.smoothing_window)
        raw_pred, threshold, threshold_text = build_initial_prediction(
            score, labels, args.threshold_policy, args.quantile, train_score
        )
        post_pred = remove_short_segments(
            fill_short_gaps(raw_pred, args.fill_gap), args.min_segment_length
        )
        pa_pred = point_adjust(post_pred, labels)

        rows.append(
            evaluate(
                base_name,
                labels,
                score,
                raw_pred,
                threshold_text,
                threshold,
                args.smoothing_window,
                0,
                1,
                False,
            )
        )
        rows.append(
            evaluate(
                f"{base_name}+PostProcess",
                labels,
                score,
                post_pred,
                threshold_text,
                threshold,
                args.smoothing_window,
                args.fill_gap,
                args.min_segment_length,
                False,
            )
        )
        if args.include_point_adjusted:
            rows.append(
                evaluate(
                    f"{base_name}+PostProcess+PointAdjust",
                    labels,
                    score,
                    pa_pred,
                    threshold_text,
                    threshold,
                    args.smoothing_window,
                    args.fill_gap,
                    args.min_segment_length,
                    True,
                )
            )

        prefix = column.replace("sdad_", "").replace("_score", "")
        output_scores[f"{prefix}_score"] = score
        output_scores[f"{prefix}_raw_pred"] = raw_pred
        output_scores[f"{prefix}_post_pred"] = post_pred
        if args.include_point_adjusted:
            output_scores[f"{prefix}_point_adjusted_pred"] = pa_pred

    metrics_df = pd.DataFrame(rows).sort_values(["point_adjusted", "f1"], ascending=[True, False])
    metrics_path = result_dir / f"{args.output_prefix}_metrics.csv"
    predictions_path = result_dir / f"{args.output_prefix}_scores.csv"
    metrics_df.to_csv(metrics_path, index=False)
    output_scores.to_csv(predictions_path, index=False)

    print(metrics_df.to_string(index=False))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="experiments/results/sdad_smd_machine_1_1")
    parser.add_argument(
        "--score-columns",
        nargs="+",
        default=["sdad_recon_score", "sdad_composite_score"],
    )
    parser.add_argument("--threshold-policy", choices=["topk", "quantile", "train_quantile"], default="topk")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--smoothing-window", type=int, default=1)
    parser.add_argument("--fill-gap", type=int, default=20)
    parser.add_argument("--min-segment-length", type=int, default=3)
    parser.add_argument("--include-point-adjusted", action="store_true")
    parser.add_argument("--output-prefix", default="postprocess")
    args = parser.parse_args()
    args.method_names = {
        "sdad_recon_score": "SDAD-Reconstruction",
        "sdad_composite_score": "SDAD-Composite",
        "sdad_dimnorm_topk_score": "SDAD-DimNorm-TopK",
        "sdad_dimnorm_mean_score": "SDAD-DimNorm-Mean",
        "sdad_transformer_code_score": "SDAD-Transformer-Code",
        "sdad_hybrid_dimnorm_transformer_score": "SDAD-Hybrid-DimNorm-Transformer",
        "sdad_gpt_code_score": "SDAD-GPT-Code",
        "sdad_hybrid_dimnorm_gpt_score": "SDAD-Hybrid-DimNorm-GPT",
        "sdad_vq_distance_score": "SDAD-VQ-Distance",
        "sdad_vq_position_rarity_score": "SDAD-VQ-Position-Rarity",
        "sdad_vq_transition_rarity_score": "SDAD-VQ-Transition-Rarity",
        "sdad_vq_native_hybrid_score": "SDAD-VQ-Native-Hybrid",
        "sdad_vq_weighted_fusion_score": "SDAD-VQ-Weighted-Fusion",
        "sdad_vq_iforest_fusion_score": "SDAD-VQ-IForest-Fusion",
        "sdad_vq_pca_fusion_score": "SDAD-VQ-PCA-Fusion",
        "mlp_ae_score": "MLP-AE",
        "lstm_ae_score": "LSTM-AE",
        "conv_ae_score": "Conv-AE",
        "sdad_token_rarity_score": "SDAD-Token-Rarity",
        "sdad_transition_rarity_score": "SDAD-Transition-Rarity",
    }
    run(args)


if __name__ == "__main__":
    main()
