from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


DEFAULT_FEATURE_COLUMNS = [
    "sdad_recon_score",
    "sdad_dimnorm_topk_score",
    "sdad_dimnorm_mean_score",
    "sdad_composite_score",
    "sdad_vq_distance_score",
    "sdad_vq_position_rarity_score",
    "sdad_vq_transition_rarity_score",
]


def robust_feature_scale(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.quantile(train_x, 0.50, axis=0)
    high = np.quantile(train_x, 0.95, axis=0)
    scale = high - center
    fallback = np.std(train_x, axis=0) + 1e-12
    scale = np.where(scale <= 1e-12, fallback, scale)
    return np.maximum((train_x - center) / scale, 0.0), np.maximum((test_x - center) / scale, 0.0)


def evaluate_scores(method: str, score: np.ndarray, labels: np.ndarray) -> dict[str, float | int | str]:
    labels = labels.astype(int)
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


def pca_reconstruction_score(train_x: np.ndarray, test_x: np.ndarray, variance: float) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=variance, svd_solver="full")
    pca.fit(train_x)
    train_recon = pca.inverse_transform(pca.transform(train_x))
    test_recon = pca.inverse_transform(pca.transform(test_x))
    return ((train_x - train_recon) ** 2).mean(axis=1), ((test_x - test_recon) ** 2).mean(axis=1)


def run(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir)
    train_scores_path = result_dir / "train_scores.csv"
    test_scores_path = result_dir / "scores.csv"
    train_df = pd.read_csv(train_scores_path)
    test_df = pd.read_csv(test_scores_path)
    labels = test_df["label"].to_numpy(dtype=int)

    feature_columns = [col for col in args.feature_columns if col in train_df.columns and col in test_df.columns]
    if not feature_columns:
        raise ValueError("No usable VQ feature columns were found")

    train_x_raw = train_df[feature_columns].to_numpy(dtype=np.float64)
    test_x_raw = test_df[feature_columns].to_numpy(dtype=np.float64)
    train_x, test_x = robust_feature_scale(train_x_raw, test_x_raw)

    weighted_train = train_x.mean(axis=1)
    weighted_test = test_x.mean(axis=1)

    iforest = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=min(args.max_samples, len(train_x)),
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    iforest.fit(train_x)
    iforest_train = -iforest.score_samples(train_x)
    iforest_test = -iforest.score_samples(test_x)

    pca_train, pca_test = pca_reconstruction_score(train_x, test_x, args.pca_variance)

    train_df["sdad_vq_weighted_fusion_score"] = weighted_train
    test_df["sdad_vq_weighted_fusion_score"] = weighted_test
    train_df["sdad_vq_iforest_fusion_score"] = iforest_train
    test_df["sdad_vq_iforest_fusion_score"] = iforest_test
    train_df["sdad_vq_pca_fusion_score"] = pca_train
    test_df["sdad_vq_pca_fusion_score"] = pca_test

    train_df.to_csv(train_scores_path, index=False)
    test_df.to_csv(test_scores_path, index=False)

    metrics = pd.DataFrame(
        [
            evaluate_scores("SDAD-VQ-Weighted-Fusion", weighted_test, labels),
            evaluate_scores("SDAD-VQ-IForest-Fusion", iforest_test, labels),
            evaluate_scores("SDAD-VQ-PCA-Fusion", pca_test, labels),
        ]
    ).sort_values("f1", ascending=False)
    metrics.to_csv(result_dir / "vq_fusion_metrics.csv", index=False)

    print(f"feature_columns: {feature_columns}")
    print(metrics.to_string(index=False))
    print(f"Saved VQ fusion scores to: {result_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="experiments/results/sdad_smd_machine_1_1")
    parser.add_argument("--feature-columns", nargs="+", default=DEFAULT_FEATURE_COLUMNS)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--contamination", default="auto")
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=123)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
