from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

REFERENCES_MD = """# References

The references are grouped by their role in the technical material.

## Core Representation

[1] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural Discrete Representation Learning. NeurIPS, 2017. https://arxiv.org/abs/1711.00937

## Time-Series Anomaly Detection and Public Benchmarks

[2] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network. KDD, 2019. https://doi.org/10.1145/3292500.3330672

[3] Shreshth Tuli, Giuliano Casale, and Nicholas R. Jennings. TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data. PVLDB, 2022. https://arxiv.org/abs/2201.07284

[4] Julien Audibert, Pietro Michiardi, Frederic Guyard, Sebastien Marti, and Maria A. Zuluaga. USAD: UnSupervised Anomaly Detection on Multivariate Time Series. KDD, 2020. https://dl.acm.org/doi/10.1145/3394486.3403392

[5] Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. ICLR, 2022. https://openreview.net/forum?id=LzQQ89U1qm_

[6] Hang Zhao, Yujing Wang, Juanyong Duan, Congrui Huang, Defu Cao, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, and Qi Zhang. Multivariate Time-series Anomaly Detection via Graph Attention Network. ICDM, 2020. https://arxiv.org/abs/2009.02040

## Baselines and Evaluation

[7] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Isolation Forest. ICDM, 2008. https://doi.org/10.1109/ICDM.2008.17

[8] Ian T. Jolliffe. Principal Component Analysis. Springer, 2002. https://doi.org/10.1007/b98835

[9] Alexander Lavin and Subutai Ahmad. Evaluating Real-Time Anomaly Detection Algorithms: The Numenta Anomaly Benchmark. ICMLA, 2015. https://doi.org/10.1109/ICMLA.2015.141

[10] Renjie Wu and Eamonn J. Keogh. Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress. ICDE, 2022. https://arxiv.org/abs/2009.13807
"""

REFERENCES_BIB = r"""@inproceedings{oord2017vqvae,
  title = {Neural Discrete Representation Learning},
  author = {van den Oord, Aaron and Vinyals, Oriol and Kavukcuoglu, Koray},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017},
  url = {https://arxiv.org/abs/1711.00937}
}

@inproceedings{su2019omnianomaly,
  title = {Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network},
  author = {Su, Ya and Zhao, Youjian and Niu, Chenhao and Liu, Rong and Sun, Wei and Pei, Dan},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages = {2828--2837},
  year = {2019},
  doi = {10.1145/3292500.3330672}
}

@article{tuli2022tranad,
  title = {TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data},
  author = {Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R.},
  journal = {Proceedings of the VLDB Endowment},
  year = {2022},
  url = {https://arxiv.org/abs/2201.07284}
}

@inproceedings{audibert2020usad,
  title = {USAD: UnSupervised Anomaly Detection on Multivariate Time Series},
  author = {Audibert, Julien and Michiardi, Pietro and Guyard, Frederic and Marti, Sebastien and Zuluaga, Maria A.},
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages = {3395--3404},
  year = {2020},
  doi = {10.1145/3394486.3403392}
}

@inproceedings{xu2022anomalytransformer,
  title = {Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
  author = {Xu, Jiehui and Wu, Haixu and Wang, Jianmin and Long, Mingsheng},
  booktitle = {International Conference on Learning Representations},
  year = {2022},
  url = {https://openreview.net/forum?id=LzQQ89U1qm_}
}

@inproceedings{zhao2020mtadgat,
  title = {Multivariate Time-series Anomaly Detection via Graph Attention Network},
  author = {Zhao, Hang and Wang, Yujing and Duan, Juanyong and Huang, Congrui and Cao, Defu and Tong, Yunhai and Xu, Bixiong and Bai, Jing and Tong, Jie and Zhang, Qi},
  booktitle = {IEEE International Conference on Data Mining},
  year = {2020},
  url = {https://arxiv.org/abs/2009.02040}
}

@inproceedings{liu2008isolationforest,
  title = {Isolation Forest},
  author = {Liu, Fei Tony and Ting, Kai Ming and Zhou, Zhi-Hua},
  booktitle = {IEEE International Conference on Data Mining},
  pages = {413--422},
  year = {2008},
  doi = {10.1109/ICDM.2008.17}
}

@book{jolliffe2002pca,
  title = {Principal Component Analysis},
  author = {Jolliffe, Ian T.},
  publisher = {Springer},
  year = {2002},
  doi = {10.1007/b98835}
}

@inproceedings{lavin2015nab,
  title = {Evaluating Real-Time Anomaly Detection Algorithms: The Numenta Anomaly Benchmark},
  author = {Lavin, Alexander and Ahmad, Subutai},
  booktitle = {IEEE International Conference on Machine Learning and Applications},
  year = {2015},
  doi = {10.1109/ICMLA.2015.141}
}

@inproceedings{wu2022benchmarks,
  title = {Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress},
  author = {Wu, Renjie and Keogh, Eamonn J.},
  booktitle = {IEEE International Conference on Data Engineering},
  pages = {1479--1480},
  year = {2022},
  url = {https://arxiv.org/abs/2009.13807}
}
"""


def draw_box(ax, xy, width, height, title, subtitle="", face="#F7FAFC", edge="#2D3748") -> None:
    x, y = xy
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.022",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", fontsize=10, weight="bold")
    if subtitle:
        ax.text(x + width / 2, y + height * 0.32, subtitle, ha="center", va="center", fontsize=8.1, color="#4A5568")


def arrow(ax, start, end) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.15,
            color="#2D3748",
            shrinkA=3,
            shrinkB=3,
        )
    )


def markdown_table(df: pd.DataFrame, columns: list[str], floatfmt: str = ".4f") -> str:
    out = df[columns].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: format(value, floatfmt))
    return out.to_markdown(index=False)


def read_first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readline().strip()


def dataset_sample_block(root: Path) -> str:
    base = root / "repos_paper" / "TranAD" / "data" / "SMD"
    train_sample = read_first_line(base / "train" / "machine-1-1.txt")
    test_sample = read_first_line(base / "test" / "machine-1-1.txt")
    label_sample = read_first_line(base / "labels" / "machine-1-1.txt")
    n_dims = len(train_sample.split(","))
    return f"""## Data File Example

Each SMD data row is a comma-separated multivariate time step. The selected `machine-1-1` files have `{n_dims}` variables per row. The label file contains one binary value per test time step, where `1` indicates anomaly and `0` indicates normal.

Train sample from `repos_paper/TranAD/data/SMD/train/machine-1-1.txt`:

```text
{train_sample}
```

Test sample from `repos_paper/TranAD/data/SMD/test/machine-1-1.txt`:

```text
{test_sample}
```

Label sample from `repos_paper/TranAD/data/SMD/labels/machine-1-1.txt`:

```text
{label_sample}
```
"""


def build_pipeline_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 8.4))
    ax.set_xlim(0, 1.025)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "SDAD: Single-stage VQ-based Time-Series Anomaly Detection",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color="#1A202C",
    )
    ax.text(
        0.5,
        0.915,
        "The submitted technical solution uses VQ representation and reconstruction scoring; no Transformer stage is included in the final pipeline.",
        ha="center",
        va="center",
        fontsize=10,
        color="#4A5568",
    )

    ax.text(0.045, 0.79, "Training path", ha="left", va="center", fontsize=11, weight="bold", color="#2F855A")
    ax.text(0.045, 0.44, "Detection path", ha="left", va="center", fontsize=11, weight="bold", color="#2B6CB0")

    train_y = 0.67
    infer_y = 0.32
    width = 0.100
    height = 0.13
    train_x = [0.050, 0.225, 0.400, 0.575, 0.750]
    infer_x = [0.050, 0.225, 0.400, 0.575, 0.750, 0.895]

    draw_box(ax, (train_x[0], train_y), width, height, "Normal Train Data", "SMD train split", "#E6FFFA", "#2C7A7B")
    draw_box(ax, (train_x[1], train_y), width, height, "Normalize + Window", "fit train scaler", "#EDF2F7")
    draw_box(ax, (train_x[2], train_y), width, height, "Train VQ-VAE", "learn codebook", "#EBF8FF", "#2B6CB0")
    draw_box(ax, (train_x[3], train_y), width, height, "Train Error Stats", "per-dim median/q95", "#F0FFF4", "#2F855A")
    draw_box(ax, (train_x[4], train_y), width, height, "Train Threshold", "score q=0.995", "#F0FFF4", "#2F855A")

    for left, right in zip(train_x[:-1], train_x[1:]):
        arrow(ax, (left + width, train_y + height / 2), (right, train_y + height / 2))

    draw_box(ax, (infer_x[0], infer_y), width, height, "Test Data", "same variables", "#E6FFFA", "#2C7A7B")
    draw_box(ax, (infer_x[1], infer_y), width, height, "Normalize + Window", "use train scaler", "#EDF2F7")
    draw_box(ax, (infer_x[2], infer_y), width, height, "VQ Reconstruct", "encode, quantize, decode", "#EBF8FF", "#2B6CB0")
    draw_box(ax, (infer_x[3], infer_y), width, height, "DimNorm Top-k", "robust per-dim score", "#FFF5F5", "#C53030")
    draw_box(ax, (infer_x[4], infer_y), width, height, "Threshold", "train quantile", "#FEFCBF", "#975A16")
    draw_box(ax, (infer_x[5], infer_y), width, height, "Anomaly Segments", "gap fill + short-run filter", "#F0FFF4", "#2F855A")

    for left, right in zip(infer_x[:-1], infer_x[1:]):
        arrow(ax, (left + width, infer_y + height / 2), (right, infer_y + height / 2))

    arrow(ax, (train_x[2] + width / 2, train_y), (infer_x[2] + width / 2, infer_y + height))
    ax.text(train_x[2] + width / 2, 0.54, "shared VQ codebook", ha="center", va="center", fontsize=8.8, color="#4A5568")

    arrow(ax, (train_x[3] + width / 2, train_y), (infer_x[3] + width / 2, infer_y + height))
    ax.text(train_x[3] + width / 2, 0.54, "normal error distribution", ha="center", va="center", fontsize=8.8, color="#4A5568")

    arrow(ax, (train_x[4] + width / 2, train_y), (infer_x[4] + width / 2, infer_y + height))
    ax.text(train_x[4] + width / 2, 0.54, "deployment threshold", ha="center", va="center", fontsize=8.8, color="#4A5568")

    note = FancyBboxPatch(
        (0.13, 0.095),
        0.74,
        0.09,
        boxstyle="round,pad=0.016,rounding_size=0.018",
        linewidth=1.0,
        edgecolor="#A0AEC0",
        facecolor="#F7FAFC",
    )
    ax.add_patch(note)
    ax.text(
        0.5,
        0.14,
        "Stage-2 Transformer/GPT was tested, but code-sequence scores did not yield stable gains; the final solution keeps the single-stage VQ scoring pipeline.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#2D3748",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "method_pipeline.png", dpi=220)
    fig.savefig(output_dir / "method_pipeline.pdf")
    plt.close(fig)


def build_result_figures(output_dir: Path, primary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    machines = primary["machine"].tolist()
    x = np.arange(len(machines))

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.bar(x, primary["f1"], color="#2B6CB0", width=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels(machines, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("SDAD Main Result with Segment-level Point Adjustment")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(primary["f1"]):
        ax.text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "main_result_f1_by_machine.png", dpi=220)
    plt.close(fig)

    comparison = comparison.sort_values("machine")
    machines = comparison["machine"].tolist()
    x = np.arange(len(machines))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13.4, 4.8))
    ax.bar(x - width / 2, comparison["best_sdad_f1"], width, label="SDAD", color="#2B6CB0")
    ax.bar(x + width / 2, comparison["best_light_deep_f1"], width, label="Best lightweight deep baseline", color="#C05621")
    ax.set_xticks(x)
    ax.set_xticklabels(machines, rotation=20, ha="right")
    ax.set_ylim(0, max(0.8, float(comparison[["best_sdad_f1", "best_light_deep_f1"]].to_numpy().max()) + 0.12))
    ax.set_ylabel("F1")
    ax.set_title("SDAD vs Lightweight Deep Baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_comparison_f1.png", dpi=220)
    plt.close(fig)


CORE_SDAD_METHODS = [
    "SDAD-Reconstruction",
    "SDAD-DimNorm-Mean",
    "SDAD-DimNorm-TopK",
    "SDAD-Composite",
]


def ordered_methods(df: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["_method_order"] = out["method"].map({name: index for index, name in enumerate(method_order)}).fillna(999)
    return out.sort_values(["_method_order", "method"]).drop(columns=["_method_order"])


def build_sdad_light_comparison(results: Path, summary: pd.DataFrame) -> pd.DataFrame:
    raw = summary[(summary["source"] == "raw_topk") & summary["method"].isin(CORE_SDAD_METHODS)].copy()
    best_sdad = raw.sort_values(["machine", "f1"], ascending=[True, False]).groupby("machine", as_index=False).first()
    best_sdad = best_sdad.rename(
        columns={
            "method": "best_sdad_method",
            "f1": "best_sdad_f1",
            "auroc": "best_sdad_auroc",
            "auprc": "best_sdad_auprc",
        }
    )[["machine", "best_sdad_method", "best_sdad_f1", "best_sdad_auroc", "best_sdad_auprc"]]

    light_summary = pd.read_csv(results / "light_deep_baselines_cuda" / "light_deep_baseline_summary.csv")
    best_light = (
        light_summary.sort_values(["machine", "f1"], ascending=[True, False]).groupby("machine", as_index=False).first()
    )
    best_light = best_light.rename(
        columns={
            "method": "best_light_deep_method",
            "f1": "best_light_deep_f1",
            "auroc": "best_light_deep_auroc",
            "auprc": "best_light_deep_auprc",
        }
    )[["machine", "best_light_deep_method", "best_light_deep_f1", "best_light_deep_auroc", "best_light_deep_auprc"]]

    comparison = best_sdad.merge(best_light, on="machine", how="inner")
    comparison["sdad_minus_light_deep_f1"] = comparison["best_sdad_f1"] - comparison["best_light_deep_f1"]
    comparison["sdad_wins"] = comparison["sdad_minus_light_deep_f1"] > 0
    comparison.to_csv(results / "sdad_vs_light_deep_best_raw.csv", index=False)
    return comparison


def build_ablation_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = summary[(summary["source"] == "raw_topk") & summary["method"].isin(CORE_SDAD_METHODS)].copy()
    raw_avg = raw.groupby("method")[["precision", "recall", "f1", "auroc", "auprc"]].mean().reset_index()
    raw_avg = ordered_methods(raw_avg, CORE_SDAD_METHODS)

    post_methods = [f"{method}+PostProcess+PointAdjust" for method in CORE_SDAD_METHODS]
    post = summary[
        (summary["source"] == "train_quantile_postprocess") & summary["method"].isin(post_methods)
    ].copy()
    post["method"] = post["method"].str.replace("+PostProcess+PointAdjust", "", regex=False)
    post_avg = post.groupby("method")[["precision", "recall", "f1", "auroc", "auprc"]].mean().reset_index()
    post_avg = ordered_methods(post_avg, CORE_SDAD_METHODS)
    return raw_avg, post_avg


def load_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = root / "experiments" / "results"
    primary = pd.read_csv(results / "sdad_vq_light_cuda_primary_table.csv")
    summary = pd.read_csv(results / "sdad_vq_light_cuda_summary.csv")
    comparison = build_sdad_light_comparison(results, summary)
    raw_ablation, post_ablation = build_ablation_tables(summary)
    traditional = pd.read_csv(results / "smd_baseline_multi_machine.csv")
    traditional = traditional[["machine", "method", "f1", "auroc", "auprc"]].sort_values(
        ["machine", "f1"], ascending=[True, False]
    )
    deep_avg = pd.read_csv(results / "light_deep_baselines_cuda" / "light_deep_baseline_average.csv")
    return primary, comparison, traditional, deep_avg, raw_ablation, post_ablation


def build_materials(output_dir: Path, root: Path = ROOT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary, comparison, traditional, deep_avg, raw_ablation, post_ablation = load_tables(root)
    primary_mean = primary[["precision", "recall", "f1", "auroc", "auprc"]].mean()
    comparison_mean = comparison[["best_sdad_f1", "best_light_deep_f1"]].mean()
    machines = primary["machine"].tolist()
    machine_text = ", ".join(f"`{machine}`" for machine in machines)

    primary.to_csv(output_dir / "main_results_point_adjusted.csv", index=False)
    comparison.to_csv(output_dir / "sdad_vs_lightweight_deep_baselines.csv", index=False)
    traditional.to_csv(output_dir / "traditional_baselines.csv", index=False)
    deep_avg.to_csv(output_dir / "lightweight_deep_baseline_average.csv", index=False)
    raw_ablation.to_csv(output_dir / "ablation_raw_topk_average.csv", index=False)
    post_ablation.to_csv(output_dir / "ablation_postprocess_point_adjusted_average.csv", index=False)

    build_pipeline_figure(output_dir)
    build_result_figures(output_dir, primary, comparison)
    data_sample = dataset_sample_block(root)

    (output_dir / "03_references.md").write_text(REFERENCES_MD, encoding="utf-8")
    (output_dir / "references.bib").write_text(REFERENCES_BIB, encoding="utf-8")

    index_text = """# SDAD Patent Materials Index

This folder contains only patent-oriented technical materials, figures, and result tables.
It does not contain experiment code, model checkpoints, or intermediate arrays.

## Files

| File | Purpose |
|---|---|
| `01_technical_disclosure_summary.md` | Technical disclosure draft summary |
| `02_experiment_report.md` | Experiment setting and result summary |
| `03_references.md` | Literature references grouped by technical role |
| `references.bib` | BibTeX entries for report writing |
| `method_pipeline.png` | Main method pipeline figure |
| `method_pipeline.pdf` | PDF version of the pipeline figure |
| `main_result_f1_by_machine.png` | Main SDAD F1 result figure |
| `baseline_comparison_f1.png` | SDAD vs lightweight deep baselines figure |
| `main_results_point_adjusted.csv` | Main result table |
| `sdad_vs_lightweight_deep_baselines.csv` | Deep baseline comparison table |
| `ablation_raw_topk_average.csv` | Raw top-k ablation table |
| `ablation_postprocess_point_adjusted_average.csv` | Point-adjusted ablation table |
| `traditional_baselines.csv` | Traditional baseline results |
| `lightweight_deep_baseline_average.csv` | Deep baseline average results |
"""
    (output_dir / "00_file_index.md").write_text(index_text, encoding="utf-8")

    disclosure = f"""# Technical Disclosure Summary

## Technical Title

Vector-quantized representation based anomaly detection method for multivariate time-series data.

## Technical Field

The method relates to time-series data analysis, machine monitoring, server monitoring, and unsupervised anomaly detection.

## Technical Problem

Multivariate monitoring series often contain many variables with different scales and local anomaly patterns. Methods based only on global reconstruction error may dilute local variable-level anomalies. The proposed method introduces a vector-quantized normal-pattern codebook and dimension-normalized scoring to make local abnormal dimensions more visible.

## Technical Solution

The method trains a VQ-VAE on normal time-series windows to learn a discrete codebook of normal patterns. During detection, test windows are encoded, quantized, and reconstructed through the VQ bottleneck. Reconstruction error is computed at each time step and variable dimension. The error of each variable is normalized using the training error distribution, and the largest top-k normalized dimensions are aggregated as the anomaly score. A train-quantile threshold and segment postprocessing are then applied to output anomaly intervals.

The representation backbone is related to vector-quantized latent representation learning [1]. The evaluation setting follows the public multivariate time-series anomaly detection protocol commonly used by SMD, TranAD, USAD, Anomaly Transformer, and MTAD-GAT studies [2-6].

## Final Pipeline Scope

The final technical solution is a single-stage VQ anomaly detection pipeline. A second-stage Transformer/GPT module over VQ codes was implemented and tested, but it is not included in the final technical solution because it did not provide stable improvement under the current SMD setting.

Empirical observations:

| Variant | Observation |
|---|---|
| Transformer-Code | Average raw top-k F1: `0.1900` |
| DimNorm + Transformer hybrid | Average raw top-k F1: `0.3169`; unstable across machines and more complex |
| GPT-style code model | On `machine-1-3`, GPT-Code F1: `0.1114`; Hybrid-DimNorm-GPT F1: `0.1187` |

The single-stage VQ score is more stable and interpretable because anomalies are reflected directly by VQ reconstruction error and local dimension-level error increases.

## Key Technical Points

1. Learning a normal-pattern VQ codebook from normal multivariate time-series windows.
2. Computing variable-wise reconstruction error through the VQ bottleneck.
3. Normalizing reconstruction error by variable-specific training distributions.
4. Aggregating the top-k abnormal dimensions as the final anomaly score.
5. Applying a train-quantile threshold without using anomaly labels during training.
6. Postprocessing predicted anomaly points into anomaly segments.

## Experimental Effect

On `{len(machines)}` selected SMD machines, the best SDAD scoring head obtains an average raw top-k F1 of `{comparison_mean['best_sdad_f1']:.4f}`, compared with `{comparison_mean['best_light_deep_f1']:.4f}` for the best lightweight deep autoencoder baseline. With train-quantile thresholding and segment-level point adjustment, the main method obtains an average F1 of `{primary_mean['f1']:.4f}`.

## Reference Positioning

The method is positioned as a lightweight VQ reconstruction and scoring method for multivariate time-series anomaly detection. It is not a Transformer detector such as TranAD or Anomaly Transformer [3,5], and it does not rely on graph attention modeling as MTAD-GAT does [6]. The closest technical basis is VQ representation learning [1] combined with reconstruction-based unsupervised anomaly scoring.
"""
    (output_dir / "01_technical_disclosure_summary.md").write_text(disclosure, encoding="utf-8")

    report = f"""# Experiment Report

## Dataset

Dataset: SMD server machine dataset.

Machines used in this validation:

{machine_text}.

Training uses only the normal training split. Test labels are used only for evaluation.

{data_sample}

## Method

Main method: `SDAD-DimNorm-TopK`.

Workflow:

1. Normalize the multivariate series using training statistics.
2. Build sliding windows.
3. Train VQ-VAE on normal training windows.
4. Reconstruct test windows through the VQ bottleneck.
5. Compute variable-wise reconstruction error.
6. Normalize each variable's error with the training error distribution.
7. Aggregate the top-k normalized dimensions as the anomaly score.
8. Apply train-quantile thresholding and anomaly segment postprocessing.

## Baselines

| Type | Methods |
|---|---|
| Traditional unsupervised | ZScoreMax, PCARecon, IsolationForest |
| Lightweight deep reconstruction | MLP-AE, LSTM-AE, Conv-AE |

The baseline set covers simple statistical scoring, PCA reconstruction, Isolation Forest, and lightweight autoencoder families. Stronger published deep methods such as TranAD, USAD, Anomaly Transformer, and MTAD-GAT are listed in `03_references.md` as related work for report writing.

## Main Result

Point-adjusted segment-level average:

| Metric | Value |
|---|---:|
| Precision | {primary_mean['precision']:.4f} |
| Recall | {primary_mean['recall']:.4f} |
| F1 | {primary_mean['f1']:.4f} |
| AUROC | {primary_mean['auroc']:.4f} |
| AUPRC | {primary_mean['auprc']:.4f} |

{markdown_table(primary, ['machine', 'method', 'precision', 'recall', 'f1', 'auroc', 'auprc'])}

## Ablation Study

The ablation compares the main scoring components under the same VQ-VAE backbone.

Raw top-k average:

{markdown_table(raw_ablation, ['method', 'precision', 'recall', 'f1', 'auroc', 'auprc'])}

Train-quantile thresholding with segment postprocessing and point adjustment:

{markdown_table(post_ablation, ['method', 'precision', 'recall', 'f1', 'auroc', 'auprc'])}

## Comparison with Lightweight Deep Baselines

Average raw top-k F1:

| Method group | Mean F1 |
|---|---:|
| SDAD best scoring head | {comparison_mean['best_sdad_f1']:.4f} |
| Best lightweight deep baseline | {comparison_mean['best_light_deep_f1']:.4f} |

{markdown_table(comparison, ['machine', 'best_sdad_method', 'best_sdad_f1', 'best_light_deep_method', 'best_light_deep_f1', 'sdad_minus_light_deep_f1'])}

## Stage-2 Result and Decision

The final reported method does not include a Transformer/GPT second stage. Stage-2 code-sequence prediction was tested, but its anomaly score was less stable than the single-stage VQ reconstruction score. The added module increases complexity without producing a consistent improvement, so the final technical solution remains single-stage.

## References

See `03_references.md` for the grouped reference list and `references.bib` for BibTeX entries.
"""
    (output_dir / "02_experiment_report.md").write_text(report, encoding="utf-8")

    manifest = pd.DataFrame(
        {
            "file": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        }
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "deliverables" / "patent_materials"))
    args = parser.parse_args()
    build_materials(Path(args.output_dir), ROOT)
    print(f"Built patent materials at: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
