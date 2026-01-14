#!/usr/bin/env python3
"""
按数据集汇总 runs/ 下的 test_predictions.csv，合并所有预测后计算 CCC/MSE/R2。
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REQUIRED_COLUMNS = ("true_v", "pred_v", "true_a", "pred_a")
FOLD_PATTERN = re.compile(r"fold([1-5])(?!\d)")


@dataclass
class DatasetMetrics:
    dataset: str
    model: str
    num_files: int
    num_samples: int
    ccc_v: float
    ccc_a: float
    ccc_avg: float
    mse_v: float
    mse_a: float
    mse_avg: float
    r2_v: float
    r2_a: float
    r2_avg: float


def concordance_correlation_coefficient(observed: np.ndarray, predicted: np.ndarray) -> float:
    mean_observed = np.mean(observed)
    mean_predicted = np.mean(predicted)
    covariance = np.cov(observed, predicted, ddof=1)[0, 1]
    obs_variance = np.var(observed, ddof=1)
    pred_variance = np.var(predicted, ddof=1)
    ccc = 2 * covariance / (obs_variance + pred_variance + (mean_observed - mean_predicted) ** 2)
    return float(ccc)


def mean_squared_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    mse = np.mean((observed - predicted) ** 2)
    return float(mse)


def r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    if observed.size == 0:
        return 0.0
    ss_res = np.sum((observed - predicted) ** 2)
    mean_obs = np.mean(observed)
    ss_tot = np.sum((observed - mean_obs) ** 2)
    if ss_tot == 0:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    return float(r2)


def iter_prediction_files(model_dir: Path) -> Iterable[Path]:
    files: list[Path] = []
    for path in model_dir.rglob("test_predictions.csv"):
        if FOLD_PATTERN.search(path.as_posix()):
            files.append(path)
    return files


def load_predictions(
    files: Iterable[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    true_v: list[float] = []
    pred_v: list[float] = []
    true_a: list[float] = []
    pred_a: list[float] = []
    num_files = 0

    for csv_path in sorted(files):
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue
            col_idx = {name: header.index(name) for name in REQUIRED_COLUMNS if name in header}
            if len(col_idx) != len(REQUIRED_COLUMNS):
                missing = [name for name in REQUIRED_COLUMNS if name not in col_idx]
                print(f"[WARN] 缺少列 {missing}: {csv_path}")
                continue
            for row in reader:
                if not row:
                    continue
                try:
                    true_v.append(float(row[col_idx["true_v"]]))
                    pred_v.append(float(row[col_idx["pred_v"]]))
                    true_a.append(float(row[col_idx["true_a"]]))
                    pred_a.append(float(row[col_idx["pred_a"]]))
                except (ValueError, IndexError):
                    print(f"[WARN] 无法解析行: {csv_path}")
                    continue
            num_files += 1

    return (
        np.asarray(true_v, dtype=np.float64),
        np.asarray(pred_v, dtype=np.float64),
        np.asarray(true_a, dtype=np.float64),
        np.asarray(pred_a, dtype=np.float64),
        num_files,
    )


def compute_metrics(dataset: str, model: str, model_dir: Path) -> DatasetMetrics | None:
    files = list(iter_prediction_files(model_dir))
    if not files:
        print(f"[WARN] 未找到预测文件: {model_dir}")
        return None

    true_v, pred_v, true_a, pred_a, num_files = load_predictions(files)
    num_samples = int(true_v.size)
    if num_samples == 0:
        print(f"[WARN] 空数据集: {model_dir}")
        return None

    ccc_v = concordance_correlation_coefficient(true_v, pred_v)
    ccc_a = concordance_correlation_coefficient(true_a, pred_a)
    mse_v = mean_squared_error(true_v, pred_v)
    mse_a = mean_squared_error(true_a, pred_a)
    r2_v = r2_score(true_v, pred_v)
    r2_a = r2_score(true_a, pred_a)

    return DatasetMetrics(
        dataset=dataset,
        model=model,
        num_files=num_files,
        num_samples=num_samples,
        ccc_v=ccc_v,
        ccc_a=ccc_a,
        ccc_avg=(ccc_v + ccc_a) / 2,
        mse_v=mse_v,
        mse_a=mse_a,
        mse_avg=(mse_v + mse_a) / 2,
        r2_v=r2_v,
        r2_a=r2_a,
        r2_avg=(r2_v + r2_a) / 2,
    )


def format_table(rows: list[DatasetMetrics]) -> str:
    header = (
        "Dataset | Model | Files | Samples | CCC_V | CCC_A | CCC_Avg | "
        "MSE_V | MSE_A | MSE_Avg | R2_V | R2_A | R2_Avg"
    )
    lines = [header, "-" * len(header)]
    for item in rows:
        lines.append(
            f"{item.dataset} | {item.model} | {item.num_files} | {item.num_samples} | "
            f"{item.ccc_v:.4f} | {item.ccc_a:.4f} | {item.ccc_avg:.4f} | "
            f"{item.mse_v:.4f} | {item.mse_a:.4f} | {item.mse_avg:.4f} | "
            f"{item.r2_v:.4f} | {item.r2_a:.4f} | {item.r2_avg:.4f}"
        )
    return "\n".join(lines)


def write_csv(output_path: Path, rows: list[DatasetMetrics]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "model",
                "num_files",
                "num_samples",
                "ccc_v",
                "ccc_a",
                "ccc_avg",
                "mse_v",
                "mse_a",
                "mse_avg",
                "r2_v",
                "r2_a",
                "r2_avg",
            ]
        )
        for item in rows:
            writer.writerow(
                [
                    item.dataset,
                    item.model,
                    item.num_files,
                    item.num_samples,
                    f"{item.ccc_v:.6f}",
                    f"{item.ccc_a:.6f}",
                    f"{item.ccc_avg:.6f}",
                    f"{item.mse_v:.6f}",
                    f"{item.mse_a:.6f}",
                    f"{item.mse_avg:.6f}",
                    f"{item.r2_v:.6f}",
                    f"{item.r2_a:.6f}",
                    f"{item.r2_avg:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate CCC/MSE/R2 by dataset using all test_predictions.csv.",
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("runs"),
        help="Runs directory (default: runs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    rows: list[DatasetMetrics] = []
    allowed_datasets = {"IEMOCAP", "CCSEMO"}
    for dataset_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if dataset_dir.name not in allowed_datasets:
            continue
        for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            metrics = compute_metrics(dataset_dir.name, model_dir.name, model_dir)
            if metrics is not None:
                rows.append(metrics)

    if not rows:
        print("[WARN] 没有可用的结果。")
        return

    print(format_table(rows))
    if args.output is not None:
        write_csv(args.output, rows)
        print(f"\n[INFO] 写出 CSV: {args.output}")


if __name__ == "__main__":
    main()
