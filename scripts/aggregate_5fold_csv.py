#!/usr/bin/env python3
"""
Aggregate CCC/MSE/R2 (Average) and append mean±std to a shared README.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def parse_best_results(file_path: Path) -> dict[str, float | None]:
    """Parse best_results.txt and return V/A values for CCC/MSE/R2."""
    content = file_path.read_text(encoding="utf-8")

    def parse_section(metric: str) -> tuple[float | None, float | None]:
        section = re.search(
            rf"Best Test Results \({metric}.*?\):(.*?)(?=Best Test Results|\Z)",
            content,
            re.DOTALL,
        )
        if not section:
            return None, None
        v_match = re.search(r"V \(Valence\):\s+([-\d.]+)", section.group(1))
        a_match = re.search(r"A \(Arousal\):\s+([-\d.]+)", section.group(1))
        v_val = float(v_match.group(1)) if v_match else None
        a_val = float(a_match.group(1)) if a_match else None
        return v_val, a_val

    ccc_v, ccc_a = parse_section("CCC")
    mse_v, mse_a = parse_section("MSE")
    r2_v, r2_a = parse_section("R2")
    return {
        "CCC_V": ccc_v,
        "CCC_A": ccc_a,
        "MSE_V": mse_v,
        "MSE_A": mse_a,
        "R2_V": r2_v,
        "R2_A": r2_a,
    }


def normalize_experiment_name(dir_name: str) -> str:
    if "fold" not in dir_name:
        return dir_name
    cleaned = re.sub(r"fold\d+", "", dir_name)
    cleaned = re.sub(r"[_-]{2,}", "_", cleaned).strip("_-")
    return cleaned or dir_name


def find_best_results_file(base_dir: Path) -> Path | None:
    candidate = base_dir / "best_results.txt"
    if candidate.exists():
        return candidate
    candidate = base_dir / "exp_1" / "best_results.txt"
    if candidate.exists():
        return candidate
    for exp_dir in sorted(base_dir.glob("exp_*")):
        candidate = exp_dir / "best_results.txt"
        if candidate.exists():
            return candidate
    return None


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None
    std = float(np.std(values, ddof=1))
    return mean, std


def aggregate_runs(runs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dataset_name = runs_dir.name

    for model_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        exp_map: dict[str, dict[str, object]] = {}
        fold_dirs = [
            p for p in model_dir.iterdir()
            if p.is_dir() and re.search(r"fold(\d+)", p.name)
        ]

        if fold_dirs:
            for fold_dir in sorted(fold_dirs):
                fold_match = re.search(r"fold(\d+)", fold_dir.name)
                if not fold_match:
                    continue
                best_results = find_best_results_file(fold_dir)
                if not best_results:
                    continue

                metrics = parse_best_results(best_results)
                exp_name = normalize_experiment_name(fold_dir.name)
                exp_entry = exp_map.setdefault(
                    exp_name,
                    {
                        "folds": set(),
                        "CCC_V": [],
                        "CCC_A": [],
                        "MSE_V": [],
                        "MSE_A": [],
                        "R2_V": [],
                        "R2_A": [],
                    },
                )
                exp_entry["folds"].add(int(fold_match.group(1)))

                for key in ("CCC_V", "CCC_A", "MSE_V", "MSE_A", "R2_V", "R2_A"):
                    value = metrics.get(key)
                    if value is not None:
                        exp_entry[key].append(value)
        else:
            exp_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
            for exp_dir in sorted(exp_dirs):
                best_results = find_best_results_file(exp_dir)
                if not best_results:
                    continue
                metrics = parse_best_results(best_results)
                exp_entry = exp_map.setdefault(
                    exp_dir.name,
                    {
                        "folds": set(),
                        "CCC_V": [],
                        "CCC_A": [],
                        "MSE_V": [],
                        "MSE_A": [],
                        "R2_V": [],
                        "R2_A": [],
                    },
                )
                exp_entry["folds"].add(1)
                for key in ("CCC_V", "CCC_A", "MSE_V", "MSE_A", "R2_V", "R2_A"):
                    value = metrics.get(key)
                    if value is not None:
                        exp_entry[key].append(value)

            if not exp_map:
                best_results = find_best_results_file(model_dir)
                if best_results:
                    metrics = parse_best_results(best_results)
                    exp_entry = exp_map.setdefault(
                        model_dir.name,
                        {
                            "folds": set(),
                            "CCC_V": [],
                            "CCC_A": [],
                            "MSE_V": [],
                            "MSE_A": [],
                            "R2_V": [],
                            "R2_A": [],
                        },
                    )
                    exp_entry["folds"].add(1)
                    for key in (
                        "CCC_V",
                        "CCC_A",
                        "MSE_V",
                        "MSE_A",
                        "R2_V",
                        "R2_A",
                    ):
                        value = metrics.get(key)
                        if value is not None:
                            exp_entry[key].append(value)

        for exp_name in sorted(exp_map.keys()):
            exp_entry = exp_map[exp_name]
            num_folds = len(exp_entry["folds"])
            ccc_v_mean, ccc_v_std = mean_std(exp_entry["CCC_V"])
            ccc_a_mean, ccc_a_std = mean_std(exp_entry["CCC_A"])
            mse_v_mean, mse_v_std = mean_std(exp_entry["MSE_V"])
            mse_a_mean, mse_a_std = mean_std(exp_entry["MSE_A"])
            r2_v_mean, r2_v_std = mean_std(exp_entry["R2_V"])
            r2_a_mean, r2_a_std = mean_std(exp_entry["R2_A"])

            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_dir.name,
                    "experiment": exp_name,
                    "num_folds": num_folds,
                    "ccc_v_mean": ccc_v_mean,
                    "ccc_v_std": ccc_v_std,
                    "ccc_a_mean": ccc_a_mean,
                    "ccc_a_std": ccc_a_std,
                    "mse_v_mean": mse_v_mean,
                    "mse_v_std": mse_v_std,
                    "mse_a_mean": mse_a_mean,
                    "mse_a_std": mse_a_std,
                    "r2_v_mean": r2_v_mean,
                    "r2_v_std": r2_v_std,
                    "r2_a_mean": r2_a_mean,
                    "r2_a_std": r2_a_std,
                }
            )

    return rows


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.4f} ± N/A"
    return f"{mean:.4f} ± {std:.4f}"


def append_readme(rows: list[dict[str, object]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset = rows[0]["dataset"] if rows else "Unknown"
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        model = str(row["model"])
        grouped.setdefault(model, []).append(row)

    lines: list[str] = []
    lines.append(f"## {dataset}")
    lines.append("")
    lines.append("Metric format: mean ± std (std uses ddof=1; single run => N/A)")
    lines.append("")

    lines.append("| Model | CCC_V | CCC_A | MSE_V | MSE_A | R2_V | R2_A |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for model in sorted(grouped.keys()):
        entries = grouped[model]
        if len(entries) == 1:
            row = entries[0]
            display_name = model
            ccc_v = format_mean_std(row["ccc_v_mean"], row["ccc_v_std"])
            ccc_a = format_mean_std(row["ccc_a_mean"], row["ccc_a_std"])
            mse_v = format_mean_std(row["mse_v_mean"], row["mse_v_std"])
            mse_a = format_mean_std(row["mse_a_mean"], row["mse_a_std"])
            r2_v = format_mean_std(row["r2_v_mean"], row["r2_v_std"])
            r2_a = format_mean_std(row["r2_a_mean"], row["r2_a_std"])
            lines.append(
                f"| {display_name} | {ccc_v} | {ccc_a} | {mse_v} | {mse_a} | "
                f"{r2_v} | {r2_a} |"
            )
            continue
        for row in sorted(entries, key=lambda item: str(item["experiment"])):
            display_name = f"{model}:{row['experiment']}"
            ccc_v = format_mean_std(row["ccc_v_mean"], row["ccc_v_std"])
            ccc_a = format_mean_std(row["ccc_a_mean"], row["ccc_a_std"])
            mse_v = format_mean_std(row["mse_v_mean"], row["mse_v_std"])
            mse_a = format_mean_std(row["mse_a_mean"], row["mse_a_std"])
            r2_v = format_mean_std(row["r2_v_mean"], row["r2_v_std"])
            r2_a = format_mean_std(row["r2_a_mean"], row["r2_a_std"])
            lines.append(
                f"| {display_name} | {ccc_v} | {ccc_a} | {mse_v} | {mse_a} | "
                f"{r2_v} | {r2_a} |"
            )
    lines.append("")

    new_block = "\n".join(lines)
    if not output_file.exists():
        output_file.write_text(new_block, encoding="utf-8")
        return
    existing = output_file.read_text(encoding="utf-8")
    separator = "" if existing.endswith("\n") else "\n"
    output_file.write_text(existing + separator + new_block, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate CCC/MSE/R2 averages and std to a shared README."
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        required=True,
        help="Dataset runs directory, e.g., runs/CCSEMO or runs/IEMOCAP",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output README path (default: runs/5fold_stats.md)",
    )
    args = parser.parse_args()

    if not args.runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {args.runs_dir}")

    output_file = args.output or (Path("runs") / "5fold_stats.md")
    rows = aggregate_runs(args.runs_dir)
    if not rows:
        raise RuntimeError(f"No results found under: {args.runs_dir}")
    append_readme(rows, output_file)
    print(f"Appended README block to: {output_file}")


if __name__ == "__main__":
    main()
