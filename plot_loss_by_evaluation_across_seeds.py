# -*- coding: utf-8 -*-
"""Plot train/validation/test loss vs evaluation count across seeds."""

import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = ROOT / "runs" / "results"
OUTPUT_DIR = ROOT / "runs" / "plots" / "loss_by_evaluation_across_seeds"
SEEDS = [0, 1, 2]
LOSS_PLOTS = {
    "train_loss": "Train Loss",
    "validation_loss": "Validation Loss",
    "test_loss": "Test Loss",
}


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def load_seed_rows(seed):
    csv_path = RESULTS_DIR / f"evaluation_results_seed_{seed}.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite_points(rows, loss_column):
    points = []
    for row in rows:
        evaluation_count = int(row["evaluation_count"])
        loss = parse_float(row[loss_column])
        if math.isfinite(loss):
            points.append((evaluation_count, loss))
    return points


def plot_loss_column(loss_column, title):
    plot_path = OUTPUT_DIR / f"{loss_column}_by_evaluation_all_seeds.png"

    plt.figure(figsize=(9, 5))
    for seed in SEEDS:
        rows = load_seed_rows(seed)
        points = finite_points(rows, loss_column)
        if not points:
            continue
        x_values, y_values = zip(*points)
        plt.plot(
            x_values,
            y_values,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"seed {seed}",
        )

    plt.xlabel("Evaluation Count")
    plt.ylabel(title)
    plt.title(f"{title} vs Evaluation Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    for loss_column, title in LOSS_PLOTS.items():
        created.append(plot_loss_column(loss_column, title))

    print("Created plots:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
