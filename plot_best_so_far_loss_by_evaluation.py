# -*- coding: utf-8 -*-
"""Plot best-so-far loss curves against fitness evaluation count.

For loss, "best" means the smallest finite loss observed so far.
The script reads runs/results/evaluation_results_seed_*.csv and writes
PNG/CSV outputs under runs/plots/best_so_far_loss_by_evaluation.
"""

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
OUTPUT_DIR = ROOT / "runs" / "plots" / "best_so_far_loss_by_evaluation"
LOSS_COLUMNS = ["test_loss", "train_loss", "validation_loss"]


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def cumulative_min(values):
    best = math.inf
    output = []
    for value in values:
        if math.isfinite(value) and value < best:
            best = value
        output.append(best)
    return output


def load_rows(seed):
    path = RESULTS_DIR / f"evaluation_results_seed_{seed}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_best_so_far_csv(seed, evaluation_counts, best_losses):
    path = OUTPUT_DIR / f"best_so_far_loss_by_evaluation_seed_{seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "evaluation_count",
                "best_test_loss_so_far",
                "best_train_loss_so_far",
                "best_validation_loss_so_far",
            ]
        )
        for index, evaluation_count in enumerate(evaluation_counts):
            writer.writerow(
                [
                    evaluation_count,
                    best_losses["test_loss"][index],
                    best_losses["train_loss"][index],
                    best_losses["validation_loss"][index],
                ]
            )
    return path


def plot_seed(seed, evaluation_counts, best_losses):
    path = OUTPUT_DIR / f"best_so_far_loss_by_evaluation_seed_{seed}.png"
    plt.figure(figsize=(9, 5))

    labels = {
        "test_loss": "best test loss so far",
        "train_loss": "best train loss so far",
        "validation_loss": "best validation loss so far",
    }
    for column in LOSS_COLUMNS:
        y_values = best_losses[column]
        finite_points = [
            (x, y)
            for x, y in zip(evaluation_counts, y_values)
            if math.isfinite(y)
        ]
        if not finite_points:
            continue
        x, y = zip(*finite_points)
        plt.plot(x, y, marker="o", markersize=3, linewidth=2, label=labels[column])

    plt.xlabel("Evaluation Count")
    plt.ylabel("Best Loss So Far")
    plt.title(f"Best-so-far Loss per Fitness Evaluation (seed={seed})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created = []

    for csv_path in sorted(RESULTS_DIR.glob("evaluation_results_seed_*.csv")):
        seed = csv_path.stem.removeprefix("evaluation_results_seed_")
        rows = load_rows(seed)
        evaluation_counts = [int(row["evaluation_count"]) for row in rows]
        best_losses = {
            column: cumulative_min([parse_float(row[column]) for row in rows])
            for column in LOSS_COLUMNS
        }
        created.append(write_best_so_far_csv(seed, evaluation_counts, best_losses))
        created.append(plot_seed(seed, evaluation_counts, best_losses))

    print("Created files:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
