# -*- coding: utf-8 -*-
"""Backfill accuracy columns and plot best-so-far accuracy across seeds.

Existing experiment CSV files saved loss and score, but did not save accuracy
explicitly. This script reconstructs:

- train_accuracy / validation_accuracy from the Keras training log
- test_accuracy from score * test_loss when test_loss is finite

It writes augmented CSV files and best-so-far accuracy plots.
"""

import csv
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = ROOT / "runs" / "results"
LOG_PATH = ROOT / "runs" / "ga_seed_0_1_2_20260710_0131.out.log"
ACCURACY_RESULTS_DIR = ROOT / "runs" / "results" / "accuracy"
PLOTS_DIR = ROOT / "runs" / "plots" / "best_so_far_accuracy_across_seeds"
SEEDS = [0, 1, 2]


RUN_RE = re.compile(r"=== Run \d+/\d+ \| seed=(\d+) ===")
CHROMOSOME_RE = re.compile(r"Choromosome Number:\s+\d+\s+\[")
ACCURACY_RE = re.compile(r"accuracy:\s*([0-9.]+)\s*-\s*val_loss:.*?val_accuracy:\s*([0-9.]+)")


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def parse_training_accuracies_from_log():
    by_seed = {seed: [] for seed in SEEDS}
    current_seed = None
    current_train_accuracy = None
    current_validation_accuracy = None
    in_evaluation = False

    def finish_current():
        nonlocal current_train_accuracy, current_validation_accuracy, in_evaluation
        if in_evaluation and current_seed is not None:
            by_seed[current_seed].append(
                {
                    "train_accuracy": current_train_accuracy,
                    "validation_accuracy": current_validation_accuracy,
                }
            )
        current_train_accuracy = None
        current_validation_accuracy = None
        in_evaluation = False

    with LOG_PATH.open(encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            run_match = RUN_RE.search(line)
            if run_match:
                finish_current()
                current_seed = int(run_match.group(1))
                continue

            if CHROMOSOME_RE.search(line):
                finish_current()
                in_evaluation = True
                continue

            accuracy_match = ACCURACY_RE.search(line)
            if in_evaluation and accuracy_match:
                current_train_accuracy = float(accuracy_match.group(1))
                current_validation_accuracy = float(accuracy_match.group(2))

    finish_current()
    return by_seed


def backfill_seed(seed, parsed_accuracies):
    input_path = RESULTS_DIR / f"evaluation_results_seed_{seed}.csv"
    output_path = ACCURACY_RESULTS_DIR / f"evaluation_results_with_accuracy_seed_{seed}.csv"

    with input_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for index, row in enumerate(rows):
        parsed = parsed_accuracies[index] if index < len(parsed_accuracies) else {}
        score = parse_float(row.get("score"))
        test_loss = parse_float(row.get("test_loss"))
        row["test_accuracy"] = score * test_loss if math.isfinite(score) and math.isfinite(test_loss) else ""
        row["train_accuracy"] = parsed.get("train_accuracy", "")
        row["validation_accuracy"] = parsed.get("validation_accuracy", "")

    fieldnames = list(rows[0].keys())
    for column in ["test_accuracy", "train_accuracy", "validation_accuracy"]:
        if column not in fieldnames:
            fieldnames.append(column)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path, rows


def cumulative_max(rows, accuracy_column):
    best = -math.inf
    points = []
    for row in rows:
        evaluation_count = int(row["evaluation_count"])
        accuracy = parse_float(row.get(accuracy_column))
        if math.isfinite(accuracy) and accuracy > best:
            best = accuracy
        if math.isfinite(best):
            points.append((evaluation_count, best))
    return points


def plot_accuracy_column(all_rows, accuracy_column, title):
    plot_path = PLOTS_DIR / f"best_so_far_{accuracy_column}_by_evaluation_all_seeds.png"
    plt.figure(figsize=(9, 5))

    for seed in SEEDS:
        points = cumulative_max(all_rows[seed], accuracy_column)
        if not points:
            continue
        x_values, y_values = zip(*points)
        plt.plot(
            x_values,
            y_values,
            marker="o",
            markersize=3,
            linewidth=2,
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
    ACCURACY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    parsed_by_seed = parse_training_accuracies_from_log()
    all_rows = {}
    created = []

    for seed in SEEDS:
        output_path, rows = backfill_seed(seed, parsed_by_seed[seed])
        all_rows[seed] = rows
        created.append(output_path)

    created.append(plot_accuracy_column(all_rows, "train_accuracy", "Best-so-far Train Accuracy"))
    created.append(plot_accuracy_column(all_rows, "validation_accuracy", "Best-so-far Validation Accuracy"))
    created.append(plot_accuracy_column(all_rows, "test_accuracy", "Best-so-far Test Accuracy"))

    print("Created files:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
