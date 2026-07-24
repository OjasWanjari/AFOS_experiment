# -*- coding: utf-8 -*-
"""Plot the top-3 activation functions saved by the GA experiment."""

import ast
import csv
import os
import textwrap
from pathlib import Path

import numpy as np
import tensorflow as tf

from activation import get_activation


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = ROOT / "runs" / "results"
PLOTS_DIR = ROOT / "runs" / "plots" / "top3_activation_functions"


def evaluate_activation(chromosome, x_values):
    x_tensor = tf.constant(x_values, dtype=tf.float32)
    y_tensor = get_activation(chromosome, x_tensor)
    return y_tensor.numpy()


def finite_xy(chromosome, x_values):
    y_values = evaluate_activation(chromosome, x_values)
    mask = np.isfinite(y_values)
    return x_values[mask], y_values[mask]


def load_top3_rows(seed):
    csv_path = RESULTS_DIR / f"top3_activation_functions_seed_{seed}.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_individual(seed, row, x_values):
    rank = int(row["rank"])
    chromosome = ast.literal_eval(row["chromosome"])
    expression = row["activation_expression"]
    x, y = finite_xy(chromosome, x_values)

    path = PLOTS_DIR / f"top3_activation_seed_{seed}_rank_{rank}.png"
    plt.figure(figsize=(8, 5))
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.plot(x, y, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.title(
        "\n".join(
            textwrap.wrap(
                f"seed={seed} rank={rank} score={float(row['score']):.4f}: {expression}",
                width=75,
            )
        )
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_seed_overlay(seed, rows, x_values):
    path = PLOTS_DIR / f"top3_activation_functions_seed_{seed}.png"
    plt.figure(figsize=(9, 6))
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.5)

    for row in rows:
        rank = int(row["rank"])
        chromosome = ast.literal_eval(row["chromosome"])
        x, y = finite_xy(chromosome, x_values)
        plt.plot(
            x,
            y,
            linewidth=2,
            label=f"rank {rank}: {row['chromosome']} score={float(row['score']):.4f}",
        )

    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.title(f"Top 3 activation functions (seed={seed})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    x_values = np.linspace(-5.0, 5.0, 2001, dtype=np.float32)

    created = []
    for seed in [0, 1, 2]:
        rows = load_top3_rows(seed)
        created.append(plot_seed_overlay(seed, rows, x_values))
        for row in rows:
            created.append(plot_individual(seed, row, x_values))

    print("Created plots:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
