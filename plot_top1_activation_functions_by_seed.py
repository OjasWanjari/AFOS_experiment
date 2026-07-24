# -*- coding: utf-8 -*-
"""Plot the top-1 activation function for each seed on one figure."""

import ast
import csv
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from activation import get_activation


RESULTS_DIR = ROOT / "runs" / "results"
OUTPUT_DIR = ROOT / "runs" / "plots" / "top1_activation_functions"


def load_top1(seed):
    path = RESULTS_DIR / f"top3_activation_functions_seed_{seed}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    top1 = sorted(rows, key=lambda row: int(row["rank"]))[0]
    return {
        "seed": seed,
        "chromosome": ast.literal_eval(top1["chromosome"]),
        "chromosome_text": top1["chromosome"],
        "expression": top1["activation_expression"],
        "score": float(top1["score"]),
    }


def evaluate(chromosome, x_values):
    x_tensor = tf.constant(x_values, dtype=tf.float32)
    y_tensor = get_activation(chromosome, x_tensor)
    y_values = y_tensor.numpy()
    mask = np.isfinite(y_values)
    return x_values[mask], y_values[mask]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_values = np.linspace(-5.0, 5.0, 2001, dtype=np.float32)
    top1_rows = [load_top1(seed) for seed in [0, 1, 2]]

    plot_path = OUTPUT_DIR / "top1_activation_functions_by_seed.png"
    csv_path = OUTPUT_DIR / "top1_activation_functions_by_seed.csv"

    plt.figure(figsize=(10, 6))
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.5)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "chromosome", "activation_expression", "score"])
        for row in top1_rows:
            x, y = evaluate(row["chromosome"], x_values)
            plt.plot(
                x,
                y,
                linewidth=2,
                label=f"seed {row['seed']}: {row['chromosome_text']} score={row['score']:.4f}",
            )
            writer.writerow([row["seed"], row["chromosome_text"], row["expression"], row["score"]])

    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.title("Top-1 activation function by seed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Created files:")
    print(plot_path)
    print(csv_path)


if __name__ == "__main__":
    main()
