# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:28:20 2022

@author: ar2806
"""
import argparse
import csv
import json
import math
import os
import random
import shutil

import numpy as np
import tensorflow as tf

from activation import chromosome_to_expression
from ga_utils import create_initial_pop, next_generation
from fitness import fitness


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _seed_label(seed):
    return "none" if seed is None else str(seed)


def _json_chromosome(chromosome):
    if chromosome is None:
        return ""
    return json.dumps([int(gene) for gene in chromosome])


def _prepare_output_dir(*parts):
    output_dir = os.path.join(*parts)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_best_model(model, seed, output_dir=os.path.join("runs", "models")):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"best_model_seed_{_seed_label(seed)}")
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    elif os.path.exists(model_path):
        os.remove(model_path)
    model.save(model_path, overwrite=True, include_optimizer=True)
    print("Best model:", model_path)
    return model_path


def save_evaluation_results(seed, rows, output_dir=os.path.join("runs", "results")):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"evaluation_results_seed_{_seed_label(seed)}.csv")
    fieldnames = [
        "seed",
        "generation",
        "evaluation_count",
        "chromosome",
        "activation_expression",
        "score",
        "test_loss",
        "train_loss",
        "validation_loss",
        "test_accuracy",
        "train_accuracy",
        "validation_accuracy",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Evaluation results CSV:", csv_path)
    return csv_path


def save_top_activation_functions(seed, rows, top_n=3, output_dir=os.path.join("runs", "results")):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"top{top_n}_activation_functions_seed_{_seed_label(seed)}.csv")
    top_rows = sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:top_n]

    fieldnames = [
        "rank",
        "seed",
        "generation",
        "evaluation_count",
        "chromosome",
        "activation_expression",
        "score",
        "test_loss",
        "train_loss",
        "validation_loss",
        "test_accuracy",
        "train_accuracy",
        "validation_accuracy",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(top_rows, start=1):
            output_row = {field: row.get(field, "") for field in fieldnames}
            output_row["rank"] = rank
            writer.writerow(output_row)

    print(f"Top {top_n} activation functions CSV:", csv_path)
    return csv_path


def save_generation_score_history(
    seed,
    generation_scores,
    generation_test_losses,
    generation_train_losses,
    generation_validation_losses,
    output_dir=os.path.join("runs", "results"),
):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"generation_score_history_seed_{_seed_label(seed)}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_score", "best_test_loss", "best_train_loss", "best_validation_loss"])
        for generation, values in enumerate(
            zip(generation_scores, generation_test_losses, generation_train_losses, generation_validation_losses),
            start=1,
        ):
            score, test_loss, train_loss, validation_loss = values
            writer.writerow([generation, score, test_loss, train_loss, validation_loss])
    print("Generation score history CSV:", csv_path)
    return csv_path


def save_results_summary(results, output_dir=os.path.join("runs", "results")):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "results_summary.csv")
    best_by_seed_path = os.path.join(output_dir, "best_by_seed.csv")

    summary_fields = [
        "seed",
        "best_chromosome",
        "best_activation_expression",
        "best_score",
        "best_test_loss",
        "best_train_loss",
        "best_validation_loss",
        "best_test_accuracy",
        "best_train_accuracy",
        "best_validation_accuracy",
        "best_generation",
        "best_evaluation",
        "best_model_path",
        "top3_activation_functions_csv",
        "generation_score_history_csv",
        "evaluation_results_csv",
        "loss_plot",
        "evaluation_loss_plot",
        "train_validation_generation_loss_plot",
        "train_validation_evaluation_loss_plot",
    ]
    best_fields = [
        "seed",
        "best_chromosome",
        "best_activation_expression",
        "best_score",
        "best_test_loss",
        "best_train_loss",
        "best_validation_loss",
        "best_test_accuracy",
        "best_train_accuracy",
        "best_validation_accuracy",
        "best_generation",
        "best_evaluation",
        "best_model_path",
        "top3_activation_functions_csv",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in summary_fields})

    with open(best_by_seed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=best_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in best_fields})

    print("Results summary CSV:", summary_path)
    print("Best by seed CSV:", best_by_seed_path)
    return summary_path, best_by_seed_path


def save_loss_history(seed, generation_losses, output_dir=os.path.join("runs", "plots")):
    os.makedirs(output_dir, exist_ok=True)
    seed_label = "none" if seed is None else str(seed)
    csv_path = os.path.join(output_dir, f"loss_history_seed_{seed_label}.csv")
    plot_path = os.path.join(output_dir, f"loss_history_seed_{seed_label}.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "loss"])
        for generation, loss in enumerate(generation_losses, start=1):
            writer.writerow([generation, loss])

    finite_points = [
        (generation, loss)
        for generation, loss in enumerate(generation_losses, start=1)
        if math.isfinite(loss)
    ]
    if not finite_points:
        print("Loss plot was skipped because no finite loss values were recorded.")
        print("Loss history CSV:", csv_path)
        return None, csv_path

    matplotlib_config_dir = os.path.join("runs", ".matplotlib")
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_config_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generations, losses = zip(*finite_points)
    plt.figure(figsize=(8, 5))
    plt.plot(generations, losses, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.title(f"Best Test Loss per Generation (seed={seed_label})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Loss plot:", plot_path)
    print("Loss history CSV:", csv_path)
    return plot_path, csv_path


def save_evaluation_loss_history(seed, evaluation_losses, output_dir=os.path.join("runs", "plots")):
    os.makedirs(output_dir, exist_ok=True)
    seed_label = "none" if seed is None else str(seed)
    csv_path = os.path.join(output_dir, f"evaluation_loss_history_seed_{seed_label}.csv")
    plot_path = os.path.join(output_dir, f"evaluation_loss_history_seed_{seed_label}.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_count", "loss"])
        for evaluation_count, loss in enumerate(evaluation_losses, start=1):
            writer.writerow([evaluation_count, loss])

    finite_points = [
        (evaluation_count, loss)
        for evaluation_count, loss in enumerate(evaluation_losses, start=1)
        if math.isfinite(loss)
    ]
    if not finite_points:
        print("Evaluation loss plot was skipped because no finite loss values were recorded.")
        print("Evaluation loss history CSV:", csv_path)
        return None, csv_path

    matplotlib_config_dir = os.path.join("runs", ".matplotlib")
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_config_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation_counts, losses = zip(*finite_points)
    plt.figure(figsize=(9, 5))
    plt.plot(evaluation_counts, losses, marker="o", markersize=3)
    plt.xlabel("Evaluation Count")
    plt.ylabel("Loss")
    plt.title(f"Test Loss per Fitness Evaluation (seed={seed_label})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Evaluation loss plot:", plot_path)
    print("Evaluation loss history CSV:", csv_path)
    return plot_path, csv_path


def save_train_validation_loss_history(
    seed,
    x_values,
    train_losses,
    validation_losses,
    x_column,
    x_label,
    file_prefix,
    title,
    output_dir=os.path.join("runs", "plots"),
):
    os.makedirs(output_dir, exist_ok=True)
    seed_label = "none" if seed is None else str(seed)
    csv_path = os.path.join(output_dir, f"{file_prefix}_seed_{seed_label}.csv")
    plot_path = os.path.join(output_dir, f"{file_prefix}_seed_{seed_label}.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([x_column, "train_loss", "validation_loss"])
        for x, train_loss, validation_loss in zip(x_values, train_losses, validation_losses):
            writer.writerow([x, train_loss, validation_loss])

    train_points = [
        (x, loss)
        for x, loss in zip(x_values, train_losses)
        if math.isfinite(loss)
    ]
    validation_points = [
        (x, loss)
        for x, loss in zip(x_values, validation_losses)
        if math.isfinite(loss)
    ]
    if not train_points and not validation_points:
        print(f"{title} plot was skipped because no finite loss values were recorded.")
        print(f"{title} CSV:", csv_path)
        return None, csv_path

    matplotlib_config_dir = os.path.join("runs", ".matplotlib")
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_config_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    if train_points:
        train_x, train_y = zip(*train_points)
        plt.plot(train_x, train_y, marker="o", markersize=3, label="train loss")
    if validation_points:
        validation_x, validation_y = zip(*validation_points)
        plt.plot(validation_x, validation_y, marker="o", markersize=3, label="validation loss")
    plt.xlabel(x_label)
    plt.ylabel("Loss")
    plt.title(f"{title} (seed={seed_label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"{title} plot:", plot_path)
    print(f"{title} CSV:", csv_path)
    return plot_path, csv_path


def run_ga(pop_size=30, gen=50, seed=None):
    if seed is not None:
        set_seed(seed)

    #chromosome_shape = [7,18,18]
    #term_threshold = 0.80
    population = create_initial_pop(pop_size)
    highest_accuracy = []
    generation_losses = []
    evaluation_losses = []
    generation_train_losses = []
    generation_validation_losses = []
    evaluation_train_losses = []
    evaluation_validation_losses = []
    evaluation_rows = []
    best_chromosome = None
    best_activation_expression = None
    best_score = float("-inf")
    best_test_loss = None
    best_train_loss = None
    best_validation_loss = None
    best_test_accuracy = None
    best_train_accuracy = None
    best_validation_accuracy = None
    best_generation = None
    best_evaluation = None
    best_model_path = None
    for i in range(gen):
        print("Start of Generation No: ", i+1)
        generation_number = i + 1
        evaluation_start = len(evaluation_rows)

        def record_evaluation(
            serial,
            chromosome,
            score,
            test_loss,
            train_loss,
            validation_loss,
            test_accuracy,
            train_accuracy,
            validation_accuracy,
            model,
        ):
            nonlocal best_chromosome, best_activation_expression, best_score
            nonlocal best_test_loss, best_train_loss, best_validation_loss
            nonlocal best_test_accuracy, best_train_accuracy, best_validation_accuracy
            nonlocal best_generation, best_evaluation, best_model_path
            evaluation_count = evaluation_start + serial + 1
            activation_expression = chromosome_to_expression(chromosome)
            evaluation_rows.append({
                "seed": seed,
                "generation": generation_number,
                "evaluation_count": evaluation_count,
                "chromosome": _json_chromosome(chromosome),
                "activation_expression": activation_expression,
                "score": score,
                "test_loss": test_loss,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "test_accuracy": test_accuracy,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            })
            if score > best_score:
                best_score = score
                best_chromosome = chromosome[:]
                best_activation_expression = activation_expression
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_validation_loss = validation_loss
                best_test_accuracy = test_accuracy
                best_train_accuracy = train_accuracy
                best_validation_accuracy = validation_accuracy
                best_generation = generation_number
                best_evaluation = evaluation_count
                best_model_path = save_best_model(model, seed)

        (
            population,
            scores,
            losses,
            train_losses,
            validation_losses,
            test_accuracies,
            train_accuracies,
            validation_accuracies,
        ) = fitness(
            population,
            on_evaluation=record_evaluation,
        )
        evaluation_losses.extend(losses)
        evaluation_train_losses.extend(train_losses)
        evaluation_validation_losses.extend(validation_losses)
        for pop,sc in zip(population,scores):
            print(pop, sc)
        print("End of a Generation...")
        highest_accuracy.append(max(scores))
        generation_losses.append(min(losses))
        generation_train_losses.append(min(train_losses))
        generation_validation_losses.append(min(validation_losses))
        '''for (pop,score) in zip(population,scores):
            if score >= term_threshold:
                print('Expected Result is Achieved')
                print(pop)
                is_loop = False
                break
        if not is_loop:
            break'''
        population = next_generation(population, scores, pop_size=pop_size)
        print('completion of a generation')
        #if termination criteria doesn't pass then
        #new_population = next_generation(pop_with_score), it includes cross over and mutation
        #this new population will be used to check fitness function again
        #when termination criteria passes, show the best population
    print("Highest Accuracy over all generations:", highest_accuracy)
    print("Lowest Loss over all generations:", generation_losses)
    print("Loss by evaluation count:", evaluation_losses)
    print("Train Loss by generation:", generation_train_losses)
    print("Validation Loss by generation:", generation_validation_losses)
    print("Train Loss by evaluation count:", evaluation_train_losses)
    print("Validation Loss by evaluation count:", evaluation_validation_losses)
    print("Best chromosome:", best_chromosome)
    print("Best activation expression:", best_activation_expression)
    print("Best score:", best_score)
    print("Best test loss:", best_test_loss)
    print("Best train loss:", best_train_loss)
    print("Best validation loss:", best_validation_loss)
    print("Best test accuracy:", best_test_accuracy)
    print("Best train accuracy:", best_train_accuracy)
    print("Best validation accuracy:", best_validation_accuracy)
    print("Best generation:", best_generation)
    print("Best evaluation:", best_evaluation)
    print("Best model path:", best_model_path)
    evaluation_results_csv = save_evaluation_results(seed, evaluation_rows)
    top3_activation_functions_csv = save_top_activation_functions(seed, evaluation_rows, top_n=3)
    generation_score_history_csv = save_generation_score_history(
        seed,
        highest_accuracy,
        generation_losses,
        generation_train_losses,
        generation_validation_losses,
    )
    loss_plot, loss_csv = save_loss_history(seed, generation_losses)
    evaluation_loss_plot, evaluation_loss_csv = save_evaluation_loss_history(seed, evaluation_losses)
    train_validation_generation_loss_plot, train_validation_generation_loss_csv = save_train_validation_loss_history(
        seed=seed,
        x_values=list(range(1, len(generation_train_losses) + 1)),
        train_losses=generation_train_losses,
        validation_losses=generation_validation_losses,
        x_column="generation",
        x_label="Generation",
        file_prefix="train_validation_loss_by_generation",
        title="Train and Validation Loss per Generation",
    )
    train_validation_evaluation_loss_plot, train_validation_evaluation_loss_csv = save_train_validation_loss_history(
        seed=seed,
        x_values=list(range(1, len(evaluation_train_losses) + 1)),
        train_losses=evaluation_train_losses,
        validation_losses=evaluation_validation_losses,
        x_column="evaluation_count",
        x_label="Evaluation Count",
        file_prefix="train_validation_loss_by_evaluation",
        title="Train and Validation Loss per Fitness Evaluation",
    )
    return {
        "seed": seed,
        "best_chromosome": _json_chromosome(best_chromosome),
        "best_activation_expression": best_activation_expression,
        "best_score": best_score,
        "best_test_loss": best_test_loss,
        "best_train_loss": best_train_loss,
        "best_validation_loss": best_validation_loss,
        "best_test_accuracy": best_test_accuracy,
        "best_train_accuracy": best_train_accuracy,
        "best_validation_accuracy": best_validation_accuracy,
        "best_generation": best_generation,
        "best_evaluation": best_evaluation,
        "best_model_path": best_model_path,
        "top3_activation_functions_csv": top3_activation_functions_csv,
        "highest_accuracy": highest_accuracy,
        "generation_score_history_csv": generation_score_history_csv,
        "evaluation_results_csv": evaluation_results_csv,
        "generation_losses": generation_losses,
        "loss_plot": loss_plot,
        "loss_csv": loss_csv,
        "evaluation_losses": evaluation_losses,
        "evaluation_loss_plot": evaluation_loss_plot,
        "evaluation_loss_csv": evaluation_loss_csv,
        "generation_train_losses": generation_train_losses,
        "generation_validation_losses": generation_validation_losses,
        "evaluation_train_losses": evaluation_train_losses,
        "evaluation_validation_losses": evaluation_validation_losses,
        "train_validation_generation_loss_plot": train_validation_generation_loss_plot,
        "train_validation_generation_loss_csv": train_validation_generation_loss_csv,
        "train_validation_evaluation_loss_plot": train_validation_evaluation_loss_plot,
        "train_validation_evaluation_loss_csv": train_validation_evaluation_loss_csv,
    }


def run_multiple_experiments(pop_size=5, gen=10, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2]

    results = []
    for run_index, seed in enumerate(seeds, start=1):
        print(f"=== Run {run_index}/{len(seeds)} | seed={seed} ===")
        result = run_ga(pop_size=pop_size, gen=gen, seed=seed)
        results.append(result)
    save_results_summary(results)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.seeds:
        run_multiple_experiments(
            pop_size=args.pop_size,
            gen=args.generations,
            seeds=args.seeds,
        )
    elif args.runs > 1:
        run_multiple_experiments(
            pop_size=args.pop_size,
            gen=args.generations,
            seeds=list(range(args.runs)),
        )
    else:
        result = run_ga(
            pop_size=args.pop_size,
            gen=args.generations,
            seed=0,
        )
        save_results_summary([result])
