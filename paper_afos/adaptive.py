"""Fitness-driven GA probabilities, reconstructed from saved generation history."""
import math


def validate_adaptive_config(search):
    adaptive = search.get("adaptive_probabilities")
    if adaptive is None:
        return
    if not isinstance(adaptive, dict) or type(adaptive.get("enabled")) is not bool:
        raise ValueError("adaptive_probabilities.enabled must be a boolean")
    fields = {"enabled", "window_generations", "improvement_target",
              "max_crossover_probability", "max_mutation_probability", "epsilon"}
    if set(adaptive) - fields:
        raise ValueError("Unknown adaptive_probabilities setting")
    if not adaptive["enabled"]:
        return
    if fields - set(adaptive):
        raise ValueError("Missing adaptive_probabilities setting")
    window = adaptive["window_generations"]
    if type(window) is not int or window < 1:
        raise ValueError("window_generations must be a positive integer")
    for field in fields - {"enabled", "window_generations"}:
        value = adaptive[field]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid adaptive_probabilities.{field}")
    if adaptive["improvement_target"] <= 0 or adaptive["epsilon"] <= 0:
        raise ValueError("improvement_target and epsilon must be positive")
    if not search["crossover_probability"] <= adaptive["max_crossover_probability"] <= 1:
        raise ValueError("max_crossover_probability must be between the base probability and 1")
    if not search["mutation_probability"] <= adaptive["max_mutation_probability"] <= 1:
        raise ValueError("max_mutation_probability must be between the base probability and 1")


def generation_probabilities(search, history, seed, generation, best_score):
    """Return probabilities for generation + 1, using only this seed's past scores.

    B_g is the best score seen through generation g. After a warmup of w
    generations, r = (B_g - B_(g-w)) / max(B_(g-w), epsilon). A clipped
    r / improvement_target interpolates from maximum rates at stagnation to
    base rates at or above the improvement target.
    No random draws or mutable controller state are needed, including on resume.
    """
    def finite_score(value):
        return value if value is not None and math.isfinite(value) and value > 0 else 0.0

    previous = [row for row in history
                if row["seed"] == seed and row["generation"] < generation]
    best = max([finite_score(best_score)] + [finite_score(row["best_score"]) for row in previous])
    info = dict(best_so_far_score=best, probability_mode="fixed",
                fitness_reference_generation=None, fitness_reference_score=None,
                fitness_relative_improvement=None, improvement_fraction=None,
                crossover_probability=search["crossover_probability"],
                mutation_probability=search["mutation_probability"])
    adaptive = search.get("adaptive_probabilities")
    if not adaptive or not adaptive["enabled"]:
        return info
    reference_generation = generation - adaptive["window_generations"]
    if reference_generation < 1:
        info["probability_mode"] = "warmup"
        return info
    if not any(row["generation"] == reference_generation for row in previous):
        raise ValueError("Missing reference generation for adaptive probabilities")
    reference = max(finite_score(row["best_score"]) for row in previous
                    if row["generation"] <= reference_generation)
    improvement = (best - reference) / max(reference, adaptive["epsilon"])
    fraction = min(1.0, max(0.0, improvement / adaptive["improvement_target"]))
    pc_max = adaptive["max_crossover_probability"]
    pm_max = adaptive["max_mutation_probability"]
    # Convex interpolation keeps both rates within their configured bounds.
    info.update(probability_mode="adaptive",
                fitness_reference_generation=reference_generation,
                fitness_reference_score=reference,
                fitness_relative_improvement=improvement,
                improvement_fraction=fraction,
                crossover_probability=(1 - fraction) * pc_max + fraction * search["crossover_probability"],
                mutation_probability=(1 - fraction) * pm_max + fraction * search["mutation_probability"])
    return info
