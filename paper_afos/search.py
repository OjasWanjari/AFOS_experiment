"""Typed chromosomes and GA operators for paper Sections 3.1.1-3.1.7."""
import math

# Indices deliberately differ from the legacy activation.py. Always save the schema.
UNARY = (
    "x", "neg", "exp", "abs", "exp_neg", "min_zero", "relu", "sin",
    "cos", "sinh", "tanh", "asin", "atan", "asinh", "atanh", "erf",
    "sigmoid", "x_erf", "softplus",
)
BINARY = (
    "add", "mul", "max", "min", "max_relu", "max_minzero",
    "min_relu", "min_minzero", "mul_exp", "mul_sigmoid", "mul_erf",
)


def limits(chromosome):
    if len(chromosome) not in (3, 5, 7):
        raise ValueError("Paper chromosome length must be 3, 5 or 7")
    n = (len(chromosome) - 1) // 2
    return [len(BINARY)] * n + [len(UNARY)] * (n + 1)


def validate(chromosome):
    if any(type(g) is not int or not 0 <= g < n
           for g, n in zip(chromosome, limits(chromosome))):
        raise ValueError(f"Invalid paper chromosome: {chromosome}")
    return chromosome


def random_chromosome(rng):
    length = rng.choice((3, 5, 7))
    return [rng.randrange(n) for n in limits([0] * length)]


def tree(chromosome):
    """Fig. 2 shapes: B(U,U), B(B(U,U),U), B(B(U,U),B(U,U))."""
    c = validate(chromosome)
    u = lambda i: ("u", UNARY[c[i]])
    b = lambda i, left, right: ("b", BINARY[c[i]], left, right)
    if len(c) == 3:
        return b(0, u(1), u(2))
    if len(c) == 5:
        return b(0, b(1, u(2), u(3)), u(4))
    return b(0, b(1, u(3), u(4)), b(2, u(5), u(6)))


def expression(chromosome):
    unary = ("x", "(-x)", "exp(x)", "abs(x)", "exp(-x)", "min(x,0)",
             "max(x,0)", "sin(x)", "cos(x)", "sinh(x)", "tanh(x)",
             "asin(x)", "atan(x)", "asinh(x)", "atanh(x)", "erf(x)",
             "sigmoid(x)", "(x*erf(x))", "log(1+exp(x))")
    binary = ("({a}+{b})", "({a}*{b})", "max({a},{b})", "min({a},{b})",
              "max(max({a},0),{b})", "max(min({a},0),{b})",
              "min(max({a},0),{b})", "min(min({a},0),{b})",
              "({a}*exp({b}))", "({a}*sigmoid({b}))", "({a}*erf({b}))")
    def visit(node):
        if node[0] == "u":
            return unary[UNARY.index(node[1])]
        return binary[BINARY.index(node[1])].format(a=visit(node[2]), b=visit(node[3]))
    return visit(tree(chromosome))


def crossover(a, b, rng):
    # Same cut position as the old string encoding, but never remap unary genes
    # modulo the binary vocabulary. Only type-valid cuts are eligible.
    cuts = []
    for cut in range(min(len(a), len(b))):
        left, right = a[:cut] + b[cut:], b[:cut] + a[cut:]
        # Check semantic types as well as numeric validity.
        ta, tb = [n == len(BINARY) for n in limits(a)], [n == len(BINARY) for n in limits(b)]
        if (ta[:cut] + tb[cut:] != [n == len(BINARY) for n in limits(left)] or
                tb[:cut] + ta[cut:] != [n == len(BINARY) for n in limits(right)]):
            continue
        cuts.append(cut)
    cut = rng.choice(cuts)  # cut 0 (whole chromosome exchange) is always valid
    return a[:cut] + b[cut:], b[:cut] + a[cut:]


def mutate(chromosome, rng):
    child = chromosome[:]
    position = rng.randrange(len(child))
    choices = [g for g in range(limits(child)[position]) if g != child[position]]
    child[position] = rng.choice(choices)
    return child


def next_population(population, scores, config, rng):
    size, nparents = config["population_size"], config["selected_parents"]
    # Preserve multiplicity: the paper does not specify genotype deduplication.
    ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
    parents = [population[i][:] for i in ranked[:nparents]]
    result = [p[:] for p in parents]
    children = []
    for _ in range(config["max_crossover_children"] // 2):
        a, b = rng.sample(parents, 2)
        if rng.random() < config["crossover_probability"]:
            children.extend(crossover(a, b, rng))
    mutants = []
    for _ in range(config["max_mutation_children"]):
        candidate = rng.choice(population)
        if rng.random() < config["mutation_probability"]:
            mutants.append(mutate(candidate, rng))
    result.extend(children)
    result.extend(mutants)
    random_count = size - len(result)
    if random_count < config["minimum_random_individuals"]:
        raise ValueError("Inconsistent GA population allocation")
    result.extend(random_chromosome(rng) for _ in range(random_count))
    return result, {"parents": len(parents), "crossover": len(children),
                    "mutation": len(mutants), "random": random_count}


def fitness_score(loss, accuracy):
    if loss is None or accuracy is None or not math.isfinite(loss) or not math.isfinite(accuracy) or loss <= 0:
        return 0.0
    return accuracy / loss


def select_top(rows, count, metric="validation_accuracy"):
    """Unique expressions, pooled across search runs, ranked without test access."""
    valid = [r for r in rows if r.get("status") == "completed" and
             r.get(metric) is not None and math.isfinite(r[metric])]
    seen, selected = set(), []
    for row in sorted(valid, key=lambda r: (-r[metric], r["validation_loss"], r["seed"], r["evaluation_count"])):
        if row["activation_expression"] not in seen:
            seen.add(row["activation_expression"])
            selected.append(dict(row, rank=len(selected) + 1))
            if len(selected) == count:
                break
    return selected
