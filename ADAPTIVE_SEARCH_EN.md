# Fitness-driven crossover and mutation probabilities

[Japanese version](ADAPTIVE_SEARCH_JA.md)

Use `paper_parameters_adaptive.json` to calculate the probabilities for generating the next population from each seed's improvement in best-so-far fitness. This is an extension to the original fixed-probability experiment. At 0% improvement, crossover is 100% and mutation is 80%. At 1% improvement or higher, they return to the original 80% and 20%. Performance gains have not yet been evaluated through full training runs.

## Conditions retained for comparison

`paper_parameters.json` still uses fixed crossover and mutation probabilities of 0.8 and 0.2. The adaptive configuration differs only by the added `search.adaptive_probabilities` section.

The following conditions remain the same: 30 individuals, 40 generations, seeds 0–4, six retained parents, at most six crossover children, at most one mutation child, at least 17 random individuals, CNN architecture, data split, training settings, fitness calculation, and retraining settings. Match the previous runtime environment as well. The same seeds do not imply the same populations or subsequent random-number sequences once the probabilities change.

## Changes from the previous code

The comparison baseline is the fixed-probability implementation in commit
[`54982d93`](https://github.com/OjasWanjari/AFOS_experiment/commit/54982d93cd36eec2c1623f2c1107c7fb6da5ced2),
immediately before this change. Line references below point to the code saved with this guide. Existing experiment `source_snapshot` directories have not been edited.

| File and location | Before | Change |
| --- | --- | --- |
| [run_paper.py:21](run_paper.py#L21) | No adaptive-control imports | Import `generation_probabilities` and `validate_adaptive_config` |
| [run_paper.py:82, load_config()](run_paper.py#L82) | Validate the existing configuration | Also validate the lookback window, improvement target, and probability limits |
| [run_paper.py:177, search()](run_paper.py#L177) | Use fixed rates after every generation | Calculate improvement and next-generation rates from the same seed's generation history |
| [run_paper.py:180](run_paper.py#L180) | Pass `s` directly to `next_population()` | Copy `s` into `generation_config`, replacing only the two probabilities |
| [run_paper.py:185](run_paper.py#L185) | No probability-update log | Log improvement, applied probabilities, and offspring counts; also save them in generation history |
| [run_paper.py:193](run_paper.py#L193) | No adaptive information for the final generation | Leave probabilities and next-generation number empty when no offspring are generated |
| [paper_afos/adaptive.py:5](paper_afos/adaptive.py#L5) | File did not exist | Add `validate_adaptive_config()` |
| [paper_afos/adaptive.py:34](paper_afos/adaptive.py#L34) | File did not exist | Add `generation_probabilities()`, including warmup, seed isolation, and zero-denominator handling |
| [paper_parameters_adaptive.json:22](paper_parameters_adaptive.json#L22) | File did not exist | Copy the original configuration and add only `search.adaptive_probabilities` |
| [tests/test_adaptive_search.py](tests/test_adaptive_search.py) | File did not exist | Add probability, population-allocation, fixed-mode equivalence, and resume tests |
| [RUN_GUIDE_JA.md](RUN_GUIDE_JA.md) / [RUN_GUIDE_EN.md](RUN_GUIDE_EN.md) | No adaptive-search instructions | Link the adaptive guides and list `adaptive.py` as a required file |
| [ADAPTIVE_SEARCH_JA.md](ADAPTIVE_SEARCH_JA.md) / [ADAPTIVE_SEARCH_EN.md](ADAPTIVE_SEARCH_EN.md) | Files did not exist | Document the behavior, code changes, and execution steps in Japanese and English |

The main change is the configuration passed to next-generation creation. These excerpts are from `search()`.

Before:

```python
state["population"], counts = next_population(
    state["population"], [r["score"] for r in done], s,
    random.Random(training_seed(seed, generation, 999999)),
)
```

After (calculate rates after each generation; generate offspring only when another generation remains):

```python
entry.update(generation_probabilities(
    s, state["generations"], seed, generation, entry["best_score"],
))
if generation < s["generations"]:
    generation_config = dict(
        s,
        crossover_probability=entry["crossover_probability"],
        mutation_probability=entry["mutation_probability"],
    )
    state["population"], counts = next_population(
        state["population"], [r["score"] for r in done], generation_config,
        random.Random(training_seed(seed, generation, 999999)),
    )
```

The crossover, mutation, parent-selection, and random-population-filling operators in `paper_afos/search.py` are unchanged. The CNN and training code in `paper_afos/model.py` and the original `paper_parameters.json` are also unchanged. The current `run_paper.py` always imports `adaptive.py`, so include that new file even when running in fixed mode.

Compared with the first adaptive proposal discussed before this version, stagnation crossover has changed from 50% to 100%. The configuration key changed from `min_crossover_probability` to `max_crossover_probability`, and the crossover formula changed from `0.5 + 0.3 * q` to `1.0 - 0.2 * q`. The mutation formula, 1% improvement target, and three-generation window remain the same. Replace any configuration from that earlier proposal with the current configuration and start a new experiment.

## Calculation

Fitness remains `validation_accuracy / validation_loss`. Let `B_g` be the largest recorded score from generations 1 through g of the current seed. Early-stopped candidates remain included according to the existing score definition; candidates failing with nonfinite metrics receive zero as before. Test metrics are not used.

The supplied configuration calculates:

```text
r = (B_g - B_(g-3)) / max(B_(g-3), 1e-12)
q = min(1, max(0, r / 0.01))

next-generation crossover probability = 1.0 - 0.2 * q
next-generation mutation probability  = 0.8 - 0.6 * q
```

`r` is the relative gain across the entire three-generation interval, not an average per generation or an accuracy percentage-point change. Since it uses cumulative maxima, it cannot be negative. A lower current-generation best does not erase a previously recorded best.

| Relative improvement over three generations | Crossover | Mutation |
| --- | --- | --- |
| 0% | 1.00 | 0.80 |
| 0.25% | 0.95 | 0.65 |
| 0.5% | 0.90 | 0.50 |
| 0.75% | 0.85 | 0.35 |
| 1% or higher | 0.80 | 0.20 |

Generation 1 is initialized randomly as before. After generations 1–3, there is no complete comparison window, so offspring use crossover 0.8 and mutation 0.2. The first adaptive update occurs after generation 4: compare `B_4` with `B_1` and use the resulting rates to generate generation 5. Each seed uses only its own history. If the reference fitness is zero, epsilon supplies the denominator; if the current best is also zero, the improvement is zero.

Mutation still generates either zero or one child, even at probability 0.8. Crossover has three attempts, each producing two children. At zero improvement, all three attempts occur, producing six crossover children. With six retained parents and zero or one mutation child, random generation supplies 18 or 17 individuals to reach 30. As improvement increases and both probabilities decrease, the expected number of random individuals increases. The two probabilities govern independent decisions and need not sum to one.

## Configuration

Edit `search.adaptive_probabilities` in `paper_parameters_adaptive.json`.

| Key | Supplied value | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Set to `false` to use fixed probabilities |
| `window_generations` | `3` | Lookback interval for cumulative best fitness |
| `improvement_target` | `0.01` | Relative improvement needed to return to the original rates; 0.01 means 1% |
| `max_crossover_probability` | `1.0` | Crossover at stagnation; the lower endpoint is the original `crossover_probability` |
| `max_mutation_probability` | `0.8` | Mutation at stagnation; the lower endpoint is the original `mutation_probability` |
| `epsilon` | `1e-12` | Minimum denominator to avoid division by zero |

Rates are recalculated directly from improvement every generation; this does not add or subtract a fixed step from the previous rates. Variation in training also affects the recorded best fitness, so this control rule does not itself establish a true performance gain.

## Execution

Run the following from the repository root using the existing environment described in [RUN_GUIDE_EN.md](RUN_GUIDE_EN.md). Inspect the configuration without loading TensorFlow or data:

```powershell
.\.venv-gpu\Scripts\python.exe run_paper.py --config paper_parameters_adaptive.json --dry-run
```

Start search in a new output directory:

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --output runs/paper_adaptive_fitness
```

Omit `--stage search` to continue through retraining. Resume an interrupted adaptive run with the same configuration and source:

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --resume runs/paper_adaptive_fitness
```

Start a new experiment rather than switching a previous fixed-probability run to adaptive mode. Existing resume checks reject configuration or source changes. To resume an older run, use its original `source_snapshot` and configuration. The standard `--smoke` run has only one generation and therefore does not exercise adaptive updates.

## Saved records and implementation

Generation history in `results/generation_history.csv` and `checkpoint.json` includes:

- `best_score`: the current generation's maximum fitness.
- `best_so_far_score`: the cumulative maximum for the current seed.
- `fitness_reference_generation` / `fitness_reference_score`: the comparison generation and its cumulative maximum.
- `fitness_relative_improvement` / `improvement_fraction`: r and its normalized, clipped fraction q.
- `probability_mode`: `fixed`, `warmup`, `adaptive`, or `search_complete` for the final generation.
- `next_generation`: the generation created using the recorded rates and counts.
- `crossover_probability` / `mutation_probability`: the probabilities actually used to generate that next population.
- `parents` / `crossover` / `mutation` / `random`: actual next-generation population counts.

For example, rates and counts on generation 4's row describe generation 5's creation. The final generation has no offspring, so its probabilities and next-generation number are empty. Rates and improvement are also written to `run.log`. Saved generation scores are sufficient to reconstruct the rates on resume; no separate mutable controller state is needed.

- `paper_afos/adaptive.py`: configuration validation and fitness-based probability calculation.
- `run_paper.py`: generation-end calculation, application to offspring creation, and recording.
- `paper_afos/search.py`: existing population-generation operators.
- `tests/test_adaptive_search.py`: probability, validation, fixed-mode equivalence, and resume tests.

Run adaptive tests without model training:

```powershell
python -m unittest discover -s tests -p test_adaptive_search.py -v
```

Include existing population-generation and recording regression tests (16 tests total):

```powershell
python -m unittest tests.test_adaptive_search tests.test_paper.SearchTests tests.test_paper.RecordingTests -v
```

All 16 tests and the configuration dry run have passed. With model training replaced by a test implementation, interruptions during candidate evaluation and immediately before committing a generation update both resume to the same final state and generation CSV as an uninterrupted run. These checks do not measure accuracy or fitness gains from full training.
