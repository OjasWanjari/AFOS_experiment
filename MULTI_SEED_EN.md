# Run multiple seeds sequentially from JSON

[日本語](MULTI_SEED_JA.md)

`run_paper.py` reads `search.seeds` and runs **each search seed in a fresh Python process**.
It waits for that process to exit before starting the next seed. Workers do not run in parallel.
The coordinator does not import TensorFlow or load training data, so it does not retain training allocations between seeds.

Previously, one TensorFlow process handled all seeds. This change addresses runs that succeed individually but run out of memory when combined.
It does not reduce the memory required by an individual seed or guarantee that OOM cannot occur within a seed.

## Configure seeds 2, 3, and 4

Change these two fields inside `search` in `paper_parameters_adaptive.json`. Keep the other settings as intended for the experiment.

```json
"seeds": [2, 3, 4],
"replications": 3
```

`post_search.seeds` specifies retraining seeds independently and is not changed automatically.
To reuse previous baseline results instead of retraining the six baselines, set `post_search.baselines` to `[]` before starting.

## Start the run

Activate the environment and run these commands from the repository, on Linux or Windows:

```bash
python run_paper.py --config paper_parameters_adaptive.json --dry-run
python -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --output runs/adaptive_seeds234_01
```

With 40 generations, 30 individuals, and three seeds, `search_evaluations` is 3600.
The first command only checks the configuration. The second runs seeds 2, 3, and 4 in order.
Use an output directory that does not already exist. The final `Finished:` message indicates completion of the requested stage.

To continue through retraining, replace `--stage search` with `--stage all` in the second command.
Retraining uses a separate process for each activation-function/retraining-seed trial.
Repeated TensorFlow initialization and data loading add startup overhead.

## Outputs and selection

All seeds share one output directory:

- `results/evaluation_results_seed_2.csv`, etc.: per-seed candidate results.
- `results/generation_history.csv`: generation histories and adaptive probabilities across all seeds.
- `results/selected_top15.csv`: functions selected across all seeds configured in this run.
- `checkpoint.json`: current seed/generation, completed evaluations, and retraining trials.
- `run.log` / `results/events.csv`: progress. `worker_started` / `worker_completed` record process IDs.

Previous seed 0/1 results in other folders are not added to selection automatically.
Search settings, fitness calculations, adaptive probabilities, and training-seed derivation are unchanged.

## Resume after interruption or OOM

A failed worker stops the queue immediately. The next seed does not start automatically.
On Ctrl+C, the coordinator terminates and waits for its training process.

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --resume runs/adaptive_seeds234_01
```

Use the original configuration, source, and environment. Completed seeds/evaluations are retained.
An interrupted candidate or retraining trial restarts its training from scratch.
After search completes, use `--stage retrain --resume runs/adaptive_seeds234_01` to add retraining.

**Runs started with the previous code cannot be resumed directly with this changed source.**
Source verification remains enforced. Resume old runs with their recorded `source_snapshot`, or start seeds 2–4
in a new output directory using the updated program. Do not modify historical checkpoints or results.

## Implementation and checks

`run_isolated()` and `wait_for_worker()` in `run_paper.py` schedule workers and wait for process exit.
`search(..., one_seed=True)` returns after one search seed; `retrain(..., one_trial=True)` returns after one retraining trial.
A final isolated process aggregates CSV outputs and generates plots. Do not pass the internal `--_worker` option manually.

```bash
python -m unittest tests.test_seed_processes tests.test_adaptive_search tests.test_paper.SearchTests tests.test_paper.RecordingTests -v
```

Automated tests use real processes with training doubles to check seeds 2–4, pooled selection, retraining, and failure/resume behavior.
Check GPU memory use on the target PC during its actual run.
