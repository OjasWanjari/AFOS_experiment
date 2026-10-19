# Retrain the best functions pooled across all search seeds

`retrain_paper.py` reads saved search results, **selects the global top 15 functions across the search seeds, and runs only retraining and test evaluation**. It never starts a search. ReLU, Swish, Mish, GELU, SERF, and ELU are also retrained under the same conditions.

The default workload is **(15 discovered functions + 6 baselines) × 5 retraining seeds = 105 trials**, each lasting 50 epochs. Each trial starts with a new model. Search weights and previous seed-0 or seed-4 retraining results are not reused.

## Updated workflow: use the new seed-4 results (2026-10-19)

This guide selects the newly added `paper_full_results/paper_full_seeds4` as the source for search seed 4. Its results are pooled with search seeds 0–3 to select one common set of 15 functions, each evaluated with retraining seeds 0–4. This does not retrain only seed 4.

**Resuming the previous `retrain_runs/pooled_top15_prepared` does not incorporate the new search results.** Candidates in `plan.json` are frozen when the plan is created. Use the new output directory `retrain_runs/pooled_top15_updated`. If you have already created a plan there, verify its source as described in Section 2 before resuming. If it uses different sources, choose another unused output directory.

## Relationship to the paper

Reference: the user-provided PDF, *Activation Function Optimization Scheme for Image Classification*, arXiv:2409.04915v1, Section 4.1, printed pages 15–16, and Tables 2, 3, and 6.

| Item | Default behavior |
| --- | --- |
| Candidate ranking | **Validation accuracy** of completed 15-epoch search evaluations, never fitness or test accuracy |
| Candidate count | 15 functions across all search seeds combined |
| Retraining | From scratch for 50 epochs, five repetitions per function (seeds 0–4) |
| CNN | Table 2: six convolution layers with 28, 32, 64, 64, 128, 128 filters |
| Optimization | SGD, learning rate 0.01, momentum 0.9, batch size 64, he_uniform, same padding |
| Loss | Categorical cross-entropy |
| Data | CIFAR-10, using the saved split: 45,000 training, 5,000 validation, 10,000 test examples |
| Evaluation | Evaluate the final model after 50 epochs; do not select a best epoch afterward |
| Aggregation | Mean and sample standard deviation (ddof=1), only when all five test evaluations succeed |

**The test set is not used for training.** Training uses the training split, with the validation split monitored during training. The test set is used only for the final evaluation.

The paper does not fully specify pooling across search repetitions, deduplication, exact seeds, split retention during retraining, or the standard deviation convention. This implementation retains the existing project's assumptions and makes these choices explicit:

- Keep the highest-validation-accuracy record for each **exact expression string**. Algebraic equivalence is not checked: `a*b` and `b*a` may remain separate candidates.
- Break ties by lower validation loss, then lower search seed, then lower evaluation number.
- Normalize by `/255`; use dropout 0.2, split seed 1234, no augmentation, and retraining seeds 0–4.
- Stop and record NaN/Inf outcomes. Do not apply the search-only first-epoch validation-accuracy threshold of 0.25 during retraining.
- Disable XLA/JIT, enable GPU memory growth, and use a separate process per trial to address the observed memory problems. These are execution changes; identical numerical results across hardware/library versions and complete elimination of OOM are not guaranteed.

## 1. Prepare the environment and files

Use the TensorFlow environment that ran your original searches. Python 3.10 or newer, TensorFlow/Keras, NumPy, and scikit-learn are required. Do not install the Windows DirectML requirements into the Linux/CUDA experiment environment.

Copy the repository to the other PC, or at least copy:

```text
retrain_paper.py
run_paper.py                 # Shared configuration-validation and JSON/CSV helpers
paper_parameters.json
paper_afos/                 # Entire folder, including retraining.py and retrain_worker.py
paper_full_results/
  paper_full_seeds0/
    manifest.json
    split_indices.npz
    results/evaluation_results_seed_0.csv
  paper_full_seeds1/         # Search seed 1
  paper_full_seeds2/         # Search seed 2, also contains a partial seed-3 run
  paper_full_seeds3 and 4/   # Use its 1,200 seed-3 records; also contains old seed 4
  paper_full_seeds4/         # Use this newly added search seed 4
    manifest.json
    split_indices.npz
    results/evaluation_results_seed_4.csv
```

Each run needs `manifest.json`, `split_indices.npz`, and `results/evaluation_results_seed_*.csv`. Keep `paper_full_seeds3 and 4`: it is still needed for seed 3. Do not delete that directory to exclude old seed 4. The explicit `--source` option chooses new seed 4 without pooling both seed-4 trajectories. Search results were not included in the code push to GitHub; copy them separately to the execution PC.

`run_paper.py` is imported only for helpers; no search function is called. Copying just `retrain_paper.py` is insufficient. Run every command below from the repository root.

Example using the existing Linux environment:

```bash
cd /home/victor/AFOS_experiment
source /home/victor/myafos/bin/activate
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Example using the existing Windows PowerShell environment:

```powershell
cd D:\vscode\AFOS-main
.\.venv-gpu\Scripts\Activate.ps1
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

The following commands use the activated environment's `python`. Alternatively, replace `python` with that environment's Python executable. Workers use the same executable. A real training run downloads CIFAR-10 if it is absent from the Keras cache.

## 2. Reselect the top 15 using the new seed-4 source

```bash
python retrain_paper.py --results-root paper_full_results --source "4=paper_full_results/paper_full_seeds4" --output retrain_runs/pooled_top15_updated --plan-only
```

This does not import TensorFlow, start GPU training, or download the dataset. Specify an output directory that does not already exist.

- `selected_top15.csv`: expressions, chromosomes, originating search seeds, evaluation numbers, and validation metrics.
- `selection_audit.json`: chosen source runs and information about excluded or missing evaluations.
- `plan.json`: frozen selection, 105 trials, configuration, and source-code hashes.
- `source_snapshot/` and `split_indices.npz`: copies of the execution code and dataset split.

When several runs contain the same search seed, choose **one trajectory with the largest contiguous record count**, starting at evaluation 1. This does not pick a trajectory by its accuracy. For the supplied seed-3 data, the 1,200-record run takes precedence over the 382-record partial run.

Old and new seed 4 both contain 1,200 records but have different contents. Omitting `--source` therefore produces `Ambiguous equal-length trajectories for seed 4`. Include the override in the command above. To explicitly select seed 3 as well, specify both sources (this creates a separate new plan):

```bash
python retrain_paper.py --results-root paper_full_results --source "3=paper_full_results/paper_full_seeds3 and 4" --source "4=paper_full_results/paper_full_seeds4" --output retrain_runs/pooled_explicit_updated --plan-only
```

All configured search seeds, 0–4 by default, must be present. Chosen runs must agree on training conditions, primitive ordering, and the saved data split. Only records with `status=completed`, the full search epoch count, and finite losses/accuracies are eligible for ranking.

New seed 4 contains 1,200 evaluations: 596 completed all 15 epochs, 130 stopped for low accuracy, and 474 stopped with nonfinite values. None ended at zero epochs. The earlier 851 zero-epoch records belong to old seed 4; selecting the new source removes that old-source warning. Nonfinite evaluations remain excluded from candidate selection.

Before training, inspect `retrain_runs/pooled_top15_updated/selection_audit.json`. For `seed: 4`, check that `selected_source` ends in `paper_full_seeds4`, `recorded` is 1200, `eligible` is 596, and `zero_epoch_records` is 0. `plan.json` should contain 105 trials, retraining seeds `[0, 1, 2, 3, 4]`, and 50 epochs.

For the data checked on 2026-10-19, rank 1 is `(erf(x)*sigmoid(sinh(x)))` (82.50% validation accuracy), and rank 2 is `(sigmoid(x)*erf(erf(x)))` from new seed 4 (81.94%). One function is replaced compared with the old top-15 set. These validation metrics help verify selection; they are not test accuracies from the new retraining run.

To inspect an existing plan without training, use the following command. It does not reselect candidates:

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume --plan-only
```

## 3. Start retraining

Execute the plan prepared above:

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume
```

Or prepare a plan and begin training in one command, using another new directory:

```bash
python retrain_paper.py --results-root paper_full_results --source "4=paper_full_results/paper_full_seeds4" --output retrain_runs/pooled_new_updated
```

GPU execution is the default. If TensorFlow cannot find a GPU, the trial fails rather than silently using the CPU. To use the CPU, add `--device cpu` **when preparing the plan**.

Run only the first trial to inspect behavior and duration:

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume --max-trials 1
```

Resume without `--max-trials` to continue. Each trial writes progress to `trials/<function>_seed_<seed>/attempt_0001/worker.log` and `histories/`. Use `tail -f <worker.log path>` on Linux or `Get-Content <worker.log path> -Wait` in PowerShell to follow it.

Do not add `--source` or `--results-root` to `--resume`: it uses the inputs frozen in the output directory. Always specify the newly prepared output directory; resuming an old directory trains its old selection.

If only metrics are needed, add `--no-save-models` **when creating a new plan**. To use a different configuration, pass `--config <JSON file>`. Distinguish experiments with modified settings from the default paper protocol.

## 4. Resume after interruption or an error

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume
```

Completed trials are skipped. An OOM, Python exception, or abnormal worker termination stops the coordinator immediately to avoid a cascade of failures. After investigating and resolving the cause, explicitly retry infrastructure failures:

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume --retry-errors
```

- A failed trial restarts from **epoch 1 with the same seed**. Mid-training weights and optimizer state are not restored.
- Previous logs/results remain in `attempt_0001/`; retries use `attempt_0002/`, etc. Aggregation uses the latest attempt.
- `nonfinite` / `nonfinite_test` are recorded as terminal numerical outcomes and execution continues to the next trial. Even `--retry-errors` does not retry them. This avoids reporting only lucky successful reruns.
- A model-save failure does not erase valid test metrics. The trial retains `status=completed` with `model_save_status=error`, and execution stops. A normal `--resume` proceeds to the next trial; retrying only model serialization is not implemented.
- A forced termination may leave `run.lock`. Remove only that file **after confirming both coordinator and worker have exited**. Do not run two coordinators against the same output directory.
- Resume uses the frozen candidates, settings, and source snapshot. Editing original search files or working-tree code does not reselect candidates. Use a new output directory to change settings, device, or selection. Editing frozen files fails hash verification.

Each worker exits after one trial, releasing that process's GPU allocations. This does not prevent a single trial from exceeding available memory or another process from occupying the GPU. Set `CUDA_VISIBLE_DEVICES` before launch if needed. If you change the environment before resuming, inspect each attempt's `environment.json` and disclose the change when reporting results.

## 5. Read the outputs

| File | Contents |
| --- | --- |
| `results/retraining_results.csv` | Latest state of all 105 trials; unstarted trials are `pending`; accuracies are fractions from 0 to 1 |
| `results/retraining_summary.csv` | Success count, mean, and sample standard deviation per function; mean/SD stay blank until all five evaluations succeed |
| `status.json` | Overall completed/failed/pending counts and selection warnings |
| `trials/<trial>/attempt_0001/result.json` | Trial metrics, timestamps, and error information |
| `trials/<trial>/attempt_0001/histories/<trial>.csv` | Per-epoch loss/accuracy/val_loss/val_accuracy; raw NaN/Inf are retained |
| `trials/<trial>/attempt_0001/worker.log` | TensorFlow and training output |
| `trials/<trial>/attempt_0001/environment.json` | Python, TensorFlow/Keras and other package versions, device information; recorded after TF initialization |
| `trials/<trial>/attempt_0001/model.keras` | Model with a valid test evaluation, when model saving is enabled |
| `error.txt` / `save_error.txt` | Exception details within the affected attempt, if applicable |

Nonfinite values become null/blank in aggregate JSON/CSV files. A mean of 0.825 is 82.5%; an SD of 0.0014 is 0.14 percentage points. Any warnings present when the plan was created remain even if retraining succeeds. The old seed-4 warning about 851 zero-epoch evaluations does not apply to the new seed-4 source used here.

The legacy runner for new seed 4 failed during plotting after training with `No module named 'matplotlib'`. Its saved search CSVs remain usable for candidate selection. The retraining-only runner produces CSV/JSON summaries without plotting, so it does not require `matplotlib`. ELU's NaN in that legacy run is a separate numerical issue; numerical failures may still occur during the new retraining run.

Exit codes: 0 = the requested invocation finished normally (planning or `--max-trials` may leave pending trials); 1 = configuration/runtime error; 2 = processing finished but numerical failures or model-save errors remain; 130 = user interruption. Check `all_evaluations_completed` and `model_save_errors` in `status.json` for full completion.

## Small smoke test

```bash
python retrain_paper.py --results-root paper_full_results --source "4=paper_full_results/paper_full_seeds4" --output retrain_runs/smoke_updated --smoke-test --device cpu
python retrain_paper.py --output retrain_runs/smoke_updated --resume
```

This trains the top candidate and ReLU for one epoch each, seed 0, on 32 synthetic training examples. It does not download CIFAR-10. The plan is labeled `smoke=true`. **Do not compare these accuracies with the paper.** The second command also checks that completed trials are skipped.

Automated tests:

```bash
python -m unittest tests.test_pooled_retraining -v
```
