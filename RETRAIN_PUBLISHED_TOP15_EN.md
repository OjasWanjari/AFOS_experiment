# Evaluate the published Top 15 under the existing experiment conditions

`retrain_published_top15.py` evaluates the **15 fixed functions in Table 3, printed page 16, of the supplied arXiv:2409.04915v1 PDF**. Each function is trained from scratch for **50 epochs with seeds 0, 1, 2, 3, 4: 75 trials total**. It does not run a search, reselect candidates, or repeat the six baseline functions.

## What is held fixed

The reference is the pooled retraining output imported as `paper_full_results/retrain_runs/pooled_top15_updated`.

- Copy the entire configuration from its frozen plan: SGD learning rate 0.01, momentum 0.9, batch size 64, architecture, dropout, initialization, and all other settings.
- Copy the saved `split_indices.npz` byte-for-byte, retaining the membership and order of 45,000 training and 5,000 validation examples. The 10,000 test examples are used only for final evaluation.
- Copy the reference's frozen model, data loader, primitive evaluator, training, stopping, recording, coordinator and summary code. Verify their hashes. Only worker environment recording/checking is updated.
- Initialize a fresh model for each seed. Evaluate the final 50-epoch model, without selecting a best epoch or reusing search/retraining weights.
- Retain JIT disabled, GPU memory growth, and one fresh process per trial.
- Keep NaN/Inf as failed outcomes. Do not report a five-seed mean from only successful repetitions.

**Run the actual evaluation on the original Linux PC, in the original virtual environment.** The imported run recorded Python 3.10.12, TensorFlow 2.21.0, Keras 3.12.4, NumPy 2.2.6, scikit-learn 1.7.2, and an RTX 4080 SUPER. Reuse that environment; do not upgrade packages or recreate it from guessed requirements.

The guard compares Python/package versions, OS family, TensorFlow CUDA/cuDNN build metadata, GPU model/count, XLA flags, and saved runtime settings. A mismatch stops before data loading/training. Original GPU driver versions and every environment variable were not recorded; this guard does not certify complete environment identity or bitwise GPU determinism.

## 1. Required files

Use the updated repository, including `retrain_published_top15.py`, `retrain_paper.py`, `run_paper.py`, and the entire `paper_afos/` directory. Keep the previous retraining output:

```text
previous_retraining_output/
  plan.json
  plan.sha256
  split_indices.npz
  source_snapshot/
  trials/.../attempt_.../
    result.json
    environment.json
    worker.log
```

Preparation checks environments recorded by completed reference trials. Keep those trial records. Previous `model.keras` weights are not needed, and search CSVs are not used to reselect functions.

## 2. Prepare a plan without training

Run from the repository root. For the imported Windows layout:

```powershell
python retrain_published_top15.py --reference-run paper_full_results/retrain_runs/pooled_top15_updated --output retrain_runs/published_top15_matched --plan-only
```

On the original Linux PC, if the previous output is under `retrain_runs/pooled_top15_updated`:

```bash
cd /home/victor/AFOS_experiment
source /home/victor/myafos/bin/activate
python retrain_published_top15.py --reference-run retrain_runs/pooled_top15_updated --output retrain_runs/published_top15_matched --plan-only
```

`--reference-run` must identify the **previous retraining output**, not a search directory. `--output` must be new. Use `--resume` for a prepared output.

Preparation does not import TensorFlow, download data, or start training. It freezes formulas/configuration in `selected_top15.csv` and `plan.json`.

Alternatively, prepare on Windows and copy the **entire** `retrain_runs/published_top15_matched` folder to the original Linux PC. After preparation, the original reference folder is not required to run the frozen plan. The original absolute reference path in the plan is provenance, not a runtime input path.

## 3. Verify the original environment without training

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --check-environment
```

Success prints `Recorded environment matches`. Details are saved to `environment_preflight.json`. This loads TensorFlow but no dataset and performs no training. Resolve discrepancies by using the original machine/environment/GPU; do not bypass the guard and label a different environment as a matched comparison.

## 4. Run one trial, then the rest

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --max-trials 1
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume
```

The first trial is `paper_01_seed_0` (EELU-4). The second command skips completed trials and continues. To start all 75 at once, use only the second command. Every worker repeats the environment check before loading data.

Do not combine `--resume` with `--reference-run`. Resume uses frozen inputs and code.

## 5. Failures and resumption

After fixing an interruption, OOM, or environment/runtime failure:

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --retry-errors
```

- Retry the same seed from epoch 1 and preserve previous attempts.
- `nonfinite` and `nonfinite_test` are terminal numerical outcomes; even this flag does not retry them.
- If a forced stop leaves `run.lock`, remove only that file after verifying that both coordinator and worker have exited.
- Never launch two coordinators against the same output directory.
- Exit code 2 may indicate numerical failures. Read `status.json` for completed/failed/pending counts.

## Function IDs and formula interpretation

`paper_01` through `paper_15` follow **Table 3's printed row order**, not the EELU numbering and not the previous `discovered_XX` functions.

| ID | Published function | Published mean +/- SD (%) |
|---|---|---:|
| paper_01 | EELU-4: erf(x) * softplus(x) | 82.74 +/- 0.14 |
| paper_02 | EELU-2: -x * erf(exp(-x)) | 82.67 +/- 0.30 |
| paper_03 | EELU-1: x * erf(exp(x)) | 82.66 +/- 0.29 |
| paper_04 | EELU-3: x * erf(exp(x))**2 | 82.48 +/- 0.61 |
| paper_05 through paper_15 | Table 3 rows 5 through 15; all formulas in selected_top15.csv | Recorded in that CSV |

For EELU-2, follow the **mathematical formula** in Tables 3/4. The implementation column in Table 4 repeats EELU-1 and is not used.

EELU-3 uses the existing operator tree `(x * erf(exp(x))) * erf(exp(x))`. It is mathematically equivalent to `x * erf(exp(x))**2`, but floating-point evaluation order may differ. Softplus uses the same `tf.nn.softplus` as the previous candidate run. No exponential clipping, gradient clipping, or learning-rate changes are introduced.

## Outputs and comparison

- `selected_top15.csv`: fixed functions, chromosomes, Table 3 row numbers and reported paper values.
- `reference_audit.json`: reference plan hash, preserved code hashes and checked environment records.
- `results/retraining_results.csv`: status and test metrics for 75 trials; accuracies are fractions, not percentages.
- `results/retraining_summary.csv`: mean and sample SD only when all five repetitions succeed; otherwise blank.
- `trials/<ID>/attempt_0001/`: logs, environment check, epoch histories, results, and successfully saved models.

Compare the previous `pooled_top15_updated/results/` against the new `published_top15_matched/results/`, pairing identical retraining seeds. Distinguish the paper's published numbers from the newly measured results for its functions.

Continue to disclose the previous candidate 07/10 algebraic duplication, candidate 13's one failure, and ELU's five failures. These 75 trials do not fill those previous failures.
