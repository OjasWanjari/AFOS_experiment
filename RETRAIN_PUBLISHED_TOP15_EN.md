# Evaluate the published Top 15 under the existing experiment conditions

`retrain_published_top15.py` evaluates the **15 fixed functions in Table 3, printed page 16, of the supplied arXiv:2409.04915v1 PDF**. Each function is trained from scratch for **50 epochs with seeds 0, 1, 2, 3, 4: 75 trials total**. It does not run a search, reselect candidates, or repeat the six baseline functions.

**If you use the supplied execution ZIP, follow section A to get started.** It includes a prepared 75-trial plan, frozen source code, saved data split, and both language guides. No additional copies of previous search or retraining results are required. Section B is the alternative workflow for creating a new plan from another reference run.

## A. Run from the ZIP (use this workflow for the supplied bundle)

### A-1. Obtain the ZIP on the original Linux PC

Open the [execution ZIP on GitHub](https://github.com/OjasWanjari/AFOS_experiment/blob/codex/pooled-retraining/retrain_runs/published_top15_matched_bundle.zip) and use **Download raw file** to save `published_top15_matched_bundle.zip`. This is different from the repository-wide archive obtained through "Code → Download ZIP". If you download on Windows, transfer the file to the original Linux PC.

The following example uses a new directory, `/home/victor/afos_published_top15_eval`. Choose a directory that does not exist yet to avoid overwriting an existing experiment.

```bash
mkdir /home/victor/afos_published_top15_eval
cd /home/victor/afos_published_top15_eval
```

Copy the ZIP into this directory. Alternatively, download directly on Linux with the command below. For a private repository requiring authentication, use your authenticated browser instead.

```bash
curl -fL -o published_top15_matched_bundle.zip "https://github.com/OjasWanjari/AFOS_experiment/raw/refs/heads/codex/pooled-retraining/retrain_runs/published_top15_matched_bundle.zip"
```

### A-2. Activate the original environment and extract the ZIP

```bash
source /home/victor/myafos/bin/activate
python -m zipfile -e published_top15_matched_bundle.zip .
cd AFOS_published_top15
```

If your environment is elsewhere, substitute the path to the environment used for the previous retraining run. Python, TensorFlow, CUDA, and the CIFAR-10 data itself are not included in the ZIP. Training uses the existing Keras dataset cache or downloads CIFAR-10 if it is absent.

The extracted layout is:

```text
/home/victor/afos_published_top15_eval/
  published_top15_matched_bundle.zip
  AFOS_published_top15/                  <- Run subsequent commands here
    retrain_published_top15.py
    retrain_paper.py
    run_paper.py
    paper_afos/
    RETRAIN_PUBLISHED_TOP15_JA.md
    RETRAIN_PUBLISHED_TOP15_EN.md
    retrain_runs/
      published_top15_matched/           <- Prepared plan and output directory
        plan.json
        plan.sha256
        selected_top15.csv
        split_indices.npz
        reference_audit.json
        source_snapshot/
        results/
        status.json
```

**Use `--resume` from the very first invocation after extraction.** Here it means "use the prepared plan," even when no training has started. Do not recreate the plan with `--reference-run`, and skip sections B-1 and B-2.

### A-3. Verify the bundled plan and environment without training

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --plan-only
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --check-environment
```

The first command should print `75 trials, seeds 0..4, 50 epochs` and verifies the plan, source and split hashes. The second should print `Recorded environment matches`. Neither command trains a model. If the environment differs, resolve the machine/GPU/virtual-environment discrepancy before starting the evaluation.

### A-4. Run the first trial, then continue with the rest

Start with the following command. This is a **full, official 50-epoch trial for EELU-4 with seed 0**, not a shortened smoke test.

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --max-trials 1
```

After it finishes, inspect the status. A successful first trial leaves one `completed` trial and 74 `pending` trials.

```bash
cat retrain_runs/published_top15_matched/status.json
```

Continue with the remaining trials:

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume
```

Completed trials are skipped. To start all 75 at once, omit the first command with `--max-trials 1` and run only this command. These commands run in the foreground; keep the terminal open during training.

### A-5. Monitor, resume, and locate the results

In a separate terminal, enter the same `AFOS_published_top15` directory to inspect progress:

```bash
cat retrain_runs/published_top15_matched/status.json
tail -f retrain_runs/published_top15_matched/trials/paper_01_seed_0/attempt_0001/worker.log
```

`status.json` is updated primarily when a trial ends. Follow `worker.log` for current epoch progress. The example above follows the first trial; for subsequent trials, use the log path printed by the coordinator.

To resume, **reuse the directory containing your existing results; do not extract the ZIP again**. In a new terminal, restore the environment and working directory:

```bash
source /home/victor/myafos/bin/activate
cd /home/victor/afos_published_top15_eval/AFOS_published_top15
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume
```

If an interruption, OOM or environment/runtime failure was recorded, resolve its cause and append `--retry-errors` to the final command. NaN/Inf outcomes are retained as numerical failures and are not automatically retried. See B-5 for handling a stale lock.

Results are written to `retrain_runs/published_top15_matched/` **inside this extracted directory**, not to the original `AFOS_experiment/retrain_runs/` directory.

- `results/retraining_results.csv`: results for all 75 trials.
- `results/retraining_summary.csv`: five-seed means and sample standard deviations.
- `trials/`: logs, histories, environment records, result JSON files and successfully saved models.

For analysis on another PC, copy the entire `published_top15_matched` output directory. Re-extracting the ZIP over a run that has started could overwrite its plan/summary files; resume using the existing output instead.

Common mistakes: for `Output already exists`, check that you included `--resume`. If `retrain_published_top15.py` cannot be found, check that you entered the extracted `AFOS_published_top15` directory.

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

## B-1. Alternative without the ZIP: required files for a new plan

**Skip B-1 and B-2 when using the ZIP's prepared plan.** These steps are only for creating a new plan from another reference run.

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

## B-2. Prepare a plan without training

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

## B-3. Verify the original environment without training

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --check-environment
```

Success prints `Recorded environment matches`. Details are saved to `environment_preflight.json`. This loads TensorFlow but no dataset and performs no training. Resolve discrepancies by using the original machine/environment/GPU; do not bypass the guard and label a different environment as a matched comparison.

## B-4. Run one trial, then the rest

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --max-trials 1
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume
```

The first trial is `paper_01_seed_0` (EELU-4). The second command skips completed trials and continues. To start all 75 at once, use only the second command. Every worker repeats the environment check before loading data.

Do not combine `--resume` with `--reference-run`. Resume uses frozen inputs and code.

## B-5. Failures and resumption

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
