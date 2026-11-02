# Diagnosing NaN/Inf in failed retraining trials

[日本語](DIAGNOSE_RETRAINING_JA.md)

These runs investigate failures and do not update the official results or averages. Use the original Linux PC, RTX 4080 SUPER, and `/home/victor/myafos` virtual environment. The runner checks the recorded software versions, GPU model, and runtime settings before training. It stops if those checks fail; this check does not establish NVIDIA driver identity or guarantee bitwise reproducibility.

## Download and extract the execution ZIP

Download [nonfinite_diagnostics_bundle.zip](retrain_runs/nonfinite_diagnostics_bundle.zip) from this repository. On the GitHub file page, use **Download raw file**, then copy the ZIP to the original Linux PC. The ZIP contains the files needed to run the prepared diagnostic plans; you do not need to copy the original result folders or clone the repository for this workflow.

Extract it into a new directory, without overwriting existing results:

```bash
source /home/victor/myafos/bin/activate
mkdir /home/victor/afos_diagnostics
cd /home/victor/afos_diagnostics
# Copy nonfinite_diagnostics_bundle.zip into this directory.
python -m zipfile -e nonfinite_diagnostics_bundle.zip .
cd AFOS_diagnostics
```

Run all commands below from `AFOS_diagnostics`. The extracted package contains:

- `published/`: a prepared plan for 25 failed trials of the published functions.
- `previous/`: a prepared plan for six failed trials: discovered candidate 13, seed 1, and ELU, seeds 0–4.
- Both execution guides. Each plan directory contains its runner, environment helper, saved data split, original training source snapshot, and integrity hashes.

The ZIP does not include Python, TensorFlow, CUDA, or CIFAR-10. Keep the original environment rather than installing current package versions. Recorded versions include Python 3.10.12, TensorFlow 2.21.0, Keras 3.12.4, NumPy 2.2.6, and scikit-learn 1.7.2. CIFAR-10 is read from the existing Keras cache, or downloaded if absent.

## Start with paper_13, seed 0

`paper_13 = max(x, erf(x))` contains no exponential operation and failed during epoch 1 for all five seeds. It is the first suggested diagnostic case.

First, rerun the original training procedure without inserting operation-level checks:

```bash
python published/diagnose_retraining.py --output published --resume --trial paper_13_seed_0 --mode replay
```

Then rerun with operation-level NaN/Inf checks:

```bash
python published/diagnose_retraining.py --output published --resume --trial paper_13_seed_0 --mode check
```

Both modes retain the original 50-epoch setting. If the failure does not recur, training continues through all 50 epochs and test evaluation. Test data are used only for evaluation, not for training.

`check` uses `tf.debugging.enable_check_numerics()` to raise an exception when a checked operation outputs a nonfinite value. It retains graph execution, the model, SGD, the saved data split, seed, shuffle settings, and Dropout. Instrumentation can change graph optimization and execution behavior, so its failure location or timing may differ from the original run. Checks can also increase runtime and memory use.

## Run other failed cases

```bash
python published/diagnose_retraining.py --output published --resume --trial paper_08_seed_0 --trial paper_10_seed_0 --trial paper_11_seed_0 --trial paper_12_seed_0 --mode check
python previous/diagnose_retraining.py --output previous --resume --trial elu_seed_0 --mode replay
python previous/diagnose_retraining.py --output previous --resume --trial elu_seed_0 --mode check
python previous/diagnose_retraining.py --output previous --resume --trial discovered_13_seed_1 --mode check
```

For each case, use both `replay` and `check` when comparing reproduction with the instrumented result. The published functions and ELU also accept seeds 1–4. Only seed 1 failed for discovered candidate 13, so its other seeds are not eligible in this diagnostic plan. Candidate 13 originally failed in epoch 50 and must be retrained from epoch 1.

Each trial runs in a separate process. Only explicitly supplied `--trial` IDs run; supplying several runs them sequentially. A runtime error stops the remaining sequence. List eligible trials without starting training:

```bash
python published/diagnose_retraining.py --output published --resume --plan-only
python previous/diagnose_retraining.py --output previous --resume --plan-only
```

## Inspect the outputs

For example, the first checked run writes to `published/diagnostics/paper_13_seed_0/check/attempt_0001/`.

| File | Contents |
|---|---|
| `result.json` | Outcome, exception, and last recorded train/validation/test phase, epoch (1-based), and batch (0-based). The train phase alone does not distinguish forward from backward computation. |
| `error.txt` | Exception traceback, when an exception occurs. Inspect the operation name, type, shape, and creation stack. |
| `worker.log` | TensorFlow messages and training output. The coordinator prints this path when starting a trial. |
| `batches.jsonl` | Batch start/end events and Keras metrics. A batch-end loss may be a cumulative average rather than that batch's individual loss. |
| `epochs.jsonl` | Epoch metrics, if an epoch completes far enough to invoke the callback. |
| `variables.json` / `failure_or_final_variables.npz` | Captured model weights and optimizer state at stopping or completion, with finite-value counts and maximum absolute finite values. These are not pre-failure snapshots, raw gradients, or complete resumable training checkpoints. |
| `environment.json` / `environment_check.json` | Environment details and comparison with the reference, when environment inspection completes. |

Early failures may produce only a subset of these files. An exit code of 0 means a diagnostic outcome was collected, not necessarily that training succeeded.

| `result.json` status | Meaning |
|---|---|
| `nonfinite` / `nonfinite_test` | The original loss/accuracy checks detected a nonfinite metric. |
| `numerics_check_triggered` | An operation-level check stopped execution. Inspect `error.txt`. |
| `not_reproduced` | All planned epochs and test evaluation completed without reproducing a nonfinite metric. |
| `runtime_error` | Environment mismatch, OOM, or another exception. Distinguish this from the numerical failure under investigation. |

Use the operation and stack in `error.txt` to locate the forward layer, gradient computation (for example, `gradient_tape`), or SGD update. If the stage remains ambiguous, further measurement of the relevant inputs, outputs, and gradients is needed.

**The first detected nonfinite intermediate is not automatically the cause of a NaN final loss.** For example, an exponential may overflow to Inf while a subsequent `erf` saturates to a finite output. Compare the ordinary replay with the checked run, and report cases that stop only with instrumentation enabled.

## Retry, interrupt, and return the results

Repeating a command creates `attempt_0002` and subsequent directories. Each attempt restarts training from the beginning with the same seed. Existing official results and previous diagnostic attempts are preserved. `--resume` reuses the diagnostic plan; it does not restore partially trained weights.

Do not run multiple coordinators against the same output directory simultaneously. Ctrl+C also terminates the worker. If a forced termination leaves `diagnostic.lock`, confirm that both coordinator and worker have stopped before deleting that lock file. Partial logs remain available.

Return the complete `published/diagnostics/` and `previous/diagnostics/` folders for analysis. Changes such as gradient clipping or a lower learning rate should be evaluated as separate experiments after investigating the failure.

## Prepare your own plan (unnecessary when using the ZIP)

This alternative requires the repository and the original result folders, including their plans, source snapshots, split files, and per-trial results/environment logs. From the repository root:

```bash
python diagnose_retraining.py --reference-run paper_full_results/published_top15_matched --output diagnostic_runs/published --plan-only
python diagnose_retraining.py --reference-run paper_full_results/retrain_runs/pooled_top15_updated --output diagnostic_runs/previous --plan-only
```

Each output directory must be new. After preparation, run the copied `diagnose_retraining.py` inside that output directory with `--resume`, as in the ZIP workflow. Keep the prepared plans, source snapshots, and hash files unchanged so integrity verification continues to pass.
