# Retraining results for the updated pooled Top 15

Analysis date: 2026-10-22. The newly supplied saved results were audited and reaggregated. No additional training was performed, and the source data were not modified.

English translation of [the Japanese report](README_JA.md).

**All 105 trials have terminal outcomes. Of these, 99 completed 50 epochs of training and test evaluation, while six stopped because of NaN values. Fourteen of the 15 discovered candidates have results for all five seeds. The highest mean test accuracy was 83.582 ± 0.460%.**

## Inputs and verification

- Run: `paper_full_results/retrain_runs/pooled_top15_updated`. The Top 15 selected across all five searches, including the new seed-4 search, and six baseline functions were each evaluated with retraining seeds 0–4.
- `selected_top15.csv` exactly matches the Top 15 from the previous updated analysis. The plan hash, saved source code, source search CSVs and manifests, and training/validation split hashes were checked.
- All 105 aggregate rows were checked against each trial's `result.json` and training history. Means and sample standard deviations were recalculated. All recorded metrics were finite throughout the 50 epochs of the 99 completed trials, and 99 saved models were present.
- There were 105 distinct process IDs. Runtime settings were GPU execution, JIT disabled, and memory growth enabled, using TensorFlow 2.21.0 / Keras 3.12.4.
- Worker logs contained OOM matches in zero trials and tracebacks in zero trials. All 105 trials had layout optimizer warnings; these warnings should be distinguished from the trials' final statuses.
- `paper_full_results/runs` contains search records in a different format: 50 evaluations and 10 generations for each of seeds 0–2. These are not results from retraining a common set of 15 functions for 50 epochs with five seeds, so they are not combined with this analysis.

Audit record: [audit.json](audit.json).

## Completion status

| Group | Planned | Completed | Stopped with nonfinite values |
|---|---:|---:|---:|
| 15 discovered candidates | 75 | 74 | 1 |
| Six baseline functions | 30 | 25 | 5 |
| Total | 105 | 99 | 6 |

`all_evaluations_completed=false` means that not every trial succeeded. No trials are pending or running. The number of discovered candidates with all five successful repetitions increased from 0/15 to 14/15.

## Results for discovered functions

Mean ± SD refers to test accuracy across retraining seeds 0–4 for the same function. SD is the sample standard deviation. Candidate IDs reflect validation-accuracy rank during search; the table below is ordered by mean test accuracy after retraining.

| Mean rank | Search candidate | Function | Completed | Mean ± SD (%) |
|---:|---|---|---:|---:|
| 1 | discovered_01 | `(erf(x)*sigmoid(sinh(x)))` | 5/5 | 83.582 ± 0.460 |
| 2 | discovered_03 | `(atan(x)*sigmoid(sinh(x)))` | 5/5 | 83.260 ± 0.422 |
| 3 | discovered_02 | `(sigmoid(x)*erf(erf(x)))` | 5/5 | 83.162 ± 0.418 |
| 4 | discovered_10 | `(erf(x)*sigmoid(x))` | 5/5 | 82.982 ± 0.316 |
| 5 | discovered_04 | `(asinh(x)*sigmoid(sinh(x)))` | 5/5 | 82.934 ± 0.607 |
| 5 | discovered_07 | `(sigmoid(x)*erf(x))` | 5/5 | 82.934 ± 0.239 |
| 7 | discovered_05 | `(sigmoid(x)*erf(sin(x)))` | 5/5 | 82.892 ± 0.490 |
| 8 | discovered_06 | `(asinh(x)*sigmoid(x))` | 5/5 | 82.642 ± 0.100 |
| 9 | discovered_08 | `(sigmoid(x)*erf(sinh(x)))` | 5/5 | 82.438 ± 0.361 |
| 10 | discovered_12 | `(max(min(sin(x),0),x)*erf(sigmoid(x)))` | 5/5 | 82.384 ± 0.460 |
| 11 | discovered_14 | `min((log(1+exp(x))*erf(x)),(atan(x)*erf(atan(x))))` | 5/5 | 82.374 ± 0.382 |
| 12 | discovered_09 | `max(min((exp(x)*erf(tanh(x))),0),(x*sigmoid(max(x,0))))` | 5/5 | 82.140 ± 0.516 |
| 13 | discovered_15 | `(sin(x)*sigmoid(asinh(x)))` | 5/5 | 81.920 ± 0.385 |
| 14 | discovered_11 | `max(min((log(1+exp(x))*erf(x)),0),min(tanh(x),sin(x)))` | 5/5 | 81.050 ± 0.493 |
| — | discovered_13 | `((atan(x)*min(x,0))+tanh(x))` | 4/5 | Incomplete |

The best candidate was `(erf(x)*sigmoid(sinh(x)))`. Its test accuracies for seeds 0–4 were 83.82%, 83.75%, 82.77%, 83.69%, and 83.88%.

All 14 candidates with complete results exceeded ReLU's mean, and eight exceeded GELU's mean. The best candidate exceeded ReLU by 4.758 percentage points and GELU, the strongest successfully evaluated baseline, by 1.092 percentage points. These are observed differences, not conclusions from statistical significance tests.

Candidates 7 and 10 are mathematically identical but have different expression strings, so they were evaluated as separate candidates. The set should not be interpreted as 15 mathematically distinct functions.

## Baseline functions and comparison with the paper

| Function | This experiment: mean ± SD (%) | Paper: mean ± SD (%) | Mean difference (percentage points) |
|---|---:|---:|---:|
| relu | 78.824 ± 0.601 | 81.37 ± 0.46 | -2.546 |
| swish | 81.794 ± 0.545 | 81.46 ± 0.60 | +0.334 |
| mish | 81.124 ± 0.601 | 82.23 ± 0.27 | -1.106 |
| gelu | 82.490 ± 0.448 | 81.75 ± 0.26 | +0.740 |
| serf | 80.462 ± 0.592 | 82.16 ± 0.22 | -1.698 |
| elu | Incomplete (0/5) | 82.29 ± 0.57 | — |

Paper reference: *Activation Function Optimization Scheme for Image Classification*, arXiv:2409.04915v1, Table 6, Base CNN. Both studies use five repetitions for functions with complete results, but identical data splits and execution environments are not guaranteed.

The highest mean in this experiment, 83.582%, is numerically 0.842 percentage points above the paper's EELU-4 result of 82.74 ± 0.14%. However, this comparison involves different functions and execution environments. No retraining results for EELU-1–4 in the current environment are available in this analysis. The discrepancies between this experiment and the paper for ReLU, SERF, and Mish also remain unresolved, so superiority over the paper has not been established.

## The six failed trials

| Trial | First epoch with nonfinite values | Training loss | Validation loss |
|---|---:|---|---|
| discovered_13_seed_1 | 50 | nan | nan |
| elu_seed_0 | 1 | nan | nan |
| elu_seed_1 | 1 | nan | nan |
| elu_seed_2 | 1 | nan | nan |
| elu_seed_3 | 1 | nan | nan |
| elu_seed_4 | 1 | nan | nan |

Candidate 13 with seed 1 had finite metrics through epoch 49 and produced NaN at epoch 50. A history containing 50 rows must not, by itself, be treated as evidence of success. ELU produced NaN in the first epoch for every seed. No test evaluations were saved for any of these six failed trials.

The successful repetitions of candidate 13 (seeds 0, 2, 3, and 4) achieved 81.31%, 79.95%, 81.84%, and 81.57%. A four-run average that excludes the failure must not be reported as the official five-run mean.

The lowest test accuracy among the 99 completed trials was 78.02%; none of these completed trials showed the previously observed collapse to approximately 10% accuracy with finite metrics. Intermediate activations and gradients at the origin of the NaN values were not saved, so the root causes of the ELU and candidate-13 failures remain undetermined.

## Next checks

1. Investigate candidate 13 seed 1 and ELU by recording the first operations or gradients that become nonfinite. Treat any changes to the experimental conditions as a separate experiment.
2. Evaluate EELU-1–4 as fixed comparison functions in the same environment with the same five seeds, enabling direct comparison with the current candidates.
3. Investigate why baselines such as ReLU perform below the paper's results and how rankings change between the short search evaluations and 50-epoch retraining.

## Source files

- [Per-trial results](../../paper_full_results/retrain_runs/pooled_top15_updated/results/retraining_results.csv)
- [Per-function summary](../../paper_full_results/retrain_runs/pooled_top15_updated/results/retraining_summary.csv)
- [Run status](../../paper_full_results/retrain_runs/pooled_top15_updated/status.json)
- [Frozen candidates](../../paper_full_results/retrain_runs/pooled_top15_updated/selected_top15.csv)
- [Audit script](analyze.py)
