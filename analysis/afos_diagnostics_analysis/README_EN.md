# Analysis of AFOS_diagnostics Results

[日本語](README_JA.md)

Analysis date: November 2, 2026. This report examines the diagnostic records dated October 29, 2026, returned in `analysis/AFOS_diagnostics/`. It analyzes saved logs and arrays; no training was rerun during this analysis. The Japanese report's “Next investigations” section is omitted from this English version.

**Nonfinite values were detected in convolution and sinh operations during the forward pass, and in the derivative of atan during the backward pass. Several cases showed rapidly increasing loss and NaN model weights. The original failure of discovered candidate 13, seed 1, was not reproduced in its diagnostic run.**

## 1. Coverage and outcomes

The diagnostic plan contains 31 originally failed cases: 25 for published functions and six for discovered candidate 13 and ELU. The returned records cover **seven distinct cases and nine attempts**, not all 31 cases.

- `replay`: reruns the original training procedure without adding operation-level numerical checks.
- `check`: adds NaN/Inf checks to operations. This changes the computation graph, so the numerical trajectory is not guaranteed to match replay.
- Outcomes: three `nonfinite`, five `numerics_check_triggered`, and one `not_reproduced` attempt.
- All nine saved environment checks reported a match. The plan, code, and data-split hashes were revalidated. Every variable in the saved NPZ files was checked against `variables.json` for its finite-element count and maximum finite absolute value.

Batch numbers below are **one-based** for readability. The logs use zero-based `batch` values, so the table reports `batch + 1`. All eight attempts with numerical abnormalities stopped in epoch 1.

| Case | Mode | Detection point | Observed result |
|---|---|---|---|
| Published 08, seed 0 | check | Batch 13 | Backward pass at activation layer 6: `Square` inside the atan derivative produced +Inf |
| Published 10, seed 0 | check | Batch 5 | Forward pass: convolution layer 6 produced ±Inf |
| Published 11, seed 0 | check | Batch 7 | Forward pass: `Sinh` at activation layer 5 produced ±Inf |
| Published 12, seed 0 | check | Batch 5 | Forward pass: convolution layer 4 produced ±Inf |
| Published 13, seed 0 | replay × 2 | Batch 9 in both attempts | NaN training loss reproduced; no operation-level check result was returned |
| ELU, seed 0 | replay | Batch 14 | NaN training loss reproduced |
| ELU, seed 0 | check | Batch 15 | Forward pass: convolution layer 3 produced ±Inf |
| Discovered candidate 13, seed 1 | check | Completed 50 epochs | Failure not reproduced; test evaluation completed |

**The reported detection operation must be distinguished from the operation that first generated a nonfinite value anywhere in the execution.** Variables saved after an exception are not pre-failure checkpoints and may include updates executed before the exception was returned. These snapshots alone cannot establish the order in which forward-pass, backward-pass, and parameter-update abnormalities arose.

## 2. Interpretation by function

### Published 08: overflow in the square used by the atan derivative

Function:

`(tanh(x) + max(x,0)) * sigmoid(softplus(x) * erf(atan(x)))`

The [error stack](../AFOS_diagnostics/published/diagnostics/paper_08_seed_0/check/attempt_0001/error.txt) includes `_AtanGrad` and reports `gradient_tape/.../paper_activation_5_1/Square_1`. The input to the square is the `BiasAdd` output of convolution layer 6.

Mathematically, `d atan(x)/dx = 1/(1+x²)`. The largest finite float32 value is approximately `3.4×10^38`. Even a finite input can therefore overflow when squared if its absolute value exceeds approximately `1.84×10^19`. The diagnostic check detected +Inf in this square operation.

Recorded training loss rose sharply before termination: `37.59 → 264.96 → 461,457.97`. All saved model weights remained finite, but their maximum absolute value reached **approximately 1.42×10^20**. This is evidence of extreme growth in training values, far beyond ordinary small rounding differences.

However, even when `x² = Inf`, `1/(1+Inf)` can be zero. **An infinite intermediate in the derivative does not establish that the final gradient was NaN, or that this square alone caused the original failure.** The available records do not resolve that distinction.

### Published 10 and 12, and ELU: infinite convolution outputs and rapidly growing training values

Functions:

- Published 10: `max(sin(x)*sigmoid(sin(x)), x*erf(x)+tanh(x))`
- Published 12: `x*erf(x)+sin(x)`
- ELU: `x` on the positive branch and `exp(x)-1` on the negative branch, with alpha = 1.

For these cases, the check reported nonfinite values in **convolution outputs**, rather than inside the activation function. The last recorded training losses were:

| Case and mode | Last three or four recorded loss values |
|---|---|
| Published 10, check | 16.72 → 62.77 → 499.30 → 1.91×10^9 |
| Published 12, check | 15.79 → 91.28 → 2,828.36 → 6.52×10^14 |
| ELU, check | 94.10 → 541,305.56 → 5.66×10^20 |
| ELU, replay | 6,894.16 → 1.59×10^16 → NaN |

These are Keras aggregated losses recorded in `batches.jsonl`; **they are not necessarily losses for individual batches in isolation**.

Rapidly increasing loss, extremely large finite weights, and NaN weights and SGD momentum states strongly support an interpretation of numerical divergence during training. Large activations, weights, and gradient updates may have amplified each other until computations exceeded the float32 range. Raw gradients and layer inputs were not saved, so the update that initiated divergence remains unidentified.

The learning rate was 0.01 and momentum was 0.9, but no comparison in these records establishes either setting as the sole cause. **The ELU result does not show that the exponential on ELU's negative branch overflowed.** The reported detection point was a convolution.

### Published 11: infinite sinh output in the forward pass

Function:

`(tanh(x) + max(x,0)) * sigmoid(softplus(x) * erf(sinh(x)))`

The [error stack](../AFOS_diagnostics/published/diagnostics/paper_11_seed_0/check/attempt_0001/error.txt) reports ±Inf in `Sinh` at activation layer 5. The magnitude of sinh grows exponentially with the absolute input, making overflow possible for large inputs.

The complete function contains `erf(sinh(x))`, whose forward output can saturate to a finite value because `erf(±Inf)=±1`. In the backward pass, however, the chain-rule term `erf'(sinh(x))*cosh(x)` can potentially lead to an indeterminate numerical product corresponding to `0*Inf`. **This is a plausible mechanism, not a multiplication observed to fail in this log.** NaN weights in the saved snapshot also do not establish that causal sequence by themselves.

### Published 13: failure reproduced twice; the responsible operation remains unidentified

Function: `max(x,erf(x))`.

Both replay attempts recorded NaN training loss at batch 9. In attempt 1, the preceding losses were `3,663.67 → 9.02×10^8 → NaN`; in attempt 2, they were `3,892.61 → 8.66×10^8 → NaN`. Every saved model-weight and momentum-state element was NaN in both attempts.

**The returned folder contains two replay attempts and no check result for paper_13.** Divergence was reproduced, but the initial failing operation was not identified. The result shows that numerical failure can arise during network training even when the activation function contains neither exp nor sinh.

### Discovered candidate 13: the failure was not reproduced in check mode

Function: `atan(x)*min(x,0)+tanh(x)`. This is a different function from published 13 above.

The [result record](../AFOS_diagnostics/previous/diagnostics/discovered_13_seed_1/check/attempt_0001/result.json) shows completion of 50 epochs and 35,200 training batches. All saved model weights and momentum states were finite. **This was one diagnostic run for seed 1, not a diagnostic rerun of all five seeds.**

| Metric | Final diagnostic value |
|---|---:|
| Training accuracy / loss | 86.3756% / 0.383498 |
| Validation accuracy / loss | 80.9000% / 0.605185 |
| Test accuracy / loss | 81.1500% / 0.618287 |

This completion does not mean that the original epoch-50 failure has been fixed. The graph changes introduced by check mode may have changed the training trajectory. Even the two published-13 replay attempts, using the same seed and mode, did not produce identical loss values. Bitwise reproducibility was therefore not observed across these executions. GPU nondeterminism is a possible contributor, but its role cannot be established from these records alone.

**The diagnostic test accuracy of 81.15% must remain separate from the original Top 15 aggregate.** It should not replace the original failed attempt or be combined with the original successful seeds as a uniform five-seed result.

## 3. Verification of saved weights

The model has 306,234 weight elements, including biases, and 306,234 SGD momentum-state elements. The NaN counts below were the same for each of those two groups, verified by rereading the NPZ arrays. These saved arrays contained no ±Inf: all nonfinite elements were NaN. This is a separate observation from the infinite intermediate tensors reported by the error messages.

| Case and mode | NaN model weights / 306,234 | Maximum absolute finite model weight |
|---|---:|---:|
| Published 08, check | 0 | 1.42×10^20 |
| Published 10, check | 306,234 | No finite elements |
| Published 11, check | 233,859 | 0.8034, finite subset only |
| Published 12, check | 306,234 | No finite elements |
| Published 13, replay, both attempts | 306,234 | No finite elements |
| ELU, replay | 171,402 | 4.33×10^14, finite subset only |
| ELU, check | 262,410 | 1.20×10^19, finite subset only |
| Discovered candidate 13, check | 0 | 1.9249 |

These are states saved after termination or successful completion, not weights immediately before the abnormality or raw gradients.

## 4. Interpretation limits and log details

- **No termination due to OOM was recorded in these nine attempts.** Their statuses and logs show numerical-check stops, nonfinite training metrics, or successful completion. They should be distinguished from earlier GPU memory failures.
- `layout failed ... LayoutOptimizer` appears in all nine attempts, including the candidate-13 run that completed 50 epochs. This message alone does not establish the cause of NaN; training continued after it.
- The final context in the failed replay `result.json` files is validation, but `batches.jsonl` records NaN training loss earlier. **The abnormality did not first appear during validation.** Validation ran after training was stopped, and its final context was saved.
- The `704/704` display in `worker.log` does not prove that all 704 training batches were processed. Callback records show nine completed training batches for each failed paper_13 replay and 14 for the failed ELU replay.
- Accuracy can remain finite when loss is NaN. The failed replays recorded validation accuracy of 9.6%. Finite accuracy is not evidence that the model is numerically healthy.
- Environment matching applies only to the checked fields: Python and package versions, GPU model, TensorFlow build information, selected XLA settings, and related recorded configuration. It does not guarantee identical GPU drivers or deterministic computation.

## 5. Analysis files

- [audit.json](audit.json): extracted results for each attempt, verified variable statistics, cases without returned results, and SHA-256 hashes of the input files.
- [analyze.py](analyze.py): aggregation script using Python and NumPy. It does not import TensorFlow or start training.
- [Diagnostic source records](../AFOS_diagnostics/): saved plans, code snapshots, logs, and variable arrays used by the analysis.

To regenerate the audit from the repository root with Python and NumPy installed:

```bash
python analysis/afos_diagnostics_analysis/analyze.py
```
