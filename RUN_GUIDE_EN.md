# AFOS: Required Files and Execution Guide

[Japanese version](RUN_GUIDE_JA.md)

This guide covers **activation-function search and retraining on CIFAR-10 using the paper's base CNN**, through `run_paper.py`.
This is a separate entry point from the legacy `main.py` experiment. It does not include the paper's generalization experiments on other architectures or datasets.
See [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) (in Japanese) for reproduction assumptions, including parameters not specified in the paper.

## 1. Required files

Keep the following files in the same relative locations as in this repository. Cloning the entire repository is the easiest option.

| File | Purpose |
| --- | --- |
| `run_paper.py` | Entry point for search, retraining, saving results, and final plotting |
| `paper_parameters.json` | Population size, generations, seeds, training settings, and other parameters |
| `paper_afos/__init__.py` | Python package definition |
| `paper_afos/search.py` | Chromosomes, mathematical expressions, genetic algorithm, and fitness |
| `paper_afos/model.py` | CNN, activation functions, training, and data loading |
| `paper_afos/recording.py` | CSV recording of epoch histories and execution events |
| `requirements-paper-windows.txt` | Windows dependencies validated on the development PC |

Optional supporting files:

- `tests/test_paper.py`: Automated tests.
- `benchmark_paper_generation.py`: GPU timing of 30 individuals for one generation, followed by runtime estimates.
- `benchmark_paper_candidate.py`: GPU timing of a single specified individual. Uses its own simplified output format.
- `PAPER_REPRODUCTION.md`: Details of the paper-to-code correspondence, implementation assumptions, and saved outputs.

The legacy `main.py`, `fitness.py`, `activation.py`, `ga_utils.py`, and `utils.py` are not required by this entry point.
Previous `runs/` directories, images, slides, and the paper PDF are also not required to execute it.
Create a new virtual environment on each PC instead of copying one from another machine.

## 2. Get the source code

Open Windows PowerShell in the parent directory where you want to store the repository. Git and 64-bit Python 3.10 are required.

```powershell
git clone https://github.com/OjasWanjari/AFOS_experiment.git
cd AFOS_experiment
```

Run all subsequent commands from the repository root.
The existing checkout on the development PC is `D:\vscode\AFOS-main`. If you already have the source, you do not need to clone it again.

## 3. Create the Python environment (once on a new PC)

```powershell
py -3.10 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-paper-windows.txt
.\.venv-gpu\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The validated configuration uses Python 3.10, TensorFlow CPU 2.10.0, and the DirectML plugin.
If the GPU list printed by the last command is `[]`, no GPU has been detected. Check your GPU driver and dependencies before starting the full experiment.
Compatibility with a new PC is not guaranteed. Migrating to Python 3.11 or later, Keras 3, or a Linux CUDA environment is outside the scope of this guide.
To run a check intentionally on the CPU, use `--device cpu` in place of `--device gpu` in the relevant commands below.

If `.venv-gpu` already works on your current PC, you do not need to recreate it or reinstall its packages.
Activating the virtual environment is not necessary: the commands explicitly invoke its Python executable.

## 4. Check the configuration and run a short test

Inspect the configuration without training or downloading data:

```powershell
.\.venv-gpu\Scripts\python.exe run_paper.py --dry-run
```

Use synthetic data to check search, retraining, saving, and final plotting:

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --smoke --device gpu --output runs/paper_smoke_check
```

Results produced with `--smoke` are not scientific experiment results.
A new run is rejected if the output directory already exists; choose a different name in that case.

Run the automated tests (using the CPU):

```powershell
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -p test_paper.py -v
```

## 5. Run the full experiment

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --device gpu --output runs/paper_full
```

The default configuration uses 30 individuals × 40 generations × seeds 0–4, totaling **6,000 search evaluations**.
The top 15 discovered functions and six existing activation functions are then each trained from scratch five times for up to 50 epochs: 105 retraining trials in total.
Each search evaluation runs for up to 15 epochs. Early stopping means that runtime varies between individuals.

Keras downloads CIFAR-10 on the first full run, which requires an internet connection.
The default cache location is `.keras/datasets/` under the user's home directory, or under `KERAS_HOME` if that environment variable is set.
The split contains 45,000 training images, 5,000 validation images, and 10,000 test images.
Keep the PC running, prevent sleep, and leave the training terminal open while the experiment runs.
Make sure the output drive has free space for the saved models as well as the results.

## 6. View live logs and CSV results

In a separate PowerShell terminal, from the same repository root:

```powershell
Get-Content runs/paper_full/run.log -Wait -Tail 30
```

Pressing `Ctrl+C` in this log-viewing terminal only stops viewing the log. Keep it separate from the training terminal.

| Path within the specified output directory | Update timing and contents |
| --- | --- |
| `histories/*.csv` | After each epoch: training/validation loss and accuracy |
| `results/evaluation_results_seed_*.csv` | After each individual: chromosome, expression, score, loss, accuracy, and status |
| `results/best_by_seed.csv`, `results/top3_seed_*.csv` | Best and top-ranked candidates, updated after each individual |
| `results/generation_history.csv` | Best metrics after each generation |
| `results/selected_top15.csv` | Retraining candidates selected after all searches finish |
| `results/retraining_results.csv`, `results/comparison_summary.csv` | Metrics, including test results, and summaries after each retraining trial |
| `results/events.csv` | Timestamped start, completion, interruption, error, and plotting events |
| `plots/` | Figures generated together once, after all requested runs finish |
| `models/` | Best search model for each seed and models from successful retraining trials |

CSV records are at epoch, individual, and generation granularity. Per-minibatch metrics and predictions for every image are not recorded.
Test metrics are not computed during search; they are evaluated only after retraining.
The run also saves JSON checkpoints, configuration and source snapshots, and an NPZ file containing the data-split indices.

## 7. Interrupt and resume

If you press `Ctrl+C` in the training terminal, records of completed individuals and the partial epoch CSV history are retained.
Resume with the same configuration and source code:

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --device gpu --resume runs/paper_full
```

The interrupted trial starts again from the beginning. Its previous partial history is moved to `histories/attempts/`.
You cannot specify `--output` and `--resume` together.
If you started with a custom configuration, supply the same `--config`; if it was a synthetic-data test, also supply `--smoke`.
Changes to the configuration or source cause resume to be rejected, preventing results from different conditions from being mixed.
Runs started with an older version require the code and configuration in that run's `source_snapshot/` directory.

For the default full run, plotting reads the saved CSVs only after search, retraining, and baseline comparisons for all seeds have finished.
No plotting is triggered by an interruption or exception. Resuming the same completed results does not regenerate existing figures.
If plotting failed or figures were deleted, plotting is retried on resume.

## 8. Run search and retraining separately (optional)

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --stage search --device gpu --output runs/paper_separate
.\.venv-gpu\Scripts\python.exe -u run_paper.py --stage retrain --device gpu --resume runs/paper_separate
```

With this split workflow, figures are generated once after all search runs finish, and again after the subsequent retraining runs finish.
If you want plotting only at the end of the entire workflow, use the full-run command in Section 5.

## 9. Estimate runtime on your PC (optional)

```powershell
.\.venv-gpu\Scripts\python.exe -u benchmark_paper_generation.py --output runs/paper_generation_timing
```

This runs only seed 0, with 30 individuals for one generation, and saves `time_estimate.json` and `TIME_ESTIMATE.md` in addition to the usual CSVs and log.
It does not run retraining. The estimate extrapolates from the initial generation; it is not a guaranteed completion time and does not account for later changes in early-stopping frequency or post-search retraining time.
After an interruption, add `--resume` to the same command.
Keep the `.config.json` file created alongside the output directory as well.

## 10. What is and is not stored on GitHub

The source code, configuration, dependency list, automated tests, and this guide are stored in the repository.
Virtual environments, downloaded datasets, and new `runs/` outputs are covered by `.gitignore` and are not automatically saved to GitHub.
Back up the CSVs, figures, and models you need after the experiment. These new outputs are separate from historical results already tracked by Git.
