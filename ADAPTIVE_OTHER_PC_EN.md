# Running adaptive AFOS on another PC

[Japanese version](ADAPTIVE_OTHER_PC_JA.md) · [Adaptive probabilities and code changes](ADAPTIVE_SEARCH_EN.md)

This guide targets the **Linux / NVIDIA GPU environment** recorded in the previous `paper_full_results` experiment. Run the commands in a Linux terminal (Bash), in order. They are not Windows PowerShell commands. Do not copy the development PC's Windows virtual environment to Linux.

## 1. Environment identified from previous records

| Item | Recorded value | Evidence |
| --- | --- | --- |
| OS | Linux x86_64, kernel `6.8.0-138-generic`, glibc 2.35 | Search manifests |
| Python | 3.10.12 | Search manifests |
| TensorFlow | 2.21.0 | Search manifests |
| Keras | 3.12.4 | Search manifests |
| NumPy | 2.2.6 | Search manifests |
| scikit-learn | 1.7.2 | Search manifests |
| matplotlib | 3.10.9 | Seed 0 manifest |
| GPU | NVIDIA GeForce RTX 4080 SUPER | Retraining preflight record |
| TensorFlow build | CUDA 12.5.1, cuDNN 9 | The same retraining record |

Search evidence comes from the five experiment manifests, including `paper_full_results/paper_full_seeds0/manifest.json`. GPU and CUDA/cuDNN information comes from `paper_full_results/published_top15_matched/environment_preflight.json`. **The GPU model was recorded during retraining; it does not directly establish the search GPU model.** CUDA/cuDNN values are TensorFlow build information, not the installed GPU driver version. The inspected records do not establish the Linux distribution name or the search-time driver version.

The steps below pin the recorded main Python packages. A different GPU may work, but identical performance, timing, or numerical results are not guaranteed.

## 2. Obtain the code

Use a Linux PC with Git, Python 3.10, and an NVIDIA driver. Check:

```bash
git --version
python3.10 --version
nvidia-smi
```

`nvidia-smi` should display the GPU and driver version. Resolve missing drivers or inaccessible GPUs before continuing. Containers also need GPU access enabled.

From the parent directory where you want to keep the code:

```bash
git clone --depth 1 --single-branch --branch codex/adaptive-fitness-probabilities https://github.com/OjasWanjari/AFOS_experiment.git AFOS_adaptive
cd AFOS_adaptive
git branch --show-current
git rev-parse HEAD
```

The branch should be `codex/adaptive-fitness-probabilities`. Choose another destination name if `AFOS_adaptive` already exists. Run all remaining commands from this repository directory.

The main files are `run_paper.py`, `paper_parameters_adaptive.json`, the `paper_afos/` package, and `tests/test_adaptive_search.py`. Previous models and result CSVs are not needed to start a new search.

## 3. Prepare Python

### A. Reuse the previous environment, if available

Activate that environment and check its versions. The search log mentions `/home/victor/myafos/`, but that path may not exist on the target PC. Use its actual environment path:

```bash
# Replace this placeholder with the actual environment path.
source /path/to/existing-environment/bin/activate
python --version
python -m pip check
```

Skip installation in section B when reusing an existing environment. Do not upgrade it without checking the intended versions.

### B. Create an environment on a new PC

With Python 3.10 and its venv support available:

```bash
python3.10 -m venv .venv-afos
source .venv-afos/bin/activate
python -m pip install --upgrade pip
python -m pip install "tensorflow[and-cuda]==2.21.0" "keras==3.12.4" "numpy==2.2.6" "scikit-learn==1.7.2" "matplotlib==3.10.9"
python -m pip check
```

TensorFlow documents `tensorflow[and-cuda]` for NVIDIA GPU installation on Linux / WSL2. A GPU driver is still required. These commands pin the recorded main packages; they do not reproduce every historical dependency or GPU runtime component. If installation reports a conflict, inspect it rather than removing version pins silently. See the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

**Do not use `requirements-paper-windows.txt` on Linux.** It describes the Windows TensorFlow 2.10 / DirectML environment.

### Check versions and GPU visibility

```bash
python -c "import sys, importlib.metadata as m; print(sys.version); print({p: m.version(p) for p in ['tensorflow','keras','numpy','scikit-learn','matplotlib']})"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU')); print('Build:', tf.sysconfig.get_build_info())"
```

Do not start GPU training if TensorFlow prints `GPUs: []`. A successful `nvidia-smi` check alone does not prove TensorFlow can use the GPU.

## 4. Keras 3 model filenames (already updated)

The `run_paper.py` on this branch already uses model filenames supported by Keras 3.
**No manual source edit is needed. Continue directly to section 5.**

| Model | Saved path |
| --- | --- |
| Best search model for each seed | `models/best_search_seed_{seed}.keras` |
| Retrained model | `models/{name}_seed_{seed}.keras` |

The two changes are the filenames passed to `model.save()` in `search()` and `retrain()` in `run_paper.py`.
Both now end in `.keras`. Fitness, training settings, and probability calculations are unchanged.
See the [Keras migration guide](https://keras.io/guides/migrating_to_keras_3/#saving-a-model-in-the-tf-savedmodel-format).

Keep the execution source unchanged after starting a run. Resuming an experiment started before this update requires the source originally used for that experiment.

## 5. Inspect or edit experiment settings

Open `paper_parameters_adaptive.json` in a text editor.

| Setting | Supplied value | Meaning |
| --- | --- | --- |
| `search.generations` | `40` | Generations per seed; change to 60 for a 60-generation search |
| `search.population_size` | `30` | Individuals per generation |
| `search.seeds` / `search.replications` | `[0,1,2,3,4]` / `5` | Search seeds and their count |
| `search.adaptive_probabilities.enabled` | `true` | Enable adaptive probabilities |
| `window_generations` | `3` | Best-fitness comparison interval |
| `improvement_target` | `0.01` | Return to base rates at 1% relative improvement |
| `max_crossover_probability` | `1.0` | 100% crossover at zero improvement |
| `max_mutation_probability` | `0.8` | 80% mutation at zero improvement |
| `training.epochs` | `15` | Maximum training epochs per search candidate, not generations |
| `post_search.epochs` | `50` | Maximum retraining epochs |

The probability-control keys starting with `window_generations` are inside `search.adaptive_probabilities`. Base probabilities remain `search.crossover_probability=0.8` and `search.mutation_probability=0.2`. Both rates decrease continuously between 0% and 1% improvement. For a generation-count change, edit only `search.generations`. When changing seeds, update `replications` to match their count. Retraining seeds are configured separately under `post_search`.

Check the configuration without training or downloading data:

```bash
python run_paper.py --config paper_parameters_adaptive.json --dry-run
```

Expect `search_evaluations: 6000` for 30 individuals × 40 generations × 5 seeds, or 9000 for 60 generations. Check `enabled: true` and `improvement_target: 0.01`. **Omitting `--config paper_parameters_adaptive.json` runs the fixed-probability configuration.**

## 6. Run short checks before full training

Check probabilities and interruption/resume behavior without model training:

```bash
python -m unittest discover -s tests -p test_adaptive_search.py -v
```

After it reports `OK`, check GPU training, model saving, and output generation using synthetic data:

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --smoke --device gpu --output runs/adaptive_smoke_01
```

Expect a final `Finished:` message. Smoke results are not scientific results. The standard smoke run has only one generation and therefore does not reach adaptive updates; the automated tests check adaptive calculation, application, and resume behavior. Use a fresh output name, such as `_02`, if the previous directory exists.

## 7. Start the experiment

### Search only

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --output runs/adaptive_fitness_01
```

This runs all configured seeds, saves their results, and generates final plots. The supplied configuration has 6000 evaluations. Early stopping makes training time vary by candidate.

### Search followed by retraining and baseline comparison

Use this instead of the search-only command; do not run both simultaneously:

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage all --device gpu --output runs/adaptive_fitness_01
```

The default retraining stage runs the selected 15 functions and six baselines across five seeds: 105 trials. The first real-data run needs internet access to download CIFAR-10. The program creates its output directory; do not create `runs/adaptive_fitness_01` in advance.

Keep the PC awake and the training terminal open. For SSH, use your usual persistent terminal/session mechanism if the connection may close. Do not run `git pull` or edit execution code/settings while an experiment is running.

## 8. Monitor progress and inspect results

In another terminal, enter the same repository directory and run:

```bash
tail -n 30 -f runs/adaptive_fitness_01/run.log
```

`Ctrl+C` in this monitoring terminal stops only the log viewer. A probability update may look like:

```text
next_generation=5 probability_mode=adaptive fitness_improvement=0.005 crossover_probability=0.900000 mutation_probability=0.500000
```

This means 0.5% relative improvement over three generations, with generation 5 created using 90% crossover and 50% mutation. After generations 1–3, `warmup` uses the base 80% / 20% rates. The first adaptive update occurs after generation 4 and affects generation 5. The final generation has no offspring, so its probability fields are empty.

| Location inside the output directory | Contents |
| --- | --- |
| `run.log` | Progress, probabilities, errors |
| `results/generation_history.csv` | Fitness, improvement, next-generation probabilities, and offspring counts |
| `results/evaluation_results_seed_*.csv` | Candidate functions, loss, accuracy, fitness |
| `results/selected_top15.csv` | Selected functions |
| `results/retraining_results.csv` / `comparison_summary.csv` | Retraining results and aggregates, when run |
| `histories/` | Epoch histories |
| `plots/` | Plots generated after the requested stage completes |
| `models/` | Saved models |
| `checkpoint.json` | Resume state |
| `manifest.json` / `source_snapshot/` | Execution environment, configuration, and source |

## 9. Interrupt, resume, or add retraining later

Use `Ctrl+C` in the training terminal to interrupt. Resume with the same source, configuration, and environment, replacing `--output` with `--resume`:

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --resume runs/adaptive_fitness_01
```

Use `--stage all` to continue through retraining. Completed candidate evaluations are retained; an interrupted candidate starts its training again from scratch. Source/configuration changes cause resume to be rejected. A different generation count also requires a new experiment.

After completing search only, start retraining without repeating search:

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage retrain --device gpu --resume runs/adaptive_fitness_01
```

In a new terminal, return to the repository and reactivate `.venv-afos` with `source .venv-afos/bin/activate`, or activate the existing environment you chose. Copy the entire output directory when transferring results. Resuming also requires the original source/configuration and checkpoint, not just CSV files.

## 10. Troubleshooting

| Symptom | What to check |
| --- | --- |
| `python3.10: command not found` | Locate/install Python 3.10 and venv support on the target PC; do not silently substitute the latest Python |
| `No matching distribution` / dependency conflict | Python version, x86_64 platform, pinned packages; do not mix Windows requirements into Linux |
| `No GPU detected` / `GPUs: []` | Both `nvidia-smi` and TensorFlow GPU visibility; drivers, CUDA dependencies, container GPU access |
| `Invalid filepath extension` | Check that the source being executed uses the updated `.keras` filenames described in section 4 |
| Output directory already exists | Use a new name for a new experiment, or `--resume` to continue |
| `Resume config differs` | Restore the starting configuration, or use a new output for different settings |
| `Source changed since this run` | Restore the recorded `source_snapshot`; do not apply filename changes or updates after starting |
| GPU out of memory | Check other GPU processes. Batch-size/JIT changes are experiment changes and should be recorded in a new run |
| Probabilities initially stay fixed | The first three generation-end updates are warmup; smoke ends before adaptation |

## Windows / WSL2

The historical records describe Linux. For native Windows, use the [Windows setup guide](RUN_GUIDE_EN.md) and explicitly pass `--config paper_parameters_adaptive.json` when running. The Windows TensorFlow 2.10 / DirectML setup does not need this guide's Linux installation. Model filenames are already updated on this branch. For WSL2, create the environment inside Linux, verify the NVIDIA driver/GPU setup, and pass the same tests and smoke check before full training.

## Verification scope

This guide was prepared from the recorded environment and current source. Configuration and adaptive-control tests have passed on the development PC. **A fresh Linux installation and GPU training on the target PC have not been executed as part of preparing this guide.** Run the environment checks and short checks before starting the full experiment.
