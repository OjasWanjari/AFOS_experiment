# 別のPCで適応型AFOSを実行する手順

[English version](ADAPTIVE_OTHER_PC_EN.md) · [適応確率の仕様と変更箇所](ADAPTIVE_SEARCH_JA.md)

この手順書は、前回の `paper_full_results` に記録された **Linux・NVIDIA GPU環境** を基準にしています。
コマンドは実行先のLinux端末（Bash）で、上から順に実行してください。Windows PowerShellのコマンドではありません。
現在のWindows PCの仮想環境を、別PCへコピーする必要はありません。

## 1. 前回の記録から確認できた環境

| 項目 | 記録された値 | 確認元 |
| --- | --- | --- |
| OS | Linux、x86_64、kernel `6.8.0-138-generic`、glibc 2.35 | 各探索の `manifest.json` |
| Python | 3.10.12 | 各探索の `manifest.json` |
| TensorFlow | 2.21.0 | 各探索の `manifest.json` |
| Keras | 3.12.4 | 各探索の `manifest.json` |
| NumPy | 2.2.6 | 各探索の `manifest.json` |
| scikit-learn | 1.7.2 | 各探索の `manifest.json` |
| matplotlib | 3.10.9 | seed 0の `manifest.json` |
| GPU | NVIDIA GeForce RTX 4080 SUPER | 再学習時の `environment_preflight.json` |
| TensorFlowのビルド | CUDA 12.5.1、cuDNN 9 | 同じ再学習時の記録 |

探索の確認元は `paper_full_results/paper_full_seeds0/manifest.json` などの5探索フォルダーです。
GPU名とCUDA/cuDNNの確認元は `paper_full_results/published_top15_matched/environment_preflight.json` です。
**GPU名は再学習時の記録であり、探索時のGPU機種を直接証明するものではありません。**
CUDA/cuDNNの値もTensorFlowのビルド情報で、GPUドライバーのバージョンではありません。
Linuxのディストリビューション名と探索時のドライバーバージョンは、今回確認した記録からは確定できません。

以降では同じ主要Pythonパッケージを指定します。別GPUでも試せますが、前回と同じ速度・数値結果になる保証はありません。

## 2. コードを取得する

GitとPython 3.10、NVIDIAドライバーが利用できるLinux PCを用意します。
まず端末で次を確認してください。

```bash
git --version
python3.10 --version
nvidia-smi
```

`nvidia-smi` にはGPU名とドライバーバージョンが表示されます。コマンドが見つからない、またはGPUに接続できない場合は、
そのPCのドライバー設定を先に確認してください。コンテナー内では、コンテナーへのGPU公開も必要です。

コードを保存したい親フォルダーで実行します。

```bash
git clone --depth 1 --single-branch --branch codex/adaptive-fitness-probabilities https://github.com/OjasWanjari/AFOS_experiment.git AFOS_adaptive
cd AFOS_adaptive
git branch --show-current
git rev-parse HEAD
```

ブランチ名が `codex/adaptive-fitness-probabilities` なら正しい取得先です。
既に `AFOS_adaptive` がある場合は別のフォルダー名を使ってください。既存の実験フォルダーを削除する必要はありません。
以降のコマンドは、この `AFOS_adaptive` フォルダー内で実行します。

必要な主なファイル：

- `run_paper.py`：実行入口。
- `paper_parameters_adaptive.json`：適応型の設定。
- `paper_afos/`：モデル・個体生成・適応確率・記録の処理。
- `tests/test_adaptive_search.py`：学習なしで確率と再開処理を確認するテスト。

前回のモデルや結果CSVは、新しい探索を始めるためには不要です。

## 3. Python環境を用意する

### A. 前回使った環境が実行先に残っている場合

その環境を有効にして、次の確認へ進みます。探索ログには `/home/victor/myafos/` が記録されていますが、
実行先にも同じパスが存在するとは限りません。実際の環境の場所を使ってください。

```bash
# 実際の場所に置き換える。例のまま実行しない。
source /path/to/existing-environment/bin/activate
python --version
python -m pip check
```

既存環境を使う場合、次のBのインストールは不要です。バージョンを確認し、既存環境を無断で更新しないでください。

### B. 新しいPCで環境を作る場合

Python 3.10のvenvが利用できる状態で実行します。

```bash
python3.10 -m venv .venv-afos
source .venv-afos/bin/activate
python -m pip install --upgrade pip
python -m pip install "tensorflow[and-cuda]==2.21.0" "keras==3.12.4" "numpy==2.2.6" "scikit-learn==1.7.2" "matplotlib==3.10.9"
python -m pip check
```

`tensorflow[and-cuda]` はLinux／WSL2のNVIDIA GPU向けにTensorFlowが案内するインストール方式です。
GPUドライバーは別途必要です。このコマンドは前回の全依存環境を完全複製するものではなく、
記録された主要バージョンを指定した新規環境作成例です。インストールの競合が出た場合は、
勝手にバージョン指定を外さず、エラーメッセージを確認してください。
[TensorFlow公式インストール手順](https://www.tensorflow.org/install/pip)

**Linuxでは `requirements-paper-windows.txt` を使いません。** これはWindows用のTensorFlow 2.10／DirectML構成です。

### 共通：バージョンとGPUを確認する

```bash
python -c "import sys, importlib.metadata as m; print(sys.version); print({p: m.version(p) for p in ['tensorflow','keras','numpy','scikit-learn','matplotlib']})"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU')); print('Build:', tf.sysconfig.get_build_info())"
```

`GPUs: []` ならTensorFlowからGPUを使えません。GPUを使う本番実行は、一覧にGPUが表示されてから始めます。
`nvidia-smi` が成功しても、TensorFlowのGPU認識が成功するとは限りません。

## 4. Keras 3用のモデル保存名（変更済み）

このブランチの `run_paper.py` は、Keras 3で保存できるように変更済みです。
**手動でコードを変更する必要はありません。そのまま手順5へ進んでください。**

| 保存するモデル | 保存先 |
| --- | --- |
| 探索中の各seedの最良モデル | `models/best_search_seed_{seed}.keras` |
| 再学習モデル | `models/{name}_seed_{seed}.keras` |

変更箇所は `run_paper.py` の探索処理 `search()` と再学習処理 `retrain()` にある `model.save()` の保存名です。
両方に `.keras` を追加しました。fitness・学習条件・確率計算は変更していません。
[Keras公式移行ガイド](https://keras.io/guides/migrating_to_keras_3/#saving-a-model-in-the-tf-savedmodel-format)

実験開始後は実行コードを変更しないでください。変更前に開始した実験を再開する場合は、その実験で使用したソースが必要です。

## 5. 実験設定を確認・変更する

`paper_parameters_adaptive.json` をテキストエディターで開きます。

| 場所 | 初期値 | 意味 |
| --- | --- | --- |
| `search.generations` | `40` | 各seedの世代数。60世代ならここを60にする |
| `search.population_size` | `30` | 1世代の個体数 |
| `search.seeds` / `search.replications` | `[0,1,2,3,4]` / `5` | 実行seedとその数 |
| `search.adaptive_probabilities.enabled` | `true` | 適応型を有効にする |
| `window_generations` | `3` | 最高fitnessを比較する世代間隔 |
| `improvement_target` | `0.01` | 1%改善で元の確率に戻す |
| `max_crossover_probability` | `1.0` | 改善0%の交叉確率100% |
| `max_mutation_probability` | `0.8` | 改善0%の突然変異確率80% |
| `training.epochs` | `15` | 探索時の最大学習エポック数。世代数とは別 |
| `post_search.epochs` | `50` | 再学習時の最大エポック数 |

`window_generations` 以降の確率制御キーは `search.adaptive_probabilities` の中です。
元の確率は `search.crossover_probability=0.8`、`search.mutation_probability=0.2` です。
改善0〜1%の間は両方の確率が連続的に下がります。
世代数だけ変えたい場合、`search.generations` 以外を変更する必要はありません。
seedを変更する場合は `replications` もその数に合わせます。再学習seedは `post_search` で別に設定します。

設定の読み込み確認（学習・ダウンロードなし）：

```bash
python run_paper.py --config paper_parameters_adaptive.json --dry-run
```

40世代・30個体・5seedなら `search_evaluations` は6000です。60世代なら9000になります。
`enabled` が `true`、改善基準が `0.01` になっていることも確認します。
**`--config paper_parameters_adaptive.json` を省略すると固定確率の実験になります。**

## 6. 本番前の短い確認

確率計算と中断再開を、学習なしで確認します。

```bash
python -m unittest discover -s tests -p test_adaptive_search.py -v
```

`OK` を確認したら、人工データでGPU学習・モデル保存・結果出力を確認します。

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --smoke --device gpu --output runs/adaptive_smoke_01
```

最後に `Finished:` と出れば完了です。結果は本番成績には使いません。
標準の `--smoke` は1世代だけなので、適応確率が動くところまでは通りません。
適応計算とその適用・再開は上の自動テストが確認します。
同じ保存先がある場合は `_02` などの新しい名前を使ってください。

## 7. 本番を開始する

### 探索だけ実行する場合

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --output runs/adaptive_fitness_01
```

指定した全seedの探索と結果保存・最終描画で終了します。
40世代の設定なら6000評価です。各個体は早期終了する場合があるため、所要時間は個体によって異なります。

### 探索から再学習・既存関数比較まで実行する場合

上の探索専用コマンドの代わりに、次を使います。両方を同時には実行しません。

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage all --device gpu --output runs/adaptive_fitness_01
```

探索後、上位15関数と既存6関数を5seedずつ再学習します。標準設定では計105試行です。
初回の本番実行ではCIFAR-10をダウンロードするため、インターネット接続が必要です。
保存先はプログラムが作るので、事前に `runs/adaptive_fitness_01` を作らないでください。

PCをスリープさせず、実行中の端末を閉じないでください。SSH接続なら、接続が切れても実行を維持できる
端末セッション管理ツールなど、実行先で普段使っている方法を利用してください。
本番実行中は `git pull` や設定・コードの編集を行わず、開始時の状態を保ちます。

## 8. 進捗・確率・結果を確認する

別の端末で同じリポジトリへ移動し、実行します。

```bash
tail -n 30 -f runs/adaptive_fitness_01/run.log
```

ログ閲覧側の `Ctrl+C` は閲覧だけを終了します。
確率の更新例：

```text
next_generation=5 probability_mode=adaptive fitness_improvement=0.005 crossover_probability=0.900000 mutation_probability=0.500000
```

これは「3世代で0.5%改善したため、第5世代を交叉90%・突然変異50%で生成」という意味です。
第1〜3世代終了時は `warmup` で元の80%・20%。第4世代終了時に最初の適応計算を行い、第5世代生成へ反映します。
最終世代には次世代がないので、CSVの確率欄は空欄です。

| 出力先内の場所 | 内容 |
| --- | --- |
| `run.log` | 進捗、確率、エラー |
| `results/generation_history.csv` | 世代ごとのfitness・改善率・次世代用確率・生成個体数 |
| `results/evaluation_results_seed_*.csv` | 個体ごとの関数・loss・accuracy・fitness |
| `results/selected_top15.csv` | 探索で選ばれた関数 |
| `results/retraining_results.csv` / `comparison_summary.csv` | 再学習を実行した場合の結果と集計 |
| `histories/` | 各エポックの学習履歴 |
| `plots/` | 指定工程の終了後に生成するグラフ |
| `models/` | 保存モデル |
| `checkpoint.json` | 再開用状態 |
| `manifest.json` / `source_snapshot/` | 実行時の環境・設定・ソース |

## 9. 中断・再開と、後から再学習する方法

学習用端末で `Ctrl+C` を押して中断します。再開時は、開始時と同じコード・設定・環境を使い、
`--output` を `--resume` に置き換えます。

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --resume runs/adaptive_fitness_01
```

探索と再学習を続けるなら `--stage all` を使います。
完了済み個体は再利用されますが、中断時に学習途中だった個体は最初から学習し直します。
設定・ソースが変わると再開は拒否されます。世代数を変更する場合も新しい実験として開始してください。

探索だけを完了した後、再学習を追加する場合：

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage retrain --device gpu --resume runs/adaptive_fitness_01
```

端末を開き直したら、リポジトリへ移動して `source .venv-afos/bin/activate` を再実行します。
既存環境を使った場合は、その環境のactivateを使ってください。
結果を別PCに移すときは出力フォルダー全体を保存します。途中再開にはCSVだけでなくチェックポイントと元のコード・設定が必要です。

## 10. よくある問題

| 表示・状況 | 確認すること |
| --- | --- |
| `python3.10: command not found` | Python 3.10の設置場所・venv対応を実行先で確認。無指定の最新版Pythonへ置き換えない |
| `No matching distribution` / 依存競合 | Pythonのバージョン、x86_64環境、指定パッケージを確認。Windows用依存ファイルを混ぜない |
| `No GPU detected` / `GPUs: []` | `nvidia-smi` とTensorFlowの一覧を両方確認。GPUドライバー・CUDA依存・GPU公開を確認 |
| `Invalid filepath extension` | 実行中のコードが手順4の `.keras` 保存名に対応した版か確認 |
| 出力フォルダーが既に存在 | 新しい実験は別名にする。続きからなら `--resume` |
| `Resume config differs` | 開始時の設定に戻す。別条件なら新規出力先で実行 |
| `Source changed since this run` | 保存された `source_snapshot` の実行コードに戻す。開始後に保存名修正や更新をしない |
| GPUメモリー不足 | 他プロセスのGPU使用を確認。batch sizeやJIT設定の変更は実験条件の変更として記録し、新規実行にする |
| 最初は確率が変わらない | 最初の3世代終了時は正常な `warmup`。`--smoke` は適応前に終了する |

## Windows／WSL2で実行する場合

前回の記録はLinuxです。Windowsネイティブで実行する場合は [Windows環境の準備](RUN_GUIDE_JA.md) を使い、
実行時には必ず適応型の `--config paper_parameters_adaptive.json` を指定してください。
WindowsのTensorFlow 2.10／DirectML環境では、この手順書のLinux向けインストールは不要です。モデル保存名はこのブランチで変更済みです。
WSL2ではLinux側で環境を作ります。NVIDIAドライバーとGPU対応を確認し、同じテスト・smoke確認を通してから実行します。

## この手順の確認範囲

前回の環境記録と現在のコードを確認して作成しました。設定確認と適応制御のテストは開発PCで通過済みです。
**実行先Linux PCでの新規インストール・GPU学習は、まだこの手順では実測していません。**
必ず手順3の環境確認と手順6の短い実行確認を、本番より先に行ってください。
