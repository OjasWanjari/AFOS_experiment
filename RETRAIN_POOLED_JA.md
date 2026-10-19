# 全探索シードの上位関数を再学習する

`retrain_paper.py` は保存済みの探索結果を読み、**全探索シードをまとめた上位15関数だけを選び直して、再学習とテスト評価を実行**します。探索処理は実行しません。比較用のReLU、Swish、Mish、GELU、SERF、ELUも同じ条件で再学習します。

既定値は **(15候補 + 6比較関数) × 5再学習シード = 105回**、各50エポックです。新しいモデルを毎回作ります。探索時の重みや過去のseed 0・seed 4の再学習結果は引き継ぎません。

## 今回の更新：新しいseed 4を使う（2026-10-19）

この手順では、追加された `paper_full_results/paper_full_seeds4` を探索seed 4の入力に採用します。seed 0〜3の探索結果と統合して共通の上位15関数を選び、その各関数を再学習seed 0〜4で評価します。seed 4だけを再学習する手順ではありません。

**以前の `retrain_runs/pooled_top15_prepared` を再開しても、新しい探索結果は反映されません。** `plan.json`の候補は作成時に固定されるため、今回は新しい出力先 `retrain_runs/pooled_top15_updated` を使います。既にその名前で計画を作成している場合は、第2節で採用元を確認してから再開してください。異なる採用元の計画なら、さらに別の未作成ディレクトリ名を使ってください。

## 論文との対応

参照：ユーザー提供PDF *Activation Function Optimization Scheme for Image Classification*、arXiv:2409.04915v1、4.1節・本文15–16ページ・Table 2 / Table 3 / Table 6。

| 項目 | このプログラムの既定動作 |
| --- | --- |
| 候補の順位 | 探索で15エポック完走した記録の **validation accuracy** 順。fitnessやtest accuracyでは選ばない |
| 候補数 | 全探索シードを合わせて15関数 |
| 再学習 | 初期重みから50エポック、各関数5回（seed 0–4） |
| CNN | Table 2の6畳み込み層：28, 32, 64, 64, 128, 128フィルタ |
| 最適化 | SGD、学習率0.01、momentum 0.9、batch size 64、he_uniform、same padding |
| 損失 | categorical cross-entropy |
| データ | CIFAR-10。保存された分割をそのまま使用：訓練45,000、検証5,000、テスト10,000 |
| 評価 | 50エポック終了時のモデルでテスト評価。最良エポックの選び直しはしない |
| 集計 | 5回すべてのテスト評価が成功した場合に平均・標本標準偏差（ddof=1）を出力 |

**テストデータで学習するわけではありません。** モデルは訓練データで学習し、検証データを監視に使います。テストデータは学習後の評価のみです。

論文には、探索反復間の候補の統合方法、重複除去、具体的なseed、再学習時の分割維持、標準偏差の定義などが完全には指定されていません。ここでは既存実装の前提を維持し、次を明示的な実装上の選択とします。

- 同じ**数式文字列**の記録は、検証精度が最も高い1件にまとめます。数学的な同値性の判定はしません。例：`a*b` と `b*a` は別候補として残る場合があります。
- 同率の場合は、検証損失が低い順 → 探索seedが小さい順 → 評価番号が小さい順です。
- `/255`正規化、dropout 0.2、分割seed 1234、データ拡張なし、再学習seed 0–4を採用します。
- NaN/Infの発生は記録して停止します。探索時の「初回検証精度0.25以下」の打ち切りは再学習では使いません。
- 実行時のOOM対策としてXLA/JITを無効化し、GPU memory growthと試行ごとの別プロセスを使用します。これは実行方法の変更です。ハードウェアやライブラリー間の数値の完全一致、OOMの完全解消は保証しません。

## 1. 実行環境とファイルを用意する

以前の探索を実行できたTensorFlow環境を使用してください。Python 3.10以上、TensorFlow/Keras、NumPy、scikit-learnが必要です。Linux/CUDAの実験には、Windows用DirectMLのrequirementsを使用しないでください。

別PCへはリポジトリ全体、または少なくとも次をコピーします。

```text
retrain_paper.py
run_paper.py                 # 設定検証・JSON/CSV出力の共通ヘルパーとして使用
paper_parameters.json
paper_afos/                 # フォルダー全体（新しいretraining.py/retrain_worker.pyも含む）
paper_full_results/
  paper_full_seeds0/
    manifest.json
    split_indices.npz
    results/evaluation_results_seed_0.csv
  paper_full_seeds1/         # 探索seed 1
  paper_full_seeds2/         # 探索seed 2（seed 3の途中版も含む）
  paper_full_seeds3 and 4/   # 探索seed 3の1,200件を使用。旧seed 4も含む
  paper_full_seeds4/         # 今回追加された探索seed 4を使用
    manifest.json
    split_indices.npz
    results/evaluation_results_seed_4.csv
```

各runについて `manifest.json`、`split_indices.npz`、`results/evaluation_results_seed_*.csv` が必要です。`paper_full_seeds3 and 4`はseed 3の入力にも必要なので、旧seed 4を除外する目的でフォルダーごと削除しないでください。新しいseed 4を`--source`で明示すると、旧seed 4との二重集計を避けられます。探索結果はGitHubへのコードのpushには含めていないため、別PCにも別途コピーしてください。

`run_paper.py`はヘルパーのimportだけに使い、探索関数は呼び出しません。`retrain_paper.py`だけを単独コピーしても動きません。以下のコマンドはすべてリポジトリのルートから実行します。

Linuxの既存環境の例：

```bash
cd /home/victor/AFOS_experiment
source /home/victor/myafos/bin/activate
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Windows PowerShellの既存環境の例：

```powershell
cd D:\vscode\AFOS-main
.\.venv-gpu\Scripts\Activate.ps1
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

以下では有効化した環境の`python`を使用します。仮想環境を有効化できない場合は`python`をその環境のPython実行ファイルのパスに置き換えてください。子プロセスも同じPythonを使います。初回の本学習では、KerasキャッシュにCIFAR-10がなければダウンロードが発生します。

## 2. 新しいseed 4を指定して上位15候補を選び直す

```bash
python retrain_paper.py --results-root paper_full_results --source "4=paper_full_results/paper_full_seeds4" --output retrain_runs/pooled_top15_updated --plan-only
```

この処理ではTensorFlowをimportせず、GPU学習やデータセットのダウンロードを行いません。`--output`には未作成のディレクトリを指定します。

- `selected_top15.csv`：上位15候補の数式、染色体、探索seed、評価番号、検証精度。
- `selection_audit.json`：どの探索結果を採用したか、除外・欠落に関する情報。
- `plan.json`：候補・105試行・設定・ソースコードのハッシュを固定した実行計画。
- `source_snapshot/`と`split_indices.npz`：実行するコードとデータ分割のコピー。

同じseedの探索結果が複数ある場合は、評価番号が1から連続した記録のうち**件数が最大の軌跡を1つ**採用します。検証精度が高い軌跡を選ぶ処理ではありません。今回のseed 3では、382件の途中版より1,200件の版を採用します。

新旧のseed 4はどちらも1,200件で内容が異なるため、`--source`を省略すると `Ambiguous equal-length trajectories for seed 4` で停止します。上のコマンドの指定を必ず含めてください。seed 3の採用元も明示したい場合は、次のように両方指定できます（上とは別の新規計画です）。

```bash
python retrain_paper.py --results-root paper_full_results --source "3=paper_full_results/paper_full_seeds3 and 4" --source "4=paper_full_results/paper_full_seeds4" --output retrain_runs/pooled_explicit_updated --plan-only
```

既定の探索seed 0–4がすべて必要です。選んだrun間で学習条件、関数の番号付け、保存されたデータ分割が一致しなければ停止します。`status=completed`、所定エポック完走、損失・精度が有限値という条件を満たす記録だけを順位付けします。

新しいseed 4は1,200件中、15エポック完走596件、低精度打ち切り130件、非有限値による打ち切り474件で、0エポックの記録は0件です。以前の「0エポック851件」は旧seed 4の結果です。新しい入力を指定すれば、その旧結果に由来する警告は出ません。非有限値の記録は引き続き候補選定から除外します。

学習開始前に `retrain_runs/pooled_top15_updated/selection_audit.json` で、`seed: 4` の `selected_source` が `paper_full_seeds4`、`recorded` が1200、`eligible` が596、`zero_epoch_records` が0になっていることを確認してください。`plan.json`には105試行、再学習seed `[0, 1, 2, 3, 4]`、50エポックが保存されます。

2026-10-19時点のデータでは、全体1位は `(erf(x)*sigmoid(sinh(x)))`（検証精度82.50%）、2位は新seed 4の `(sigmoid(x)*erf(erf(x)))`（81.94%）です。旧計画と比べて1関数が入れ替わります。これらは選定確認用の検証精度であり、今回の再学習後のテスト精度ではありません。

作成済み計画を学習せずに読み直す場合は、次を使います。候補の再選定は行いません。

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume --plan-only
```

## 3. 再学習を開始する

上で作成した計画をそのまま実行：

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume
```

計画作成から学習開始までを一度に行う場合（別の未作成ディレクトリ）：

```bash
python retrain_paper.py --results-root paper_full_results --source "4=paper_full_results/paper_full_seeds4" --output retrain_runs/pooled_new_updated
```

既定はGPUです。GPUが認識されなければエラーとして停止します。CPUで実行する場合は**計画作成時**に`--device cpu`を指定します。

最初の1試行だけを実行して動作・所要時間を確認する場合：

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume --max-trials 1
```

続けるときは`--max-trials`を外して再開します。各試行の進捗は`trials/<関数名>_seed_<番号>/attempt_0001/worker.log`と`histories/`に出力されます。Linuxでは`tail -f <worker.logのパス>`、PowerShellでは`Get-Content <worker.logのパス> -Wait`で確認できます。

`--resume`では出力先に固定された入力を使用するため、`--source`や`--results-root`を追加しないでください。以前の出力先を指定すると以前の候補を学習するので、必ず今回作成した出力先を指定します。

メトリクスだけが必要でモデルを保存しない場合は、**新規計画の作成時**に`--no-save-models`を追加します。設定ファイルを変更する場合は`--config <JSONファイル>`を指定します。既定条件以外の実験は論文条件とは区別してください。

## 4. 中断・エラーから再開する

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume
```

完了した試行はスキップします。OOM、Python例外、子プロセス異常終了などがあれば、連続失敗を避けるためその場で実行を止めます。原因を確認・解消してから、失敗した試行の再実行を明示します。

```bash
python retrain_paper.py --output retrain_runs/pooled_top15_updated --resume --retry-errors
```

- 失敗試行は同じseedで**1エポック目から**やり直します。途中の重みやoptimizer状態からは復元しません。
- 前回のログ・結果は`attempt_0001/`に残り、再実行は`attempt_0002/`などに保存します。集計は最新attemptを使用します。
- `nonfinite` / `nonfinite_test`は数値的失敗として保存し、次の試行に進みます。`--retry-errors`でも自動再実行しません。成功するまで再試行して平均を良く見せることを避けます。
- モデル保存だけの失敗では、テスト精度を消しません。`status=completed`に加えて`model_save_status=error`を記録して実行を停止します。通常の`--resume`で次の試行へ進めますが、このモデルの保存だけを再試行する機能はありません。
- 強制終了後に`run.lock`が残った場合、**親プロセスと子プロセスの両方が終了したことを確認してから**そのファイルだけを削除します。同じ出力先を同時実行しないでください。
- 再開時には固定済みの候補・設定・コードを使用します。元の探索結果や作業中のコードを変更しても再選択されません。設定・GPU/CPU・候補を変更したい場合は新しい出力先を使います。固定ファイルを直接編集するとハッシュ検証で停止します。

1試行が終わるごとに子プロセスを終了するため、そのプロセスのGPUメモリを解放できます。ただし、単一試行のピークメモリ不足や他プロセスとのGPU競合は残り得ます。使用GPUは必要に応じて起動前の`CUDA_VISIBLE_DEVICES`で指定してください。環境を変更して再開した場合、各attemptの`environment.json`を確認し、実験報告にその違いを記載してください。

## 5. 出力の読み方

| ファイル | 内容 |
| --- | --- |
| `results/retraining_results.csv` | 全105試行の最新状態。未着手は`pending`。精度は0–1の比率 |
| `results/retraining_summary.csv` | 関数ごとの成功回数、平均、標本標準偏差。5回そろうまでは平均・標準偏差を空欄にする |
| `status.json` | 全体の完了・失敗・未実行件数、候補選定の警告 |
| `trials/<trial>/attempt_0001/result.json` | その試行の結果、開始終了時刻、エラー内容 |
| `trials/<trial>/attempt_0001/histories/<trial>.csv` | エポックごとのloss/accuracy/val_loss/val_accuracy。NaN/Infは生の値を残す |
| `trials/<trial>/attempt_0001/worker.log` | TensorFlowと学習のログ |
| `trials/<trial>/attempt_0001/environment.json` | Python、TensorFlow/Keras等のバージョンとデバイス情報（TF初期化後に記録） |
| `trials/<trial>/attempt_0001/model.keras` | 有効なテスト評価が得られたモデル（保存有効時） |
| `error.txt` / `save_error.txt` | 該当attemptの例外詳細（発生時のみ） |

NaN/Infは集計用JSON/CSVではnull/空欄になります。平均0.825は82.5%、標準偏差0.0014は0.14パーセントポイントです。計画作成時に警告があった場合は、再学習が成功してもその警告は残ります。今回指定する新しいseed 4には、旧seed 4の0エポック851件の警告は該当しません。

新seed 4の旧ランナーによる実行は、学習後のグラフ生成で `No module named 'matplotlib'` になりましたが、保存済み探索CSVは候補選定に利用できます。今回の再学習専用ランナーはCSV/JSONの集計までを行い、グラフを生成しないため、`matplotlib`は必要ありません。旧実行のELUのNaNは別の問題で、今回も数値的失敗が起こる可能性はあります。

終了コード：0＝この起動で予定した処理を正常終了（`--plan-only`や`--max-trials`による未実行分はあり得る）、1＝設定または実行エラー、2＝処理を終えたが数値的失敗またはモデル保存エラーが残る、130＝ユーザーによる中断。完全終了は`status.json`の`all_evaluations_completed`と`model_save_errors`で確認します。

## 小規模な動作確認

```bash
python retrain_paper.py --results-root paper_full_results --source "4=paper_full_results/paper_full_seeds4" --output retrain_runs/smoke_updated --smoke-test --device cpu
python retrain_paper.py --output retrain_runs/smoke_updated --resume
```

これは上位1候補とReLUをseed 0で各1エポック、ランダム生成した32枚の訓練データで実行します。CIFAR-10のダウンロードは不要です。計画には`smoke=true`が記録されます。**精度を論文との比較に使用しないでください。** 2つ目のコマンドで完了済みの試行が再実行されないことも確認できます。

自動テスト：

```bash
python -m unittest tests.test_pooled_retraining -v
```
