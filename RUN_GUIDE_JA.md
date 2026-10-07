# AFOS：必要なファイルと実行手順

この手順は `run_paper.py` による **CIFAR-10・論文のベースCNNの探索／再学習** 用です。
旧実験の `main.py` とは別の入口です。論文の他モデル・他データセットの汎化実験は含みません。
論文に未記載のパラメーターなど、再現上の仮定は [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) を確認してください。

## 1. 必要なファイル

次のファイルを、リポジトリ内と同じ位置関係で揃えます。通常はリポジトリ全体をcloneしてください。

| ファイル | 用途 |
| --- | --- |
| `run_paper.py` | 実行入口。探索、再学習、保存、終了後の描画 |
| `paper_parameters.json` | 個体数・世代数・seed・学習条件など |
| `paper_afos/__init__.py` | Pythonパッケージの定義 |
| `paper_afos/search.py` | 染色体、数式、遺伝的アルゴリズム、評価値 |
| `paper_afos/model.py` | CNN、活性化関数、学習、データ読み込み |
| `paper_afos/recording.py` | エポック履歴と実行イベントのCSV記録 |
| `requirements-paper-windows.txt` | このPCで確認したWindows用依存パッケージ |

任意の補助ファイル：

- `tests/test_paper.py`：自動テスト。
- `benchmark_paper_generation.py`：30個体・1世代のGPU実測と時間予測。
- `benchmark_paper_candidate.py`：指定した1個体のGPU時間計測。専用の簡易出力形式を使用。
- `PAPER_REPRODUCTION.md`：論文との対応、実装上の仮定、保存内容の詳細。

この実行入口に旧 `main.py`、`fitness.py`、`activation.py`、`ga_utils.py`、`utils.py` は不要です。
過去の `runs/`、画像・スライド、論文PDFも実行に必須ではありません。
仮想環境は別PCからコピーせず、そのPCで作り直してください。

## 2. ソースを取得する

Windows PowerShellを開き、保存したい親フォルダーで実行します。Gitと64bit版Python 3.10が必要です。

```powershell
git clone https://github.com/OjasWanjari/AFOS_experiment.git
cd AFOS_experiment
```

以降のコマンドは、このリポジトリのルートで実行してください。
このPCの既存フォルダーは `D:\vscode\AFOS-main` です。すでに取得済みならcloneは不要です。

## 3. Python環境を作る（新しいPCで初回のみ）

```powershell
py -3.10 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-paper-windows.txt
.\.venv-gpu\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

このPCで検証した構成はPython 3.10、TensorFlow CPU 2.10.0とDirectMLプラグインです。
末尾のGPU一覧が `[]` ならGPUが認識されていません。本番を始める前にGPUドライバーと依存環境を確認してください。
新しいPCでの互換性は保証しません。Python 3.11以降・Keras 3・Linux向けCUDA環境への置換はこの手順の対象外です。
CPUで確認したいときだけ、以降の実行コマンドに `--device cpu` を指定します。

現在のPCで `.venv-gpu` が動作している場合、作成・再インストールは不要です。
仮想環境のactivateも不要です（各コマンドでそのPythonを直接指定しています）。

## 4. 設定と短い動作試験

設定のみ確認（学習・データダウンロードなし）：

```powershell
.\.venv-gpu\Scripts\python.exe run_paper.py --dry-run
```

人工データで探索・再学習・保存・最終描画を確認：

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --smoke --device gpu --output runs/paper_smoke_check
```

`--smoke`の成績は科学的な実験結果ではありません。
出力先がすでに存在すると新規実行は拒否されるため、別の名前を指定してください。

自動テスト（CPU使用）：

```powershell
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -p test_paper.py -v
```

## 5. 本番をすべて実行する

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --device gpu --output runs/paper_full
```

標準設定は30個体 × 40世代 × seed 0〜4で **6,000回の探索評価** です。
その後、発見した上位15関数と既存6関数を各5回、最大50エポックで再学習します（計105試行）。
探索は最大15エポックで、早期打ち切りがあるため実行時間は個体により変わります。

CIFAR-10は初回の本番実行時にKerasがダウンロードします。初回はインターネット接続が必要です。
通常のキャッシュ先はユーザーホームの `.keras/datasets/` です（`KERAS_HOME`を設定した場合はその配下）。
Train 45,000枚、Validation 5,000枚、Test 10,000枚を使用します。
学習を続けるにはPCを起動し、スリープを避け、実行中の端末を閉じないでください。
モデルも保存するため、出力ドライブの空き容量を確保してください。

## 6. 実行中のログ・CSVを見る

別のPowerShellで、同じリポジトリのルートから：

```powershell
Get-Content runs/paper_full/run.log -Wait -Tail 30
```

このログ閲覧用端末の `Ctrl+C` は閲覧だけを終了します。学習用端末とは区別してください。

| 保存先（指定した出力フォルダー内） | 更新タイミング・内容 |
| --- | --- |
| `histories/*.csv` | 各エポック終了時。Train/Validation loss・accuracy |
| `results/evaluation_results_seed_*.csv` | 個体終了時。染色体・数式・score・loss・accuracy・状態 |
| `results/best_by_seed.csv`, `top3_seed_*.csv` | 個体終了時に更新する最良・上位候補 |
| `results/generation_history.csv` | 世代終了時の最高成績 |
| `results/selected_top15.csv` | 全探索終了後に選んだ再学習対象 |
| `results/retraining_results.csv`, `comparison_summary.csv` | 再学習試行終了時。Testを含む成績と集計 |
| `results/events.csv` | 開始・完了・中断・エラー・描画の時刻付き記録 |
| `plots/` | 全実行終了後に一度だけまとめて作る図 |
| `models/` | 各seedの探索中最良モデルと再学習の成功モデル |

CSVはエポック／個体／世代単位です。ミニバッチ単位の指標や全画像の予測は記録しません。
Testの成績は探索中には計算せず、再学習終了後にのみ計算します。
再開用JSON、設定・ソースのスナップショット、データ分割NPZも保存されます。

## 7. 中断と再開

学習用端末で `Ctrl+C` を押した場合、完了済み個体の記録と途中のエポックCSVを残します。
同じ設定・同じソースのまま再開します：

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --device gpu --resume runs/paper_full
```

中断した試行は最初からやり直します。以前の途中履歴は `histories/attempts/` に退避します。
`--output` と `--resume` は同時指定できません。
独自設定で始めた場合は同じ `--config` を、人工データ試験の場合は同じ `--smoke` を指定してください。
設定やソースを変更すると、異なる条件の混在防止のため再開を拒否します。
過去バージョンの実行は、その出力先の `source_snapshot/` のコードと設定が必要です。

標準実行では、全seedの探索・再学習・既存関数との比較が完了してからCSVを読み込み、一括描画します。
中断・例外時には描画しません。完了済みの同じ結果を再開しても図は再生成しません。
描画中に失敗した場合や図が削除された場合は再開時に再描画します。

## 8. 探索と再学習を分ける場合（任意）

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --stage search --device gpu --output runs/paper_separate
.\.venv-gpu\Scripts\python.exe -u run_paper.py --stage retrain --device gpu --resume runs/paper_separate
```

この場合だけは、探索の全実行終了後と、後日の再学習の全実行終了後に、それぞれ一括描画します。
全工程の最後だけ描画したい場合は手順5の標準実行を使用してください。

## 9. 自分のPCで時間を見積もる場合（任意）

```powershell
.\.venv-gpu\Scripts\python.exe -u benchmark_paper_generation.py --output runs/paper_generation_timing
```

seed 0・30個体・1世代だけ実行し、通常のCSV・ログに加えて `time_estimate.json` と
`TIME_ESTIMATE.md` を保存します。再学習は行いません。初期世代からの外挿なので、
後半の早期打ち切り率の変化や再学習時間を含む、確定した終了日時ではありません。
中断後は同じコマンドに `--resume` を追加してください。
出力先の隣に作成される `.config.json` も保持してください。

## 10. GitHubに含めるもの・含めないもの

プログラム、設定、依存一覧、自動テスト、この手順書を保存します。
仮想環境・ダウンロードデータ・新規の `runs/` は `.gitignore` の対象で、自動的にはGitHubへ保存されません。
実験後は必要なCSV・図・モデルを別途バックアップしてください。既に追跡されている過去の結果とは区別してください。
