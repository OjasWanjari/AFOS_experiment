# 複数seedをJSONから順番に実行する

[English](MULTI_SEED_EN.md)

`run_paper.py` はJSONの `search.seeds` を読み、**1つのseedにつき1つの独立したPythonプロセス**で探索します。
seedが終了するとプロセスの終了を待ってから次のseedを開始します。並列実行ではありません。
呼び出し元はTensorFlowや学習データを読み込まないため、seed間で学習用プロセスのメモリを持ち越しません。

以前は同じTensorFlowプロセスで全seedを処理していました。個別実行なら成功するが連続実行でOOMになる状況に対応する変更です。
単一seed自体に必要なGPUメモリは減らさないため、1つのseed内で発生するOOMまで防ぐものではありません。

## seed 2・3・4を実行する設定

`paper_parameters_adaptive.json` の `search` 内の2項目を変更します。他の設定はそのままにします。

```json
"seeds": [2, 3, 4],
"replications": 3
```

`post_search.seeds` は再学習用seedで、探索seedとは別です。探索seedを変更しても自動変更しません。
前回の既存関数の結果を使い、既存6関数を再学習しない場合は、実験開始前に `post_search.baselines` を `[]` にします。

## 実行コマンド

LinuxでもWindowsでも、仮想環境を有効にし、リポジトリ内で実行します。

```bash
python run_paper.py --config paper_parameters_adaptive.json --dry-run
python -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --output runs/adaptive_seeds234_01
```

40世代・30個体・3seedなら `search_evaluations` は3600です。
最初のコマンドは設定確認のみです。2つ目でseed 2、3、4を順に探索します。
保存先には未作成のフォルダー名を指定してください。最後に `Finished:` と出れば要求した工程が完了しています。

探索後の再学習まで続ける場合は、上の2つ目のコマンドの `--stage search` を `--stage all` に置き換えます。
再学習は「活性化関数 × 再学習seed」の1試行ごとに独立プロセスを使います。
データ読み込みやTensorFlow初期化を繰り返すため、その分の起動時間が増えます。

## 出力と選抜

全seedの結果は指定した1つの出力フォルダーへ保存します。

- `results/evaluation_results_seed_2.csv` など：seed別の個体評価。
- `results/generation_history.csv`：全seedの世代履歴・適応確率。
- `results/selected_top15.csv`：この実行で指定した全seedをまとめて選ぶ上位関数。
- `checkpoint.json`：現在のseed・世代・完了済み個体と再学習試行。
- `run.log` / `results/events.csv`：処理の進捗。`worker_started` / `worker_completed` にプロセスIDを記録。

別フォルダーにある以前のseed 0・1の結果は、自動では選抜対象に加わりません。
探索条件、fitness計算、適応確率、学習用seedの計算方法は変更していません。

## 中断・OOM後の再開

途中で子プロセスが失敗すると、その場で停止します。次のseedを勝手に開始しません。
Ctrl+Cの場合も、呼び出し元は学習プロセスを終了させ、その終了を待ちます。

```bash
python -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --resume runs/adaptive_seeds234_01
```

開始時と同じ設定・ソース・環境を使います。完了済みseedと個体は再実行せず、チェックポイントから続行します。
学習途中だった個体や再学習試行は、最初から学習し直します。
探索完了後に再学習だけを行う場合は `--stage retrain --resume runs/adaptive_seeds234_01` を使います。

**変更前のプログラムで開始した実験を、この変更後のソースでそのまま再開することはできません。**
既存のソース一致チェックを維持しています。古い実験はその `source_snapshot` のコードで再開し、
この方式を使うseed 2・3・4は新しい出力先で開始してください。過去の結果やチェックポイントを書き換える必要はありません。

## 変更箇所と確認方法

`run_paper.py` の `run_isolated()` と `wait_for_worker()` が順次実行・終了待ちを担当します。
`search(..., one_seed=True)` は1seedの探索後、`retrain(..., one_trial=True)` は1再学習試行後に戻ります。
最後の独立プロセスでCSV集計・描画を行います。内部用の `--_worker` は手動指定しません。

```bash
python -m unittest tests.test_seed_processes tests.test_adaptive_search tests.test_paper.SearchTests tests.test_paper.RecordingTests -v
```

自動テストは学習を代替した実プロセスでseed 2・3・4、選抜、再学習、失敗後の再開を確認します。
実行先GPUでのメモリ使用量は、そのPCの実行時に確認してください。
