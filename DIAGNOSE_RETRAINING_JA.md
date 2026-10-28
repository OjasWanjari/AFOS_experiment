# NaN/Infが発生した試行の診断

[English](DIAGNOSE_RETRAINING_EN.md)

これは原因調査専用の再実行です。公式の結果・平均値を更新しません。前回と同じLinux PC、RTX 4080 SUPER、仮想環境 `/home/victor/myafos` を使用します。保存された環境と異なる場合は学習前に停止します。

## 配布ZIPを使う

`retrain_runs/nonfinite_diagnostics_bundle.zip` を元のLinux PCにコピーし、新しいフォルダへ展開します。既存の結果へ上書きしないでください。

GitHubでは [実行用ZIP](retrain_runs/nonfinite_diagnostics_bundle.zip) のページで **Download raw file** を選択してください。ZIPには実行に必要な計画とコード、日英の手順書が含まれるため、元の結果フォルダのコピーやリポジトリのcloneは不要です。

```bash
source /home/victor/myafos/bin/activate
mkdir /home/victor/afos_diagnostics
cd /home/victor/afos_diagnostics
# このフォルダに nonfinite_diagnostics_bundle.zip をコピー
python -m zipfile -e nonfinite_diagnostics_bundle.zip .
cd AFOS_diagnostics
```

ZIPには `published/`（論文関数の25失敗）と `previous/`（候補13 seed 1・ELUの5シード、計6失敗）の計画、保存済み分割・学習コードが含まれます。Python、TensorFlow、CUDA、CIFAR-10本体は含みません。データは元のKerasキャッシュを使用し、なければ取得します。

## 最初にpaper_13 seed 0を調べる

`paper_13 = max(x, erf(x))` はexpを含まず、全5シードが1エポック目に失敗したため、最初の診断対象とします。

まず同じ学習処理を再実行し、数値検査を挿入しない状態で再現するか確認します。正式な50エポック設定なので、再現しなければ50エポックまで続きます。

```bash
python published/diagnose_retraining.py --output published --resume --trial paper_13_seed_0 --mode replay
```

続いて、演算ごとのNaN/Inf検査を有効にして実行します。

```bash
python published/diagnose_retraining.py --output published --resume --trial paper_13_seed_0 --mode check
```

`check` は `tf.debugging.enable_check_numerics()` を使い、非有限値が演算の出力に現れた時点で例外を記録します。元のgraph実行・モデル・SGD・データ分割・seed・シャッフル設定・Dropoutを維持します。検査自体で計算グラフや最適化が変化するため、元の実行と同じ経路・タイミングになるとは限りません。

## その他の失敗ケース

```bash
python published/diagnose_retraining.py --output published --resume --trial paper_08_seed_0 --trial paper_10_seed_0 --trial paper_11_seed_0 --trial paper_12_seed_0 --mode check
python previous/diagnose_retraining.py --output previous --resume --trial elu_seed_0 --mode replay
python previous/diagnose_retraining.py --output previous --resume --trial elu_seed_0 --mode check
python previous/diagnose_retraining.py --output previous --resume --trial discovered_13_seed_1 --mode check
```

論文の対象関数とELUはseed 1〜4も同様に指定できます。今回の候補13で失敗したのはseed 1のみで、他のシードはこの診断計画の対象外です。候補13は50エポック目に失敗したため最初からの再学習に時間がかかります。試行ごとに別プロセスで実行します。診断対象は明示した `--trial` のみです。対象一覧は以下で確認できます。

```bash
python published/diagnose_retraining.py --output published --resume --plan-only
python previous/diagnose_retraining.py --output previous --resume --plan-only
```

## 保存される情報

例えば `published/diagnostics/paper_13_seed_0/check/attempt_0001/` に次を保存します。

- `result.json`: 再現状況、例外、最後に記録されたtrain/validation/testの区分、epoch（1始まり）、batch（0始まり）。train区分だけではforward/backwardの区別はできません。
- `error.txt`: 完全な例外スタック。検査が作成された演算名・型・形状などを確認します。
- `worker.log`: TensorFlowと学習処理のログ。
- `batches.jsonl`: バッチ開始・終了の記録と、その時点のKeras集計指標。終了時のlossは累積平均の場合があり、単一バッチ損失とは限りません。
- `epochs.jsonl`: エポックの集計値。
- `variables.json` / `failure_or_final_variables.npz`: 停止時または終了時のモデル重み・optimizer状態と、有限値の数・最大絶対値。これは異常発生直前の状態や生の勾配ではなく、完全な再開用チェックポイントでもありません。
- `environment.json` / `environment_check.json`: 実行環境と参照元との一致確認。

終了コード0でも「学習成功」とは限りません。診断として停止結果を回収できたことを意味します。

| result.jsonのstatus | 意味 |
|---|---|
| nonfinite / nonfinite_test | 元の損失・精度チェックで非有限値を検出 |
| numerics_check_triggered | 演算の数値検査で停止。error.txtを解析する |
| not_reproduced | 設定した全エポック・テストまで非有限値を再現しなかった |
| runtime_error | 環境不一致、OOM、その他の例外。診断対象の数値異常と混同しない |

`error.txt` の演算名やスタックがforwardの層、`gradient_tape`等の勾配計算、SGD更新のどこに対応するかを調べます。名称だけで曖昧な場合は断定せず、該当演算の入力・出力・勾配を個別に計測する次段階へ進みます。

**最初に検出した非有限値が、最終損失NaNの原因と直ちに確定するわけではありません。** 例えばexpがInfになっても、その後のerfが有限値へ飽和する場合があります。replayでの再現とcheckの停止箇所を合わせて解釈してください。検査でのみ止まる場合も報告します。

## 再実行・中断

同じコマンドを再度実行すると `attempt_0002` 以降に保存し、同じseedで最初からやり直します。元の公式結果も前の診断結果も上書きしません。`--resume` は診断計画を再利用する意味で、途中の学習状態を復元するものではありません。

同じ出力で複数のcoordinatorを同時起動しないでください。Ctrl+Cはworkerも停止させます。強制終了で `diagnostic.lock` が残った場合は、両プロセスが終了したと確認してからそのロックだけを削除します。部分的な診断記録は残ります。

実行後は `published/diagnostics/` と `previous/diagnostics/` をフォルダごと戻してください。これらを読むことで発生演算・段階を絞り込めます。学習率の変更やクリッピング等の対策は、原因調査後に別実験として行います。

## 自分で診断計画を作る場合（ZIP利用時は不要）

リポジトリ直下で、元の再学習フォルダを指定します。

```bash
python diagnose_retraining.py --reference-run paper_full_results/published_top15_matched --output diagnostic_runs/published --plan-only
python diagnose_retraining.py --reference-run paper_full_results/retrain_runs/pooled_top15_updated --output diagnostic_runs/previous --plan-only
```

出力フォルダは新規である必要があります。作成後は出力内の `diagnose_retraining.py` と `--resume` を使用します。
