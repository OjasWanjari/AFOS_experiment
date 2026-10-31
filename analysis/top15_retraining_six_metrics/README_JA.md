# Top 15：学習・検証・テストのaccuracyとloss

15関数それぞれについて6パネル（2行×3列）の図を作成。計15図・90パネル。保存済み再学習結果を使用し、追加学習は行っていない。

| | 学習用 | 検証用 | テスト用 |
|---|---|---|---|
| 上段 | accuracyのエポック推移 | accuracyのエポック推移 | 最終accuracyをシード別に表示 |
| 下段 | lossのエポック推移 | lossのエポック推移 | 最終lossをシード別に表示 |

学習・検証パネルは横軸がepoch 1〜50、5シードを重ねた折れ線。テストは成功時の50エポック終了後に1回だけ評価されており、各エポックの値は存在しない。そのためテストパネルのみ横軸をseed 0〜4にして、最終評価値を点と数値で表示。点同士は結ばない。
テストの値を1〜50エポックに複製したり、補間して架空の曲線にすることはしていない。

accuracyは%表示、lossはcategorical cross-entropy。全図でaccuracyの縦軸0〜100%、lossの縦軸0〜2を統一。線形軸を使用。
候補番号は前回と同じ統合Top 15の選抜順位。候補7と10は数学的に同一だが保存IDどおり別図。

## 失敗の扱い

候補13 seed 1はepoch 50で学習lossと検証lossがNaNになったため欠損として表示。同じepochの有限なaccuracyは残し、失敗epochの×印を付けた。
この試行のテストは実行されていないため、テストパネルにNot evaluatedと表示。0へ置換しない。
テスト評価は75試行中74試行分。学習・検証履歴は75試行×50epochの3,750記録。

## 出力と元データ

- [全15関数の閲覧ページ](index.html)
- `candidate_01_six_metrics.png`〜`candidate_15_six_metrics.png`：各関数6パネル。SVGも同梱。
- `train_validation_by_epoch.csv`：学習・検証の各epoch指標。accuracyは0〜1。NaNは空欄。
- `final_test_by_seed.csv`：各候補・各seedの最終test値。accuracyは0〜1。未評価は空欄。
- `audit.json`：入力ハッシュ・照合結果。
- `plot_metrics.py`：再生成スクリプト（PythonとMatplotlib）。

元データ：`paper_full_results/retrain_runs/pooled_top15_updated`。各試行の採用attemptはresults/retraining_results.csvに従う。
計画・分割ハッシュ、Top 15の一致、試行情報、epochの連番、履歴最終値とresult.jsonの一致、test値と集計CSVの一致を確認した。
再生成時には元の結果フォルダと前回Top 15選択CSVも必要。
