# 新しい統合Top 15の再学習結果

分析日：2026-10-22。追加された保存結果を検査・再集計した。追加学習や元データの変更は行っていない。

**105試行すべてに終了結果があり、99試行で50エポックの学習・テスト評価が完了した。6試行はNaNで停止した。発見15候補のうち14候補で5シードがそろい、最高平均テスト精度は83.582±0.460%だった。**

## 入力と検証

- 対象：`paper_full_results/retrain_runs/pooled_top15_updated`。新seed 4を含む全5探索から選定したTop 15と既存6関数を、各seed 0〜4で評価した。
- `selected_top15.csv`は前回の更新版分析のTop 15と完全一致。計画のハッシュ、保存コード、探索元CSV・manifest、訓練・検証分割のハッシュを照合した。
- 105件の集計行を各試行の`result.json`・学習履歴と照合し、平均・標本標準偏差を再計算した。99件は全50エポックの指標が有限で、保存モデルも99個存在する。
- 105個の異なるプロセスID。実行設定はGPU、JIT無効、memory growth有効。TensorFlow 2.21.0 / Keras 3.12.4。
- workerログのOOM該当：0試行、Traceback：0試行。layout optimizerの警告は105試行にあるが、試行終了状態とは区別する。
- `paper_full_results/runs`は別形式の探索記録で、seed 0〜2各50評価・10世代。共通15関数を各5シードで50エポック再学習した結果ではないため合算しない。

検査記録：[audit.json](audit.json)。

## 完了状況

| 対象 | 予定 | 完了 | 非有限値で停止 |
|---|---:|---:|---:|
| 発見15候補 | 75 | 74 | 1 |
| 既存6関数 | 30 | 25 | 5 |
| 合計 | 105 | 99 | 6 |

`all_evaluations_completed=false`は全件成功でないことを示す。未着手・実行中の試行はない。5回そろった候補は0/15から14/15に更新された。

## 発見関数の結果

平均±SDは同じ関数の再学習seed 0〜4に対するテスト精度。SDは標本標準偏差。候補番号は探索時の検証精度順位で、下表は再学習後の平均順。

| 平均順位 | 探索候補 | 関数 | 完了 | 平均±SD (%) |
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
| — | discovered_13 | `((atan(x)*min(x,0))+tanh(x))` | 4/5 | 未完了 |

最高候補は`(erf(x)*sigmoid(sinh(x)))`。seed 0〜4は83.82%, 83.75%, 82.77%, 83.69%, 83.88%。

完了14候補はすべてReLUの平均を上回り、うち8候補はGELUの平均を上回った。最高候補の差はReLU比+4.758ポイント、最良baselineのGELU比+1.092ポイント。これは観測差で、有意差検定による結論ではない。

候補7と10は数学的に同一だが数式文字列が異なり、別候補として実行されている。15個の数学的に異なる関数と解釈しない。

## 既存関数と論文の比較

| 関数 | 今回 平均±SD (%) | 論文 平均±SD (%) | 平均差 (pt) |
|---|---:|---:|---:|
| relu | 78.824 ± 0.601 | 81.37 ± 0.46 | -2.546 |
| swish | 81.794 ± 0.545 | 81.46 ± 0.60 | +0.334 |
| mish | 81.124 ± 0.601 | 82.23 ± 0.27 | -1.106 |
| gelu | 82.490 ± 0.448 | 81.75 ± 0.26 | +0.740 |
| serf | 80.462 ± 0.592 | 82.16 ± 0.22 | -1.698 |
| elu | 未完了 (0/5) | 82.29 ± 0.57 | — |

論文参照：Activation Function Optimization Scheme for Image Classification, arXiv:2409.04915v1, Table 6 Base CNN。両者とも完了関数は5回だが、論文と同じ分割・実行環境まで保証されない。

今回最高83.582%は論文EELU-4の82.74±0.14%より数値上0.842ポイント高い。ただし異なる関数・実行環境の比較であり、EELU-1〜4を今回の環境で再学習した結果はない。ReLU・SERF・Mishの論文との差も残るため、論文に対する優位性は未確定。

## 失敗した6試行

| 試行 | 最初の非有限値エポック | 訓練損失 | 検証損失 |
|---|---:|---|---|
| discovered_13_seed_1 | 50 | nan | nan |
| elu_seed_0 | 1 | nan | nan |
| elu_seed_1 | 1 | nan | nan |
| elu_seed_2 | 1 | nan | nan |
| elu_seed_3 | 1 | nan | nan |
| elu_seed_4 | 1 | nan | nan |

候補13のseed 1は49エポックまで有限で、50エポック目にNaNになった。50という履歴行数だけで成功と判断してはいけない。ELUは全seedで1エポック目にNaNとなった。いずれもテスト評価は保存されていない。

候補13の成功分（seed 0,2,3,4）は81.31%, 79.95%, 81.84%, 81.57%。失敗を除いた4回平均を5回の正式な平均として報告しない。

完了99試行の最低テスト精度は78.02%で、以前の有限値10%崩壊は今回の完了試行にはない。NaNの発生源となった中間活性・勾配は保存されていないため、ELUや候補13の根本原因は未確定。

## 次に確認する点

1. 候補13 seed 1とELUの失敗原因を、最初に非有限値になる演算・勾配の記録で調べる。条件を変える場合は別実験として区別する。
2. EELU-1〜4を固定比較対象として同じ環境・同じ5シードで評価し、今回の候補と直接比較する。
3. ReLUなどの基準関数が論文より低い理由と、短期探索順位から50エポック評価順位への変化を調べる。

## 元ファイル

- [試行別結果](../../paper_full_results/retrain_runs/pooled_top15_updated/results/retraining_results.csv)
- [関数別集計](../../paper_full_results/retrain_runs/pooled_top15_updated/results/retraining_summary.csv)
- [実行状態](../../paper_full_results/retrain_runs/pooled_top15_updated/status.json)
- [固定した候補](../../paper_full_results/retrain_runs/pooled_top15_updated/selected_top15.csv)
- [再検査スクリプト](analyze.py)
