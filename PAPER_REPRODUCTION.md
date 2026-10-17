# 論文のAFOS探索・再学習を実行する

`run_paper.py` は `paper_parameters.json` を読み込み、論文の **CIFAR-10 / ベースCNNの探索と再学習** を実行する。旧実験の `main.py` と出力は保持する。

**複数PC・複数runの探索結果をまとめて再学習だけ実行する場合**は、新しい `retrain_paper.py` を使用する。全シードの上位15候補を固定し、各再学習を別プロセスで実行する。手順は [日本語版](RETRAIN_POOLED_JA.md) / [English](RETRAIN_POOLED_EN.md) を参照。

参照: ユーザー提供の *Activation Function Optimization Scheme for Image Classification*（arXiv:2409.04915v1）、3.1節、4.1節、Fig. 1–2、Table 2、Appendix A / Table A.1。

論文のResNet50・AlexNet・VGG16・MobileNet・CCTや、他の7データセットを使った汎化検証は別実験であり、この実装には含まれない。論文中のaccuracyを再現したという意味ではない。本番学習・結果の一致は未検証。

## 実装した条件

| 項目 | 設定 |
| --- | --- |
| 個体数 / 世代数 / 探索反復 | 30 / 40 / 5回 |
| 親の選択 | score順の上位6個体 |
| 交叉 | 一点交叉、確率80%、子は最大6個体 |
| 突然変異 | 全個体から候補を選択、一点変異、確率20%、子は最大1個体 |
| ランダム生成 | 次世代30個体に不足する分、最低17個体 |
| 単項 / 二項関数 | 論文3.1.1節の19 / 11種類 |
| 探索時学習 | 最大15エポック、バッチ64 |
| 最適化 | SGD、学習率0.01、momentum 0.9 |
| 初期化 / パディング | he_uniform / same |
| 損失 | categorical cross-entropy |
| 打ち切り | 損失・評価値のNaN/Inf、または1エポック後のValidation accuracyが0.25以下 |
| fitness | Validation accuracy / Validation loss |
| CIFAR-10分割 | Train 45,000 / Validation 5,000 / Test 10,000 |
| 探索後 | Validation accuracy順の上位15関数を50エポックで最初から学習、各5回 |
| 比較対象 | ReLU、Swish、Mish、GELU、SERF、ELUも同じCNN・学習条件で各5回 |

CNNはTable 2に合わせる。各Convの直後に探索関数を適用するパラメーターなしの層を置く。

```text
32×32×3
Conv(28,3×3) → AF → Conv(32,3×3) → AF → MaxPool(2×2) → Dropout
Conv(64,3×3) → AF → Conv(64,3×3) → AF → MaxPool(2×2) → Dropout
Conv(128,3×3) → AF → Conv(128,3×3) → AF → MaxPool(2×2) → Dropout
Flatten(2048) → Dense(10,Softmax)
```

学習可能パラメーターは306,234。中間Dense、Batch Normalization、残差接続、データ拡張は入れない。探索中にはTestのloss・accuracyを計算しない。Testは再学習終了後だけ評価し、その値で候補を選び直さない。

## 論文で未指定・矛盾する点と採用した解釈

完全に一意な再現はPDFだけではできない。次の仮定は `manifest.json` にも保存する。

- seedは探索・再学習とも0〜4、分割seedは1234、Dropout率は0.2。これらの具体値は論文に未記載。画像はfloat32にして255で割り、one-hotラベルを使う。
- 最初のConvはTable 2の **28フィルタ** を採用。32への訂正は行わない。
- 本文の「core unitsは最大2」とFig. 2(c)の二項ノード3個は一致しない。Fig. 2(a–c)の深さ2までの3形状（3/5/7遺伝子）を採用。`11^3 × 19^4 = 173,457,251` 通りという最大形状は「1.7億超」と整合する。5遺伝子は左側を合成する形状。
- 各形状を一様に選び、その中の各関数を一様に選ぶ。木全体の組合せを一様には抽出しない。
- 一点交叉は二項/単項の型を壊さない切断点を選ぶ。切断位置0も許す。異なる長さでは全体交換だけになる場合がある。この有効点制約の詳細は論文にない。旧コードのような剰余による関数番号の変換は行わない。
- 交叉は親6個体から異なる2個体を独立に3回抽出。親には重複した染色体を許す。交叉・突然変異が発火しなかった枠はランダム生成に回す。突然変異は必ず元と違う同型の関数を選ぶ。
- asin/atanhなどの定義域をクリップしない。NaN/Infが発生した個体はscore 0と失敗理由を記録する。softplusだけは数学的に同じ安定実装を使用する。
- 探索後は5回の探索全体から、完走した評価記録をValidation accuracy順に並べ、数式文字列が同じものを除いて15件選ぶ。論文では反復間の集約・重複処理が未指定。数学的に同値でも文字列が違う関数は区別される。
- 再学習も45,000枚で学習し5,000枚をValidationとして監視する。optimizer等は探索と同じ。学習中の低accuracyによる打ち切りは再学習では使わず、非有限値のみ中断する。
- 標準偏差は標本標準偏差（ddof=1）。5回のいずれかが失敗した関数は平均・標準偏差を空欄にし、成功回数を併記する。
- 再開のため個体ごとの学習seedを探索seed・世代・個体番号から固定的に生成する。ハードウェア間のビット単位の一致は保証しない。

## Windowsでの準備

このPCで確認した環境はPython 3.10、TensorFlow CPU 2.10.0 + DirectMLプラグイン。DirectMLは開発終了した旧環境であり、新しいPCのGPUとの互換性は個別に確認する。Python 3.11以降やKeras 3をこの手順で使わない。

```powershell
py -3.10 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-paper-windows.txt
.\.venv-gpu\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

仮想環境を別PCからコピーせず作り直す。LinuxのCUDA環境は別の依存構成が必要。`run_paper.py` 自体にはWindowsの固定パスはない。

## 実行

作業ディレクトリはリポジトリのルート。まず設定確認（データ取得・学習なし）：

```powershell
.\.venv-gpu\Scripts\python.exe run_paper.py --dry-run
```

人工データでの短い動作確認（論文の実験結果ではない）：

```powershell
.\.venv-gpu\Scripts\python.exe run_paper.py --smoke --device gpu --output runs/paper_smoke
```

本番：探索6,000個体評価、その後、発見15関数×5回と既存6関数×5回の計105回を各50エポック学習する。

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --output runs/paper_full
```

出力先は**存在しない新しいディレクトリ**を指定する。指定なしでは時刻付きのディレクトリを自動作成する。既存出力先を指定すると上書きせずエラーになる。GPUが見つからなければ停止する。CPUを意図的に使う場合は `--device cpu`。

探索と再学習を分ける場合：

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --stage search --output runs/paper_full
.\.venv-gpu\Scripts\python.exe -u run_paper.py --stage retrain --resume runs/paper_full
```

中断後の再開：

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --resume runs/paper_full
```

各個体評価・再学習試行の終了時にチェックポイントを保存する。中断した試行だけ最初からやり直す。設定・ソースが変更されている場合は混在を防ぐため再開を拒否する。再開時も同じ `--config` と、動作確認なら `--smoke` を指定する。GPU・ライブラリ環境も合わせること。

リアルタイムログは端末と出力先の `run.log` に保存される。TensorFlowネイティブ側の初期化ログは端末のみに出る場合がある。

```powershell
Get-Content runs/paper_full/run.log -Wait -Tail 30
```

## 保存内容

- `manifest.json`: 実際の設定、仮定、Python・ライブラリ・デバイス情報、遺伝子番号の定義、ソースハッシュ
- `source_snapshot/`: 実行時ソースと有効設定のコピー
- `split_indices.npz`: Train/Validationの実際の画像インデックス
- `model_summary.txt`: CNN各層・パラメーター数
- `checkpoint.json`, `status.json`: 再開情報、終了状態、再学習失敗数
- `results/evaluation_results_seed_*.csv`: 探索の全評価。Test列は意図的に存在しない
- `results/best_by_seed.csv`, `top3_seed_*.csv`, `generation_history.csv`: 最良個体、上位3関数、世代履歴
- `results/selected_top15.csv`: Validation accuracyで選んだ関数と数式
- `results/retraining_results.csv`, `comparison_summary.csv`: 再学習と既存関数のtrain/validation/test成績、平均・標準偏差
- `histories/`: 個体・再学習ごとのエポック履歴。各エポック終了直後にCSVへ追記
- `histories/attempts/`: 中断した試行を再実行するときに退避する以前の履歴
- `results/events.csv`: 開始、個体評価完了、再学習完了、中断・エラー、描画開始／完了の日時付き履歴
- `results/activation_curve_samples.csv`: 最終描画に用いた活性化関数のx・yと染色体・数式
- `plots/`: 探索のbest-so-far損失グラフ、選ばれた関数の図、再学習のTest成績比較
- `plots_complete.json`: 描画済みCSVのハッシュと図の一覧（同じ結果を再開しても再描画しない）
- `models/`: 各seedの探索中最良モデル、再学習の全成功モデル

### 実行中のCSV記録と終了後の一括描画

`run_paper.py` は学習中に描画しない。各エポックのTrain/Validation loss・accuracyを
`histories/*.csv` に直ちに追記し、個体評価が終わるたびに評価一覧・best・top3のCSVを更新する。
世代終了時に世代履歴、再学習試行終了時にTestを含む成績と比較集計を更新する。
Testは探索や各エポックでは評価せず、従来どおり再学習後だけ評価する。
ミニバッチ単位の指標・画像ごとの予測は記録対象ではない。
履歴CSVでは`nan`・`inf`をそのまま残し、結果一覧では非有限値を空欄にして`status`で失敗を示す。

標準の`--stage all`では**全seedの探索・発見関数の再学習・既存関数との比較がすべて完了した後に一度だけ**、
保存したCSVを読み込んで図をまとめて作成する。例外や中断では描画しない。
`--stage search`を明示した場合は探索のみが今回の実行範囲なので、全探索終了後に一括描画する。
その後`--stage retrain`を実行すれば全再学習終了後に更新結果を一括描画する。
同じ完了済み結果を`--resume`しても図が揃っていれば再描画しない。
描画中の失敗や図ファイルの削除があれば、再開時に描画を再試行する。

CSVは成績・履歴用で、モデルはSavedModel、再開情報と設定はJSONのまま併存する。
中断した試行はエポック途中からではなく最初から再学習する。変更前の実行結果は変更しない。
ソースハッシュが変わるため、旧版で始めた実行をこの新版でそのまま再開することはできない。

新しい染色体番号は旧 `activation.py` と互換ではない。旧描画スクリプトで解釈しないこと。保存モデルは次のように読む。

```python
import tensorflow as tf
import paper_afos.model  # registers custom activation layers/functions
model = tf.keras.models.load_model("runs/paper_full/models/discovered_01_seed_0")
```

検証：

```powershell
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -p test_paper.py -v
```

## 実装時の検証結果

- 13件の自動テストが成功：候補関数全種類の数値、CNNの層構成、GAの型と個体数、Validation選択、打ち切り、モデル再読込、中断後の再開と連続実行の一致、エポックCSVの即時記録・中断履歴保持、完了後描画・重複描画抑止・描画失敗時の再試行。
- CSV逐次記録版のCPU人工データ試験で探索6個体・再学習2試行を完了し、最後にのみ描画。完了済み実行の再開では描画がスキップされることも確認。
- CPUの人工データ試験で探索・再学習・保存が完了。
- GTX 1080 Ti / DirectMLの人工データ試験で探索を完了し、別プロセスで再開して再学習とReLU比較を完了。
- 本番CIFAR-10の6,000評価・50エポック再学習はまだ実行していない。人工データのaccuracyは研究結果として使用しない。
