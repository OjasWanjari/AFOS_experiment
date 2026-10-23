# 論文のTop 15を今回の候補と同じ条件で評価する

`retrain_published_top15.py` は、提供された論文 **arXiv:2409.04915v1、16ページ、表3** の15関数を固定して、**各seed 0〜4で50エポック、合計75試行**を行います。新しい探索や候補の再選択はしません。既存の6基準関数は再実行せず、前回の結果と比較します。

**配布済みの実行用ZIPを使う場合は「A. ZIPから実行する」だけで開始できます。** ZIPには作成済みの75試行計画・保存コード・データ分割・日英の手順書が入っています。以前の探索結果や再学習結果を追加コピーする必要はありません。「B」は、別の参照元から計画を新しく作る場合の手順です。

## A. ZIPから実行する（今回はこちらを使用）

### A-1. 元のLinux PCにZIPを用意する

[実行用ZIPのGitHubページ](https://github.com/OjasWanjari/AFOS_experiment/blob/codex/pooled-retraining/retrain_runs/published_top15_matched_bundle.zip)を開き、**Download raw file**から `published_top15_matched_bundle.zip` を保存します。GitHubの「Code → Download ZIP」で取得するリポジトリ全体のZIPとは別です。Windowsでダウンロードした場合は、元のLinux PCにファイルを転送してください。

以下は、元のLinux PCの `/home/victor/afos_published_top15_eval` に新しく展開する例です。既存の実験フォルダへの上書きを避けるため、まだ存在しないフォルダ名を使います。

```bash
mkdir /home/victor/afos_published_top15_eval
cd /home/victor/afos_published_top15_eval
```

このフォルダへZIPをコピーします。Linux PCで直接ダウンロードする場合は、次のコマンドでも取得できます（非公開リポジトリで認証が必要な場合はブラウザから取得してください）。

```bash
curl -fL -o published_top15_matched_bundle.zip "https://github.com/OjasWanjari/AFOS_experiment/raw/refs/heads/codex/pooled-retraining/retrain_runs/published_top15_matched_bundle.zip"
```

### A-2. 前回の仮想環境を有効化し、ZIPを展開する

```bash
source /home/victor/myafos/bin/activate
python -m zipfile -e published_top15_matched_bundle.zip .
cd AFOS_published_top15
```

仮想環境の場所が異なる場合は、前回の再学習で使った環境のパスに置き換えます。ZIPにはPython・TensorFlow・CUDAやCIFAR-10データ本体は含まれません。CIFAR-10は既存のKerasキャッシュを使い、未保存なら学習開始時にダウンロードされます。

展開後は次の構成です。

```text
/home/victor/afos_published_top15_eval/
  published_top15_matched_bundle.zip
  AFOS_published_top15/                  ← 以後のコマンドを実行する場所
    retrain_published_top15.py
    retrain_paper.py
    run_paper.py
    paper_afos/
    RETRAIN_PUBLISHED_TOP15_JA.md
    RETRAIN_PUBLISHED_TOP15_EN.md
    retrain_runs/
      published_top15_matched/           ← 作成済み計画・結果保存先
        plan.json
        plan.sha256
        selected_top15.csv
        split_indices.npz
        reference_audit.json
        source_snapshot/
        results/
        status.json
```

**ZIPを展開した直後から `--resume` を付けます。** ここでのresumeは「同梱された計画を使う」という意味で、初回の学習でも必要です。`--reference-run` を指定した計画作成や「B-1・B-2」は実行しないでください。

### A-3. 同梱された計画と実行環境を確認する（学習なし）

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --plan-only
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --check-environment
```

1つ目は `75 trials, seeds 0..4, 50 epochs` と表示され、計画・コード・分割のハッシュを確認します。2つ目は `Recorded environment matches` と表示されることを確認してください。両方とも学習しません。環境が不一致の場合は、本評価を開始せず、元のPC・GPU・仮想環境を確認します。

### A-4. 1試行を実行してから、残りを続ける

まず次を実行します。これは動作確認だけの短縮学習ではなく、**EELU-4・seed 0の正式な50エポック学習**です。

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --max-trials 1
```

終了後に状態を確認します。成功した場合は `completed` が1件、`pending` が74件です。

```bash
cat retrain_runs/published_top15_matched/status.json
```

続けて残りを実行します。

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume
```

完了済みの試行はスキップされます。最初から全75試行を開始する場合は、`--max-trials 1` を付けたコマンドを省略して、このコマンドだけを実行します。上記はフォアグラウンド実行なので、学習中はターミナルを閉じないでください。

### A-5. 進捗確認・再開・結果の場所

別のターミナルから、同じ `AFOS_published_top15` フォルダに移動して確認できます。

```bash
cat retrain_runs/published_top15_matched/status.json
tail -f retrain_runs/published_top15_matched/trials/paper_01_seed_0/attempt_0001/worker.log
```

`status.json` は主に試行終了時に更新されます。実行中のエポックは `worker.log` で確認してください。上記ログは最初の試行用です。次の試行はcoordinatorが表示したログのパスに置き換えます。

再開するときは、**ZIPを展開し直さず、すでに結果が入っている同じフォルダ**を使います。新しいターミナルでは次のように環境と作業場所を戻します。

```bash
source /home/victor/myafos/bin/activate
cd /home/victor/afos_published_top15_eval/AFOS_published_top15
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume
```

中断・OOM・環境エラーなどが記録されている場合は、原因を解消したうえで最後のコマンドに `--retry-errors` を追加します。NaN/Infは数値的失敗として保存され、自動再試行しません。ロックが残った場合の対応は「B-5」を参照してください。

結果は、この展開先の `retrain_runs/published_top15_matched/` に保存されます。元の `AFOS_experiment/retrain_runs/` には保存されません。

- `results/retraining_results.csv`: 各75試行の結果。
- `results/retraining_summary.csv`: 各関数の5シード平均・標本標準偏差。
- `trials/`: 試行別ログ、履歴、環境、結果JSON、成功時のモデル。

分析用に別PCへ戻す場合は、`published_top15_matched` フォルダ全体をコピーしてください。学習を開始したフォルダへZIPを再展開すると計画・集計ファイルを上書きするおそれがあるため、再開には既存の出力を使います。

よくある操作ミス：`Output already exists` は `--resume` の付け忘れ、`retrain_published_top15.py` が見つからない場合は `AFOS_published_top15` への移動漏れを確認してください。

## 引き継ぐ条件

比較対象は `paper_full_results/retrain_runs/pooled_top15_updated` に取り込まれた再学習です。

- 参照元の `plan.json` から設定全体をコピーします。学習率0.01、SGD momentum 0.9、batch size 64、CNN、dropout、初期化などを新しく指定しません。
- **保存された `split_indices.npz` をそのままコピー**します。45,000件の訓練データ、5,000件の検証データの所属・順序を引き継ぎます。テスト10,000件は学習に使いません。
- 参照元の `source_snapshot` にあるモデル・データ読み込み・演算・学習・集計コードをコピーし、ハッシュで一致を確認します。workerの変更は環境情報の記録・照合です。
- 新規モデルをseed 0〜4で初期化し、50エポック終了時のモデルをテストします。探索時や以前のモデルの重み、検証精度最大のエポックは使いません。
- JIT無効、GPU memory growth有効、試行ごとに別プロセスという実行方式も引き継ぎます。
- NaN/Infは失敗として保存して次の試行に進みます。成功した回だけを使った正式な5回平均は出しません。

**本評価は前回実行したLinux PCの同じ仮想環境で行ってください。** 今回保存された環境はPython 3.10.12、TensorFlow 2.21.0、Keras 3.12.4、NumPy 2.2.6、scikit-learn 1.7.2、RTX 4080 SUPERです。新しい環境を推測で作り直したり、ライブラリを更新したりせず、既存の環境を使います。

環境チェックはPython・主要パッケージのバージョン、OS系統、TensorFlowのCUDA/cuDNNビルド情報、GPU機種・台数、XLAフラグ、保存された実行設定を照合します。不一致ならデータ読み込み・学習前に停止します。元のGPUドライバのバージョンやすべての環境変数は保存されていないため、完全な環境同一性やGPU計算のビット単位の再現性まで保証するチェックではありません。

## B-1. ZIPを使わずに新しい計画を作る場合：必要なファイル

**このB-1・B-2は、同梱計画を使うZIP実行では不要です。** 別の参照元から計画を新しく作る場合にのみ実行します。

更新済みリポジトリに加え、前回の再学習出力を用意します。

```text
retrain_published_top15.py
retrain_paper.py
run_paper.py
paper_afos/                         # フォルダ全体
前回の再学習出力/
  plan.json
  plan.sha256
  split_indices.npz
  source_snapshot/                 # 保存コード全体
  trials/.../attempt_.../
    result.json
    environment.json
    worker.log
```

準備処理は成功した試行の保存環境を照合します。参照元の試行フォルダを残してください。重みファイル `model.keras` はこの準備に不要です。過去の探索CSVから候補を選び直す処理もありません。

## B-2. 計画を作る（学習なし）

すべてリポジトリ直下で実行します。Windowsに取り込んだ結果を参照する例：

```powershell
python retrain_published_top15.py --reference-run paper_full_results/retrain_runs/pooled_top15_updated --output retrain_runs/published_top15_matched --plan-only
```

Linux PCでは、元の出力が `retrain_runs/pooled_top15_updated` にある場合、次のようにします。

```bash
cd /home/victor/AFOS_experiment
source /home/victor/myafos/bin/activate
python retrain_published_top15.py --reference-run retrain_runs/pooled_top15_updated --output retrain_runs/published_top15_matched --plan-only
```

`--reference-run` は**前回の再学習出力**のパスです。探索出力や新規出力のパスを指定しないでください。`--output` は存在しない新規フォルダを指定します。作成済みなら `--resume` を使います。

この段階はTensorFlowを読み込まず、GPU学習・データダウンロードを行いません。設定や数式は `plan.json` と `selected_top15.csv` に固定されます。

Windowsで作成した `retrain_runs/published_top15_matched` を**フォルダ全体**Linux PCへコピーしても構いません。コピー後の実行には元の参照フォルダは不要です。計画内の参照元の絶対パスは来歴記録であり、実行時の入力パスではありません。

## B-3. 元のPCで環境を確認する（学習なし）

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --check-environment
```

`Recorded environment matches` と表示されれば、保存された項目が一致しています。結果は `environment_preflight.json` に記録します。この確認ではデータを読み込まず、学習しません。

不一致の場合は表示された項目を確認し、元のPC・元の仮想環境・同じGPUを使ってください。一致チェックを回避して別環境で実行した結果を、同じ条件の比較として扱わないでください。

## B-4. まず1試行、その後に残りを実行

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --max-trials 1
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume
```

最初の試行は `paper_01_seed_0`、すなわちEELU-4です。2番目のコマンドは完了済み試行を飛ばして続行します。最初から全75試行を行う場合は2番目だけで構いません。各workerでも毎回、学習前に環境を再確認します。

`--resume` に `--reference-run` を付けないでください。`--resume` は保存済みの候補・コード・分割・設定を使います。

## B-5. 失敗時と再開

通常の中断・OOM・実行環境エラーでは、原因解消後に以下を使います。

```bash
python retrain_published_top15.py --output retrain_runs/published_top15_matched --resume --retry-errors
```

- 失敗試行は同じseedで第1エポックから再実行し、以前のattemptを残します。
- `nonfinite` / `nonfinite_test` は確定した数値的失敗として保持し、このオプションでも再実行しません。
- 強制終了で `run.lock` が残った場合は、coordinatorとworkerの両方が停止したことを確認してから、そのロックファイルのみを削除します。
- 同じ出力フォルダで2つの実行を同時に起動しないでください。
- 終了コード2は数値的失敗が含まれる場合などに使われます。`status.json` で完了・失敗・未実行の件数を確認します。

## 関数番号と数式の扱い

`paper_01`〜`paper_15` は表3の掲載順です。前回の `discovered_01`〜`discovered_15` とは別の関数群です。

| ID | 論文名・数式 | 論文の平均±SD（%） |
|---|---|---:|
| paper_01 | EELU-4: erf(x) × softplus(x) | 82.74 ± 0.14 |
| paper_02 | EELU-2: −x × erf(exp(−x)) | 82.67 ± 0.30 |
| paper_03 | EELU-1: x × erf(exp(x)) | 82.66 ± 0.29 |
| paper_04 | EELU-3: x × erf(exp(x))² | 82.48 ± 0.61 |
| paper_05〜paper_15 | 表3の5〜15番目。全式は selected_top15.csv に保存 | 同CSVに記録 |

EELU-2は表3・4の**数式**を採用します。表4のTensorFlow実装欄はEELU-1と重複しているため採用しません。

EELU-3は既存の演算木で `(x * erf(exp(x))) * erf(exp(x))` と表します。数式としては `x * erf(exp(x))**2` と同じですが、浮動小数点演算の順序は異なり得ます。softplusは前回の候補と同じ `tf.nn.softplus` を使います。expのクリッピング、勾配クリッピング、学習率変更などは追加しません。

## 出力と比較

- `selected_top15.csv`: 固定した論文15関数、染色体、表3の掲載順、論文の報告値。
- `reference_audit.json`: 参照元の計画ハッシュ、維持したコードのハッシュ、照合した環境記録。
- `results/retraining_results.csv`: 75試行の状態・テスト精度。精度は0〜1。
- `results/retraining_summary.csv`: 5回すべて成功した関数の平均と標本標準偏差。未完了の関数は平均・SDが空欄。
- `trials/<ID>/attempt_0001/`: ログ、環境照合、エポック履歴、結果、成功時のモデル。

今回の15候補の結果は前回の `pooled_top15_updated/results/`、論文15関数の新しい結果は `published_top15_matched/results/` です。この2つを同じ再学習seedで比較してください。論文掲載の数値と、今回新たに実行した論文関数の数値は区別します。

同じ式の重複（前回の候補07・10）と、前回の候補13の1失敗・ELUの5失敗は引き続き明記してください。新しい75試行を実行しても、それらの失敗が埋まるわけではありません。
