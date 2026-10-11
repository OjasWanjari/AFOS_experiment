# AFOS：再学習のみの再開と保存エラー・OOMへの対処

作成日：2026年10月11日。今回添付されたチェックポイント、ソース、ログ、学習履歴を確認した結果に基づく手順です。コマンドは元のLinux GPUマシン上のBash用です。

## 1. 結論と保存状態

**探索をやり直さず、`--stage retrain` で再学習だけを実行できます。** 先にモデル保存先を修正し、コード変更の履歴を残した復旧用コピーを作ります。

| 確認項目 | 今回の保存ファイルで確認した状態 |
| --- | --- |
| 探索 | 1,200 / 1,200件、`search_complete: true` |
| 探索結果の内訳 | completed 600件、early_stopped 161件、nonfinite 439件 |
| 選択された活性化関数 | 15件 |
| 再学習の完了記録 | 0件、`retraining_complete: false` |
| 最初の再学習 | discovered_01、seed 0、50エポックの履歴あり |
| 今回の停止原因 | 拡張子のないパスへの `model.save()` |
| 記録された環境 | Python 3.10.12、TensorFlow 2.21.0、Keras 3.12.4 |
| 再学習の予定 | 発見した関数15件＋ベースライン6件、各seed 0、各50エポック |

`events.csv` では、2026年10月11日10:34:14（日本時間）に最初の再学習の終了、10:34:17に保存エラーが記録されています。学習自体の途中で停止したケースではありません。

ただし、このプログラムはモデル保存の後に再学習結果をチェックポイントへ追加します。保存前にプロセスが終了したため、**再開すると discovered_01 はエポック1から学習し直します。** CSVには重みやオプティマイザ状態がないため、50エポック目から復元できません。`best_search_seed_0.keras` は探索用であり、失敗した再学習モデルの代用にはなりません。

## 2. 今回の探索結果について確認できたこと

- 評価番号は1〜1,200まで連続しています。
- 全1,200件の履歴CSVについて、エポック数と最終損失・精度がチェックポイントと一致しました。
- `epochs_completed=0` の評価は0件でした。
- `events.csv` に `training_failed` はなく、`run.log` に例外捕捉メッセージ `An error occurred in train_candidate:` はありませんでした。
- 添付された5つのPythonソースのSHA-256と設定は、`manifest.json` と一致しました。

探索コードには、あらゆる通常の例外を `nonfinite` に変換する処理が残っています。しかし、今回の記録ではその処理によって探索候補がスキップされた形跡はありません。439件の `nonfinite` をすべてOOMと解釈してはいけません。この問題を理由に今回の探索を破棄する必要は認められません。

これは保存記録の整合性確認であり、別PCで報告されたOOMや、記録されていないGPU状態まで否定するものではありません。

## 3. 保存先の正しい修正

`retrain()` 内では、元の246行目に相当する保存処理を次にします。

```python
model.save(
    str(root / "models" / f"{name}_seed_{seed}.keras"),
    include_optimizer=True,
)
```

一方、`search()` 内の探索最良モデルは次の名前で正しいです。

```python
model.save(
    str(root / "models" / f"best_search_seed_{seed}.keras"),
    include_optimizer=True,
)
```

**`best_search_seed_{seed}.keras` を再学習側に使わないでください。** 同じseedの各候補が同じファイルに保存され、探索時の最良モデルも上書きされる可能性があります。再学習では `discovered_01_seed_0.keras`、`discovered_02_seed_0.keras`、`relu_seed_0.keras` のように分かれる必要があります。

`.keras` はモデルと学習状態を保存する形式です。今回の用途では `model.export()` への置き換えは必要ありません。[Keras公式保存仕様](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/)

## 4. ソースの一致チェックを維持して復旧する

元のスクリプトは、コードを変更すると `Source changed since this run` で再開を拒否します。保存先だけの修正でも同じです。チェックを削除したり、元のmanifestのハッシュを履歴なしで書き換えたりしないでください。

同梱・別途提供の **`prepare_retrain_recovery.py`** を使います。この補助スクリプトは次の処理を行います。

1. 探索完了、選択候補、元の設定とソースハッシュを確認する。
2. 実験フォルダ全体を、新しい別フォルダへコピーする。
3. コピー側の保存先を修正する。指定した場合だけXLAを無効にする。
4. 元のmanifest・チェックポイント・ソースを `retrain_recovery_provenance/` に保存する。
5. コピー側のmanifestに変更履歴と新しいソースハッシュを記録する。

**元の実験フォルダは変更せず、学習も開始しません。** 元の検索結果が新しいコードで作られたと偽装するのではなく、どの時点でコードを変更したかを記録します。現在のプロジェクトで行った他の編集は取り込まず、保存済み `source_snapshot` を基にします。

### 必要なフォルダ構成

元のLinux環境にある完全な実験フォルダを優先してください。

```text
paper_full/
├── checkpoint.json
├── manifest.json
├── status.json
├── run.log
├── split_indices.npz
├── model_summary.txt
├── source_snapshot/
│   ├── run_paper.py
│   ├── paper_parameters.json
│   └── paper_afos/
│       ├── __init__.py
│       ├── model.py
│       ├── recording.py
│       └── search.py
├── results/          # events.csv、評価CSV、selected_top15.csvなど
├── histories/        # search_0_*.csv、discovered_01_0.csvなど
└── models/           # best_search_seed_0.kerasなど
```

今回の添付ファイルはWindowsのTemp直下に並んでいます。そのままでは上の構成ではありません。**Tempフォルダ全体を `--source` に指定しないでください。** 元のLinuxフォルダがない場合は、新しい専用フォルダに上の対応でファイルをコピーして構成を復元してください。元のファイル名・内容・改行を保ち、実験以外のTempファイルをコピーしないでください。

### 準備コマンド

補助スクリプトを `/home/victor/AFOS_experiment/prepare_retrain_recovery.py` にコピーします。実験への書き込みを止め、全体をコピーできる空き容量を確保してから実行します。

```bash
source /home/victor/myafos/bin/activate
cd /home/victor/AFOS_experiment

python prepare_retrain_recovery.py \
  --source /home/victor/AFOS_experiment/runs/paper_full \
  --destination /home/victor/AFOS_experiment/runs/paper_full_retrain_recovery
```

これは保存先だけを直す手順です。コピー先は未作成のパスにしてください。既に復旧用コピーを作った場合は、準備処理を繰り返さず、次の再開コマンドを使います。

OOM対策として最初からXLAも無効にする場合は、**上の準備コマンドに `--disable-xla` を追加**してください。これは任意の実行設定変更であり、保存エラー修正そのものには不要です。

## 5. 再学習だけを開始する

準備が成功したら、新しいPythonプロセスで実行します。

```bash
RUN=/home/victor/AFOS_experiment/runs/paper_full_retrain_recovery

python "$RUN/source_snapshot/run_paper.py" \
  --config "$RUN/source_snapshot/paper_parameters.json" \
  --resume "$RUN" \
  --stage retrain \
  --device gpu
```

- フォルダを別名にした場合は `RUN` を変更します。補助スクリプトが出力するコマンドも利用できます。
- `--output` と `--smoke` は付けません。今回は本実験です。
- 必要なら `--quiet` でエポック表示を抑制できます。
- 同じ結果フォルダに対して複数プロセスを同時に実行しないでください。
- 元の仮想環境を使ってください。ソースハッシュ検査はパッケージの同一性までは保証しません。

今回の最初の学習表示は `Retrain discovered_01 seed=0 epochs=50` になるはずです。冒頭に `Search evaluations: 1200` と表示されても、探索のやり直しではありません。`--stage retrain` により探索処理は呼び出されません。

記録済みの候補・seedの組はスキップされます。今回は0件なので、15関数＋6ベースラインの計21試行が対象です。途中で停止しても、完了記録のない試行から同じコマンドで再開できます。既存の中断履歴は `histories/attempts/` に退避されます。

## 6. OOMへの対処と原因の切り分け

**保存時の拡張子エラーと、GPUメモリ不足は別問題です。** 今回の保存記録が示す最後の失敗は前者です。別PCでもOOMが発生したという報告だけでは、XLAのメモリリークとは断定できません。

モデルは306,234パラメータと小規模ですが、学習には中間テンソル、勾配、演算用領域、実行時リソースも必要です。元のコードは `clear_session()` と `del model` を使っていますが、それだけで長時間のメモリ増加が防げているとは確認できていません。

対処は次の順で進めます。

1. 保存先を直し、探索とは別の新しいPythonプロセスで再学習する。
2. 別ターミナルで `nvidia-smi -l 5` を実行し、開始時・候補切替時・失敗時のVRAM使用量と他プロセスを確認する。
3. 再現する場合、元の保存フォルダから別名の復旧コピーを作り、準備時に `--disable-xla` を指定して比較する。既存の復旧フォルダを上書きしない。
4. XLAを無効にしても試行を重ねるほど増える場合、候補・seedごとに子プロセスで学習し、保存・チェックポイント更新後に子プロセスを終了する実装を検討する。
5. 新しいプロセスの最初の候補でもOOMになる場合は、その試行の最大メモリ使用量とGPU空き容量を調べる。プロセス分離だけでは解決しない場合がある。

`--disable-xla` はコピー側の `model.compile(..., jit_compile=False)` を設定し、履歴を残します。速度や数値結果が変わる可能性があり、OOM解消を保証しません。補助スクリプトは候補ごとの子プロセス分離を実装していません。[Keras公式compile仕様](https://keras.io/api/models/model_training_apis/)

バッチサイズ64からの変更は実験条件の変更なので、この復旧手順には含めていません。OOMを `nonfinite`・スコア0として保存して次へ進む方法は使わないでください。評価できなかった候補と数値的に発散した候補を区別する必要があります。今回の補助スクリプトは探索の例外処理を書き換えませんが、`--stage retrain` では探索を実行しません。

## 7. 完了確認とよくあるエラー

```bash
cat "$RUN/status.json"
```

正常終了時は `status: completed`、`stage: retrain`、`search_complete: true`、`retraining_complete: true` を確認します。`failed_retraining_runs` と `results/retraining_results.csv` も確認してください。完了フラグは、全試行が良好な結果を得たことまでは意味しません。

再開コマンドを実行しても、実行中は `status.json` が以前の失敗状態や `recovery_prepared` のままの場合があります。元のrunnerは開始時には更新せず、終了・例外発生時に書き換えるためです。進捗はコンソール・`run.log`・`results/events.csv` で確認します。

| エラー | 対処 |
| --- | --- |
| Source changed since this run | 復旧フォルダの `source_snapshot/run_paper.py` を実行する。追加編集していないか確認する。 |
| Snapshot hashes do not match manifest | 準備元のsnapshotが編集済み。元の保存ソースを復元する。 |
| Resume config differs from manifest | 復旧snapshotの `paper_parameters.json` を指定する。 |
| Destination already exists | 準備済みなら再開だけを実行する。別の準備をする場合は新しい名前を使う。 |
| Search is not complete | 古い第39世代のチェックポイントではなく、今回の探索完了版を使う。 |
| No GPU detected | 元のGPU対応仮想環境とGPU認識を確認する。 |

誤った `best_search_seed_0.keras` への保存を既に成功させていた場合、上書きされた重みはファイル名修正だけでは戻りません。必要ならバックアップから復元します。今回添付されたモデルの活性化関数設定は、探索の最良スコア候補（評価422）と一致しました。ただし、これは重み全体の同一性の検証ではありません。

## 8. 検証範囲

今回の添付データの状態・ソースハッシュ・設定・全探索履歴の整合性を確認し、補助スクリプトによる保存先修正と任意のXLA無効化が添付ソースに適用できることを検証しています。元のLinux GPUでの再学習やOOM解消は未検証です。
