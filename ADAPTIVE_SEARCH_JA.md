# fitnessの改善率に応じた交叉・突然変異確率

[English version](ADAPTIVE_SEARCH_EN.md)

`paper_parameters_adaptive.json` を指定すると、各seedの最高fitnessの改善率から、
次世代を生成する交叉確率と突然変異確率を計算します。これは今回追加した実験方式で、
元の論文の固定確率方式とは区別します。改善0%で交叉100%・突然変異80%、
1%以上で元の交叉80%・突然変異20%に戻します。性能向上はまだ検証していません。

## 同じ条件で比較する

`paper_parameters.json` は固定確率（交叉0.8・突然変異0.2）のままです。
現在の標準設定と適応型設定の差は `search.adaptive_probabilities` の追加だけです。
30個体、40世代、seed 0〜4、親6個体、交叉の子は最大6個、突然変異の子は最大1個、
ランダム個体は最低17個、CNN、データ分割、学習条件、fitnessの計算、再学習条件は同じです。
実行環境も前回と揃えてください。同じseedでも、確率が変われば生成個体やその後の乱数列は変わります。

## 前回のコードから変更した箇所

比較元は、**前回のAFOS実験で実際に使われ、結果フォルダーに保存された `source_snapshot`** です。
今回のfitness改善率の図に使ったseed 0〜4（seed 4は更新後）の記録を対象に、以下を照合しました。
Gitの直前コミットや、会話中の途中の適応案を比較元にはしていません。
以下では、今回の適応化で追加・変更した処理を説明します。

| 図に採用したseed | 実験結果フォルダー（この下の `source_snapshot/` を比較） | 保存された実行設定のsearch seeds |
| --- | --- | --- |
| 0 | `paper_full_results/paper_full_seeds0/` | `[0]` |
| 1 | `paper_full_results/paper_full_seeds1/` | `[1]` |
| 2 | `paper_full_results/paper_full_seeds2/` | `[2, 3, 4]` |
| 3 | `paper_full_results/paper_full_seeds3 and 4/` | `[3, 4]` |
| 4（更新後） | `paper_full_results/paper_full_seeds4/` | `[4]` |

各フォルダーの `manifest.json` に記録された `source_sha256` と、保存コードの実際のSHA-256を照合しました。
対象は `run_paper.py` と `paper_afos/{__init__,model,recording,search}.py` の5ファイル×5フォルダーで、
**25件すべて一致**しました。実験記録とスナップショットは変更していません。
これらの結果フォルダーのパスはローカルの実験記録の所在を示し、このブランチへの結果データの追加を意味しません。

以下の「旧行」は更新後seed 4のスナップショット内の行番号です。seed 0・3の `run_paper.py` は
このファイルと同じSHA-256です。「新行」のリンクは現在のコードです。

### 今回の適応化で追加・変更した処理

| ファイル・場所 | 前回 | 今回の変更 |
| --- | --- | --- |
| [run_paper.py：新21行目](run_paper.py#L21) | 旧20〜22行目には適応制御のimportなし | `generation_probabilities` と `validate_adaptive_config` を追加 |
| [run_paper.py：load_config()、新82行目](run_paper.py#L82) | 旧66行目からの関数は固定確率など既存設定だけを検証 | 適応制御の世代数・改善基準・確率上限も検証 |
| [run_paper.py：search()、新177行目](run_paper.py#L177) | 旧199〜200行目は世代の最高成績を記録するだけ | 同じseedの世代履歴から改善率と次世代の確率を計算 |
| [run_paper.py：新180行目](run_paper.py#L180) | 旧202〜203行目で `next_population()` に固定設定 `s` を渡す | `s` をコピーし、その世代の確率だけ差し替えた `generation_config` を渡す |
| [run_paper.py：新185行目](run_paper.py#L185) | 旧204行目では生成個体数だけを世代履歴に追加 | 改善率・確率・生成個体数を出力。世代履歴にも保存 |
| [run_paper.py：新193行目](run_paper.py#L193) | 旧206行目以降の最終世代処理には適応情報なし | 次世代を作らない最終世代は確率・次世代番号を空欄にする |
| [paper_afos/adaptive.py：5行目](paper_afos/adaptive.py#L5) | ファイルなし | `validate_adaptive_config()` を新設 |
| [paper_afos/adaptive.py：34行目](paper_afos/adaptive.py#L34) | ファイルなし | `generation_probabilities()` を新設。改善率、初期3世代、seedの分離、0除算対策を実装 |
| [paper_parameters_adaptive.json：22行目](paper_parameters_adaptive.json#L22) | 実験時の設定は交叉0.8・突然変異0.2の固定 | 現在の標準設定をコピーし、`search.adaptive_probabilities` を追加 |
| [tests/test_adaptive_search.py](tests/test_adaptive_search.py) | ファイルなし | 確率計算、個体数、固定方式との一致、中断再開などのテストを追加 |
| [RUN_GUIDE_JA.md](RUN_GUIDE_JA.md) / [RUN_GUIDE_EN.md](RUN_GUIDE_EN.md) | 適応型の案内なし | 適応型説明書へのリンクと、必要な `adaptive.py` の案内を追加 |
| [ADAPTIVE_SEARCH_JA.md](ADAPTIVE_SEARCH_JA.md) / [ADAPTIVE_SEARCH_EN.md](ADAPTIVE_SEARCH_EN.md) | ファイルなし | この変更の仕様・差分・実行手順を日本語と英語で追加 |

中心となる差分は、次世代生成へ渡す設定です。以下は `search()` 内の抜粋です。

変更前（更新後seed 4の `source_snapshot/run_paper.py`、旧202〜203行目。改行のみ整理）：

```python
state["population"], counts = next_population(
    state["population"], [r["score"] for r in done], s,
    random.Random(training_seed(seed, generation, 999999)),
)
```

変更後（確率の計算は世代終了時、次世代生成は最終世代以外で実行）：

```python
entry.update(generation_probabilities(
    s, state["generations"], seed, generation, entry["best_score"],
))
if generation < s["generations"]:
    generation_config = dict(
        s,
        crossover_probability=entry["crossover_probability"],
        mutation_probability=entry["mutation_probability"],
    )
    state["population"], counts = next_population(
        state["population"], [r["score"] for r in done], generation_config,
        random.Random(training_seed(seed, generation, 999999)),
    )
```

`paper_afos/search.py`、`paper_afos/recording.py`、`paper_afos/__init__.py` は、
前回の5フォルダーすべての保存コードと、改行コードを正規化したテキストが一致します。
したがって交叉・突然変異そのもの、親選択、ランダム補充、fitness計算、エポック記録は同じです。
`run_paper.py` は `adaptive.py` を常にimportするため、固定確率で実行する場合も新しいファイルを含めて取得してください。

## 計算方法

fitnessは従来と同じ `validation_accuracy / validation_loss` です。
世代g終了時、そのseedの第1〜g世代で記録されたscoreの最大値を `B_g` とします。
学習が早期打ち切りになった個体も、既存のscoreの定義に従って含めます。
非有限値で失敗した個体のscoreは従来どおり0です。testの指標は使いません。

初期設定の計算式は次のとおりです。

```text
r = (B_g - B_(g-3)) / max(B_(g-3), 1e-12)
q = min(1, max(0, r / 0.01))

次世代の交叉確率     = 1.0 - 0.2 × q
次世代の突然変異確率 = 0.8 - 0.6 × q
```

`r` は3世代の区間全体の相対改善率です（1世代あたりの平均ではありません）。
累積最高fitnessを使うので負にはなりません。現在の世代のfitnessが下がった場合も、
過去最高からの更新がなければ停滞として扱います。

| 3世代前からの改善率 | 交叉確率 | 突然変異確率 |
| --- | --- | --- |
| 0% | 1.00 | 0.80 |
| 0.25% | 0.95 | 0.65 |
| 0.5% | 0.90 | 0.50 |
| 0.75% | 0.85 | 0.35 |
| 1%以上 | 0.80 | 0.20 |

第1世代は従来どおり全個体をランダム生成します。第1〜3世代終了時は比較対象がないため、
交叉0.8・突然変異0.2で次世代を生成します。最初の適応は第4世代終了時で、
`B_4` と `B_1` を比較し、第5世代の生成に適用します。seedが変わると履歴もリセットします。
参照fitnessが0の場合は分母をepsilonに置き換えます。現在も0なら停滞、
正のfitnessを初めて発見すればその相対改善率に応じて確率を計算します。

上限数は変えていないため、突然変異確率0.8でも生成する突然変異個体は0または1個です。
交叉は3回の抽選で各2個生成します。改善0%では交叉が必ず3回実行され、交叉の子は6個です。
この場合、親6個・交叉6個・突然変異0または1個・ランダム18または17個で合計30個になります。
改善率が上がって両方の確率が下がると、ランダム個体の期待数は増えます。
交叉・突然変異の確率は独立で、合計を1にする制約はありません。

## 変更できる設定

`paper_parameters_adaptive.json` の `search.adaptive_probabilities` を編集します。

| キー | 初期値 | 意味 |
| --- | --- | --- |
| `enabled` | `true` | `false` で固定確率に戻す |
| `window_generations` | `3` | 何世代前の累積最高fitnessと比較するか |
| `improvement_target` | `0.01` | 元の確率に戻る改善率。0.01は1% |
| `max_crossover_probability` | `1.0` | 停滞時の交叉確率。下限は従来の `crossover_probability` |
| `max_mutation_probability` | `0.8` | 停滞時の突然変異確率。下限は従来の `mutation_probability` |
| `epsilon` | `1e-12` | 0除算を防ぐ分母の下限 |

毎世代、改善率から直接確率を決定します。前世代の確率に一定量を足す方式ではありません。
最高fitnessの更新には学習のばらつきも影響するため、この制御だけで真の性能改善を保証はしません。

## 実行

まず設定を確認します。学習やデータの読み込みは行いません。

```powershell
.\.venv-gpu\Scripts\python.exe run_paper.py --config paper_parameters_adaptive.json --dry-run
```

探索のみを、新しい出力先で実行する例です。

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --output runs/paper_adaptive_fitness
```

探索から再学習まで続ける場合は `--stage search` を外します。
適応型の実行を中断した後は、同じ設定・ソースで再開します。

```powershell
.\.venv-gpu\Scripts\python.exe -u run_paper.py --config paper_parameters_adaptive.json --stage search --device gpu --resume runs/paper_adaptive_fitness
```

前回の固定確率の結果をこのコードで再開して適応型に切り替えず、新規実験として開始してください。
既存の再開チェックは設定・ソースの変更を拒否します。過去の実行を再開する場合は、
その実行の `source_snapshot` と設定を使用してください。
標準の `--smoke` は1世代だけなので、確率の適応までは通りません。

## 記録と実装

`results/generation_history.csv` と `checkpoint.json` の世代履歴に、次の値を保存します。

- `best_score`：その世代の最高fitness。
- `best_so_far_score`：そのseedでの累積最高fitness。
- `fitness_reference_generation` / `fitness_reference_score`：比較元の世代と累積最高fitness。
- `fitness_relative_improvement` / `improvement_fraction`：改善率rと、0〜1に収めたq。
- `probability_mode`：`fixed`、`warmup`、`adaptive`、最終世代は `search_complete`。
- `next_generation`：記録した確率・生成数を使った次世代の番号。
- `crossover_probability` / `mutation_probability`：その次世代の生成に実際に使用した確率。
- `parents` / `crossover` / `mutation` / `random`：次世代に入る実際の個体数。

例えば世代4の行の確率・個体数は世代5を生成した値です。最終世代には次世代がないので、
確率と次世代番号は空欄です。改善率と確率は `run.log` にも出力します。
再開時は保存された世代別scoreから同じ確率を再計算でき、別の制御状態は必要ありません。

- `paper_afos/adaptive.py`：設定検証とfitness改善率からの確率計算。
- `run_paper.py`：世代終了後の計算、次世代生成への反映、記録。
- `paper_afos/search.py`：既存の個体生成処理をそのまま使用。
- `tests/test_adaptive_search.py`：改善率、設定検証、固定方式との一致、中断再開のテスト。

学習なしのテスト：

```powershell
python -m unittest discover -s tests -p test_adaptive_search.py -v
```

既存の個体生成・記録処理も含めた確認（計16件）：

```powershell
python -m unittest tests.test_adaptive_search tests.test_paper.SearchTests tests.test_paper.RecordingTests -v
```

16件のテストと `--dry-run` は通過済みです。学習処理を代替したテストで、中断位置が
個体評価中の場合と世代更新の保存直前の場合の両方で、再開後の状態・世代CSVが連続実行と一致することを確認しています。
本番学習での精度・fitnessの改善を検証した結果ではありません。
