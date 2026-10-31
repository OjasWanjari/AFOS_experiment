"""Plot recorded best validation accuracy and minimum loss across search evaluations."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return float(value) if value else math.nan


reference = read_csv(ROOT / "analysis/paper_full_results_updated_seed4/search_evaluations_selected.csv")
assert len(reference) == 6000
curves, summaries, sources = {}, [], []
for seed in range(5):
    folder = "paper_full_seeds3 and 4" if seed == 3 else f"paper_full_seeds{seed}"
    source = ROOT / "paper_full_results" / folder / "results" / f"evaluation_results_seed_{seed}.csv"
    rows = read_csv(source)
    compared = [r for r in reference if int(r["seed"]) == seed]
    assert len(rows) == len(compared) == 1200
    assert [int(r["evaluation_count"]) for r in rows] == list(range(1,1201))
    assert not any("f1" in key.lower() for key in rows[0])
    best_accuracy, best_loss = -math.inf, math.inf
    acc_curve, loss_curve, eligible_rows = [], [], []
    best_acc_row, best_loss_row = None, None
    for row, original in zip(rows, compared):
        assert int(row["seed"]) == seed and all(row[k] == original[k] for k in row)
        acc, loss = number(row["validation_accuracy"]), number(row["validation_loss"])
        eligible = row["status"] in ("completed", "early_stopped") and math.isfinite(acc) and math.isfinite(loss) and loss >= 0
        if eligible:
            assert 0 <= acc <= 1
            eligible_rows.append(row)
            if acc > best_accuracy:
                best_accuracy, best_acc_row = acc, row
            if loss < best_loss:
                best_loss, best_loss_row = loss, row
        acc_curve.append(best_accuracy*100 if math.isfinite(best_accuracy) else math.nan)
        loss_curve.append(best_loss if math.isfinite(best_loss) else math.nan)
    finite_acc = [x for x in acc_curve if math.isfinite(x)]
    finite_loss = [x for x in loss_curve if math.isfinite(x)]
    assert all(a <= b for a,b in zip(finite_acc,finite_acc[1:]))
    assert all(a >= b for a,b in zip(finite_loss,finite_loss[1:]))
    assert best_accuracy == max(float(r["validation_accuracy"]) for r in eligible_rows)
    assert best_loss == min(float(r["validation_loss"]) for r in eligible_rows)
    curves[seed] = {"accuracy":acc_curve, "loss":loss_curve}
    summaries.append(dict(seed=seed, total_evaluations=1200, eligible_evaluations=len(eligible_rows),
        first_eligible_evaluation=int(eligible_rows[0]["evaluation_count"]),
        best_validation_accuracy_pct=best_accuracy*100, best_accuracy_at=int(best_acc_row["evaluation_count"]),
        best_accuracy_expression=best_acc_row["activation_expression"],
        minimum_validation_loss=best_loss, minimum_loss_at=int(best_loss_row["evaluation_count"]),
        minimum_loss_expression=best_loss_row["activation_expression"], f1_availability="not_recorded"))
    sources.append(dict(seed=seed, path=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))

assert sum(s["eligible_evaluations"] for s in summaries) == 3682
for metric in ("accuracy", "loss"):
    wide = [dict(evaluation_count=i+1, **{f"seed_{s}":curves[s][metric][i] if math.isfinite(curves[s][metric][i]) else "" for s in range(5)}) for i in range(1200)]
    write_csv(OUT / f"best_{metric}_by_evaluation.csv", wide)
write_csv(OUT / "seed_summary.csv", summaries)

colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6957A5"]
styles = ["-", "--", "-.", ":", "-"]
markers = ["o", "s", "^", "D", "v"]
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":12,
    "axes.spines.top":False, "axes.spines.right":False, "svg.fonttype":"none",
    "axes.labelcolor":"#24364B", "text.color":"#24364B"})
for metric in ("accuracy", "loss"):
    fig, ax = plt.subplots(figsize=(12,7.5))
    fig.subplots_adjust(left=.10, right=.97, bottom=.24, top=.79)
    title = "Best validation accuracy across five seeds" if metric == "accuracy" else "Minimum validation loss across five seeds"
    fig.text(.10,.945,title,fontsize=22,weight="bold")
    fig.text(.10,.89,"All search candidates, cumulative best at each evaluation",fontsize=14)
    for seed in range(5):
        ax.step(range(1,1201),curves[seed][metric],where="post",color=colors[seed],linestyle=styles[seed],
                linewidth=2,label=f"Seed {seed}",marker=markers[seed],markevery=[199,399,599,799,999,1199],markersize=5)
    ax.set_xlim(0,1200)
    ax.set_xticks(range(0,1201,200))
    ax.set_xlabel("Search evaluation count (within each seed)",labelpad=10)
    ax.set_ylabel("Best validation accuracy so far (%)" if metric == "accuracy" else "Minimum validation loss so far (log scale)",labelpad=10)
    if metric == "accuracy":
        ax.set_ylim(0,100)
        inset = ax.inset_axes([.32,.10,.64,.40])
    else:
        assert all(v > 0 for s in range(5) for v in curves[s][metric] if math.isfinite(v))
        ax.set_yscale("log")
        ax.set_ylim(.45,5000)
        inset = ax.inset_axes([.32,.49,.64,.45])
    ax.grid(color="#D9E0E8",alpha=.8,linewidth=.7)
    ax.set_axisbelow(True)
    for seed in range(5):
        inset.step(range(1,1201),curves[seed][metric],where="post",color=colors[seed],linestyle=styles[seed],linewidth=1.6)
    inset.set_xlim(200,1200)
    inset.set_xticks([200,400,600,800,1000,1200])
    detail_values = [v for s in range(5) for v in curves[s][metric][199:] if math.isfinite(v)]
    margin = (max(detail_values)-min(detail_values))*.15
    inset.set_ylim(min(detail_values)-margin,max(detail_values)+margin)
    inset.set_title("Detail: evaluations 200–1,200 (linear scale)",fontsize=9,pad=5)
    inset.tick_params(labelsize=8)
    inset.grid(color="#D9E0E8",linewidth=.5)
    inset.patch.set_facecolor("#FAFCFE")
    ax.legend(ncol=5,loc="upper center",bbox_to_anchor=(.5,-.19),frameon=False,fontsize=11)
    fig.text(.10,.065,"1,200 evaluation positions per seed. Numerical failures do not update the running best; initial undefined values remain blank.",fontsize=9.4)
    fig.text(.10,.03,"Updated seed 4 is used. Accuracy and loss are optimized independently and can correspond to different candidates.",fontsize=9.6)
    for ext in ("png","svg"):
        fig.savefig(OUT / f"search_{metric}_convergence.{ext}",dpi=200,facecolor="white")
    plt.close(fig)

table = "\n".join(f"| {s['seed']} | {s['best_validation_accuracy_pct']:.2f}% | {s['best_accuracy_at']} | {s['minimum_validation_loss']:.6f} | {s['minimum_loss_at']} |" for s in summaries)
readme = f"""# 探索全体の検証精度・検証損失の推移

2026-10-31作成。新seed 4を含む5探索・全6,000評価を使用。追加学習なし。

## 定義

- 横軸：各探索seedの評価番号1〜1,200。全候補を対象とする。
- 精度：それまでに観測された検証精度の最大値（%）。
- 損失：それまでに観測された検証損失の最小値（categorical cross-entropy）。小さいほど良い。
- 有限値の検証指標を持つcompleted / early_stoppedの3,682評価を更新対象にする。数値異常の2,318評価も横軸には残すが、最良値を更新しない。
- 有効な最初の評価以前は欠損。失敗を精度0・損失0に置換しない。
- 精度と損失はそれぞれ独立に最良値を計算している。fitness最大候補の精度・損失を追跡した図ではない。
- 精度は非減少、損失は非増加となる。横ばいは観測上の更新なしを表し、最適解に到達した証明ではない。

## 損失の表示

seed 2の最初の有効評価（3回目）は検証損失3286.219482で、後半の値より非常に大きい。全範囲を省略せず示すため、損失の主図は対数軸にした。
両グラフには200〜1,200回の拡大図（線形軸）を付けた。精度の主図は0〜100%の線形軸。

| Seed | 最高検証精度 | 初到達の評価番号 | 最小検証損失 | 初到達の評価番号 |
|---|---:|---:|---:|---:|
{table}

## F1スコア

探索CSVにはF1列がなく、保存された5runのmodel.pyはいずれもmetrics=["accuracy"]のみを指定していた。
探索時の画像別予測や混同行列の記録も確認できないため、各候補のmacro-F1 / weighted-F1を精度・損失から復元することはできない。F1の実測曲線は作成していない。
全クラスを対象にした単一ラベル分類のmicro-F1はaccuracyと一致するが、ユーザーが意図するF1を確認せず精度をF1へ読み替えてはいない。
F1の種類を定義し、評価ごとに予測または混同行列からF1を記録する再実行が必要。再学習モデルのF1で探索全1,200評価のF1を代用することもできない。

## 出力

- [精度PNG](search_accuracy_convergence.png) / [SVG](search_accuracy_convergence.svg)
- [損失PNG](search_loss_convergence.png) / [SVG](search_loss_convergence.svg)
- `best_accuracy_by_evaluation.csv`：評価番号×5シード、精度は%単位。
- `best_loss_by_evaluation.csv`：評価番号×5シード、損失。
- `seed_summary.csv`：最良値、到達評価番号、対応する数式。
- `audit.json`：入力と検査結果。
- `plot_metrics.py`：再生成スクリプト。

元の各探索CSVと前回分析の全6,000行を照合。各seedが1〜1,200を欠番・重複なしで持つこと、累積曲線の単調性と最終値が元記録の極値に一致することを検査した。
"""
(OUT/"README_JA.md").write_text(readme,encoding="utf-8")
(OUT/"audit.json").write_text(json.dumps(dict(date="2026-10-31",records=6000,eligible=3682,
    excluded_from_updates=2318, f1="not_recorded; no F1 curve fabricated",sources=sources),indent=2),encoding="utf-8")
print(json.dumps(summaries,indent=2))
