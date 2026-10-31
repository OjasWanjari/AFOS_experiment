"""Best recorded fitness across all candidates, independently for five searches."""
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


aggregated = read_csv(ROOT / "analysis/paper_full_results_updated_seed4/search_evaluations_selected.csv")
assert len(aggregated) == 6000
curves, summary, sources = {}, [], []
for seed in range(5):
    folder = "paper_full_seeds3 and 4" if seed == 3 else f"paper_full_seeds{seed}"
    source = ROOT / "paper_full_results" / folder / "results" / f"evaluation_results_seed_{seed}.csv"
    rows = read_csv(source)
    reference = [r for r in aggregated if int(r["seed"]) == seed]
    assert len(rows) == len(reference) == 1200
    assert [int(r["evaluation_count"]) for r in rows] == list(range(1, 1201))
    best, best_row, values, updates = -math.inf, None, [], []
    for r, ref in zip(rows, reference):
        assert int(r["seed"]) == seed and all(r[k] == ref[k] for k in r)
        score = float(r["score"])
        loss, accuracy = number(r["validation_loss"]), number(r["validation_accuracy"])
        expected = 0.0 if r["status"] == "nonfinite" or not math.isfinite(loss) or not math.isfinite(accuracy) or loss <= 0 else accuracy/loss
        assert math.isfinite(score) and score >= 0
        assert math.isclose(score, expected, rel_tol=1e-12, abs_tol=1e-12)
        if score > best:
            best, best_row = score, r
            updates.append(int(r["evaluation_count"]))
        values.append(best)
    assert all(a <= b for a,b in zip(values, values[1:]))
    assert len(values) == 1200 and values[-1] == max(float(r["score"]) for r in rows)
    curves[seed] = values
    summary.append(dict(seed=seed, evaluations=1200, best_fitness=best,
        best_first_reached_at=int(best_row["evaluation_count"]), best_generation=int(best_row["generation"]),
        best_expression=best_row["activation_expression"], best_status=best_row["status"],
        best_epochs_completed=int(best_row["epochs_completed"]),
        evaluations_without_further_improvement=1200-int(best_row["evaluation_count"]),
        record_updates_after_first=len(updates)-1))
    sources.append(dict(seed=seed, csv=str(source.relative_to(ROOT)),
                        sha256=hashlib.sha256(source.read_bytes()).hexdigest()))

wide = [dict(evaluation_count=i+1, **{f"seed_{s}_best_fitness":curves[s][i] for s in range(5)}) for i in range(1200)]
write_csv(OUT / "best_fitness_by_evaluation.csv", wide)
write_csv(OUT / "seed_summary.csv", summary)
(OUT / "audit.json").write_text(json.dumps(dict(date="2026-10-31", records=6000,
    evaluations_per_seed=1200, includes_all_recorded_statuses=True,
    nonfinite_policy="Use the original recorded score of zero; do not drop evaluation positions.",
    checks=["Raw CSVs match the selected aggregate", "Each seed covers 1..1200 without gaps",
            "All scores match the original fitness rule", "Cumulative maxima are monotonic and end at the raw maximum"],
    sources=sources), indent=2), encoding="utf-8")

colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6957A5"]
styles = ["-", "--", "-.", ":", "-"]
markers = ["o", "s", "^", "D", "v"]
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":12,
    "axes.spines.top":False, "axes.spines.right":False, "svg.fonttype":"none",
    "axes.labelcolor":"#24364B", "text.color":"#24364B"})
fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(left=.09, right=.97, bottom=.23, top=.80)
fig.text(.09, .945, "Search convergence across five seeds", fontsize=23, weight="bold")
fig.text(.09, .885, "Best fitness found among all candidates up to each evaluation", fontsize=14)
for seed in range(5):
    ax.step(range(1,1201), curves[seed], where="post", color=colors[seed], linestyle=styles[seed],
            linewidth=2.1, label=f"Seed {seed}", marker=markers[seed], markevery=[199,399,599,799,999,1199],
            markersize=5, alpha=.95)
ax.set_xlim(0,1200)
ax.set_ylim(0, math.ceil(max(s["best_fitness"] for s in summary)*10)/10+.1)
ax.set_xticks(range(0,1201,200))
ax.set_xlabel("Search evaluation count (within each seed)", labelpad=10)
ax.set_ylabel("Best fitness so far", labelpad=10)
ax.grid(color="#D9E0E8", alpha=.8, linewidth=.7)
ax.set_axisbelow(True)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5,-.18), frameon=False, fontsize=11)
fig.text(.09, .07, "Fitness = validation accuracy (0–1) / validation loss. 1,200 recorded evaluations per seed.", fontsize=10.5)
fig.text(.09, .035, "All candidates included. Failed numerical evaluations retain their recorded fitness of zero. Updated seed 4 is used.", fontsize=9.7)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"search_fitness_convergence.{ext}", dpi=200, facecolor="white")
plt.close(fig)

table = "\n".join(f"| {s['seed']} | {s['best_fitness']:.6f} | {s['best_first_reached_at']} | {s['evaluations_without_further_improvement']} |" for s in summary)
readme = f"""# 全探索候補の最高fitnessの推移

2026-10-31作成。新しいseed 4を採用した5探索・6,000評価を使用。追加学習は行っていない。

各seedについて評価番号tにおける縦軸は、そのseedの評価1〜tに記録されたscoreの最大値。Top 15だけに限定せず、全候補を対象とする。
横軸は各seed内の通算評価番号（1〜1,200）。全5曲線で1,200点があり、欠番・重複はない。世代ごとの平均や再学習epochではない。

fitnessは検証精度（0〜1）÷検証損失。非有限値などの評価には元コードが保存した0を使用し、その評価番号を削除しない。
低精度打ち切りも含めた保存scoreの累積最大値を用いる。15エポック完走候補だけに限定した以前の図とは定義を区別する。
5シードすべての最終最高値を持つ候補は15エポック完了だった。

| Seed | 最終最高fitness | 最高値を初めて記録した評価番号 | 以後更新なしの評価回数 |
|---|---:|---:|---:|
{table}

階段が上がる箇所は新たな最高値の発見、横ばい部分はそれを超える候補が見つからなかった区間。
累積最大値は定義上低下しない。横ばいであっても、探索空間の最適解への到達や、探索を続けても改善しないことを証明するものではない。
最終最高fitnessの候補が、検証精度で選抜したTop 15の1位と一致するとは限らない。

## ファイル

- [グラフPNG](search_fitness_convergence.png) / [SVG](search_fitness_convergence.svg)
- [各評価時点の5シードの値](best_fitness_by_evaluation.csv)：1,200行
- [シード別集計](seed_summary.csv)：最高値・数式・発見評価番号など
- `audit.json`：入力パスとSHA-256、照合項目
- `plot_convergence.py`：PythonとMatplotlibによる再生成スクリプト

元の探索CSV全6,000行を既存の`analysis/paper_full_results_updated_seed4/search_evaluations_selected.csv`と照合。
保存scoreが元のfitness計算規則と一致し、各累積曲線が非減少で、最終値が全候補の最大scoreと一致することを検査した。
"""
assert all(s["best_status"] == "completed" and s["best_epochs_completed"] == 15 for s in summary)
(OUT / "README_JA.md").write_text(readme, encoding="utf-8")
print(json.dumps(summary, indent=2))
