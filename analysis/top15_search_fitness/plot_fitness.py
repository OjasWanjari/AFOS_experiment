"""Plot repeated search evaluations of the selected Top 15, without training."""
import csv
import hashlib
import html
import json
import math
import os
import textwrap
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
INPUT = ROOT / "analysis/paper_full_results_updated_seed4"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


selected = read_csv(INPUT / "pooled_top15.csv")
assert selected == read_csv(ROOT / "paper_full_results/retrain_runs/pooled_top15_updated/selected_top15.csv")
assert [int(c["rank"]) for c in selected] == list(range(1, 16))
aggregated = read_csv(INPUT / "search_evaluations_selected.csv")
assert len(aggregated) == 6000
rows, sources = [], []
for seed in range(5):
    folder = "paper_full_seeds3 and 4" if seed == 3 else f"paper_full_seeds{seed}"
    path = ROOT / "paper_full_results" / folder / "results" / f"evaluation_results_seed_{seed}.csv"
    raw = read_csv(path)
    assert len(raw) == 1200
    assert [int(r["evaluation_count"]) for r in raw] == list(range(1, 1201))
    subset = [r for r in aggregated if int(r["seed"]) == seed]
    assert len(subset) == 1200
    for original, saved in zip(raw, subset):
        assert int(original["seed"]) == seed
        assert all(original[k] == saved[k] for k in original), (seed, original["evaluation_count"])
        rows.append(dict(original, source_run=folder))
    sources.append(dict(seed=seed, path=str(path.relative_to(ROOT)), sha256=digest(path)))

series, observations, summary = {}, [], []
for c in selected:
    rank, expression = int(c["rank"]), c["activation_expression"]
    selected_record = next(r for r in rows if r["seed"] == c["seed"] and r["evaluation_count"] == c["evaluation_count"])
    assert all(selected_record[k] == c[k] for k in c if k != "rank")
    for seed in range(5):
        matching = [r for r in rows if int(r["seed"]) == seed and r["activation_expression"] == expression]
        xs, ys, bests = [], [], []
        best = -math.inf
        for r in matching:
            score = float(r["score"])
            assert r["status"] == "completed" and int(r["epochs_completed"]) == 15
            accuracy, loss = float(r["validation_accuracy"]), float(r["validation_loss"])
            assert math.isfinite(score) and loss > 0 and 0 <= accuracy <= 1
            assert math.isclose(score, accuracy/loss, rel_tol=1e-12, abs_tol=1e-12)
            best = max(best, score)
            xs.append(int(r["evaluation_count"]))
            ys.append(score)
            bests.append(best)
            observations.append(dict(candidate=rank, seed=seed, evaluation_count=xs[-1],
                generation=int(r["generation"]), training_seed=int(r["training_seed"]),
                activation_expression=expression, fitness=score, best_fitness_so_far=best,
                validation_accuracy=accuracy, validation_loss=loss,
                epochs_completed=int(r["epochs_completed"]), status=r["status"], source_run=r["source_run"]))
        assert all(a <= b for a,b in zip(bests, bests[1:]))
        series[rank, seed] = xs, ys, bests
        peak_index = ys.index(max(ys)) if ys else None
        summary.append(dict(candidate=rank, seed=seed, activation_expression=expression,
            observations=len(xs), first_evaluation=xs[0] if xs else "", last_evaluation=xs[-1] if xs else "",
            first_fitness=ys[0] if ys else "", last_fitness=ys[-1] if ys else "",
            best_fitness=max(ys) if ys else "", best_at_evaluation=xs[peak_index] if xs else "",
            availability="observed" if xs else "not_observed"))

assert len(summary) == 75
coverage = {int(c["rank"]): [len(series[int(c["rank"]), s][0]) for s in range(5)] for c in selected}
assert sum(sum(n > 0 for n in counts) for counts in coverage.values()) == 16
write_csv(OUT / "fitness_observations.csv", observations)
write_csv(OUT / "fitness_summary.csv", summary)

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6957A5"]
MARKERS = ["o", "s", "^", "D", "v"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#24364B", "text.color": "#24364B", "svg.fonttype": "none"})
ymax = round(math.ceil(max(r["fitness"] for r in observations)*10)/10 + .1, 1)


def draw(ax, rank, small=False):
    for seed in range(5):
        xs, ys, bests = series[rank, seed]
        if not xs:
            continue
        ax.plot(xs, ys, color=COLORS[seed], alpha=.48, linewidth=.65 if small else .85)
        ax.scatter(xs, ys, color=COLORS[seed], marker=MARKERS[seed],
                   s=9 if small else 18, alpha=.65, zorder=3, linewidths=0)
        ax.step(xs, bests, where="post", color=COLORS[seed], linestyle="--",
                linewidth=1.5 if small else 2, zorder=4)
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
    ax.grid(color="#D9E0E8", linewidth=.65, alpha=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8 if small else 11)
    ax.set_xlabel("Search evaluation count (within each seed)", fontsize=8 if small else 12)
    ax.set_ylabel("Fitness", fontsize=9 if small else 12)


figure_files = []
for c in selected:
    rank = int(c["rank"])
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    fig.subplots_adjust(left=.08, right=.75, bottom=.19, top=.77)
    fig.text(.08, .95, f"Candidate {rank:02d} — fitness across search evaluations", fontsize=18, weight="bold")
    fig.text(.08, .875, textwrap.fill(c["activation_expression"], width=90), fontsize=12,
             family="DejaVu Sans Mono", va="center")
    draw(ax, rank)
    handles = []
    for seed in range(5):
        n = coverage[rank][seed]
        label = f"Seed {seed}: n = {n}" if n else f"Seed {seed}: not observed"
        handles.append(Line2D([], [], color=COLORS[seed] if n else "#A4ADB7", marker=MARKERS[seed] if n else "",
                              linestyle="-" if n else "", label=label))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False, fontsize=10,
              labelspacing=1.0, handlelength=1.7, borderaxespad=0)
    fig.text(.775, .39, "Dots + thin lines\nRecorded fitness\n\nDashed steps\nBest fitness so far", fontsize=10, va="top", linespacing=1.5)
    fig.text(.08, .08, "Fitness = validation accuracy (0–1) / validation loss. Each evaluation trains a fresh model for 15 epochs.", fontsize=10)
    fig.text(.08, .045, "Lines connect observations; gaps contain no measurements of this function. No curve before first or after last observation.", fontsize=9.3)
    for ext in ("png", "svg"):
        path = OUT / f"candidate_{rank:02d}_fitness.{ext}"
        fig.savefig(path, dpi=180, facecolor="white")
        figure_files.append(path)
    plt.close(fig)

fig, axes = plt.subplots(5, 3, figsize=(19, 23))
fig.subplots_adjust(left=.055, right=.985, bottom=.055, top=.885, wspace=.22, hspace=.66)
fig.suptitle("Top 15 discovered functions: fitness across search evaluations", fontsize=22, y=.987, weight="bold")
fig.text(.5, .965, "Dots / thin lines: observed fitness. Dashed steps: best fitness so far for that function and seed.", ha="center", fontsize=12)
fig.text(.5, .947, "n lists observation counts for seeds 0, 1, 2, 3, 4. Zero means not observed, not zero fitness.", ha="center", fontsize=12)
handles = [Line2D([], [], color=COLORS[s], marker=MARKERS[s], label=f"Seed {s}") for s in range(5)]
fig.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(.5, .935), frameon=False, fontsize=12)
for c, ax in zip(selected, axes.flat):
    rank = int(c["rank"])
    draw(ax, rank, small=True)
    expression = c["activation_expression"].replace("*", " * ").replace("+", " + ").replace(",", ", ")
    expression = textwrap.fill(expression, width=52, break_long_words=False, break_on_hyphens=False)
    ax.set_title(f"Candidate {rank:02d}\n{expression}\nn = {coverage[rank]}", fontsize=9, pad=8, weight="bold")
fig.text(.055, .022, "Matching uses the exact expression string. New seed 4 replaces the old run. Each seed has 1,200 total candidate evaluations.", fontsize=11)
fig.text(.055, .009, "A flat best-so-far line is not proof of convergence: repeat counts differ, and each evaluation starts training from scratch.", fontsize=11)
overview = OUT / "top15_search_fitness_overview.png"
fig.savefig(overview, dpi=150, facecolor="white")
plt.close(fig)

audit = dict(date="2026-10-31", total_search_records=len(rows), matched_top15_records=len(observations),
    possible_function_seed_pairs=75, observed_function_seed_pairs=16, unobserved_function_seed_pairs=59,
    matching="exact activation_expression string", coverage=coverage, shared_y_limits=[0, ymax],
    selected_top15_sha256=digest(INPUT / "pooled_top15.csv"), sources=sources)
(OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
table = "\n".join(f"| {rank:02d} | " + " | ".join(str(n) for n in counts) + " |" for rank, counts in coverage.items())
readme = f"""# 発見Top 15の探索評価回数に対するfitness

分析日：2026-10-31。新seed 4を採用した全5探索・6,000評価から、統合Top 15の各数式が実際に評価された記録を抽出した。追加学習は実施していない。

## グラフの読み方

- 横軸：各探索seed内の全候補の通算評価回数（1〜1,200）。その関数だけの評価番号でも、再学習epochでもない。
- 縦軸：保存されたfitness（検証精度0〜1 ÷ 検証損失）。対象の全記録について比を再計算して一致を確認した。
- 点と薄い実線：その関数を実際に評価した際のfitness。線は点を結ぶ補助で、間の評価番号にその関数の測定値があることを意味しない。
- 破線の階段：同じ関数・同じ探索seedについて、その時点までに観測された最大fitness。最初の観測より前、最後の観測より後には延長しない。
- 各関数の凡例には5シードすべてを記載。評価されていないシードは `not observed` と表示し、0や別シードの値で埋めない。
- 各点は新たに初期化したモデルの15エポック評価の結果。前回のモデルを継続学習した結果ではない。
- 候補番号は前回と同じ、全探索の検証精度による選抜順位。fitness順位ではない。
- 全15図で軸範囲を統一。数式文字列を完全一致で照合した。候補7と10は数学的に同一だが、選抜・再学習計画に合わせて別IDを維持した。

## 観測件数

該当する記録は合計{len(observations)}件で、すべて15エポック完了・有限fitnessだった。15関数×5シードの75組のうち実測ありは16組、記録なしは59組。
候補2のみ2シードに登場し、残り14関数は各1シードにのみ登場した。そのため、各関数について5本の実測曲線を作ることは保存ログだけではできない。

| 候補 | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4（新） |
|---|---:|---:|---:|---:|---:|
{table}

## 収束についての注意

固定された数式を異なる初期化で繰り返し評価しているため、この図は「同じ関数の再評価に伴うfitnessのばらつきと最高値の更新」を表す。
破線は累積最大値なので定義上は下がらない。横ばいであることだけで、学習や探索全体が収束したとは断定できない。再評価数が多い関数ほど最高値を更新する機会も多い。
特に候補15は3回、候補13は6回の観測しかなく、長期的な収束の判断には不足する。
探索全体の収束を見る場合は各seedの全候補に対する最高fitness、学習の収束を見る場合は別途作成した再学習epochのグラフを参照する。

## ファイル

- [15関数の一覧図](top15_search_fitness_overview.png)
- [個別図の閲覧ページ](index.html)
- `candidate_01_fitness.png`〜`candidate_15_fitness.png`：個別図。SVGも同梱。
- `fitness_observations.csv`：元のfitness、累積最大値、評価番号、世代、学習seed、検証指標。
- `fitness_summary.csv`：関数×seedの観測数・初回・最終・最高値。未評価の数値欄は空欄。
- `audit.json`：入力のSHA-256と件数検査。
- `plot_fitness.py`：再生成スクリプト。PythonとMatplotlibを使用。

元データ：`paper_full_results/paper_full_seeds0`、`paper_full_seeds1`、`paper_full_seeds2`、`paper_full_seeds3 and 4`のseed 3、`paper_full_seeds4`の新seed 4。
全6,000行を前回分析の`search_evaluations_selected.csv`と照合し、各seedの評価番号1〜1,200に欠番・重複がないことと選抜15関数の一致を確認した。
"""
(OUT / "README_JA.md").write_text(readme, encoding="utf-8")
cards = "\n".join(f'<section><h2>Candidate {int(c["rank"]):02d}</h2><code>{html.escape(c["activation_expression"])}</code><img loading="lazy" src="candidate_{int(c["rank"]):02d}_fitness.png" alt="Recorded search fitness for candidate {c["rank"]}"><p><a href="candidate_{int(c["rank"]):02d}_fitness.png">PNG</a> · <a href="candidate_{int(c["rank"]):02d}_fitness.svg">SVG</a></p></section>' for c in selected)
(OUT / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top 15 search fitness</title><style>body{font:16px system-ui;color:#24364b;background:#f2f5f8;max-width:1250px;margin:32px auto;padding:0 20px}section{background:white;padding:20px;margin:24px 0;border-radius:8px}img{width:100%;height:auto}code{overflow-wrap:anywhere}a{color:#0072b2}</style><h1>Top 15: fitness across search evaluations</h1><p>Within-seed evaluation counts 1–1,200. Fitness = validation accuracy / validation loss.</p><p>Dots and thin lines show observations; dashed steps show the best value so far. Unobserved seeds have no curve. Each evaluation trains a fresh model.</p><p><a href="top15_search_fitness_overview.png">Overview</a> · <a href="README_JA.md">日本語の説明</a></p>''' + cards + "</html>", encoding="utf-8")
files = figure_files + [overview, OUT/"index.html", OUT/"README_JA.md", OUT/"fitness_observations.csv", OUT/"fitness_summary.csv", OUT/"audit.json", Path(__file__)]
package = OUT / "top15_search_fitness.zip"
with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, "top15_search_fitness/" + path.name)
with zipfile.ZipFile(package) as archive:
    assert archive.testzip() is None and len(archive.namelist()) == 37
print(json.dumps({k:v for k,v in audit.items() if k != "sources"}, indent=2))
print(f"Created 15 PNG/SVG charts, overview, gallery, CSV data and ZIP: {OUT}")
