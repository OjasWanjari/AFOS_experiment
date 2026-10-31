"""Plot saved Top 15 retraining histories. No new training is performed."""
import csv
import hashlib
import html
import json
import math
import os
import statistics
import textwrap
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / "paper_full_results/retrain_runs/pooled_top15_updated"
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


plan = json.loads((RUN / "plan.json").read_text(encoding="utf-8"))
assert digest(RUN / "plan.json") == (RUN / "plan.sha256").read_text().strip()
assert digest(RUN / "split_indices.npz") == plan["split_sha256"]
selected = read_csv(RUN / "selected_top15.csv")
assert selected == read_csv(ROOT / "analysis/paper_full_results_updated_seed4/pooled_top15.csv")
assert [int(t["rank"]) for t in selected] == list(range(1, 16))
planned = {t["trial_id"]: t for t in plan["trials"]}
saved = {r["trial_id"]: r for r in read_csv(RUN / "results/retraining_results.csv")}
series, observations, summary, sources = {}, [], [], []
for candidate in selected:
    rank = int(candidate["rank"])
    for seed in range(5):
        trial = f"discovered_{rank:02d}_seed_{seed}"
        attempt = RUN / "trials" / trial / saved[trial]["attempt"]
        result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
        assert result["activation_expression"] == candidate["activation_expression"] == planned[trial]["activation_expression"]
        assert result["seed"] == seed and result["status"] == saved[trial]["status"]
        path = attempt / "histories" / f"{trial}.csv"
        history = read_csv(path)
        assert [int(h["epoch"]) for h in history] == list(range(1, result["epochs_completed"]+1))
        assert len(history) == 50
        values = []
        for h in history:
            accuracy, loss = float(h["val_accuracy"]), float(h["val_loss"])
            finite = math.isfinite(accuracy) and math.isfinite(loss) and loss > 0
            fitness = accuracy / loss if finite else math.nan
            assert not finite or (0 <= accuracy <= 1 and math.isfinite(fitness))
            values.append(fitness)
            observations.append(dict(candidate=rank, seed=seed, epoch=int(h["epoch"]),
                activation_expression=candidate["activation_expression"],
                validation_accuracy=accuracy if math.isfinite(accuracy) else "",
                validation_loss=loss if math.isfinite(loss) else "",
                validation_fitness=fitness if finite else "",
                metric_status="finite" if finite else "undefined",
                trial_status=result["status"], source=str(path.relative_to(ROOT))))
        assert (result["validation_loss"] is None and not math.isfinite(float(history[-1]["val_loss"]))) or math.isclose(float(history[-1]["val_loss"]), result["validation_loss"], rel_tol=1e-10)
        series[rank, seed] = values
        valid = [(i+1, f) for i, f in enumerate(values) if math.isfinite(f)]
        peak_epoch, peak = max(valid, key=lambda pair: pair[1])
        summary.append(dict(candidate=rank, seed=seed, observed_epochs=len(values),
            finite_epochs=len(valid), peak_epoch=peak_epoch, peak_fitness=peak,
            final_epoch_fitness=values[-1] if math.isfinite(values[-1]) else "",
            last_finite_epoch=valid[-1][0], last_finite_fitness=valid[-1][1], status=result["status"]))
        sources.append(dict(trial_id=trial, history=str(path.relative_to(ROOT)), sha256=digest(path),
                            result_sha256=digest(attempt / "result.json")))

assert len(series) == 75 and len(observations) == 3750
invalid = [(r["candidate"], r["seed"], r["epoch"]) for r in observations if r["metric_status"] != "finite"]
assert invalid == [(13, 1, 50)], invalid
write_csv(OUT / "fitness_by_epoch.csv", observations)
write_csv(OUT / "fitness_summary.csv", summary)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
    "axes.labelcolor": "#24364B", "text.color": "#24364B", "axes.titleweight": "bold"})
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6957A5"]
STYLES = ["-", "--", "-.", ":", "-"]
MARKERS = ["o", "s", "^", "D", "v"]
ymax = math.ceil(max(r["peak_fitness"] for r in summary)*10)/10 + .1


def draw(ax, rank, small=False):
    for seed in range(5):
        ax.plot(range(1, 51), series[rank, seed], label=f"Seed {seed}", color=COLORS[seed],
                linestyle=STYLES[seed], linewidth=1.5 if small else 1.9,
                marker=MARKERS[seed], markevery=5, markersize=2.6 if small else 3.8, alpha=.9)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, ymax)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.grid(axis="both", color="#D9E0E8", linewidth=.65, alpha=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9 if small else 11)
    if rank == 13:
        ax.text(.025, .96, "Seed 1: undefined at epoch 50 (NaN loss)", transform=ax.transAxes,
                va="top", fontsize=8 if small else 10, color=COLORS[1],
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
    ax.set_xlabel("Retraining epoch", fontsize=9 if small else 12)
    ax.set_ylabel("Validation fitness", fontsize=9 if small else 12)


figure_files = []
for candidate in selected:
    rank = int(candidate["rank"])
    fig, ax = plt.subplots(figsize=(11.8, 6.5))
    fig.subplots_adjust(left=.09, right=.97, bottom=.19, top=.77)
    fig.text(.09, .95, f"Candidate {rank:02d} — fitness during retraining", fontsize=18, weight="bold")
    fig.text(.09, .875, textwrap.fill(candidate["activation_expression"], width=87), fontsize=12,
             family="DejaVu Sans Mono", va="center")
    draw(ax, rank)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False, fontsize=10)
    fig.text(.09, .03, "Fitness = validation accuracy (0–1) / validation loss. Each point uses that epoch's saved metrics.", fontsize=10)
    fig.text(.97, .03, "Same data split · seeds 0–4", fontsize=9, ha="right")
    for ext in ("png", "svg"):
        path = OUT / f"candidate_{rank:02d}_fitness.{ext}"
        fig.savefig(path, dpi=180, facecolor="white")
        figure_files.append(path)
    plt.close(fig)

fig, axes = plt.subplots(5, 3, figsize=(18, 20))
fig.subplots_adjust(left=.055, right=.985, bottom=.055, top=.91, wspace=.22, hspace=.54)
fig.suptitle("Top 15 discovered functions: validation fitness during retraining", fontsize=22, y=.985, weight="bold")
fig.text(.5, .96, "Fitness = validation accuracy (0–1) / validation loss. Five seeds, fixed data split, 50 epochs.", ha="center", fontsize=13)
handles = [Line2D([], [], color=COLORS[s], linestyle=STYLES[s], marker=MARKERS[s], label=f"Seed {s}") for s in range(5)]
fig.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(.5, .949), frameon=False, fontsize=12)
for candidate, ax in zip(selected, axes.flat):
    rank = int(candidate["rank"])
    draw(ax, rank, small=True)
    expression = candidate["activation_expression"].replace("*", " * ").replace("+", " + ").replace(",", ", ")
    expression = textwrap.fill(expression, width=52, break_long_words=False, break_on_hyphens=False)
    ax.set_title(f"Candidate {rank:02d}\n{expression}", fontsize=9.4, pad=9)
fig.text(.055, .017, "Candidate numbers follow pooled search validation-accuracy ranking. Curves show observed values, without smoothing or cumulative maxima.", fontsize=11)
overview = OUT / "top15_fitness_overview.png"
fig.savefig(overview, dpi=150, facecolor="white")
plt.close(fig)

completed = [r for r in summary if r["status"] == "completed"]
audit = dict(date="2026-10-31", source=str(RUN.relative_to(ROOT)), plan_sha256=digest(RUN / "plan.json"),
    trials=len(series), epochs=len(observations), finite_fitness_epochs=len(observations)-len(invalid),
    undefined=invalid, median_peak_epoch_completed=statistics.median(r["peak_epoch"] for r in completed),
    completed_trials_with_peak_before_final=sum(r["peak_epoch"] < 50 for r in completed),
    shared_y_limits=[0, ymax], sources=sources)
(OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
readme = f"""# 発見Top 15の再学習fitness推移

分析日：2026-10-31。追加学習は実施せず、新seed 4を含む探索から選んだTop 15の保存済み再学習履歴を使用。

## グラフの読み方

- 各関数につき1枚、再学習seed 0〜4の5曲線を重ねた。候補番号は探索の検証精度による選抜順位。
- 横軸：再学習エポック（1〜50）。探索の候補評価回数（1〜1,200）とは異なる。
- 縦軸：検証精度（0〜1）÷検証損失。探索fitnessと同じ式を、各エポックの履歴に適用した事後計算。
- この値を使って再学習の重み更新や停止判定を行ったわけではない。テスト精度は使っていない。
- 各点はそのエポックの実測値。線は隣接点を結ぶ。平滑化・累積最大値への置換はしていない。
- 全15図で横軸と縦軸の範囲を統一。色に加えて線種・マーカーでシードを識別。
- 候補13 seed 1のepoch 50はval_lossがNaNのためfitness未定義。0に置換せず欠損として表示し、49までの実測値を残した。

## 確認できることと解釈

75試行×50エポックの3,750記録を照合し、3,749点が有限値、1点が未定義だった。選抜Top 15は前回分析と一致し、計画・データ分割のハッシュ、試行情報、最終検証損失を照合した。

完了74試行のうち{audit['completed_trials_with_peak_before_final']}試行で、最大fitnessは50エポック目より前に記録された。最大fitnessのエポックの中央値は{audit['median_peak_epoch_completed']:g}。
fitnessは検証精度と検証損失の比なので、精度が横ばいでも検証損失が増えると下がる。後半の横ばい・変動・低下を見分ける用途であり、単に曲線が横ばいであることだけで学習の数学的な収束を断定しない。検証損失の悪化は過学習の可能性を検討する材料だが、この比だけでは原因を確定できない。

## 出力

- [15関数の一覧図](top15_fitness_overview.png)
- [個別グラフの閲覧ページ](index.html)：各関数のPNGとSVGにリンク
- `candidate_01_fitness.png`〜`candidate_15_fitness.png`：個別グラフ（SVGも同梱）
- `fitness_by_epoch.csv`：3,750行、検証精度・検証損失・計算fitness・状態・元履歴のパス
- `fitness_summary.csv`：75試行の最大fitness、最大値のepoch、最終epoch値など
- `audit.json`：確認結果、入力ファイルのハッシュ
- `plot_fitness.py`：再生成スクリプト（PythonとMatplotlibが必要）

元データ：`paper_full_results/retrain_runs/pooled_top15_updated/trials/discovered_*_seed_*/attempt_*/histories/`。採用attemptは同runの`results/retraining_results.csv`に従う。

候補7と10は数式の因子順序が異なる同一数学関数だが、保存された候補IDどおり別図にしている。
"""
(OUT / "README_JA.md").write_text(readme, encoding="utf-8")
cards = "\n".join(f'<section><h2>Candidate {int(c["rank"]):02d}</h2><p><code>{html.escape(c["activation_expression"])}</code></p><img loading="lazy" src="candidate_{int(c["rank"]):02d}_fitness.png" alt="Five-seed validation fitness for candidate {c["rank"]}"><p><a href="candidate_{int(c["rank"]):02d}_fitness.png">PNG</a> · <a href="candidate_{int(c["rank"]):02d}_fitness.svg">SVG</a></p></section>' for c in selected)
(OUT / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top 15 retraining fitness</title><style>body{font:16px system-ui;color:#24364b;background:#f2f5f8;max-width:1180px;margin:32px auto;padding:0 20px}section{background:white;padding:20px;margin:24px 0;border-radius:8px}img{width:100%;height:auto}code{overflow-wrap:anywhere}a{color:#0072b2}</style><h1>Top 15: fitness during retraining</h1><p>Five seeds per function. Epochs 1–50. Validation fitness = validation accuracy / validation loss.</p><p>Candidate 13, seed 1: epoch 50 is undefined because validation loss is NaN. It is not plotted as zero.</p><p><a href="top15_fitness_overview.png">Overview</a> · <a href="README_JA.md">日本語の説明</a></p>''' + cards + "</html>", encoding="utf-8")
package = OUT / "top15_retraining_fitness.zip"
with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in figure_files + [overview, OUT/"index.html", OUT/"README_JA.md", OUT/"fitness_by_epoch.csv", OUT/"fitness_summary.csv", OUT/"audit.json", Path(__file__)]:
        archive.write(path, "top15_retraining_fitness/" + path.name)
with zipfile.ZipFile(package) as archive:
    assert archive.testzip() is None and len(archive.namelist()) == 37
print(json.dumps({k:v for k,v in audit.items() if k != "sources"}, indent=2))
print(f"Created 15 PNG/SVG figures, overview, gallery, CSV data and ZIP: {OUT}")
