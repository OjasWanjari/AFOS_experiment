"""Six panels per function: train/validation histories and final test metrics."""
import csv
import hashlib
import html
import json
import math
import os
import textwrap
import zipfile
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN=ROOT/"paper_full_results/retrain_runs/pooled_top15_updated"
os.environ.setdefault("MPLCONFIGDIR",str(OUT/".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


plan=json.loads((RUN/"plan.json").read_text(encoding="utf-8"))
assert digest(RUN/"plan.json")==(RUN/"plan.sha256").read_text().strip()
assert digest(RUN/"split_indices.npz")==plan["split_sha256"]
selected=read_csv(RUN/"selected_top15.csv")
assert selected==read_csv(ROOT/"analysis/paper_full_results_updated_seed4/pooled_top15.csv")
assert [int(c["rank"]) for c in selected]==list(range(1,16))
saved={r["trial_id"]:r for r in read_csv(RUN/"results/retraining_results.csv")}
planned={r["trial_id"]:r for r in plan["trials"]}
keys=dict(accuracy="train_accuracy",val_accuracy="validation_accuracy",loss="train_loss",val_loss="validation_loss")
histories,results,epoch_rows,test_rows,sources={},{},[],[],[]
for candidate in selected:
    rank=int(candidate["rank"])
    for seed in range(5):
        trial=f"discovered_{rank:02d}_seed_{seed}"
        attempt=RUN/"trials"/trial/saved[trial]["attempt"]
        result=json.loads((attempt/"result.json").read_text(encoding="utf-8"))
        assert result["activation_expression"]==candidate["activation_expression"]==planned[trial]["activation_expression"]
        assert result["seed"]==seed and result["trial_id"]==trial and result["status"]==saved[trial]["status"]
        path=attempt/"histories"/f"{trial}.csv"
        history=read_csv(path)
        assert result["epochs_completed"]==50 and [int(h["epoch"]) for h in history]==list(range(1,51))
        for key,result_key in keys.items():
            value=float(history[-1][key])
            expected=result[result_key]
            assert (expected is None and not math.isfinite(value)) or (expected is not None and math.isclose(value,expected,rel_tol=1e-10))
        for h in history:
            row=dict(candidate=rank,seed=seed,epoch=int(h["epoch"]))
            for key in keys:
                value=float(h[key])
                if math.isfinite(value):
                    assert 0<=value<=1 if "accuracy" in key else 0<=value<=2
                row[key]=value if math.isfinite(value) else ""
            row["trial_status"]=result["status"]
            epoch_rows.append(row)
        for key in ("test_accuracy","test_loss"):
            expected=None if saved[trial][key]=="" else float(saved[trial][key])
            assert result[key]==expected
            assert expected is None or (math.isfinite(expected) and 0<=expected<=(1 if key=="test_accuracy" else 2))
        if result["status"]=="completed":
            assert result["test_accuracy"] is not None and result["test_loss"] is not None
        else:
            assert (rank,seed,result["status"])==(13,1,"nonfinite")
            assert result["test_accuracy"] is None and result["test_loss"] is None
        test_rows.append(dict(candidate=rank,seed=seed,trial_status=result["status"],
            test_after_epoch=50 if result["test_accuracy"] is not None else "",
            test_accuracy=result["test_accuracy"] if result["test_accuracy"] is not None else "",
            test_loss=result["test_loss"] if result["test_loss"] is not None else "",
            source=str((attempt/"result.json").relative_to(ROOT))))
        histories[rank,seed]=history
        results[rank,seed]=result
        sources.append(dict(trial_id=trial,history=str(path.relative_to(ROOT)),history_sha256=digest(path),result_sha256=digest(attempt/"result.json")))
undefined=[(r["candidate"],r["seed"],r["epoch"],k) for r in epoch_rows for k in keys if r[k]==""]
assert undefined==[(13,1,50,"loss"),(13,1,50,"val_loss")]
assert len(epoch_rows)==3750 and len(test_rows)==75 and sum(r["test_accuracy"]!="" for r in test_rows)==74
write_csv(OUT/"train_validation_by_epoch.csv",epoch_rows)
write_csv(OUT/"final_test_by_seed.csv",test_rows)

colors=["#0072B2","#D55E00","#009E73","#CC79A7","#6957A5"]
styles=["-","--","-.",":","-"]
markers=["o","s","^","D","v"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
    "axes.spines.right":False,"svg.fonttype":"none","axes.labelcolor":"#24364B","text.color":"#24364B"})
handles=[Line2D([],[],color=colors[s],linestyle=styles[s],marker=markers[s],label=f"Seed {s}") for s in range(5)]
outputs=[]
for candidate in selected:
    rank=int(candidate["rank"])
    fig,axes=plt.subplots(2,3,figsize=(17,10.8))
    fig.subplots_adjust(left=.06,right=.975,top=.80,bottom=.16,hspace=.42,wspace=.25)
    fig.text(.06,.95,f"Candidate {rank:02d} — accuracy and loss across data splits",fontsize=23,weight="bold")
    fig.text(.06,.89,textwrap.fill(candidate["activation_expression"],width=110),fontsize=13,family="DejaVu Sans Mono",va="center")
    for i,metric in enumerate(("accuracy","loss")):
        scale=100 if metric=="accuracy" else 1
        for j,split in enumerate(("Training","Validation","Test")):
            ax=axes[i,j]
            ax.set_ylim(0,100 if metric=="accuracy" else 2)
            ax.set_ylabel("Accuracy (%)" if metric=="accuracy" else "Loss (cross-entropy)",fontsize=11)
            ax.set_title(f"{split} {metric}" + (" — final only" if j==2 else ""),fontsize=13,weight="bold",pad=11)
            ax.grid(color="#D9E0E8",linewidth=.65,alpha=.8)
            ax.set_axisbelow(True)
            if j<2:
                key=metric if j==0 else "val_"+metric
                for seed in range(5):
                    ys=[float(h[key])*scale for h in histories[rank,seed]]
                    ax.plot(range(1,51),ys,color=colors[seed],linestyle=styles[seed],linewidth=1.65,
                            marker=markers[seed],markevery=5,markersize=3,alpha=.92)
                ax.set_xlim(1,50)
                ax.set_xticks([1,10,20,30,40,50])
                ax.set_xlabel("Retraining epoch (1–50)")
                if rank==13:
                    if metric=="accuracy":
                        ax.scatter([50],[float(histories[rank,1][-1][key])*scale],color=colors[1],marker="x",s=65,linewidths=2,zorder=6,clip_on=False)
                    ax.text(.025,.04 if metric=="accuracy" else .96,
                            "Seed 1: failed epoch 50 (X)" if metric=="accuracy" else "Seed 1: epoch 50 loss = NaN",
                            transform=ax.transAxes,fontsize=8.5,color="#B34600",va="bottom" if metric=="accuracy" else "top",
                            bbox=dict(facecolor="white",edgecolor="none",alpha=.9,pad=2))
            else:
                ax.set_facecolor("#F7FAFD")
                ax.set_xlim(-.5,4.5)
                ax.set_xticks(range(5),[str(s) for s in range(5)])
                ax.set_xlabel("Retraining seed (test after epoch 50)")
                for seed in range(5):
                    value=results[rank,seed]["test_"+metric]
                    if value is None:
                        ax.text(seed,.39,"Not\nevaluated",transform=ax.get_xaxis_transform(),ha="center",va="center",fontsize=9,color=colors[seed])
                        continue
                    value*=scale
                    ax.scatter([seed],[value],marker=markers[seed],s=68,color=colors[seed],zorder=4)
                    ax.annotate(f"{value:.2f}%" if metric=="accuracy" else f"{value:.4f}",(seed,value),
                                xytext=(0,12),textcoords="offset points",ha="center",fontsize=9.2,color=colors[seed])
    fig.legend(handles=handles,ncol=5,loc="lower center",bbox_to_anchor=(.5,.085),frameon=False,fontsize=12)
    fig.text(.06,.06,"Training / validation: saved values at each epoch. Test: one final evaluation per successful seed; no per-epoch test history exists.",fontsize=11)
    fig.text(.06,.032,"Axes differ: epoch for training / validation, seed for test. Test points are not joined. No smoothing or cumulative best values.",fontsize=10)
    for ext in ("png","svg"):
        path=OUT/f"candidate_{rank:02d}_six_metrics.{ext}"
        fig.savefig(path,dpi=160,facecolor="white")
        outputs.append(path)
    plt.close(fig)

(OUT/"audit.json").write_text(json.dumps(dict(date="2026-10-31",functions=15,trials=75,epoch_records=3750,
    test_evaluated_trials=74,test_missing_trials=["discovered_13_seed_1"],nonfinite_epoch_metrics=undefined,
    plan_sha256=digest(RUN/"plan.json"),sources=sources),indent=2),encoding="utf-8")
readme="""# Top 15：学習・検証・テストのaccuracyとloss

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
"""
(OUT/"README_JA.md").write_text(readme,encoding="utf-8")
cards="\n".join(f'<section id="c{int(c["rank"])}"><h2>Candidate {int(c["rank"]):02d}</h2><code>{html.escape(c["activation_expression"])}</code><img loading="lazy" src="candidate_{int(c["rank"]):02d}_six_metrics.png" alt="Six metric panels for candidate {c["rank"]}"><p><a href="candidate_{int(c["rank"]):02d}_six_metrics.png">PNG</a> · <a href="candidate_{int(c["rank"]):02d}_six_metrics.svg">SVG</a></p></section>' for c in selected)
nav=" · ".join(f'<a href="#c{n}">{n:02d}</a>' for n in range(1,16))
(OUT/"index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top 15: six metric panels</title><style>body{font:16px system-ui;color:#24364b;background:#f2f5f8;max-width:1600px;margin:32px auto;padding:0 20px}section{background:white;padding:20px;margin:24px 0;border-radius:8px}img{width:100%;height:auto}code{overflow-wrap:anywhere}a{color:#0072b2}</style><h1>Top 15: training, validation and test metrics</h1><p>Six panels per function. Top row: accuracy. Bottom row: loss.</p><p>Training / validation use epochs 1–50. Test panels use seeds 0–4 and show only the final evaluation after epoch 50.</p><p>Candidate 13, seed 1 has no test evaluation because training failed.</p><p><a href="README_JA.md">日本語の説明</a></p><nav>''' +nav+"</nav>"+cards+"</html>",encoding="utf-8")
outputs += [OUT/"train_validation_by_epoch.csv",OUT/"final_test_by_seed.csv",OUT/"audit.json",OUT/"README_JA.md",OUT/"index.html",Path(__file__)]
with zipfile.ZipFile(OUT/"top15_retraining_six_metrics.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for path in outputs:
        archive.write(path,"top15_retraining_six_metrics/"+path.name)
with zipfile.ZipFile(OUT/"top15_retraining_six_metrics.zip") as archive:
    assert archive.testzip() is None and len(archive.namelist())==36
print("Verified: 15 functions, 75 trial histories, 3750 epoch records, 74 final test evaluations.")
print("Created 15 six-panel figures (PNG/SVG), gallery, data CSVs and ZIP.")
