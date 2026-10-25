"""Audit saved experiments and write bilingual comparison reports; no training.

Run with Python and matplotlib. Source files are read-only inputs.
"""
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from paper_afos.retraining import read_plan, sha256
from paper_afos.runtime_environment import environment_mismatches

DATE = "2026-10-25"
RUNS = {
    "discovered": ROOT / "paper_full_results/retrain_runs/pooled_top15_updated",
    "published": ROOT / "paper_full_results/published_top15_matched",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_run(root, expected_environment):
    plan = read_plan(root)
    records = read_csv(root / "results/retraining_results.csv")
    planned = {r["trial_id"]: r for r in plan["trials"]}
    assert len(records) == len(planned) == len({r["trial_id"] for r in records})
    assert set(planned) == {r["trial_id"] for r in records}
    audit = dict(planned=len(planned), environment_matches=0, models=0,
                 oom_logs=0, traceback_logs=0, layout_warning_logs=0, pids=set(),
                 failures=[], terminal_counts=dict(Counter(r["status"] for r in records)))
    grouped = {}
    for row in records:
        trial_id = row["trial_id"]
        attempt = root / "trials" / trial_id / row["attempt"]
        saved = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
        for key in ("function", "status", "activation_expression", "trial_id"):
            assert row[key] == saved[key], (trial_id, key)
        for key in ("function", "seed", "chromosome", "baseline", "activation_expression"):
            assert saved[key] == planned[trial_id][key], (trial_id, key)
        assert int(row["seed"]) == saved["seed"]
        for key in ("test_accuracy", "test_loss", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"):
            assert (float(row[key]) if row.get(key) else None) == saved.get(key), (trial_id, key)
        histories = read_csv(attempt / "histories" / f"{trial_id}.csv")
        assert len(histories) == saved["epochs_completed"] == int(row["epochs_completed"])
        assert [int(h["epoch"]) for h in histories] == list(range(1, len(histories) + 1))
        log = (attempt / "worker.log").read_text(encoding="utf-8", errors="replace")
        env = json.loads((attempt / "environment.json").read_text(encoding="utf-8"))
        if "gpu_names" not in env:
            env["gpu_names"] = re.findall(r"name: ([^\r\n]+?), pci bus id:", log)
        assert not environment_mismatches(expected_environment, env), trial_id
        audit["environment_matches"] += 1
        if "reference_environment" in plan:
            check = json.loads((attempt / "environment_check.json").read_text(encoding="utf-8"))
            assert check["matched"] and not check["differences"]
        audit["pids"].add(saved["pid"])
        audit["oom_logs"] += any(word in log.lower() for word in ("out of memory", "resourceexhausted", "resource_exhausted"))
        audit["traceback_logs"] += "traceback" in log.lower()
        audit["layout_warning_logs"] += "layout failed" in log.lower()
        model_saved = (attempt / "model.keras").exists()
        audit["models"] += model_saved
        if saved["status"] == "completed":
            assert len(histories) == 50 and model_saved
            assert all(math.isfinite(float(h[k])) for h in histories
                       for k in ("loss", "accuracy", "val_loss", "val_accuracy"))
            assert all(math.isfinite(saved[k]) for k in ("test_accuracy", "test_loss"))
        else:
            assert saved["status"] == "nonfinite" and saved["test_accuracy"] is None
            first = next(h for h in histories if any(not math.isfinite(float(h[k]))
                         for k in ("loss", "accuracy", "val_loss", "val_accuracy")))
            audit["failures"].append(dict(trial_id=trial_id, function=saved["function"], seed=saved["seed"],
                                          epoch=int(first["epoch"]), loss=first["loss"], val_loss=first["val_loss"],
                                          accuracy_finite=math.isfinite(float(first["accuracy"])),
                                          val_accuracy_finite=math.isfinite(float(first["val_accuracy"]))))
        grouped.setdefault(row["function"], []).append(saved)
    summaries = {r["function"]: r for r in read_csv(root / "results/retraining_summary.csv")}
    computed = []
    for function, trials in grouped.items():
        assert sorted(t["seed"] for t in trials) == [0, 1, 2, 3, 4]
        trials.sort(key=lambda t: t["seed"])
        successful = [t["test_accuracy"] * 100 for t in trials if t["status"] == "completed"]
        complete = len(successful) == 5
        mean = statistics.mean(successful) if complete else None
        sd = statistics.stdev(successful) if complete else None
        summary = summaries[function]
        assert int(summary["completed"]) == len(successful)
        if complete:
            assert abs(mean - float(summary["test_accuracy_mean"]) * 100) < 1e-10
            assert abs(sd - float(summary["test_accuracy_std"]) * 100) < 1e-10
        else:
            assert summary["test_accuracy_mean"] == summary["test_accuracy_std"] == ""
        computed.append(dict(function=function, expression=trials[0]["activation_expression"],
                             completed=len(successful), mean_percent=mean, sd_pp=sd,
                             seeds=[t["test_accuracy"] * 100 if t["status"] == "completed" else None for t in trials],
                             statuses=[t["status"] for t in trials]))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["status_counts"] == audit["terminal_counts"]
    assert status["planned"] == len(planned)
    audit["unique_process_ids"] = len(audit.pop("pids"))
    audit["status_updated_at"] = status["updated_at"]
    audit["input_hashes"] = {name: sha256(root / name) for name in (
        "plan.json", "split_indices.npz", "selected_top15.csv", "status.json",
        "results/retraining_results.csv", "results/retraining_summary.csv")}
    return plan, computed, audit


def plots(discovered, published, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"discovered": "#087F8C", "published": "#5267AE", "failure": "#BA4A4A", "collapse": "#D18A25"}
    fig, ax = plt.subplots(figsize=(9, 3.3), layout="constrained")
    left = np.array([0, 0])
    for label, values, color in [("Completed, other accuracies", [74, 48], colors["discovered"]),
                                 ("Completed, test accuracy 10%", [0, 2], colors["collapse"]),
                                 ("Stopped with NaN/Inf", [1, 25], colors["failure"])]:
        bars = ax.barh([1, 0], values, left=left, color=color, label=label, height=.5)
        for b, v in zip(bars, values):
            if v >= 2:
                ax.text(b.get_x()+v/2, b.get_y()+b.get_height()/2, str(v), ha="center", va="center", color="white", weight="bold")
        left += values
    ax.annotate("1", xy=(74.5, 1.25), xytext=(74.5, 1.4), ha="center", color=colors["failure"])
    ax.set_yticks([1, 0], ["Previous discovered Top 15", "Published Top 15"])
    ax.set_xlim(0, 75)
    ax.set_xticks([0, 15, 30, 45, 60, 75])
    ax.set_xlabel("Trials (75 per group; five retraining seeds per candidate)")
    ax.set_title("Execution outcomes: finite completion and learning quality are different", loc="left", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=1, frameon=False)
    fig.savefig(OUT / "execution_status.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 8), sharex=True, layout="constrained")
    for ax, rows, title, color in zip(axs, [discovered, published],
                                    ["Previous discovered candidates", "Published functions in the same environment"],
                                    [colors["discovered"], colors["published"]]):
        for i, r in enumerate(rows):
            y = len(rows) - i - 1
            if r["mean_percent"] is not None:
                if r["function"] == "paper_05":
                    ax.text(78.05, y, "50.984 +/- 37.416% (outside zoom); 2 seeds at 10%", color=colors["collapse"], fontsize=8)
                else:
                    ax.errorbar(r["mean_percent"], y, xerr=r["sd_pp"], fmt="o", color=color, capsize=3, markersize=5)
            else:
                ax.text(78.05, y, f"No five-seed mean ({r['completed']}/5 completed)", color=colors["failure"], fontsize=9)
        ax.axvline(baselines["relu"]["mean_percent"], color="#666666", ls=":", lw=1, label="ReLU mean")
        ax.axvline(baselines["gelu"]["mean_percent"], color="#888888", ls="--", lw=1, label="GELU mean")
        ax.set_yticks(list(reversed(range(15))), [r["function"].replace("discovered_", "D") for r in rows])
        ax.set_xlim(77.9, 84.5)
        ax.set_ylim(-.7, 14.7)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("Test accuracy (%) | mean +/- sample SD, five seeds")
        ax.grid(axis="x", alpha=.2)
    axs[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Function-level comparison (zoomed axis; missing means are not zero)", fontsize=15)
    fig.savefig(OUT / "function_comparison.png", dpi=180)
    plt.close(fig)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)] +
                     ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "tmp/top15_comparison_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    paper_plan = read_plan(RUNS["published"])
    expected = paper_plan["reference_environment"]
    old_plan, all_old, old_audit = audit_run(RUNS["discovered"], expected)
    new_plan, published, new_audit = audit_run(RUNS["published"], expected)
    assert old_plan["config"] == new_plan["config"]
    assert old_plan["runtime"] == new_plan["runtime"]
    assert old_plan["split_sha256"] == new_plan["split_sha256"]
    assert new_plan["reference"]["plan_sha256"] == sha256(RUNS["discovered"] / "plan.json")
    for rel, digest in new_plan["reference"]["computation_source_sha256"].items():
        assert old_plan["source_sha256"][rel] == digest
    preflight = json.loads((RUNS["published"] / "environment_preflight.json").read_text())
    assert preflight["matched"] and not environment_mismatches(expected, preflight["environment"])
    discovered = [r for r in all_old if r["function"].startswith("discovered_")]
    baselines = {r["function"]: r for r in all_old if not r["function"].startswith("discovered_")}
    catalog = {r["function"]: r for r in new_plan["selected"]}
    for r in published:
        r.update(paper_mean_percent=catalog[r["function"]]["paper_test_mean_percent"],
                 paper_sd_pp=catalog[r["function"]]["paper_test_std_percent"],
                 eelu_name=catalog[r["function"]]["eelu_name"])
        r["delta_from_paper_pp"] = r["mean_percent"]-r["paper_mean_percent"] if r["mean_percent"] is not None else None
    d = {r["function"]: r for r in discovered}
    p = {r["function"]: r for r in published}
    comparisons = {}
    for name, comparator in [("paper_14", p["paper_14"]), ("paper_01", p["paper_01"]),
                             ("paper_04", p["paper_04"]), ("gelu", baselines["gelu"])]:
        differences = [a-b for a,b in zip(d["discovered_01"]["seeds"], comparator["seeds"])]
        comparisons[name] = dict(mean_difference_pp=statistics.mean(differences),
                                 paired_differences_pp=differences, all_five_positive=all(v>0 for v in differences))
    audit = dict(analysis_date=DATE, config_split_and_preserved_code_match=True,
                 split_sha256=old_plan["split_sha256"], reference_environment=expected,
                 runs={"discovered_with_baselines": old_audit, "published": new_audit},
                 discovered=discovered, published=published, baselines=baselines, comparisons=comparisons,
                 previous_complete_candidates_above_best_published=sum(r["mean_percent"] is not None and r["mean_percent"]>p["paper_14"]["mean_percent"] for r in discovered))
    write_json(OUT / "audit.json", audit)
    plots(discovered, published, baselines)
    write_reports(audit)
    print(json.dumps({"completion": {k:v["terminal_counts"] for k,v in audit["runs"].items()},
                      "comparisons": comparisons, "above_best_published":audit["previous_complete_candidates_above_best_published"]}, indent=2))


def write_reports(audit):
    discovered, published, baselines = audit["discovered"], audit["published"], audit["baselines"]
    d, p = ({r["function"]:r for r in rows} for rows in (discovered, published))
    old_a, new_a = audit["runs"]["discovered_with_baselines"], audit["runs"]["published"]
    def metric(r, ja):
        return f"{r['mean_percent']:.3f} ± {r['sd_pp']:.3f}" if r["mean_percent"] is not None else ("算出なし" if ja else "Not reported")
    for ja in (True, False):
        heading = "前回の探索Top 15と論文Top 15の同条件再学習比較" if ja else "Matched retraining comparison: discovered Top 15 versus published Top 15"
        parts = [f"# {heading}", f"{'分析日' if ja else 'Analysis date'}: {DATE}",
                 "[English version](README_EN.md)" if ja else "[日本語版](README_JA.md)"]
        parts += ["保存された結果を再検査・再集計した。追加学習や元データの変更は行っていない。対象は前回の新seed 4を含む探索から選ばれた15候補と、今回実行された論文表3の15関数である。" if ja else
                  "Saved results were audited and reaggregated without additional training or changes to source data. The comparison covers the previous 15 candidates selected using all searches including the updated seed 4, and the newly executed 15 functions from Table 3 of the paper."]
        parts += ["## 結論" if ja else "## Main findings"]
        parts += ["**今回の実装・記録された環境・データ分割では、前回の探索候補の方が、観測された最高平均精度と数値的な完了率の両方で良い結果を示した。** ただし、これは当該条件での比較であり、論文の手法全体や別データセットに対する普遍的な優位性を示すものではない。" if ja else
                  "**Under this implementation, recorded environment and data split, the previous discovered candidates showed both a higher observed best mean accuracy and a higher numerical completion rate.** This is a result for these conditions, not evidence of universal superiority over the paper's method or across other datasets."]
        parts += [table(["比較項目", "前回の探索Top 15", "論文Top 15の今回の再評価"] if ja else
                        ["Measure", "Previous discovered Top 15", "Published Top 15 rerun"], [
            ["予定試行" if ja else "Planned trials", "75", "75"],
            ["50エポック・テスト評価完了" if ja else "Completed 50 epochs and test evaluation", "74/75 (98.67%)", "50/75 (66.67%)"],
            ["NaN/Inf停止" if ja else "NaN/Inf stops", "1/75 (1.33%)", "25/75 (33.33%)"],
            ["5シードすべて完了した候補" if ja else "Candidates with 5 completed seeds", "14/15", "10/15"],
            ["完了試行のうちテスト精度10%" if ja else "Completed trials at 10% test accuracy", "0", "2 (paper_05, seeds 0, 3)"],
            ["最高平均±SD (%)" if ja else "Best mean ± SD (%)", f"discovered_01: {metric(d['discovered_01'],ja)}", f"paper_14: {metric(p['paper_14'],ja)}"],
        ])]
        parts += [f"前回の最良候補と今回再評価した論文関数中の最良関数の差は **+{audit['comparisons']['paper_14']['mean_difference_pp']:.3f}ポイント**。論文で最良だったEELU-4の今回の再評価値との差は **+{audit['comparisons']['paper_01']['mean_difference_pp']:.3f}ポイント**。前回の完了14候補のうち13候補が論文関数中の最高平均81.710%を上回った。" if ja else
                  f"The best discovered candidate exceeds the best rerun published function by **{audit['comparisons']['paper_14']['mean_difference_pp']:.3f} percentage points (pp)**, and the rerun of EELU-4, the paper's original leader, by **{audit['comparisons']['paper_01']['mean_difference_pp']:.3f} pp**. Thirteen of the 14 discovered candidates with complete results exceed the best published-function rerun mean of 81.710%."]
        parts += ["候補07と10は数学的に同一の関数で、前回の15候補には重複が含まれる。上記13候補にもこの2件が含まれるため、異なる13種類の関数と解釈しない。" if ja else
                  "Discovered candidates 07 and 10 are mathematically identical. The previous set contains a duplicate, and both entries are included among the 13 candidates above; these are not 13 mathematically distinct functions."]

        parts += ["## 1. 比較条件と検証範囲" if ja else "## 1. Comparison conditions and verification"]
        parts += [table(["項目", "条件・確認結果"] if ja else ["Item", "Conditions / checks"], [
            ["Data", "CIFAR-10: 45,000 train / 5,000 validation / 10,000 test"],
            ["Retraining", "50 epochs from scratch; seeds 0, 1, 2, 3, 4"],
            ["Optimization", "SGD; learning rate 0.01; momentum 0.9; batch size 64"],
            ["CNN", "6 convolutions: 28, 32, 64, 64, 128, 128; dropout 0.2; he_uniform"],
            ["Runtime", "Linux; RTX 4080 SUPER; Python 3.10.12; TensorFlow 2.21.0; Keras 3.12.4"],
            ["Packages", "NumPy 2.2.6; scikit-learn 1.7.2"],
            ["Execution", "JIT disabled; memory growth enabled; one process per trial"],
            ["Evaluation", "Final model after 50 epochs; test set excluded from training"],
        ])]
        parts += ["両計画の設定全体と実行設定、分割ファイルのSHA-256が一致した。論文側で維持対象として記録されたモデル・データ読み込み・演算・学習・集計等の保存コードは、前回の保存コードとハッシュが一致した。workerには環境記録・照合の変更がある。前回105試行（基準関数30試行を含む）と論文側75試行の計180件について、試行ID、設定された関数・seed、CSVとJSONの指標、履歴、保存モデル、環境、集計値を照合した。" if ja else
                  "The full configurations, runtime settings and split-file SHA-256 match. Code recorded as preserved in the published-function plan, including the model, data loader, evaluator, training and accounting, matches the previous snapshot hashes. The worker differs in environment recording/checking. Across 180 trials (105 previous trials, including 30 baseline trials, plus 75 published-function trials), IDs, planned functions/seeds, CSV versus JSON metrics, histories, saved models, environments and summaries were checked."]
        parts += [f"分割SHA-256: `{audit['split_sha256']}`" if ja else f"Split SHA-256: `{audit['split_sha256']}`"]
        parts += ["全180試行で記録された比較対象の環境項目が一致し、論文側75件の環境チェックも合格していた。ただし元のGPUドライバや全環境変数は未記録で、GPU計算のビット単位一致も保証されない。「同条件」は、この検証可能な範囲を意味する。" if ja else
                  "Recorded comparison environment fields match for all 180 trials, and all 75 published-function environment checks passed. Original GPU driver versions and every environment variable were not recorded, and bitwise GPU determinism is not guaranteed. Matching conditions refers to this verifiable scope."]
        parts += ["平均は5シードすべて成功した関数のみ算出し、SDは標本標準偏差（ddof=1）。失敗を0%に置換したり、成功したseedだけで正式平均を作ったりしていない。以下の差は相対的な増加率ではなく、精度のパーセントポイント差である。" if ja else
                  "Means are reported only for functions with all five successful seeds; SD is the sample standard deviation (ddof=1). Failed trials are neither replaced with 0% nor omitted to manufacture an official mean. Differences below are percentage-point differences, not relative percentage improvements."]

        parts += ["## 2. 論文Top 15の実行状況" if ja else "## 2. Execution status of the published Top 15"]
        parts += ["**全75試行に終了結果があり、未実行・実行中は0件。50件が完了、25件が数値的失敗。** `all_evaluations_completed=false` は未実行が残っている意味ではなく、全件成功していないためである。保存モデルは50個、モデル保存エラーは0件だった。" if ja else
                  "**All 75 trials have terminal outcomes: none are pending or running. Fifty completed and 25 failed numerically.** `all_evaluations_completed=false` reflects unsuccessful trials, not unfinished scheduling. There are 50 saved models and no recorded model-save errors."]
        parts += ["![Execution status](execution_status.png)"]
        parts += ["青緑は10%の2件を除く完了試行、橙は完了してもテスト精度10%だった試行、赤はNaN/Inf停止。前回の基準関数30試行はこの75対75の図に含めない。" if ja else
                  "Teal denotes completed trials other than the two at 10%; orange denotes those two completed trials; red denotes NaN/Inf stops. The previous 30 baseline trials are excluded from this 75-versus-75 figure."]
        failed_functions = sorted({r["function"] for r in new_a["failures"]})
        failure_rows = []
        for name in failed_functions:
            records = [r for r in new_a["failures"] if r["function"] == name]
            failure_rows.append([name, "0–4", "0/5", "1", str(sum(r["loss"].lower()=="nan" for r in records)),
                                 str(sum(r["loss"].lower()=="inf" for r in records)), "5"])
        parts += [table(["関数", "失敗seed", "完了", "最初の非有限値epoch", "loss NaN", "loss Inf", "val_loss NaN"] if ja else
                        ["Function", "Failed seeds", "Completed", "First nonfinite epoch", "loss NaN", "loss Inf", "val_loss NaN"], failure_rows)]
        parts += ["25件すべてで最初の非有限値は1エポック目。訓練損失は23件がNaN、2件がInfで、検証損失は25件すべてNaN。失敗時のaccuracy・val_accuracyは有限値であり、精度が数値として出ていても学習が正常とは限らない。テスト評価は実施・保存されていない。" if ja else
                  "The first nonfinite epoch is epoch 1 in all 25 failures. Training loss is NaN in 23 trials and Inf in two; validation loss is NaN in all 25. Accuracy and validation accuracy remain finite at the recorded failing epoch, so finite accuracy does not establish valid training. No test evaluation is recorded for these failed trials."]
        parts += [f"論文側75件のログにOOM・Tracebackは0件。layout optimizer警告は{new_a['layout_warning_logs']}件にあり、成功試行にも現れるため、警告だけで失敗原因とは断定できない。" if ja else
                  f"The 75 published-function logs contain zero OOM matches and zero tracebacks. Layout optimizer warnings appear in {new_a['layout_warning_logs']} logs, including successful trials; the warning alone does not identify the cause."]
        parts += ["### paper_05：有限値でも学習が崩れる例" if ja else "### paper_05: finite metrics can still accompany failed learning"]
        parts += [table(["seed", "テスト精度 (%)" if ja else "Test accuracy (%)"] , [[i,f"{v:.2f}"] for i,v in enumerate(p["paper_05"]["seeds"])])]
        parts += ["seed 0と3は50エポックを終えてもテスト精度10.00%、最終訓練損失は約2.303だった。これは10クラス分類のチャンスレベルにとどまる挙動であり、正常に高精度へ学習できたとは扱わない。一方、NaN/Infではないため、処理上はcompletedである。5回の正式平均 **50.984 ± 37.416%** にこの2回を含めている。生の予測や勾配がないため、常に特定クラスを予測した等の内部動作は断定しない。" if ja else
                  "Seeds 0 and 3 completed 50 epochs but achieved 10.00% test accuracy, with final training loss around 2.303. This is chance-level behavior for ten classes, not successful learning to high accuracy. Because metrics are finite, the execution status is nevertheless completed. Both trials remain included in the official **50.984 ± 37.416%** five-seed result. Raw predictions and gradients are unavailable, so behaviors such as always predicting one class are not established."]

        parts += ["## 3. 論文15関数：今回の再評価と論文報告値" if ja else "## 3. All 15 published functions: rerun versus reported results"]
        rows = [[r["function"]+(f" ({r['eelu_name']})" if r["eelu_name"] else ""), f"{r['completed']}/5",
                 f"{r['paper_mean_percent']:.2f} ± {r['paper_sd_pp']:.2f}", metric(r,ja),
                 f"{r['delta_from_paper_pp']:+.3f}" if r["delta_from_paper_pp"] is not None else "—"] for r in published]
        parts += [table(["論文関数", "完了", "論文の平均±SD (%)", "今回の平均±SD (%)", "今回−論文 (pp)"] if ja else
                        ["Published function", "Completed", "Reported mean ± SD (%)", "Rerun mean ± SD (%)", "Rerun − paper (pp)"], rows)]
        parts += ["論文報告値は提供PDF（arXiv:2409.04915v1）の表3から実行計画に転記された値を使用した。`paper_XX`は表3の掲載順であり、EELU番号や今回の精度順位とは異なる。" if ja else
                  "Reported values are the Table 3 values from the supplied arXiv:2409.04915v1 PDF transcribed into the frozen plan. `paper_XX` denotes Table 3 row order, not EELU numbering or rerun rank."]
        parts += ["5回完了した10関数はすべて、今回の平均が論文報告値より低い。paper_05を除いた9関数でも差は−0.440〜−3.094ポイント。論文の最良関数EELU-4は80.406%へ低下し、今回の論文関数中の最高はpaper_14の81.710%だった。これは、関数だけを移植しても論文の精度や順位がそのまま再現されなかったことを示す。" if ja else
                  "All ten functions with five completed repetitions have rerun means below their published means. Even excluding paper_05, the nine differences range from −0.440 to −3.094 pp. EELU-4, the paper's leader, achieves 80.406% here, while paper_14 leads the reruns at 81.710%. Transferring the functions into this implementation did not reproduce the paper's accuracy levels or ordering."]
        parts += ["### 論文関数のseed別テスト精度 (%)" if ja else "### Per-seed test accuracy for the published functions (%)"]
        parts += [table(["関数" if ja else "Function", "seed 0", "seed 1", "seed 2", "seed 3", "seed 4"],
                        [[r["function"]]+[f"{v:.2f}" if v is not None else ("NaN/Inf停止" if ja else "NaN/Inf stop") for v in r["seeds"]] for r in published])]

        parts += ["## 4. 前回の探索Top 15との精度比較" if ja else "## 4. Accuracy comparison with the previous discovered Top 15"]
        ranked = sorted([r for r in discovered if r["mean_percent"] is not None], key=lambda r:-r["mean_percent"])
        previous_rows = []
        for r in ranked+[d["discovered_13"]]:
            rank = 1+sum(round(o["mean_percent"],3)>round(r["mean_percent"],3) for o in ranked) if r["mean_percent"] is not None else "—"
            previous_rows.append([rank, r["function"], f"{r['completed']}/5", metric(r,ja),
                                  f"{r['mean_percent']-p['paper_14']['mean_percent']:+.3f}" if r["mean_percent"] is not None else "—"])
        parts += [table(["平均順位", "探索候補", "完了", "平均±SD (%)", "論文関数の今回最高値との差 (pp)"] if ja else
                        ["Mean rank", "Discovered candidate", "Completed", "Mean ± SD (%)", "Difference from best published rerun (pp)"], previous_rows)]
        parts += ["表示精度で同じ82.934%の候補04・07は同順位。候補13はseed 1が50エポック目にNaNとなり、5回平均は未算出。候補13の成功seed 0,2,3,4は81.31%, 79.95%, 81.84%, 81.57%。この4回平均を比較表の順位に入れていない。" if ja else
                  "Candidates 04 and 07 tie at the displayed precision of 82.934%. Candidate 13 lacks a five-run mean because seed 1 became NaN in epoch 50. Its successful seeds 0, 2, 3 and 4 achieved 81.31%, 79.95%, 81.84% and 81.57%; a four-run mean is not inserted into the ranking."]
        parts += ["![Function comparison](function_comparison.png)"]
        parts += ["図は平均±標本SD。横軸を拡大しており、paper_05の50.984%は範囲外と注記した。算出できない平均を0%で描いていない。破線は前回の基準関数の平均で、新たな基準関数の再実行ではない。" if ja else
                  "The figure shows means ± sample SD on a zoomed axis. paper_05's 50.984% is explicitly labeled as outside the view. Unavailable means are not plotted as zero. Reference lines use the previous baseline results; baselines were not rerun in the new 75 trials."]
        parts += ["### 最良関数同士を同じseedで比較" if ja else "### Best observed functions compared at the same seeds"]
        pairs = audit["comparisons"]["paper_14"]["paired_differences_pp"]
        parts += [table(["seed", "discovered_01 (%)", "paper_14 (%)", "差 / Difference (pp)"],
                        [[i,f"{d['discovered_01']['seeds'][i]:.2f}",f"{p['paper_14']['seeds'][i]:.2f}",f"{pairs[i]:+.2f}"] for i in range(5)])]
        parts += ["discovered_01は全5シードでpaper_14を上回った。平均差は+1.872ポイント。EELU-4との平均差は+3.176ポイント、今回のEELU群で最も高いEELU-3との差は+2.212ポイントだった。これらは観測差であり、有意差検定の結果ではない。最良関数はテスト結果を見て比較対象として選んでいるため、選択の影響もある。" if ja else
                  "discovered_01 exceeds paper_14 at all five seeds, with a mean difference of +1.872 pp. Its mean difference is +3.176 pp versus EELU-4 and +2.212 pp versus EELU-3, the best EELU rerun. These are observed differences, not significance-test results. Both leaders are identified after observing test results, so the comparison also involves selection effects."]

        parts += ["## 5. 基準関数との位置関係" if ja else "## 5. Baseline context"]
        parts += [table(["基準関数", "完了", "平均±SD (%)"] if ja else ["Baseline", "Completed", "Mean ± SD (%)"], [[name,f"{r['completed']}/5",metric(r,ja)] for name,r in baselines.items()])]
        parts += ["基準関数は前回の105試行に含まれる結果を参照した。今回の論文75試行で再実行していない。論文関数の最良paper_14はGELUより0.780ポイント低く、Swishより0.084ポイント低い。前回の探索候補では8候補がGELUを上回った。前回のELUは全5シード失敗のため比較可能な平均がない。" if ja else
                  "These baselines come from the previous 105-trial experiment, not new baseline runs in the 75-trial evaluation. The best published rerun, paper_14, is 0.780 pp below GELU and 0.084 pp below Swish. Eight previous discovered candidates exceeded GELU. ELU failed at all five seeds in the previous run, so it has no comparable mean."]

        parts += ["## 6. 解釈・限界・次の確認" if ja else "## 6. Interpretation, limitations, and next checks"]
        parts += ["1. **今回の実装への適合性:** 前回の候補はこのデータ分割・実装を使った検証精度で選ばれた。一方、論文関数は別実験で選ばれた固定集合である。前回候補の精度・完了率が高いことは、この条件への適合性と整合するが、差の原因を一つに特定する証拠ではない。\n2. **元論文の完全再現とは区別:** 両群の比較条件は照合したが、元論文の厳密な分割・実行環境まで同一とは確認できない。EELU-2は表4の重複した実装欄ではなく数式を使用。EELU-3は既存演算木の積の形、softplusは前回と同じ安定実装を使用している。\n3. **NaN/Infの発生源は未確定:** OOMは記録されていない。例えば失敗したpaper_13=`max(x,erf(x))`にはexpも定義域制約のある逆三角関数もないため、全失敗をexpのオーバーフローだけで説明できない。最初の非有限値が活性化・畳み込み・勾配・重み更新のどこで生じたかを記録する必要がある。\n4. **失敗を除く集計の限界:** 未完了関数を除いた群全体の平均や、10%の2試行を除いたpaper_05の平均を主結果にしない。完了率と精度を併記し、欠測を隠さない。\n5. **次の実験:** paper_08/10/11/12/13のseed 0を診断用に計測し、paper_05のseed 0・3では初期出力・損失・勾配の推移を調べる。前回の候補13 seed 1とELUも未解決。学習率変更・クリッピング等は元の記録を上書きせず、別実験として報告する。\n6. **一般化の限界:** 5シードは同じデータ分割での学習の反復であり、独立した5データセットではない。別分割・別モデルや、独立に固定した比較計画での追加検証が必要。" if ja else
                  "1. **Fit to this implementation:** The previous candidates were selected by validation accuracy using this split and implementation. The published functions are a fixed set selected in another experiment. Better accuracy and completion of the previous candidates are consistent with adaptation to these conditions, but do not establish a single cause.\n2. **Separate matched comparison from exact paper reproduction:** Conditions between these two runs were checked; identity with the original paper's precise split/environment was not established. EELU-2 follows the mathematical formula rather than Table 4's duplicated code. EELU-3 uses the existing product expression tree, and softplus uses the same stable implementation as the previous run.\n3. **NaN/Inf origins remain unknown:** No OOM is recorded. For example, failed paper_13=`max(x,erf(x))` contains neither exp nor a domain-restricted inverse trigonometric function, so exponential overflow alone cannot explain all failures. Capture the first nonfinite activation, convolution, gradient or weight update.\n4. **Avoid survivor-only conclusions:** Do not headline a group-wide average after excluding incomplete functions, or a paper_05 average after dropping its two 10% trials. Report completion and accuracy together without hiding missing outcomes.\n5. **Next experiments:** Instrument seed 0 of paper_08/10/11/12/13, and examine initial outputs, losses and gradients for paper_05 seeds 0 and 3. Previous candidate 13 seed 1 and ELU remain unresolved. Report learning-rate changes or clipping as separate experiments without overwriting the original records.\n6. **Generalization limits:** Five seeds are repeated training runs on one data split, not five independent datasets. Additional splits, models or an independently fixed comparison plan are needed."]

        parts += ["## 付録：比較した関数の式" if ja else "## Appendix: evaluated function expressions"]
        parts += ["式は実行計画に保存された表記。`log(1+exp(x))`は実装では`tf.nn.softplus`。" if ja else
                  "Expressions follow the frozen execution plans. `log(1+exp(x))` is implemented with `tf.nn.softplus`."]
        parts += [table(["前回の探索候補", "関数式"] if ja else ["Previous candidate", "Expression"], [[r["function"],f"`{r['expression']}`"] for r in discovered])]
        parts += [table(["論文関数", "関数式"] if ja else ["Published function", "Expression"], [[r["function"],f"`{r['expression']}`"] for r in published])]
        parts += ["## データと再現方法" if ja else "## Sources and reproduction"]
        parts += ["- [Previous trial results](../../paper_full_results/retrain_runs/pooled_top15_updated/results/retraining_results.csv)\n- [Previous function summary](../../paper_full_results/retrain_runs/pooled_top15_updated/results/retraining_summary.csv)\n- [Published-function trial results](../../paper_full_results/published_top15_matched/results/retraining_results.csv)\n- [Published-function summary](../../paper_full_results/published_top15_matched/results/retraining_summary.csv)\n- [Published-function execution status](../../paper_full_results/published_top15_matched/status.json)\n- [Published formulas and original reported values](../../paper_full_results/published_top15_matched/selected_top15.csv)\n- [Audit and input hashes](audit.json)\n- [Reproduction script](analyze.py)\n- Reference: *Activation Function Optimization Scheme for Image Classification*, arXiv:2409.04915v1, Table 3; previous baseline context from Table 6."]
        parts += ["再生成には両実験のplan・source_snapshot・split・trials・resultsとPython / matplotlibが必要。リポジトリ直下で次を実行する。新しい学習は行わない。" if ja else
                  "Reproduction requires both runs' plans, source snapshots, splits, trials and results, plus Python and matplotlib. Run from the repository root; this does not start training."]
        parts += ["```bash\npython analysis/top15_matched_comparison/analyze.py\n```"]
        (OUT / ("README_JA.md" if ja else "README_EN.md")).write_text("\n\n".join(parts)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
