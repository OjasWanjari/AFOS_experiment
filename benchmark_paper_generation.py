"""Run exactly 30 random individuals (seed 0, generation 1), then estimate time."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

from run_paper import load_config, main as run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {root}")
    config = load_config(Path(__file__).with_name("paper_parameters.json"))
    config["search"].update(generations=1, replications=1, seeds=[0])
    if config["search"]["population_size"] != 30:
        raise ValueError("This benchmark requires the paper's 30 individuals")
    config["benchmark_note"] = "Only seed 0 / generation 1; no retraining. Paper training and early stopping unchanged."
    config_path = root.with_suffix(".config.json")
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Benchmark configuration changed")
    else:
        write_json(config_path, config)
    import paper_afos.model as module
    original_build, original_train = module.build_model, module.train_candidate
    build_time = [0.0]
    def timed_build(*a, **kw):
        start = time.perf_counter()
        model = original_build(*a, **kw)
        build_time[0] = time.perf_counter() - start
        return model
    def timed_train(*a, **kw):
        kw["verbose"] = 2  # readable epoch lines in the live log
        start = time.perf_counter()
        metrics, history = original_train(*a, **kw)
        metrics["training_seconds"] = time.perf_counter() - start
        metrics["model_build_seconds"] = build_time[0]
        metrics["evaluation_seconds"] = metrics["training_seconds"] + build_time[0]
        print(f"TIMING individual={metrics['evaluation_seconds']:.2f}s epochs={metrics['epochs_completed']} status={metrics['status']}", flush=True)
        return metrics, history
    module.build_model, module.train_candidate = timed_build, timed_train
    started_at = datetime.now().astimezone().isoformat()
    start = time.perf_counter()
    try:
        run(["--config", str(config_path), "--stage", "search", "--device", "gpu",
             "--resume" if args.resume else "--output", str(root)])
    finally:
        module.build_model, module.train_candidate = original_build, original_train
    elapsed = time.perf_counter() - start
    state = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    rows = state["evaluations"]
    if len(rows) != 30 or not state["search_complete"] or state["retraining"]:
        raise RuntimeError("Expected exactly one completed generation and no retraining")
    times = [r["evaluation_seconds"] for r in rows]
    counts = dict(Counter(r["status"] for r in rows))
    groups = {status: {"count": n, "mean_seconds": statistics.mean(r["evaluation_seconds"] for r in rows if r["status"] == status)}
              for status, n in counts.items()}
    result = dict(started_at=started_at, finished_at=datetime.now().astimezone().isoformat(),
                  resumed=args.resume, wall_seconds_this_invocation=elapsed,
                  individuals=30, total_epochs=sum(r["epochs_completed"] for r in rows),
                  evaluation_seconds=sum(times), mean_individual_seconds=statistics.mean(times),
                  outcomes=groups,
                  one_seed_40_generations_hours=sum(times)*40/3600,
                  five_seeds_40_generations_hours=sum(times)*200/3600,
                  five_seeds_40_generations_days=sum(times)*200/86400,
                  benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  caveat="Linear extrapolation of the first random generation, not a confidence interval. Later generations retain strong parents and may have fewer early stops. No post-search retraining included.")
    if not args.resume:
        result["five_seeds_wall_scaled_days"] = elapsed*200/86400
    if "completed" in groups:
        result["all_full_length_scenario_days"] = groups["completed"]["mean_seconds"]*6000/86400
    write_json(root / "time_estimate.json", result)
    report = "# 1世代の実測による探索時間の見積もり\n\n"
    report += "GTX 1080 Ti / TensorFlow 2.10 + DirectML。30個体、seed 0、論文条件、最大15エポック。Test未評価。\n\n"
    report += f"- 個体評価の合計: {sum(times)/60:.2f}分\n- 平均1個体: {statistics.mean(times):.2f}秒\n"
    report += f"- 学習エポック合計: {result['total_epochs']}\n- 今回のプロセス全体: {elapsed/60:.2f}分（初期化・読み込み・保存・描画を含む）\n\n"
    report += "| 終了状態 | 個体数 | 平均評価時間 |\n| --- | ---: | ---: |\n"
    for status, group in groups.items():
        report += f"| {status} | {group['count']} | {group['mean_seconds']:.2f}秒 |\n"
    report += f"\n単純比例では、1 seed・40世代: **{result['one_seed_40_generations_hours']:.2f}時間**、5 seeds・40世代: **{result['five_seeds_40_generations_hours']:.2f}時間（{result['five_seeds_40_generations_days']:.2f}日）**。\n"
    if "all_full_length_scenario_days" in result:
        report += f"\n全個体が今回の完走個体の平均と同じ時間で15エポック学習する仮定では、5 seedsで約{result['all_full_length_scenario_days']:.2f}日。\n"
    report += "\n初期ランダム世代の実測であり、後続世代では良い個体が残って完走率が上がる可能性がある。統計的な予測区間ではなく、実行時間の厳密な上下限でもない。探索後の50エポック再学習・既存関数比較は含まない。\n"
    (root / "TIME_ESTIMATE.md").write_text(report, encoding="utf-8")
    with (root / "run.log").open("a", encoding="utf-8") as log:
        log.write("\nGENERATION TIMING COMPLETE\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
