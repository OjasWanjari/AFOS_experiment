"""Retrain pooled AFOS candidates only. See RETRAIN_POOLED_EN.md / RETRAIN_POOLED_JA.md."""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from paper_afos.retraining import (RETRYABLE, attempts_for, latest_result, now, prepare_run,
                                  read_plan, summarize, write_json)


def execute(output, plan, retry_errors=False, max_trials=None, quiet=False):
    output = Path(output).resolve()
    lock = output / "run.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"Run is locked: {lock}. If its coordinator AND worker have stopped, remove this stale lock.") from exc
    with handle:
        handle.write(f"pid={os.getpid()} started={now()}\n")
    launched = 0
    try:
        for trial in plan["trials"]:
            previous = latest_result(output, trial)
            if previous:
                if previous["status"] in RETRYABLE:
                    if not retry_errors:
                        print(f"Stopped at {trial['trial_id']} ({previous['status']}). Inspect its log, then use --resume --retry-errors.")
                        return 1
                else:
                    # Numeric failures are terminal outcomes, not lucky-seed retries.
                    continue
            if max_trials is not None and launched >= max_trials:
                break
            number = len(attempts_for(output, trial["trial_id"])) + 1
            attempt = output / "trials" / trial["trial_id"] / f"attempt_{number:04d}"
            attempt.mkdir(parents=True, exist_ok=False)
            write_json(attempt / "result.json", dict(trial, status="running", attempt=attempt.name, started_at=now()))
            command = [sys.executable, str(output / "source_snapshot" / "retrain_paper.py"),
                       "--worker", "--output", str(output), "--trial-id", trial["trial_id"], "--attempt", str(attempt)]
            if quiet:
                command.append("--quiet")
            print(f"[{launched + 1}] {trial['trial_id']} -> {attempt / 'worker.log'}", flush=True)
            process = None
            try:
                with (attempt / "worker.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    returncode = process.wait()
            except KeyboardInterrupt:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
                current = latest_result(output, trial)
                if current["status"] == "running":
                    current.update(status="interrupted", finished_at=now())
                    write_json(attempt / "result.json", current)
                print("Interrupted; the current trial can be restarted with --resume --retry-errors.")
                return 130
            except OSError as exc:
                current = latest_result(output, trial)
                current.update(status="worker_crash", error=repr(exc), finished_at=now())
                write_json(attempt / "result.json", current)
                return 1
            launched += 1
            current = latest_result(output, trial)
            if current["status"] == "running":
                current.update(status="worker_crash", returncode=returncode, finished_at=now())
                write_json(attempt / "result.json", current)
            summarize(output, plan)
            print(f"  {current['status']}; test_accuracy={current.get('test_accuracy')}", flush=True)
            if returncode != 0 or current["status"] in RETRYABLE:
                print("Stopped after a runtime/save failure. Inspect worker.log and error.txt before resuming.")
                return 1
        rows = summarize(output, plan)
        numeric_failures = sum(r["status"] in ("nonfinite", "nonfinite_test") for r in rows)
        print(f"Finished this invocation. Completed: {sum(r['status'] == 'completed' for r in rows)}/{len(rows)}; numeric failures: {numeric_failures}")
        return 2 if numeric_failures or any(r.get("model_save_status") == "error" for r in rows) else 0
    finally:
        try:
            summarize(output, plan)
        finally:
            lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory, or existing directory with --resume")
    parser.add_argument("--results-root", type=Path, default=Path("paper_full_results"))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("paper_parameters.json"))
    parser.add_argument("--source", action="append", default=[], metavar="SEED=RUN_DIRECTORY", help="Override trajectory choice for a search seed")
    parser.add_argument("--resume", action="store_true", help="Use the frozen selection/config/source snapshot")
    parser.add_argument("--plan-only", action="store_true", help="Prepare/inspect selection without importing TensorFlow or training")
    parser.add_argument("--device", choices=("gpu", "cpu"), default=None, help="Default: gpu. Frozen when preparing a run")
    parser.add_argument("--smoke-test", action="store_true", help="Two one-epoch trials on synthetic data; NOT paper results")
    parser.add_argument("--no-save-models", action="store_true", help="Save metrics and history only")
    parser.add_argument("--retry-errors", action="store_true", help="On resume, restart infrastructure failures from epoch 1; preserve attempts")
    parser.add_argument("--max-trials", type=int, help="Limit newly started trials in this invocation")
    parser.add_argument("--quiet", action="store_true", help="Suppress epoch progress in worker logs")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trial-id", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        from paper_afos.retrain_worker import run_worker
        return run_worker(args.output, args.trial_id, args.attempt, args.quiet)
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    if args.resume and (args.device or args.smoke_test or args.no_save_models or args.source
                        or args.results_root != Path("paper_full_results")
                        or args.config != Path(__file__).resolve().with_name("paper_parameters.json")):
        parser.error("--resume uses the frozen plan; do not supply selection, device, or model-saving options")
    try:
        if args.resume:
            plan = read_plan(args.output.resolve())
        else:
            overrides = {}
            for value in args.source:
                seed, path = value.split("=", 1)
                if int(seed) in overrides:
                    raise ValueError(f"Duplicate --source for seed {seed}")
                overrides[int(seed)] = path
            plan = prepare_run(args.output, args.results_root, args.config, overrides,
                               device=args.device or "gpu", smoke=args.smoke_test, save_models=not args.no_save_models)
        print(f"{'SMOKE TEST' if plan['smoke'] else 'Pooled retraining'}: {len(plan['selected'])} selected expressions; {len(plan['trials'])} planned trials")
        for warning in plan["warnings"]:
            print(f"WARNING: {warning}")
        if args.plan_only:
            print(f"Plan saved at {args.output.resolve() / 'plan.json'}. No training started.")
            return 0
        return execute(args.output, plan, args.retry_errors, args.max_trials, args.quiet)
    except (ValueError, KeyError, FileNotFoundError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
