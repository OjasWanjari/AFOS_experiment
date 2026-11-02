"""Evaluate the paper's fixed Top 15 against an existing pooled retraining run.

See RETRAIN_PUBLISHED_TOP15_JA.md / RETRAIN_PUBLISHED_TOP15_EN.md.
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from paper_afos.published_top15 import prepare_published_run
from paper_afos.retraining import read_plan, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-run", type=Path, help="Previous pooled retraining output; required for a new plan")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--check-environment", action="store_true", help="Verify runtime only; no data loading or training")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--environment-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    if args.resume and args.reference_run:
        parser.error("--resume uses the frozen reference; omit --reference-run")
    if not args.resume and not args.reference_run:
        parser.error("A new plan requires --reference-run")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    if args.plan_only and args.check_environment:
        parser.error("Choose --plan-only or --check-environment")
    output = args.output.resolve()
    try:
        plan = read_plan(output) if args.resume else prepare_published_run(output, args.reference_run)
        if plan.get("experiment") != "published_top15":
            raise ValueError("This output is not a published Top 15 evaluation")
        if args.environment_worker:
            from paper_afos.runtime_environment import environment_mismatches, inspect_environment
            environment = inspect_environment(plan["runtime"])
            differences = environment_mismatches(plan["reference_environment"], environment)
            write_json(output / "environment_preflight.json", dict(environment=environment,
                                                                   matched=not differences, differences=differences))
            if differences:
                raise ValueError("Environment mismatch: " + "; ".join(differences))
            print("Recorded environment matches. No dataset loaded; no training started.")
            return 0
        print(f"Published Top 15: {len(plan['trials'])} trials, seeds 0..4, 50 epochs; split {plan['split_sha256']}", flush=True)
        if args.plan_only:
            print(f"Frozen plan: {output / 'plan.json'}. No TensorFlow import or training.")
            return 0
        if args.check_environment:
            command = [sys.executable, str(output / "source_snapshot/retrain_published_top15.py"),
                       "--output", str(output), "--resume", "--environment-worker"]
        else:
            command = [sys.executable, str(output / "source_snapshot/retrain_paper.py"),
                       "--output", str(output), "--resume"]
            if args.max_trials is not None:
                command += ["--max-trials", str(args.max_trials)]
            if args.retry_errors:
                command.append("--retry-errors")
            if args.quiet:
                command.append("--quiet")
        return subprocess.call(command)
    except (ValueError, RuntimeError, KeyError, FileNotFoundError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
