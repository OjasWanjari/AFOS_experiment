"""Create an audited COPY of an AFOS run with corrected Keras save paths.

Uses only the Python standard library. Does not import TensorFlow or run training.
The source run must be stopped, with a completed search and intact source_snapshot.
"""
import argparse
import ast
import hashlib
import json
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_hashes(snapshot):
    files = [snapshot / "run_paper.py", *sorted((snapshot / "paper_afos").glob("*.py"))]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def fix_save_path(source, function, filename):
    """Change only the known filename expression inside the specified function."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function]
    if len(functions) != 1:
        raise ValueError(f"Expected exactly one {function} function")
    calls = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "save"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "model"]
    if len(calls) != 1:
        raise ValueError(f"Unrecognized {function}: expected one model.save call")
    call = calls[0]
    if not call.args:
        raise ValueError(f"Unrecognized {function} save argument")
    names = [n for n in ast.walk(call.args[0]) if isinstance(n, ast.JoinedStr)]
    allowed = ['f"{name}_seed_{seed}"', 'f"{name}_seed_{seed}.keras"',
               'f"best_search_seed_{seed}"', 'f"best_search_seed_{seed}.keras"']
    if len(names) != 1:
        raise ValueError(f"Unrecognized {function} filename expression")
    node = names[0]
    # AST column offsets are UTF-8 byte offsets.
    lines = source.encode("utf-8").splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    data = source.encode("utf-8")
    old = data[start:end].decode("utf-8")
    if old not in allowed and old.replace("'", '"') not in allowed:
        raise ValueError(f"Unsupported save filename in {function}: {old}")
    result = (data[:start] + filename.encode("utf-8") + data[end:]).decode("utf-8")
    ast.parse(result)
    return result


def disable_xla(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_model"]
    if len(functions) != 1:
        raise ValueError("Expected one build_model function")
    calls = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "compile"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "model"]
    if len(calls) != 1 or any(k.arg is None for k in calls[0].keywords):
        raise ValueError("Unrecognized model.compile call")
    call = calls[0]
    existing = [k.value for k in call.keywords if k.arg == "jit_compile"]
    lines = source.encode("utf-8").splitlines(keepends=True)
    data = source.encode("utf-8")
    if existing:
        value = existing[0]
        start = sum(map(len, lines[:value.lineno - 1])) + value.col_offset
        end = sum(map(len, lines[:value.end_lineno - 1])) + value.end_col_offset
        data = data[:start] + b"False" + data[end:]
    else:
        start = sum(map(len, lines[:call.end_lineno - 1])) + call.end_col_offset - 1
        prefix = data[:start].rstrip()
        separator = b" " if prefix.endswith(b",") else b", "
        data = prefix + separator + b"jit_compile=False" + data[start:]
    result = data.decode("utf-8")
    ast.parse(result)
    return result


def prepare(source, destination, no_xla=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Destination must be separate from the source directory tree")
    if destination.exists():
        raise ValueError("Destination already exists; choose a new directory")
    snapshot = source / "source_snapshot"
    manifest = read_json(source / "manifest.json")
    state = read_json(source / "checkpoint.json")
    if not state.get("search_complete"):
        raise ValueError("Search is not complete. Use the latest checkpoint from the retraining machine")
    if state.get("retraining_complete"):
        raise ValueError("Retraining is already marked complete; this tool resumes unfinished work only")
    if len(state.get("selected", [])) < manifest["config"]["post_search"]["top_functions"]:
        raise ValueError("Checkpoint does not contain all selected candidates")
    if read_json(snapshot / "paper_parameters.json") != manifest["config"]:
        raise ValueError("Snapshot configuration does not match manifest")
    original_hashes = source_hashes(snapshot)
    if original_hashes != manifest["source_sha256"]:
        raise ValueError("Snapshot hashes do not match manifest; restore the original snapshot first")
    # Reject links so copying and later writes cannot reach back into another run.
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Run contains symbolic links; provide a self-contained copy")
    provenance_name = "retrain_recovery_provenance"
    if (source / provenance_name).exists():
        raise ValueError("Already a recovery run: resume it directly instead of preparing it again")

    runner = (snapshot / "run_paper.py").read_bytes().decode("utf-8")
    patched = fix_save_path(runner, "search", 'f"best_search_seed_{seed}.keras"')
    patched = fix_save_path(patched, "retrain", 'f"{name}_seed_{seed}.keras"')
    model_source = (snapshot / "paper_afos" / "model.py").read_bytes().decode("utf-8")
    patched_model = disable_xla(model_source) if no_xla else model_source
    # Validate all edits before creating or writing the destination.
    shutil.copytree(source, destination)
    provenance = destination / provenance_name
    provenance.mkdir()
    shutil.copy2(source / "manifest.json", provenance / "original_manifest.json")
    shutil.copy2(source / "checkpoint.json", provenance / "original_checkpoint.json")
    shutil.copytree(snapshot, provenance / "original_source_snapshot")
    target_snapshot = destination / "source_snapshot"
    (target_snapshot / "run_paper.py").write_bytes(patched.encode("utf-8"))
    (target_snapshot / "paper_afos" / "model.py").write_bytes(patched_model.encode("utf-8"))
    updated_hashes = source_hashes(target_snapshot)
    record = dict(
        prepared_at=datetime.now(timezone.utc).isoformat(),
        source_run=str(source), destination_run=str(destination),
        original_source_sha256=original_hashes, recovery_source_sha256=updated_hashes,
        changes=["Use distinct .keras paths for search and retraining models"]
                + (["Set build_model model.compile(jit_compile=False)"] if no_xla else []),
        existing_retraining_records=len(state.get("retraining", [])),
        note="Inherited results use the original source. Only subsequent work uses recovery source. "
             "Package versions in the inherited manifest describe the original run.",
    )
    write_json(provenance / "recovery.json", record)
    manifest["source_sha256"] = updated_hashes
    manifest["retrain_recovery"] = record
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "status.json", dict(status="recovery_prepared", stage="retrain"))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Stopped run with completed search")
    parser.add_argument("--destination", type=Path, required=True, help="New, non-existing run directory")
    parser.add_argument("--disable-xla", action="store_true", help="Record and apply jit_compile=False for new work")
    args = parser.parse_args()
    try:
        destination = prepare(args.source, args.destination, args.disable_xla)
    except (ValueError, OSError, KeyError, SyntaxError) as exc:
        parser.exit(1, f"Recovery preparation failed: {exc}\nIf a partial destination exists, choose a new path.\n")
    command = [sys.executable, str(destination / "source_snapshot" / "run_paper.py"),
               "--config", str(destination / "source_snapshot" / "paper_parameters.json"),
               "--resume", str(destination), "--stage", "retrain", "--device", "gpu"]
    if read_json(destination / "manifest.json")["config"].get("smoke_test"):
        command.append("--smoke")
    print("Prepared a separate recovery run. No training has started.")
    print("Original results and source are preserved; provenance is recorded in the new run.")
    print("Run this command in Bash using your original environment:")
    print(shlex.join(command))


if __name__ == "__main__":
    main()
