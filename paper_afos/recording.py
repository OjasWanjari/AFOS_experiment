"""Incremental CSV records, independent of training and plotting libraries."""
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path


def event(root, kind, trial="", detail=""):
    path = Path(root) / "results" / "events.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(["timestamp", "event", "trial", "detail"])
        writer.writerow([datetime.now().astimezone().isoformat(), kind, trial, detail])


class EpochCSV:
    """Close/flush each epoch so a running or interrupted trial remains readable.

    Raw NaN/Inf are deliberately retained here; summary CSVs use empty cells.
    A retried trial archives its previous partial history, rather than erasing it.
    """
    fields = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy"]

    def __init__(self, root, trial):
        self.root, self.trial = Path(root), trial
        self.path = self.root / "histories" / f"{trial}.csv"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            archive = self.path.parent / "attempts"
            archive.mkdir(exist_ok=True)
            self.path.replace(archive / f"{self.trial}_{uuid.uuid4().hex}.csv")
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(self.fields)
        event(self.root, "training_started", self.trial)
        return self

    def __call__(self, row):
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writerow(row)

    def __exit__(self, kind, exc, traceback):
        event(self.root, "training_failed" if exc is not None else "training_finished",
              self.trial, repr(exc) if exc is not None else "")


def read_results(path):
    """Decode numbers and chromosome arrays without altering expression strings."""
    path = Path(path)
    if not path.exists():
        return []
    strings = {"function", "activation_expression", "status"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif key not in strings:
                try:
                    row[key] = json.loads(value)
                except (ValueError, TypeError):
                    pass
    return rows
