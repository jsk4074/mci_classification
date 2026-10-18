#!/usr/bin/env python3
"""Calculate class ROC AUC from saved predictions without running the model."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from speech_report import write_report


LABELS = ("NM", "AD", "MCI")
METHOD = (
    "ROC AUC is calculated one-vs-rest for each class from its saved probability "
    "column (p_NM, p_AD, p_MCI). Macro one-vs-rest AUC is the unweighted mean of "
    "the three class AUCs; hard predicted labels are not used."
)


def _atomic_text(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _probabilities(row: dict, filename: str, row_number: int) -> list[float]:
    try:
        values = [float(row[f"p_{label}"]) for label in LABELS]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{filename} row {row_number}: probability columns must be numeric.") from exc
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError(f"{filename} row {row_number}: probabilities must be finite and in [0, 1].")
    if not math.isclose(sum(values), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"{filename} row {row_number}: probabilities must sum to 1 within 1e-6.")
    return values


def _calculate_file(path: Path, level: str, saved: dict) -> tuple[dict, str]:
    from sklearn.metrics import roc_auc_score

    content = path.read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        identity_column = "patient_id" if level == "patient" else "path"
        required = {identity_column, "true_label", *(f"p_{label}" for label in LABELS)}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path.name}: missing required identity, true_label, or probability columns.")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path.name}: duplicate CSV column names.")
        identities, targets, probabilities = set(), [], []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path.name} row {row_number}: too many CSV fields.")
            identity = (row.get(identity_column) or "").strip()
            if not identity:
                raise ValueError(f"{path.name} row {row_number}: blank {identity_column}.")
            identity = os.path.normpath(identity) if level == "recording" else identity
            if identity in identities:
                raise ValueError(f"{path.name}: duplicate {identity_column} values.")
            identities.add(identity)
            label = (row.get("true_label") or "").strip()
            if label not in LABELS:
                raise ValueError(f"{path.name} row {row_number}: true_label must be NM, AD, or MCI.")
            targets.append(label)
            probabilities.append(_probabilities(row, path.name, row_number))
    except UnicodeError as exc:
        raise ValueError(f"{path.name}: predictions must be UTF-8 CSV.") from exc
    except csv.Error as exc:
        raise ValueError(f"{path.name}: invalid CSV formatting.") from exc
    if not targets:
        raise ValueError(f"{path.name}: no prediction rows.")
    supports = Counter(targets)
    if any(supports[label] == 0 for label in LABELS):
        raise ValueError(f"{path.name}: all three true classes must be present for one-vs-rest AUC.")
    if not isinstance(saved, dict) or type(saved.get("count")) is not int or saved["count"] != len(targets):
        raise ValueError(f"{path.name}: row count does not match metrics.json.")
    saved_classes = saved.get("per_class", {})
    if not isinstance(saved_classes, dict):
        raise ValueError(f"{path.name}: metrics.json lacks per-class supports.")
    per_class = {}
    for column, label in enumerate(LABELS):
        reference = saved_classes.get(label, {})
        if (not isinstance(reference, dict) or type(reference.get("support")) is not int
                or reference["support"] != supports[label]):
            raise ValueError(f"{path.name}: class supports do not match metrics.json.")
        auc = float(roc_auc_score(
            [target == label for target in targets],
            [values[column] for values in probabilities],
        ))
        per_class[label] = {
            "auc": auc, "positive_count": supports[label],
            "negative_count": len(targets) - supports[label],
        }
    macro = sum(values["auc"] for values in per_class.values()) / len(LABELS)
    try:
        saved_auc = float(saved["macro_ovr_auc"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"{path.name}: metrics.json lacks a valid saved macro AUC.") from exc
    if not math.isfinite(saved_auc) or not math.isclose(macro, saved_auc, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{path.name}: calculated macro AUC differs from metrics.json by more than 1e-10.")
    return {"count": len(targets), "macro_ovr_auc": macro, "per_class": per_class}, hashlib.sha256(content).hexdigest()


def calculate_auc(output_dir: Path, report_path: Path | None = None) -> dict:
    """Validate all saved predictions, publish AUC artifacts, and refresh HTML."""

    output_dir = Path(output_dir).expanduser().resolve()
    try:
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("metrics.json must contain valid UTF-8 JSON from a completed run.") from exc
    if not isinstance(metrics, dict) or metrics.get("labels") != list(LABELS):
        raise ValueError("metrics.json must declare labels in NM, AD, MCI order.")
    if type(metrics.get("best_epoch")) is not int or metrics["best_epoch"] < 1:
        raise ValueError("metrics.json must contain a positive selected best_epoch.")
    if report_path is not None:
        report_path = Path(report_path).expanduser().resolve()
        protected = {output_dir / name for name in (
            "metrics.json", "best.pt", "history.csv", "config.json", "split_manifest.csv",
            "auc_metrics.json", "auc_metrics.csv",
            *(f"{split}_{level}_predictions.csv" for split in ("validation", "test") for level in ("patient", "recording")),
        )}
        if report_path in protected:
            raise ValueError("The report path must not overwrite a saved run artifact.")
    result = {
        "labels": list(LABELS), "method": METHOD, "selected_epoch": metrics["best_epoch"],
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "prediction_sha256": {},
    }
    csv_rows = []
    for split in ("validation", "test"):
        result[split] = {}
        saved_split = metrics.get(split, {})
        if not isinstance(saved_split, dict):
            raise ValueError(f"metrics.json lacks saved {split} metrics.")
        for level in ("patient", "recording"):
            filename = f"{split}_{level}_predictions.csv"
            values, digest = _calculate_file(output_dir / filename, level, saved_split.get(level))
            result[split][level] = values
            result["prediction_sha256"][filename] = digest
            for label in LABELS:
                csv_rows.append({"split": split, "level": level, "class": label, **values["per_class"][label]})
            csv_rows.append({"split": split, "level": level, "class": "macro", "positive_count": "", "negative_count": "", "auc": values["macro_ovr_auc"]})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["split", "level", "class", "positive_count", "negative_count", "auc"])
    writer.writeheader()
    writer.writerows(csv_rows)
    # All four files and all reference metrics have passed before either output is published.
    json_text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _atomic_text(output_dir / "auc_metrics.json", json_text)
    _atomic_text(output_dir / "auc_metrics.csv", buffer.getvalue())
    write_report(output_dir)
    if report_path is not None and report_path != output_dir / "report.html":
        write_report(output_dir, report_path=report_path)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Completed run containing metrics and saved prediction CSVs")
    parser.add_argument("--report-path", type=Path, help="Optional additional HTML report copy")
    args = parser.parse_args(argv)
    try:
        result = calculate_auc(args.output_dir, args.report_path)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    for split in ("validation", "test"):
        print(f"{split}: patient macro AUC={result[split]['patient']['macro_ovr_auc']:.6f}; recording macro AUC={result[split]['recording']['macro_ovr_auc']:.6f}")
    print("Saved auc_metrics.json, auc_metrics.csv, and refreshed HTML reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
