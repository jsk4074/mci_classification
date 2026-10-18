"""Saved-probability AUC calculations must preserve completed training artifacts."""

import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT / "code"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "analysis"))

import calculate_auc as auc_module


LABELS = ("NM", "AD", "MCI")
EXPECTED = {"NM": 5 / 6, "AD": 13 / 16, "MCI": 1.0}
MACRO = 127 / 144
FIELDS = ["path", "patient_id", "true_label", "predicted_label", "p_NM", "p_AD", "p_MCI"]


def write_predictions(path, prediction_rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(prediction_rows)


def completed_fixture(output):
    examples = [
        ("NM", [0.6, 0.3, 0.1]),
        ("NM", [0.4, 0.5, 0.1]),
        ("NM", [0.2, 0.7, 0.1]),
        ("AD", [0.4, 0.5, 0.1]),
        ("AD", [0.1, 0.8, 0.1]),
        ("MCI", [0.15, 0.15, 0.7]),
    ]
    support = {"NM": 3, "AD": 2, "MCI": 1}
    saved_metric = {"count": 6, "macro_ovr_auc": MACRO,
                    "per_class": {label: {"support": support[label]} for label in LABELS}}
    metrics = {"labels": list(LABELS), "best_epoch": 4,
               **{split: {level: saved_metric for level in ("patient", "recording")}
                  for split in ("validation", "test")}}
    (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (output / "history.csv").write_text("epoch,train_loss\n4,0.25\n", encoding="utf-8")
    (output / "best.pt").write_bytes(b"Synthetic checkpoint: AUC must never deserialize this")
    (output / "config.json").write_text('{"seed": 42}\n', encoding="utf-8")
    files = {}
    for split in ("validation", "test"):
        for level in ("patient", "recording"):
            filename = f"{split}_{level}_predictions.csv"
            prediction_rows = [
                {"path": f"{split}/person_{index}/06-01.wav",
                 "patient_id": f"{split}_person_{index}", "true_label": label,
                 "predicted_label": "MCI", **dict(zip((f"p_{value}" for value in LABELS), probability))}
                for index, (label, probability) in enumerate(examples)
            ]
            write_predictions(output / filename, prediction_rows)
            files[filename] = prediction_rows
    return files


def training_bytes(output):
    return {name: (output / name).read_bytes()
            for name in ("metrics.json", "history.csv", "best.pt", "config.json")}


def test_auc_uses_probability_rankings_with_ties_and_unweighted_class_average(tmp_path, monkeypatch):
    predictions = completed_fixture(tmp_path)
    monkeypatch.setattr(auc_module, "write_report", lambda *args, **kwargs: None)
    result = auc_module.calculate_auc(tmp_path)
    for split in ("validation", "test"):
        for level in ("patient", "recording"):
            measured = result[split][level]
            assert measured["macro_ovr_auc"] == pytest.approx(MACRO)
            assert measured["macro_ovr_auc"] != pytest.approx(41 / 48)  # support-weighted mean
            for label, expected in EXPECTED.items():
                assert measured["per_class"][label]["auc"] == pytest.approx(expected)
    # NM/AD share a score vector: each positive/negative tie contributes half.
    # Replacing deliberately wrong hard labels must not affect any AUC.
    for filename, prediction_rows in predictions.items():
        for row in prediction_rows:
            row["predicted_label"] = "NM"
        write_predictions(tmp_path / filename, prediction_rows)
    repeated = auc_module.calculate_auc(tmp_path)
    assert repeated["test"] == result["test"]
    assert repeated["validation"] == result["validation"]


def test_auc_schema_provenance_and_report_refresh_preserve_training_files(tmp_path, monkeypatch):
    predictions = completed_fixture(tmp_path)
    original = training_bytes(tmp_path)
    calls = []
    monkeypatch.setattr(auc_module, "write_report", lambda *args, **kwargs: calls.append((args, kwargs)))
    custom_report = tmp_path / "exports" / "auc.html"
    result = auc_module.calculate_auc(tmp_path, report_path=custom_report)
    assert result["labels"] == list(LABELS)
    assert result["selected_epoch"] == 4
    assert "saved probability" in result["method"]
    assert "unweighted" in result["method"] and "hard predicted labels are not used" in result["method"]
    assert datetime.fromisoformat(result["created_at"]).utcoffset() == timedelta(hours=9)
    assert result["prediction_sha256"] == {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in predictions
    }
    assert result["test"]["patient"]["per_class"]["NM"]["positive_count"] == 3
    assert result["test"]["patient"]["per_class"]["MCI"]["negative_count"] == 5
    assert json.loads((tmp_path / "auc_metrics.json").read_text()) == result
    with (tmp_path / "auc_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert {(row["split"], row["level"], row["class"]) for row in rows} == {
        (split, level, label) for split in ("validation", "test")
        for level in ("patient", "recording") for label in (*LABELS, "macro")
    }
    assert calls == [((tmp_path,), {}), ((tmp_path,), {"report_path": custom_report})]
    assert training_bytes(tmp_path) == original


def test_invalid_fourth_prediction_file_publishes_nothing_and_preserves_existing_auc(tmp_path, monkeypatch):
    predictions = completed_fixture(tmp_path)
    original = training_bytes(tmp_path)
    old_json = b'{"previous_auc": "preserve"}\n'
    old_csv = b"previous,auc\nkeep,0.5\n"
    (tmp_path / "auc_metrics.json").write_bytes(old_json)
    (tmp_path / "auc_metrics.csv").write_bytes(old_csv)
    late_rows = predictions["test_recording_predictions.csv"]
    late_rows[-1].update(p_NM=0.9, p_AD=0.9, p_MCI=0.9)
    write_predictions(tmp_path / "test_recording_predictions.csv", late_rows)
    calls = []
    monkeypatch.setattr(auc_module, "write_report", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(ValueError, match="test_recording_predictions.csv.*sum to 1"):
        auc_module.calculate_auc(tmp_path)
    assert (tmp_path / "auc_metrics.json").read_bytes() == old_json
    assert (tmp_path / "auc_metrics.csv").read_bytes() == old_csv
    assert training_bytes(tmp_path) == original
    assert calls == []
    assert not list(tmp_path.glob(".*.tmp"))
