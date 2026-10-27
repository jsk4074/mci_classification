"""Check report truthfulness, patient privacy, and independent regeneration."""

import csv
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT / "code"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "analysis"))


class ReportHTML(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)

    @property
    def visible_text(self):
        return " ".join(" ".join(self.text).split())


def report_module():
    import speech_report
    return speech_report


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def metrics(confusion=None, macro_f1=0.8123):
    confusion = confusion or [[8, 1, 0], [0, 7, 2], [1, 1, 7]]
    supports = [sum(row) for row in confusion]
    count = sum(supports)
    return {
        "count": count,
        "accuracy": sum(confusion[index][index] for index in range(3)) / count,
        "balanced_accuracy": 0.8,
        "macro_f1": macro_f1,
        "loss": 0.321,
        "macro_ovr_auc": 0.9,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": ["NM", "AD", "MCI"],
        "per_class": {
            label: {"precision": 0.8, "recall": 7 / 9, "f1": 0.79, "support": supports[index]}
            for index, label in enumerate(("NM", "AD", "MCI"))
        },
    }


def save_history(output):
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "train_loss", "val_patient_loss", "val_patient_macro_f1",
            "val_patient_balanced_accuracy", "learning_rate",
        ])
        writer.writeheader()
        writer.writerows([
            {"epoch": 1, "train_loss": 2.5, "val_patient_loss": 3.21,
             "val_patient_macro_f1": 0.1111, "val_patient_balanced_accuracy": 0.2,
             "learning_rate": 0.001},
            {"epoch": 2, "train_loss": 1.4, "val_patient_loss": 0.98,
             "val_patient_macro_f1": 0.2222, "val_patient_balanced_accuracy": 0.3,
             "learning_rate": 0.001},
        ])


def test_empty_report_describes_method_without_inventing_training_results(tmp_path):
    output = tmp_path / "empty_run"
    report = report_module().write_report(output)
    assert report.is_file()
    assert report.parent == output
    document = ReportHTML(report.read_text(encoding="utf-8"))
    assert "No training results yet" in document.visible_text
    assert "No final test results yet" in document.visible_text
    assert "1 × 40 × 501" in document.visible_text
    assert not (output / "metrics.json").exists()
    assert not (output / "history.csv").exists()
    state = json.loads((output / "report_state.json").read_text())
    assert state.get("latest_validation") is None


def test_method_explains_patient_isolation_and_variable_length_audio(tmp_path):
    report = report_module().write_report(tmp_path, status="prepared")
    text = ReportHTML(report.read_text()).visible_text
    assert "Splits prepared" in text
    assert "Auto-refresh every 15 seconds while prepared or training" in text
    assert "Generated " not in text
    assert not re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}:\d{2}:\d{2}\b", text)
    assert "· Auto-refresh" not in text
    for phrase in (
        "Patient-disjoint validation", "Short clips", "Long clips during training",
        "Validation and test", "Implementation updates",
    ):
        assert phrase in text
    assert re.search(r"pad|padding", text, re.IGNORECASE)
    assert re.search(r"random.{0,40}crop|crop.{0,40}random", text, re.IGNORECASE)
    assert re.search(r"window", text, re.IGNORECASE)
    assert re.search(r"recording.{0,100}patient|patient.{0,100}recording", text, re.IGNORECASE)


def test_method_input_shape_uses_the_saved_feature_configuration(tmp_path):
    features = {
        "sample_rate": 16000, "n_mels": 40, "n_fft": 400, "hop_length": 160,
        "clip_seconds": 0.5, "top_db": 80, "eval_windows": 3,
    }
    report = report_module().write_report(tmp_path, status="prepared", features=features)
    text = ReportHTML(report.read_text()).visible_text
    assert "1 × 40 × 51" in text
    assert "Right-pad to 0.5 seconds" in text


def test_status_and_latest_validation_persist_for_regeneration(tmp_path):
    latest = {"patient": metrics(macro_f1=0.51), "recording": metrics(macro_f1=0.42)}
    features = {
        "sample_rate": 16000, "n_mels": 40, "n_fft": 400, "hop_length": 160,
        "clip_seconds": 5.0, "top_db": 80.0, "eval_windows": 5,
    }
    module = report_module()
    report = module.write_report(tmp_path, status="training", features=features,
                                 latest_validation=latest, best_epoch=3,
                                 message="Epoch three validation is available.")
    state = json.loads((tmp_path / "report_state.json").read_text())
    assert state["status"] == "training"
    assert state["features"] == features
    assert state["latest_validation"] == latest
    assert state["best_epoch"] == 3
    regenerated = module.write_report(tmp_path)
    assert regenerated == report
    text = ReportHTML(regenerated.read_text()).visible_text
    assert "Training in progress" in text
    assert "Latest epoch validation" in text
    assert "No final test results yet" in text
    assert "Epoch three validation is available." in text


@pytest.mark.parametrize("status,label", [
    ("awaiting_data", "Awaiting data"),
    ("prepared", "Splits prepared"),
    ("completed", "Training completed"),
    ("failed", "Run failed"),
])
def test_explicit_status_survives_a_standalone_report_update(tmp_path, status, label):
    module = report_module()
    module.write_report(tmp_path, status=status, message="Saved status")
    report = module.write_report(tmp_path)
    assert json.loads((tmp_path / "report_state.json").read_text())["status"] == status
    assert label in ReportHTML(report.read_text()).visible_text


def test_learning_curves_are_deterministic_and_do_not_imply_final_results(tmp_path):
    save_history(tmp_path)
    module = report_module()
    report = module.write_report(tmp_path, status="training")
    source = report.read_text()
    text = ReportHTML(source).visible_text
    assert "Learning curves" in text
    assert "No final test results yet" in text
    curves = re.findall(r"<svg\b.*?</svg>", source, re.DOTALL | re.IGNORECASE)
    assert curves, "Learning curves must be embedded locally in the report"
    regenerated = module.write_report(tmp_path).read_text()
    assert re.findall(r"<svg\b.*?</svg>", regenerated, re.DOTALL | re.IGNORECASE) == curves


def test_final_report_shows_saved_validation_and_test_confusion_and_class_metrics(tmp_path):
    validation = metrics(macro_f1=0.6123)
    test = metrics(confusion=[[19, 2, 3], [4, 23, 5], [6, 7, 29]])
    save_json(tmp_path / "metrics.json", {
        "best_epoch": 7, "labels": ["NM", "AD", "MCI"],
        "validation": {"patient": validation, "recording": validation},
        "test": {"patient": test, "recording": test},
    })
    save_history(tmp_path)
    report = report_module().write_report(tmp_path)
    text = ReportHTML(report.read_text()).visible_text
    for phrase in (
        "Selected checkpoint validation", "Final held-out test",
        "Per-class performance", "Confusion matrix", "Precision", "Recall", "Support",
    ):
        assert phrase in text
    for value in ("19", "23", "29"):
        assert re.search(rf"\b{value}\b", text)
    for label in ("NM", "AD", "MCI"):
        assert label in text
    assert "No final test results yet" not in text


@pytest.mark.parametrize("selected_epoch", [7, 6])
def test_report_class_auc_requires_matching_selected_checkpoint(tmp_path, selected_epoch):
    patient = metrics()
    save_json(tmp_path / "metrics.json", {
        "best_epoch": 7, "labels": ["NM", "AD", "MCI"],
        "test": {"patient": patient},
    })
    save_json(tmp_path / "auc_metrics.json", {
        "selected_epoch": selected_epoch, "labels": ["NM", "AD", "MCI"],
        "test": {"patient": {
            "count": patient["count"], "macro_ovr_auc": 0.9,
            "per_class": {"NM": {"auc": 0.8765}, "AD": {"auc": 0.95}, "MCI": {"auc": 0.8735}},
        }},
    })
    source = report_module().write_report(tmp_path).read_text()
    assert "Macro ROC AUC (OvR)" in ReportHTML(source).visible_text
    if selected_epoch == 7:
        assert '<th>ROC AUC (OvR)</th>' in source
        assert "0.8765" in source
        assert "unweighted mean" in source
    else:
        assert '<th>ROC AUC (OvR)</th>' not in source
        assert "0.8765" not in source
        assert "do not match the selected checkpoint" in source


def test_untrusted_status_message_and_configuration_strings_are_escaped(tmp_path):
    payload = '<script>alert("patient");</script><img src=x onerror="alert(1)">'
    save_json(tmp_path / "config.json", {
        "arguments": {"device": payload, "learning_rate": 0.001, "batch_size": 16},
        "features": {"sample_rate": 16000, "n_mels": 40, "n_fft": 400,
                     "hop_length": 160, "clip_seconds": 5, "top_db": 80, "eval_windows": 5},
    })
    report = report_module().write_report(tmp_path, status="failed", message=payload)
    source = report.read_text()
    assert payload in unescape(source)
    assert "<script>alert" not in source
    assert "<img src=x" not in source
    assert "&lt;script&gt;" in source
    assert not any(tag == "img" and attrs.get("src") == "x" for tag, attrs in ReportHTML(source).elements)


def test_report_does_not_copy_patient_identifiers_or_audio_paths_into_html(tmp_path):
    identifier = "PRIVATE_PATIENT_48a6cdb"
    audio_path = f"/private/audio/{identifier}/06-01.wav"
    for filename in (
        "split_manifest.csv", "test_patient_predictions.csv", "test_recording_predictions.csv",
    ):
        (tmp_path / filename).write_text(f"patient_id,path\n{identifier},{audio_path}\n")
    save_json(tmp_path / "dataset_snapshot.json", {
        "files": [{"patient_id": identifier, "path": audio_path, "label": "NM"}],
    })
    save_json(tmp_path / "split_summary.json", {
        "total_recordings": 9, "total_patients": 9,
        "classes": {label: {"patients": 3, "recordings": 3} for label in ("NM", "AD", "MCI")},
        "splits": {},
    })
    report = report_module().write_report(tmp_path, status="prepared")
    source = report.read_text()
    assert identifier not in source
    assert audio_path not in source


def test_artifact_links_resolve_from_a_custom_report_and_escape_path_characters(tmp_path):
    output = tmp_path / "run #1"
    output.mkdir()
    save_history(output)
    save_json(output / "config.json", {})
    (output / "best.pt").write_bytes(b"placeholder")
    custom_report = tmp_path / "reports" / "progress report.html"
    report = report_module().write_report(output, status="training", report_path=custom_report)
    assert report == custom_report
    document = ReportHTML(report.read_text())
    links = [attrs["href"] for tag, attrs in document.elements if tag == "a" and "href" in attrs]
    history_links = [href for href in links if urlsplit(href).path.endswith("history.csv")]
    assert history_links
    for href in history_links:
        parts = urlsplit(href)
        assert not parts.scheme and not parts.netloc and not parts.fragment
        assert " " not in href and "%20" in href and "%23" in href
        assert (report.parent / unquote(parts.path)).resolve() == (output / "history.csv").resolve()
    for tag, attrs in document.elements:
        if "src" in attrs:
            source = urlsplit(attrs["src"])
            assert source.scheme not in {"http", "https"} and not source.netloc
        if tag == "link" and "href" in attrs:
            source = urlsplit(attrs["href"])
            assert source.scheme not in {"http", "https"} and not source.netloc


def test_report_only_cli_requires_neither_machine_learning_dependencies_nor_audio(tmp_path):
    output = tmp_path / "run"
    report = tmp_path / "reports" / "status.html"
    script = """
import builtins
import runpy
import sys
program, output, report, missing_audio = sys.argv[1:]
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'torchvision', 'librosa', 'sklearn'}:
        raise AssertionError('Report-only must not import training dependencies: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = [program, '--report-only', '--output-dir', output,
            '--report-path', report, '--data-root', missing_audio]
runpy.run_path(program, run_name='__main__')
"""
    completed = subprocess.run([
        sys.executable, "-c", script, str(PROJECT_ROOT / "resnet18_baseline.py"),
        str(output), str(report), str(tmp_path / "unavailable_dataset"),
    ], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert report.is_file()
    assert (output / "report_state.json").is_file()
    assert not (output / "split_manifest.csv").exists()
    assert "No final test results yet" in ReportHTML(report.read_text()).visible_text
