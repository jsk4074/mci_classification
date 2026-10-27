"""Aggregate s01 reports remain truthful, private and usable without model imports."""

import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import pytest


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))
from s01_metrics import classification_metrics
from s01_report import write_report


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.text = []
        self.elements = []
        self.feed(source)

    def handle_data(self, value):
        self.text.append(value)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))

    @property
    def visible(self):
        return " ".join(" ".join(self.text).split())


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def perfect():
    return classification_metrics([0, 1, 2], [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])


def test_empty_report_imports_no_model_audio_or_plotting_libraries(tmp_path):
    script = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','transformers','numpy','soundfile','matplotlib','weasyprint'}:
        raise AssertionError('Unexpected dependency: ' + name)
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
from s01_report import write_report
write_report(sys.argv[2],status='prepared')
"""
    result = subprocess.run([sys.executable, "-c", script, str(ANALYSIS), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    source = (tmp_path / "report.html").read_text()
    text = Page(source).visible
    assert "No final test results are available" in text
    assert "No completed cross-validation summary" in text
    assert "No completed epoch history" in text
    assert "XLS-R patient-bag" in text and "ResNet" not in text
    assert "Five-fold CV combines original training and validation patients" in text
    assert "original test patients remain excluded" in text
    assert "CV uses original training patients only" not in text
    assert "do not establish domain invariance" in text
    assert not (tmp_path / "metrics.json").exists()


def test_cv_means_and_standalone_figures_do_not_mix_fold_histories(tmp_path):
    pytest.importorskip("matplotlib")
    variants = {}
    for variant, mean, std in (("A", .6, .1), ("B", .7, .05), ("C", .65, .2)):
        variants[variant] = {"folds": [{"fold": 1, "patient": perfect()}, {"fold": 2, "patient": perfect()}],
                             "mean": {"macro_f1": mean, "loss": .5},
                             "std": {"macro_f1": std, "loss": .1}}
    save(tmp_path / "cv_summary.json", {"variants": variants, "selected_variant": "B"})
    with (tmp_path / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_patient_loss", "val_patient_macro_f1", "val_patient_balanced_accuracy", "variant", "fold"])
        writer.writerow([1, .8, .9, .5, .6, "B", 1])
        writer.writerow([2, .7, .8, .6, .7, "B", 2])
    report = write_report(tmp_path, status="cv")
    text = Page(report.read_text()).visible
    assert "0.7000 ± 0.0500" in text and "Selected variant: B" in text
    assert "Multiple fold histories" in text
    assert "learning_curves.svg" not in report.read_text()
    assert "No final test results are available" in text
    assert (tmp_path / "figures/cv_variants.svg").is_file()
    assert (tmp_path / "figures/cv_variants.png").is_file()


def test_partial_cv_progress_shows_only_saved_folds_without_fabricating_selection(tmp_path):
    save(tmp_path / "cv_progress.json", {"variants": {"A": {"folds": [{"fold": 1, "patient": perfect()}]}}})
    report = write_report(tmp_path, status="cv")
    text = Page(report.read_text()).visible
    assert "Selected variant: Not selected" in text
    assert "Saved fold patient metrics" in text
    assert "Unavailable ± Unavailable" in text
    assert "No final test results are available" in text


def test_final_metrics_figures_and_messages_do_not_expose_private_rows_or_configuration(tmp_path):
    pytest.importorskip("matplotlib")
    marker = "PRIVATE_PERSON_12e7"
    metric = perfect()
    metric["patient_ids"] = [marker]
    save(tmp_path / "config.json", {"stage": "final", "variant": "B", "model_revision": f"/private/{marker}",
                                   "manifest": f"/private/{marker}.csv", "patient_id": marker})
    save(tmp_path / "summary.json", {"patient_ids": [marker], "paths": [f"/private/{marker}.wav"]})
    save(tmp_path / "metrics.json", {"best_epoch": 3, "validation": {"patient": metric}, "test": {"patient": metric}})
    save(tmp_path / "s00_metrics.json", {"test": {"patient": metric}})
    (tmp_path / "test_patient_predictions.csv").write_text(f"patient_id\n{marker}\n")
    original = (tmp_path / "metrics.json").read_bytes()
    report = write_report(tmp_path, status="completed", message=f'<script>{marker}</script> /private/{marker}.wav')
    source = report.read_text()
    text = Page(source).visible
    assert marker not in source and "<script>" not in source and "/private/" not in source
    assert "Selected checkpoint validation" in text and "Final test evaluation" in text
    assert "OvR ROC AUC" in text and "100.0%" in text
    assert "s00 test results were already known" in text
    assert "Supplementary single-instance" not in text  # no recording results were supplied
    assert (tmp_path / "metrics.json").read_bytes() == original
    assert (tmp_path / "figures/test_patient_confusion.svg").is_file()
    assert (tmp_path / "figures/test_patient_confusion.png").is_file()
    assert marker not in (tmp_path / "report_state.json").read_text()
    for figure in (tmp_path / "figures").glob("*.svg"):
        assert marker not in figure.read_text()


def test_custom_report_paths_use_escaped_local_assets_and_missing_auc_stays_unavailable(tmp_path):
    output = tmp_path / "run #1"
    output.mkdir()
    missing = classification_metrics([0, 1], [[.8, .1, .1], [.1, .8, .1]])
    save(output / "metrics.json", {"validation": {"patient": missing}})
    report = write_report(output, tmp_path / "exports" / "report.html", status="completed")
    page = Page(report.read_text())
    assert "Unavailable" in page.visible and "No final test results are available" in page.visible
    for tag, attributes in page.elements:
        if tag not in ("img", "a"):
            continue
        uri = attributes.get("src", attributes.get("href", ""))
        if not uri or uri.startswith("#"):
            continue
        parts = urlsplit(uri)
        assert not parts.scheme and not parts.netloc
        assert " " not in uri and "#" not in uri
        assert (report.parent / unquote(parts.path)).exists()


def test_optional_a4_pdf_export_is_callable_and_preserves_xlsr_html(tmp_path, monkeypatch):
    pytest.importorskip("weasyprint")
    from s01_export_pdf import export_pdf
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    report = write_report(tmp_path, status="prepared")
    before = report.read_bytes()
    result = export_pdf(report, tmp_path / "report.pdf")
    assert result == tmp_path / "report.pdf"
    assert result.read_bytes().startswith(b"%PDF")
    assert report.read_bytes() == before
    stylesheet = (ANALYSIS / "report_a4.css").read_text()
    assert "A4 portrait" in stylesheet and "s01 XLS-R MIL" in stylesheet
    assert "ResNet" not in stylesheet
