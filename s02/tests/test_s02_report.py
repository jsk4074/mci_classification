"""Saved aggregate s02 reports are factual, private and independent of models."""

import csv
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import pytest


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))
from s02_report import export_report, write_report


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.parts, self.links = [], []
        self.hidden = 0
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        if tag in ("head", "style", "script"):
            self.hidden += 1
        for key, value in attrs:
            if tag in ("a", "img") and key in ("href", "src"):
                self.links.append(value)

    def handle_endtag(self, tag):
        if tag in ("head", "style", "script"):
            self.hidden -= 1

    def handle_data(self, value):
        if not self.hidden:
            self.parts.append(value)

    @property
    def text(self):
        return " ".join(" ".join(self.parts).split())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def perfect():
    return {"count": 3, "macro_f1": 1., "balanced_accuracy": 1., "accuracy": 1.,
            "macro_ovr_auc": 1., "loss": .1,
            "per_class": {label: {"precision": 1., "recall": 1., "f1": 1., "auc": 1., "support": 1} for label in ("NM", "AD", "MCI")},
            "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "confusion_matrix_labels": ["NM", "AD", "MCI"]}


def history(path, mixed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_patient_loss", "val_patient_macro_f1", "val_patient_balanced_accuracy", "variant", "fold"])
        writer.writerow([1, .9, .8, .5, .6, "A", 1 if mixed else ""])
        writer.writerow([2, .7, .6, .7, .8, "A", 2 if mixed else ""])


def assert_local_assets(report):
    page = Page(report.read_text())
    for uri in page.links:
        if uri.startswith("#"):
            continue
        parsed = urlsplit(uri)
        assert not parsed.scheme and not parsed.netloc
        assert " " not in uri and "#" not in uri
        assert (report.parent / unquote(parsed.path)).is_file()
    return page


def test_empty_report_and_cli_need_no_models_audio_or_plotting(tmp_path):
    script = """
import builtins, sys
sys.path.insert(0,sys.argv[1])
original = builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','soundfile','numpy','matplotlib','weasyprint'}:
        raise AssertionError('Unexpected dependency: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from s02_report import main
raise SystemExit(main(['--output-dir',sys.argv[2]]))
"""
    result = subprocess.run([sys.executable, "-c", script, str(ANALYSIS), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    page = assert_local_assets(tmp_path / "report.html")
    assert "s02" in page.text and "No final test results yet" in page.text
    assert "No completed cross-validation results yet" in page.text
    assert "Selected XLS-R variant: Not selected" in page.text
    assert "Generated" not in page.text and not re.search(r"\d{4}-\d{2}-\d{2}", page.text)
    assert "not an unbiased nested-CV estimate" in page.text
    assert "partly reused known holdout" in page.text
    assert not (tmp_path / "metrics.json").exists()


def test_cohort_source_audit_counts_and_untrusted_metadata_stay_private(tmp_path):
    private = "PRIVATE_PARTICIPANT_e1a95"
    save(tmp_path / "summary.json", {"total_patients": 234, "total_recordings": 9080,
         "classes": {"NM": {"patients": 48, "recordings": 1839}, "AD": {"patients": 32, "recordings": 1065}, "MCI": {"patients": 154, "recordings": 6176}},
         "patient_ids": [private], "clinical_notes": private})
    save(tmp_path / "cohort_audit.json", {"excluded_old_mci": {"patients": 18, "recordings": 711},
         "new_mci": {"duplicate_task_slots": 30, "recordings_in_duplicate_task_slots": 61, "additional_distinct_takes": 31, "participants_with_multiple_takes": 21, "annotation_state_counts": {private: 154}},
         "source_split_task_counts": {"full_dataset_MCI": {"train": {"06": {"patients": 108, "recordings": 3240}}}},
         "notes": f"<script>{private}</script>", "created_at": "2026-10-22T08:10:00+09:00"})
    save(tmp_path / "domain_audit.json", {"headers_read": 9080, "header_errors": 0,
         "by_diagnosis_and_original_split": {"MCI": {"train": {"recordings": 4320, "header_errors": 0,
             "duration_seconds": {"min": .5, "median": 3., "max": 45.},
             "audio_formats": [{"sample_rate": 16000, "channels": 1, "encoding": "PCM_16", "recordings": 4319},
                               {"sample_rate": 96000, "channels": 1, "encoding": private, "recordings": 1}], "paths": [private]}}}})
    save(tmp_path / "config.json", {"manifest": f"/private/{private}.csv", "patient_id": private})
    (tmp_path / "test_patient_predictions.csv").write_text(f"patient_id\n{private}\n")
    protected = {file.name: file.read_bytes() for file in tmp_path.iterdir() if file.is_file()}
    report = write_report(tmp_path, status="prepared", message=private)
    source, page = report.read_text(), Page(report.read_text())
    assert private not in source and "<script>" not in source and "/private/" not in source
    assert "Included patients: 234" in page.text and "Included recordings: 9,080" in page.text
    assert "Prior MCI patients excluded 18" in page.text
    assert "Repeated task slots with distinct retained takes 30" in page.text
    assert "Recordings retained in those repeated slots 61" in page.text
    assert "Additional distinct takes retained 31" in page.text
    assert "New MCI archive train 06 108 3,240" in page.text
    assert "16,000 Hz / 1 ch / PCM_16" in page.text and "Other encoding" in page.text
    assert "config.json" not in page.text  # private provenance is not an aggregate link
    assert not re.search(r"\d{4}-\d{2}-\d{2}", page.text)
    for name, content in protected.items():
        assert (tmp_path / name).read_bytes() == content


def test_saved_final_comparison_perclass_cv_and_standalone_figures(tmp_path):
    pytest.importorskip("matplotlib")
    metric = perfect()
    missing_auc = perfect()
    missing_auc["per_class"]["AD"]["auc"] = None
    missing_auc["macro_ovr_auc"] = None
    private = "PRIVATE_PERSON_389bd"
    metric["patient_ids"] = [private]
    save(tmp_path / "metrics.json", {"selected_variant": "B", "resnet18": {"best_epoch": 2, "validation": {"patient": metric}, "test": {"patient": missing_auc, "recording": metric}},
         "xlsr_mil": {"best_epoch": 3, "validation": {"patient": metric}, "test": {"patient": metric, "recording": metric}}})
    cv = {"folds": [{"fold": i, "patient": metric, "best_epoch": i} for i in range(1, 6)],
          "mean": {"macro_f1": .6, "loss": .5}, "std": {"macro_f1": .1, "loss": .05}}
    save(tmp_path / "cv_summary.json", {"resnet18": cv,
         "xlsr_mil": {"variants": {"A": cv, "B": {**cv, "mean": {"macro_f1": .7, "loss": .4}, "std": {"macro_f1": .05, "loss": .03}}}, "selected_variant": "B"}})
    history(tmp_path / "resnet18/history.csv")
    history(tmp_path / "xlsr_mil/final/history.csv")
    save(tmp_path / "report_state.json", {"status": "prepared"})
    original = (tmp_path / "metrics.json").read_bytes()
    report = write_report(tmp_path)
    page = assert_local_assets(report)
    assert "Saved evaluation complete" in page.text
    assert "ResNet18 selected epoch: 2" in page.text and "XLS-R MIL selected epoch: 3" in page.text
    assert "0.7000 ± 0.0500" in page.text and "Selected XLS-R variant: B" in page.text
    assert "ResNet18 · final test" in page.text and "XLS-R MIL · final test" in page.text
    assert "singleton bags" in page.text and "averages recording probabilities equally" in page.text
    assert "Unavailable" in page.text and "OvR ROC AUC" in page.text
    assert "No final test results yet" not in page.text
    assert (tmp_path / "metrics.json").read_bytes() == original
    for name in ("resnet18_learning", "xlsr_mil_learning", "cv_comparison", "resnet18_test_patient_confusion", "xlsr_mil_test_recording_confusion"):
        svg = tmp_path / f"figures/{name}.svg"
        assert svg.is_file() and svg.with_suffix(".png").is_file()
        content = svg.read_text()
        assert private not in content and "<dc:date>" not in content
    for name in ("resnet18_learning", "xlsr_mil_learning"):
        content = (tmp_path / f"figures/{name}.svg").read_text()
        shape = re.search(r'<svg[^>]*width="([\d.]+)pt"[^>]*height="([\d.]+)pt"', content)
        assert shape and float(shape[1]) < 1.2 * float(shape[2])  # readable stacked panels


def test_partial_subdirectory_results_do_not_invent_means_tests_or_mix_folds(tmp_path):
    pytest.importorskip("matplotlib")
    metric = perfect()
    save(tmp_path / "resnet18/cv/fold_01/metrics.json", {"best_epoch": 4, "validation": {"patient": metric}})
    save(tmp_path / "xlsr_mil/cv_progress.json", {"variants": {"C": {"folds": [{"fold": 1, "best_epoch": 2, "patient": metric}]}}})
    save(tmp_path / "resnet18/final/metrics.json", {"best_epoch": 2, "validation": {"patient": metric}})
    history(tmp_path / "resnet18/history.csv", mixed=True)
    report = write_report(tmp_path, status="failed")
    write_report(tmp_path)  # preserve an explicitly stopped status on regeneration
    page = Page(report.read_text())
    assert "Run stopped" in page.text
    assert "No final test results yet" in page.text
    assert "Unavailable ± Unavailable" in page.text
    assert "Selected XLS-R variant: Not selected" in page.text
    assert "Mixed-fold histories were omitted" in page.text
    assert "ResNet18 · selected checkpoint validation" in page.text
    assert not (tmp_path / "figures/resnet18_learning.svg").exists()
    assert not (tmp_path / "cv_summary.json").exists() and not (tmp_path / "metrics.json").exists()


def test_custom_local_assets_a4_export_and_protected_destination(tmp_path):
    pytest.importorskip("weasyprint")
    output = tmp_path / "run #1"
    save(output / "summary.json", {"total_patients": 234, "total_recordings": 9080})
    custom = tmp_path / "exports" / "comparison.html"
    report = write_report(output, custom, status="prepared")
    assert_local_assets(report)
    original_shared_css = (ANALYSIS.parents[1] / "s01/analysis/report_a4.css").read_bytes()
    pdf = export_report(output, custom, tmp_path / "exports" / "comparison.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert (ANALYSIS.parents[1] / "s01/analysis/report_a4.css").read_bytes() == original_shared_css
    if importlib.util.find_spec("pymupdf"):
        import pymupdf
        doc = pymupdf.open(pdf)
        text = "\n".join(page.get_text() for page in doc)
        assert "s02 · ResNet18 / XLS-R MIL" in text
        assert "s01 XLS-R MIL" not in text and "Generated" not in text
        assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
        assert all(abs(page.rect.width - 595.28) < 1 and abs(page.rect.height - 841.89) < 1 for page in doc)
        assert all(page.get_text().strip() for page in doc)
        assert not doc.metadata.get("creationDate") and not doc.metadata.get("modDate")
    with pytest.raises(ValueError):
        write_report(output, output / "summary.json")
