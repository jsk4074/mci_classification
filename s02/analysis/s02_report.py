#!/usr/bin/env python3
"""Timestamp-free s02 comparison reports from saved aggregate artifacts only.

Importing this module requires only the standard library. Plotting and PDF
dependencies are loaded only when needed. Patient manifests, prediction rows,
source audio, free-form metadata and exception text never enter the report.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote


LABELS = ("NM", "AD", "MCI")
METHODS = {"resnet18": "ResNet18", "xlsr_mil": "XLS-R MIL"}
VARIANTS = ("A", "B", "C")
SCORES = ("macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auc", "loss")
STATUSES = {"awaiting_data": "Awaiting saved results", "prepared": "Cohort prepared",
            "cv": "Development cross-validation", "training": "Final training in progress",
            "completed": "Saved evaluation complete", "failed": "Run stopped"}
ALIASES = {"running": "training", "cross_validation": "cv", "final": "training", "complete": "completed"}
SHARED_ANALYSIS = Path(__file__).resolve().parents[2] / "s01" / "analysis"


def _escape(value):
    return html.escape(str(value), quote=True)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _count(value):
    value = _number(value)
    return int(value) if value is not None and value >= 0 and value == int(value) else None


def _fmt(value, kind="score"):
    value = _number(value)
    if value is None:
        return "Unavailable"
    return f"{value:,.0f}" if kind == "count" else f"{value:.1%}" if kind == "percent" else f"{value:.4f}"


def _read(path, warnings):
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    except (OSError, UnicodeError, ValueError):
        pass
    warnings.append("An incomplete aggregate artifact was omitted.")
    return {}


def _atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".s02-report-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _metric(raw):
    if not isinstance(raw, dict) or not raw:
        return {}
    result = {key: _number(raw.get(key)) for key in SCORES}
    result["count"] = _count(raw.get("count"))
    result["per_class"] = {}
    for label in LABELS:
        values = _dict(_dict(raw.get("per_class")).get(label))
        result["per_class"][label] = {key: _number(values.get(key)) for key in ("precision", "recall", "f1", "auc")}
        result["per_class"][label]["support"] = _count(values.get("support"))
    matrix = raw.get("confusion_matrix")
    order = raw.get("confusion_matrix_labels", raw.get("labels", list(LABELS)))
    if (order == list(LABELS) and isinstance(matrix, list) and len(matrix) == 3
            and all(isinstance(row, list) and len(row) == 3
                    and all(_count(cell) is not None for cell in row) for row in matrix)):
        result["confusion_matrix"] = [[int(cell) for cell in row] for row in matrix]
    return result


def _levels(raw):
    return {level: metric for level in ("patient", "recording")
            if (metric := _metric(_dict(raw).get(level)))}


def _relative(path, report):
    return _escape(quote(os.path.relpath(path, report.parent).replace(os.sep, "/"), safe="/"))


def _plotter(output, warnings):
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output / "figures" / ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot
        return pyplot
    except ImportError:
        warnings.append("Matplotlib is unavailable; saved numeric tables remain available.")
        return None


def _save(plt, figure, output, name):
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.hashsalt": "s02-aggregate-report", "svg.fonttype": "none"}):
        figure.savefig(directory / f"{name}.svg", bbox_inches="tight",
                       metadata={"Creator": "s02 aggregate comparison", "Date": None})
        figure.savefig(directory / f"{name}.png", bbox_inches="tight", dpi=160,
                       metadata={"Software": "s02 aggregate comparison"})
    plt.close(figure)
    return directory / f"{name}.svg"


def _figure(path, report, caption):
    return (f'<figure><img src="{_relative(path, report)}" alt="{_escape(caption)}">'
            f'<figcaption>{_escape(caption)} · <a href="{_relative(path, report)}">SVG</a> · '
            f'<a href="{_relative(path.with_suffix(".png"), report)}">PNG</a></figcaption></figure>')


def _table(headers, rows, caption=None):
    if not rows:
        return '<p class="empty">No saved aggregate rows are available.</p>'
    attribute = f' aria-label="{_escape(caption)}"' if caption else ""
    return ('<div class="scroll"><table' + attribute + '><thead><tr>'
            + "".join(f'<th>{_escape(value)}</th>' for value in headers) + '</tr></thead><tbody>'
            + "".join('<tr>' + "".join(f'<td>{value}</td>' for value in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def _cohort(summary, audit):
    classes = _dict(summary.get("classes"))
    rows = [[label, _fmt(_dict(classes.get(label)).get("patients"), "count"),
             _fmt(_dict(classes.get(label)).get("recordings"), "count")] for label in LABELS]
    split_rows = []
    for split in ("train", "val", "test"):
        values = _dict(_dict(summary.get("splits")).get(split))
        split_rows.append([split, _fmt(values.get("patients"), "count"), _fmt(values.get("recordings"), "count")]
                          + [_fmt(_dict(_dict(values.get("classes")).get(label)).get("patients"), "count") for label in LABELS])
    # Accept only explicitly numeric aggregate audit fields; never copy notes,
    # source strings, identities, task filenames, paths or clinical row metadata.
    excluded = _dict(audit.get("excluded_old_mci"))
    new = _dict(audit.get("new_mci"))
    audit_values = (("Prior MCI patients excluded", excluded.get("patients")),
                    ("Prior MCI recordings excluded", excluded.get("recordings")),
                    ("Repeated task slots with distinct retained takes", new.get("duplicate_task_slots")),
                    ("Recordings retained in those repeated slots", new.get("recordings_in_duplicate_task_slots")),
                    ("Additional distinct takes retained", new.get("additional_distinct_takes")),
                    ("Participants with multiple takes", new.get("participants_with_multiple_takes")))
    audit_rows = [[label, _fmt(value, "count")] for label, value in audit_values if _count(value) is not None]
    audit_table = _table(("Cohort audit", "Count"), audit_rows) if audit_rows else ""
    source_rows = []
    for source, name in (("existing_approved_NM_AD", "Retained prior NM/AD"), ("full_dataset_MCI", "New MCI archive")):
        for split in ("train", "val", "test"):
            for task in ("06", "07"):
                values = _dict(_dict(_dict(_dict(audit.get("source_split_task_counts")).get(source)).get(split)).get(task))
                if values:
                    source_rows.append([name, split, task, _fmt(values.get("patients"), "count"), _fmt(values.get("recordings"), "count")])
    source_table = '<h3>Source and task coverage</h3>' + _table(("Source cohort", "Split", "Task", "Patients", "Recordings"), source_rows) if source_rows else ""
    return ('<section id="cohort"><h2>Cohort and shared patient partitions</h2>'
            '<p>The cohort combines retained prior NM/AD recordings with the new MCI archive. Prior MCI recordings are excluded. '
            'Eligibility uses canonical 06/07 tasks. Distinct repeated takes from the same patient remain eligible; '
            'exact duplicate content and unresolved cross-patient identity conflicts must be handled before the manifest is frozen.</p>'
            f'<p>Included patients: <strong>{_fmt(summary.get("total_patients"), "count")}</strong> · '
            f'Included recordings: <strong>{_fmt(summary.get("total_recordings"), "count")}</strong>.</p>'
            + _table(("Diagnosis", "Patients", "Recordings"), rows) + audit_table + source_table
            + '<h3>Shared frozen splits</h3><p>Both methods use the same approved patient memberships. All recordings from a person stay together. '
            'Prior NM/AD split roles are preserved; new MCI patients are assigned deterministically.</p>'
            + _table(("Split", "Patients", "Recordings", "NM patients", "AD patients", "MCI patients"), split_rows) + '</section>')


def _domain(audit):
    if not audit:
        return '<section id="domain"><h2>Technical source audit</h2><p class="empty">No source-header aggregate is available.</p></section>'
    rows = []
    for label in LABELS:
        for split in ("train", "val", "test"):
            value = _dict(_dict(_dict(audit.get("by_diagnosis_and_original_split")).get(label)).get(split))
            if not value:
                continue
            formats = []
            for raw in value.get("audio_formats", []) if isinstance(value.get("audio_formats"), list) else []:
                raw = _dict(raw)
                encoding = raw.get("encoding", "")
                if not isinstance(encoding, str) or not re.fullmatch(r"(?:PCM_(?:S8|U8|16|24|32)|FLOAT|DOUBLE|ULAW|ALAW|IMA_ADPCM|MS_ADPCM|GSM610|VORBIS|OPUS)", encoding):
                    encoding = "Other encoding"
                formats.append(f'{_fmt(raw.get("sample_rate"), "count")} Hz / {_fmt(raw.get("channels"), "count")} ch / {encoding}: {_fmt(raw.get("recordings"), "count")}')
            duration = _dict(value.get("duration_seconds"))
            rows.append([label, split, _fmt(value.get("recordings"), "count"), _fmt(value.get("header_errors"), "count"),
                         " / ".join(_fmt(duration.get(key)) for key in ("min", "median", "max")), _escape("; ".join(formats) or "Unavailable")])
    return ('<section id="domain"><h2>Technical source audit</h2>'
            f'<p>Headers inspected: {_fmt(audit.get("headers_read"), "count")} · Header errors: {_fmt(audit.get("header_errors"), "count")}. '
            'Header inspection includes all splits and performs no model evaluation.</p>'
            + _table(("Diagnosis", "Split", "Recordings", "Header errors", "Duration min / median / max (s)", "Rate / channels / encoding: count"), rows)
            + '<p>These are available technical metadata. They do not establish device/site identities or verify clinical diagnoses.</p></section>')


def _load_cv(output, raw, warnings):
    result = _dict(raw).copy()
    if not _dict(result.get("resnet18")):
        baseline = _read(output / "resnet18" / "cv_summary.json", warnings)
        if not baseline:
            folds = []
            for number in range(1, 6):
                value = _read(output / "resnet18" / "cv" / f"fold_{number:02d}" / "metrics.json", warnings)
                if value:
                    folds.append({"fold": number, "best_epoch": value.get("best_epoch"),
                                  **_dict(value.get("validation"))})
            baseline = {"folds": folds} if folds else {}
        result["resnet18"] = baseline
    if not _dict(result.get("xlsr_mil")):
        directory = output / "xlsr_mil"
        result["xlsr_mil"] = _read(directory / "cv_summary.json", warnings) or _read(directory / "cv_progress.json", warnings)
    return result


def _cv(cv, output, report, warnings):
    rows, details, chart = [], [], []
    candidates = [("ResNet18", _dict(cv.get("resnet18")))]
    variants = _dict(_dict(cv.get("xlsr_mil")).get("variants"))
    candidates.extend((f"XLS-R {variant}", _dict(variants[variant])) for variant in VARIANTS if variant in variants)
    for name, value in candidates:
        if not value:
            continue
        folds = value.get("folds", []) if isinstance(value.get("folds"), list) else []
        mean, std = _dict(value.get("mean")), _dict(value.get("std"))
        rows.append([name, str(len(folds))] + [f'{_fmt(mean.get(key))} ± {_fmt(std.get(key))}' for key in ("macro_f1", "balanced_accuracy", "macro_ovr_auc", "loss")])
        if (score := _number(mean.get("macro_f1"))) is not None:
            chart.append((name, score, _number(std.get("macro_f1"))))
        for fold in folds:
            fold = _dict(fold)
            if _count(fold.get("fold")) is None:
                continue
            metric = _metric(fold.get("patient"))
            details.append([name, _fmt(fold["fold"], "count"), _fmt(fold.get("best_epoch"), "count"), _fmt(metric.get("count"), "count")]
                           + [_fmt(metric.get(key)) for key in ("macro_f1", "balanced_accuracy", "macro_ovr_auc", "loss")])
    figure = ""
    if chart and (plt := _plotter(output, warnings)):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for name, score, std in chart:
            ax.bar(name, score, yerr=std if std is not None and std >= 0 else None, capsize=5)
        ax.set(ylabel="Development CV patient macro F1", ylim=(0, max(1.02, *(score + (std or 0) + .05 for _, score, std in chart))))
        ax.grid(axis="y", alpha=.2)
        figure = _figure(_save(plt, fig, output, "cv_comparison"), report, "Development CV means; error bars are sample fold SD, not confidence intervals")
    selected = _dict(cv.get("xlsr_mil")).get("selected_variant")
    selected = selected if selected in VARIANTS else "Not selected"
    pending = '<p class="empty">No completed cross-validation results yet.</p>' if not rows else ""
    return ('<section id="cv"><h2>Shared five-fold development CV</h2>'
            f'<p>Selected XLS-R variant: <strong>{selected}</strong>. ResNet18 and all XLS-R variants use the same patient folds. '
            'Selection uses mean patient macro F1; ties use lower mean patient loss, then A → B → C. Test patients are excluded.</p>'
            + pending + figure + _table(("Method", "Saved folds", "Macro F1 mean ± SD", "Balanced accuracy mean ± SD", "Macro AUC mean ± SD", "Loss mean ± SD"), rows)
            + '<h3>Saved fold checkpoints</h3>' + _table(("Method", "Fold", "Best epoch", "Patients", "Macro F1", "Balanced accuracy", "Macro AUC", "Loss"), details)
            + '<p class="small">Only saved fold results are shown. Missing means remain unavailable; partial folds are never presented as a completed comparison. '
            'CV chooses checkpoints and the XLS-R architecture; it is development model selection, not an unbiased nested-CV estimate.</p></section>')


def _history(path, warnings):
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = []
            for raw in csv.DictReader(handle):
                epoch = _count(raw.get("epoch"))
                if epoch is None:
                    continue
                rows.append({"epoch": epoch, **{key: _number(raw.get(key)) for key in
                    ("train_loss", "val_patient_loss", "val_patient_macro_f1", "val_patient_balanced_accuracy")},
                    "variant": raw.get("variant") if raw.get("variant") in VARIANTS else None,
                    "fold": _count(raw.get("fold"))})
    except (OSError, UnicodeError, csv.Error):
        warnings.append("An incomplete epoch history was omitted.")
        return []
    if len({(row["variant"], row["fold"]) for row in rows}) > 1:
        warnings.append("Mixed-fold histories were omitted; use the separate fold reports.")
        return []
    return rows


def _learning(output, report, warnings):
    blocks = []
    for method, name in METHODS.items():
        directory = output / method
        path = directory / "history.csv"
        if not path.exists():
            path = directory / "final" / "history.csv"
        rows = _history(path, warnings)
        if not rows:
            blocks.append(f'<h3>{name}</h3><p class="empty">No completed final-fit epoch history yet.</p>')
            continue
        figure = ""
        if plt := _plotter(output, warnings):
            fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), constrained_layout=True)
            for ax, keys in zip(axes, ((("train_loss", "Training loss"), ("val_patient_loss", "Validation patient loss")),
                                       (("val_patient_macro_f1", "Validation patient macro F1"), ("val_patient_balanced_accuracy", "Validation patient balanced accuracy")))):
                for key, label in keys:
                    values = [(row["epoch"], row[key]) for row in rows if row[key] is not None]
                    if values:
                        ax.plot([x for x, _ in values], [y for _, y in values], marker="o", markersize=3, label=label)
                ax.set_xlabel("Epoch")
                ax.grid(alpha=.2)
                if ax.lines:
                    ax.legend(fontsize=9)
            axes[0].set_ylabel("Cross-entropy loss")
            axes[1].set(ylabel="Patient score", ylim=(0, 1.03))
            figure = _figure(_save(plt, fig, output, f"{method}_learning"), report, f"{name}: final training and patient-validation history")
        values = [[_fmt(row["epoch"], "count")] + [_fmt(row[key]) for key in ("train_loss", "val_patient_loss", "val_patient_macro_f1", "val_patient_balanced_accuracy")] for row in rows]
        blocks.append(f'<article><h3>{name}</h3>{figure}' + _table(("Epoch", "Train loss", "Patient validation loss", "Patient macro F1", "Patient balanced accuracy"), values) + '</article>')
    return '<section id="learning"><h2>Final-fit learning curves</h2><p>Each curve describes one method’s final fit. CV fold histories are kept separate.</p>' + "".join(blocks) + '</section>'


def _load_metrics(output, raw, warnings):
    result = _dict(raw).copy()
    for method in METHODS:
        if not _dict(result.get(method)):
            directory = output / method
            result[method] = _read(directory / "metrics.json", warnings) or _read(directory / "final" / "metrics.json", warnings)
    return result


def _evaluation(method, split, raw, output, report, warnings):
    levels = _levels(raw)
    if not levels:
        return ""
    name = METHODS[method]
    title = f'{name} · {"selected checkpoint validation" if split == "validation" else "final test"}'
    blocks = []
    for level, metric in levels.items():
        description = ("Primary patient assessment: ResNet18 averages recording probabilities equally within each patient."
                       if method == "resnet18" else "Primary patient assessment: one MIL bag prediction uses all available recordings from the patient.")
        if level == "recording":
            description = ("Supplementary recording assessment: each recording inherits the patient diagnosis. "
                           + ("XLS-R recording predictions use singleton bags. " if method == "xlsr_mil" else "")
                           + "Correlated clips are not independent patients and never select the architecture or checkpoint.")
        cards = ''.join(f'<div><span>{label}</span><strong>{_fmt(metric.get(key), "score" if key == "loss" else "percent")}</strong></div>' for key, label in
                        (("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced accuracy"), ("accuracy", "Accuracy"), ("macro_ovr_auc", "Macro OvR ROC AUC"), ("loss", "Loss")))
        rows = [[label] + [_fmt(metric["per_class"][label].get(key), "count" if key == "support" else "percent")
                          for key in ("precision", "recall", "f1", "auc", "support")] for label in LABELS]
        matrix = metric.get("confusion_matrix")
        confusion = '<p class="empty">No saved confusion matrix is available.</p>'
        if matrix is not None:
            confusion = _table(("True / predicted", *LABELS), [[label, *map(str, row)] for label, row in zip(LABELS, matrix)])
            if plt := _plotter(output, warnings):
                fig, ax = plt.subplots(figsize=(4.6, 4), constrained_layout=True)
                image = ax.imshow(matrix, cmap="Blues", vmin=0)
                fig.colorbar(image, ax=ax, label="Count")
                ax.set(xticks=range(3), xticklabels=LABELS, yticks=range(3), yticklabels=LABELS,
                       xlabel="Predicted class", ylabel="True class", title=f'{name}: {split} {level}')
                for y in range(3):
                    for x in range(3):
                        ax.text(x, y, str(matrix[y][x]), ha="center", va="center")
                confusion += _figure(_save(plt, fig, output, f'{method}_{split}_{level}_confusion'), report, f'{name}: {split} {level} confusion counts')
        blocks.append(f'<article><h3>{level.capitalize()} level · n = {_fmt(metric.get("count"), "count")}</h3><p>{description}</p>'
                      + f'<div class="cards">{cards}</div><h4>Per-class performance</h4>'
                      + _table(("Class", "Precision", "Recall", "F1", "OvR ROC AUC", "Support"), rows)
                      + '<h4>Confusion matrix</h4>' + confusion + '</article>')
    return f'<section><h2>{title}</h2>' + ''.join(blocks) + '<p class="small">AUC uses class probabilities. A class without positive and negative examples has no defined AUC; full three-class macro AUC is unavailable if any class AUC is undefined.</p></section>'


def _comparison(metrics):
    rows = []
    for split in ("validation", "test"):
        for method, name in METHODS.items():
            metric = _metric(_dict(_dict(metrics.get(method)).get(split)).get("patient"))
            if metric:
                rows.append([name, split, _fmt(metric.get("count"), "count")]
                            + [_fmt(metric.get(key)) for key in ("macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auc")])
    pending = '<p class="empty">No final test results yet.</p>' if not any(_dict(metrics.get(key)).get("test") for key in METHODS) else ''
    return ('<section id="results"><h2>Matched s02 patient results</h2><p>These are the current s02 reruns of the ResNet18 method from s00 and XLS-R MIL method from s01. '
            'Historical scores from the smaller cohort are not inserted as matched baselines. Each final checkpoint is selected on validation patients before one planned test evaluation per method.</p>'
            + pending + _table(("Method", "Split", "Patients", "Macro F1", "Balanced accuracy", "Accuracy", "Macro OvR AUC"), rows)
            + '<p class="small">Missing method results remain pending. Patient identity matching is enforced by the study runner; equal aggregate counts alone cannot establish matching.</p></section>')


CSS = '''
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#213442;font:15px/1.6 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:36px 24px}header{padding:8px 0 24px}h1{font-size:34px;line-height:1.2}h2{font-size:23px;margin-top:0}h3{margin-bottom:8px}h4{margin-bottom:5px}section{background:#fff;border:1px solid #dce4e9;border-radius:12px;padding:25px;margin-bottom:20px}article+article{border-top:1px solid #dce4e9;margin-top:24px;padding-top:20px}.badge{display:inline-block;background:#dcefe9;color:#126756;padding:4px 13px;border-radius:30px}.small,figcaption{font-size:12px;color:#526b7a}.empty{background:#f2f6f8;padding:16px;color:#526b7a}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:18px 0}.cards div{background:#eff5f7;border-radius:8px;padding:13px}.cards span{display:block;font-size:11px}.cards strong{font-size:24px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 8px;border-bottom:1px solid #dae3e9;text-align:right}th:first-child,td:first-child{text-align:left}thead th{background:#f1f5f8}img{max-width:100%;height:auto}figure{margin:18px 0}figcaption{text-align:center}a{color:#167369}nav{display:flex;gap:18px;flex-wrap:wrap}.notice{padding:15px;background:#fff3dc;border-left:4px solid #bc8e32}footer{text-align:center;font-size:12px;color:#526b7a}@media(max-width:700px){main{padding:20px 12px}section{padding:18px}.cards{grid-template-columns:1fr 1fr}}@media print{body{background:white}main{padding:0}nav{display:none}section{break-inside:auto}figure{break-inside:avoid}}
'''


def write_report(output_dir: Path, report_path: Path | None = None, *, status=None, **unused) -> Path:
    """Write only aggregate HTML/figures; never load models or patient rows."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = Path(report_path).expanduser().resolve() if report_path is not None else output / "report.html"
    if report.suffix.lower() not in (".html", ".htm"):
        raise ValueError("The report destination must be an HTML file.")
    warnings = []
    summary = _read(output / "summary.json", warnings)
    cohort = _read(output / "cohort_audit.json", warnings)
    summary = summary or cohort
    audit = _read(output / "domain_audit.json", warnings)
    cv = _load_cv(output, _read(output / "cv_summary.json", warnings), warnings)
    metrics = _load_metrics(output, _read(output / "metrics.json", warnings), warnings)
    previous = _read(output / "report_state.json", warnings)
    candidate = ALIASES.get(status, status)
    has_fits = any(_dict(metrics.get(key)).get("validation") for key in METHODS)
    has_cv = bool(_dict(cv.get("resnet18")).get("folds") or _dict(_dict(cv.get("xlsr_mil")).get("variants")))
    inferred = ("completed" if all(_dict(metrics.get(key)).get("test") for key in METHODS)
                else "training" if has_fits else "cv" if has_cv else "prepared" if summary else "awaiting_data")
    current = (candidate if candidate in STATUSES else "failed" if previous.get("status") == "failed"
               else inferred if inferred != "awaiting_data" else previous.get("status") if previous.get("status") in STATUSES else inferred)
    _atomic(output / "report_state.json", json.dumps({"status": current}, indent=2) + '\n')
    cv_html = _cv(cv, output, report, warnings)
    learning = _learning(output, report, warnings)
    panels = ''.join(_evaluation(method, split, _dict(metrics.get(method)).get(split), output, report, warnings)
                     for split in ("validation", "test") for method in METHODS)
    selected = metrics.get("selected_variant", _dict(metrics.get("xlsr_mil")).get("selected_variant", _dict(cv.get("xlsr_mil")).get("selected_variant")))
    selected = selected if selected in VARIANTS else "Not selected"
    epochs = ' · '.join(f'{name} selected epoch: {_fmt(_dict(metrics.get(method)).get("best_epoch"), "count")}' for method, name in METHODS.items())
    links = ''.join(f'<li><a href="{_relative(output / name, report)}">{name}</a></li>' for name in
                    ("summary.json", "cohort_audit.json", "domain_audit.json", "cv_summary.json", "metrics.json") if (output / name).is_file())
    warning_html = ''.join(f'<p class="notice">{_escape(item)}</p>' for item in dict.fromkeys(warnings))
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>s02 · matched speech classification comparison</title><style>{CSS}</style></head><body><main>
<header><p>s02 · NM / AD / MCI</p><h1>ResNet18 and XLS-R MIL on the expanded cohort</h1><span class="badge">{STATUSES[current]}</span><p>Matched patient splits, shared development folds, and saved aggregate results.</p><p class="small">Selected XLS-R variant: {selected} · {epochs}</p><nav><a href="#cohort">Cohort</a><a href="#method">Method</a><a href="#cv">CV</a><a href="#results">Patient results</a><a href="#limitations">Interpretation</a></nav></header>
{_cohort(summary, cohort)}
<section id="method"><h2>Method and evaluation protocol</h2><h3>ResNet18 rerun of the s00 method</h3><p>A randomly initialized single-channel ResNet18 classifies fixed-size 40-band log-mel inputs from mono 16 kHz audio. Training uses random five-second crops and equal expected class mass with equal patient mass within each class. Short clips are padded. Validation/test use up to five deterministic windows per recording; window probabilities are averaged into recording probabilities, then recordings are averaged equally within each patient.</p><h3>Frozen XLS-R rerun of the s01 method</h3><p>A pinned frozen multilingual XLS-R encoder processes up to five deterministic five-second waveform windows. Padding masks exclude padded samples and encoder frames. Valid frames are averaged within windows, then windows are averaged equally into recording representations. A trainable LayerNorm → Linear(256) → GELU → Dropout(0.3) projection and diagnosis head use all available recordings in each patient bag. A uses mean MIL; B uses learned attention MIL; C uses the same attention MIL with train-only channel variants. Validation/test always use base representations. Patient cross-entropy uses inverse training-patient class-frequency weights.</p><h3>Shared selection and once-only final evaluation</h3><p>Five-fold diagnosis-stratified CV combines the study training and validation patients; all recordings from each patient remain together. ResNet18 and XLS-R A/B/C use identical folds. Fold checkpoints use patient macro F1, with lower unweighted patient loss as the tie-breaker. Mean CV patient macro F1 selects the XLS-R variant; ties use lower mean patient loss, then A → B → C. After configuration is frozen, both final models train on the study training patients and select checkpoints on the study validation patients. Each selected method evaluates the study test patients once. Recording results are supplementary.</p></section>
{_domain(audit)}{cv_html}{learning}{_comparison(metrics)}{panels}
<section id="limitations"><h2>Interpretation and practical limitations</h2><p>The new MCI cohort and prior NM/AD cohort differ in source acquisition. Source and diagnosis are confounded; successful class separation can exploit cohort or channel cues. Resampling and channel transformations do not establish domain invariance. Cross-device or cross-site evidence requires independent devices/sites with diagnosis overlap. Diagnosis must never be treated as a measured device identity.</p><p>Prior NM/AD test outcomes were already known from s00/s01, and those patient split roles are retained. This is an expanded-cohort comparison with a partly reused known holdout, not a pristine blinded confirmatory test. Validation and development CV are used for selection and are not independent generalization estimates. Small NM/AD test supports and correlated clips limit precision; inspect AD recall and the full per-class confusion matrices. Strong MCI performance does not by itself demonstrate clinical generalization.</p><p>Report patient macro F1, balanced accuracy, AUC and supports together. Class imbalance can make accuracy misleading. Compare only the new matched s02 runs; historical scores from a different cohort do not provide a paired estimate of improvement.</p></section>
{warning_html}<section><h2>Saved aggregate artifacts</h2><ul>{links}</ul><p class="small">This report reads aggregate results only. Patient identities, source filenames, clinical metadata rows and prediction rows are not rendered.</p></section><footer>s02 comparison · local figures only · no model inference during report generation</footer></main></body></html>'''
    _atomic(report, document)
    return report


def export_report(output_dir: Path, report_path: Path | None = None, pdf_path: Path | None = None, *, status=None) -> Path:
    """Refresh aggregate HTML and publish an A4 PDF using shared export code."""
    report = write_report(output_dir, report_path, status=status)
    output = Path(output_dir).expanduser().resolve()
    destination = Path(pdf_path).expanduser().resolve() if pdf_path is not None else output / "report.pdf"
    if destination.suffix.lower() != ".pdf":
        raise ValueError("The PDF destination must end in .pdf.")
    spec = importlib.util.spec_from_file_location("s02_shared_pdf_export", SHARED_ANALYSIS / "s01_export_pdf.py")
    if spec is None or spec.loader is None:
        raise ImportError("The shared A4 PDF exporter is unavailable.")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    stylesheet = (SHARED_ANALYSIS / "report_a4.css").read_text(encoding="utf-8")
    stylesheet = stylesheet.replace("s01 XLS-R MIL · NM / AD / MCI", "s02 · ResNet18 / XLS-R MIL · NM / AD / MCI")
    with tempfile.TemporaryDirectory(prefix=".s02-print-", dir=output) as directory:
        css = Path(directory) / "report_a4.css"
        css.write_text(stylesheet, encoding="utf-8")
        return shared.export_pdf(report, destination, stylesheet=css)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--pdf-path", type=Path)
    parser.add_argument("--export-pdf", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.export_pdf or args.pdf_path is not None:
            result = export_report(args.output_dir, args.report_path, args.pdf_path)
        else:
            result = write_report(args.output_dir, args.report_path)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Report generation failed ({type(exc).__name__}).\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
