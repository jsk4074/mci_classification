"""Offline s01 XLS-R MIL report from aggregate artifacts, never patient rows.

The module imports only the standard library until a numeric figure is needed.
No model, audio decoder, prediction CSV, or patient manifest is loaded here.
Free-form messages and configuration strings are intentionally not copied into
HTML; local logs retain raw diagnostics. Progress fields are tightly whitelisted.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote
from zoneinfo import ZoneInfo


LABELS = ("NM", "AD", "MCI")
VARIANTS = ("A", "B", "C")
SPLITS = ("train", "val", "test", "unassigned")
SCORES = ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auc", "loss")
STATUSES = {"awaiting_data": "Awaiting data", "prepared": "Inputs prepared",
            "cv": "Cross-validation", "training": "Training in progress",
            "completed": "Saved evaluation complete", "failed": "Run stopped"}
ALIASES = {"running": "training", "cross_validation": "cv", "cv_running": "cv",
           "evaluating": "training", "complete": "completed"}


def _escape(value):
    return html.escape(str(value), quote=True)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _count(value):
    number = _number(value)
    return int(number) if number is not None and number >= 0 and number == int(number) else None


def _fmt(value, kind="score"):
    value = _number(value)
    if value is None:
        return "Unavailable"
    return f"{value:,.0f}" if kind == "count" else f"{value:.1%}" if kind == "percent" else f"{value:.4f}"


def _dict(value):
    return value if isinstance(value, dict) else {}


def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read(path, warnings):
    if not path.is_file():
        return {}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Aggregate object expected")
        return result
    except (ValueError, OSError, UnicodeError):
        warnings.append(f"Could not read {path.name}; incomplete information was omitted.")
        return {}


def _metric(value):
    """Strip unknown fields before they can enter HTML or persisted state."""
    if not isinstance(value, dict) or not value:
        return {}
    result = {key: _number(value.get(key)) for key in SCORES}
    result["count"] = _count(value.get("count"))
    result["per_class"] = {}
    for label in LABELS:
        values = _dict(_dict(value.get("per_class")).get(label))
        result["per_class"][label] = {key: _number(values.get(key)) for key in ("precision", "recall", "f1", "auc")}
        result["per_class"][label]["support"] = _count(values.get("support"))
    matrix = value.get("confusion_matrix")
    if (isinstance(matrix, list) and len(matrix) == 3 and
            all(isinstance(row, list) and len(row) == 3 and all(_count(cell) is not None for cell in row) for row in matrix)):
        result["confusion_matrix"] = [[int(cell) for cell in row] for row in matrix]
    return result


def _levels(value):
    value = _dict(value)
    if "macro_f1" in value:
        value = {"patient": value}
    return {level: result for level in ("patient", "recording") if (result := _metric(value.get(level)))}


def _history(path, warnings):
    if not path.is_file():
        return []
    rows = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                epoch = _count(raw.get("epoch"))
                if epoch is None:
                    continue
                row = {key: _number(raw.get(key)) for key in (
                    "train_loss", "val_patient_loss", "val_patient_macro_f1", "val_patient_balanced_accuracy")}
                row["epoch"] = epoch
                row["variant"] = raw.get("variant") if raw.get("variant") in VARIANTS else None
                row["fold"] = _count(raw.get("fold"))
                rows.append(row)
    except (OSError, UnicodeError, csv.Error):
        warnings.append("Could not read the complete epoch history.")
    contexts = {(row["variant"], row["fold"]) for row in rows}
    if len(contexts) > 1:
        warnings.append("Multiple fold histories were found. Mixed-fold learning curves were omitted; use the separate fold reports.")
        return []
    return rows


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
        if "Matplotlib is unavailable; numeric tables remain available." not in warnings:
            warnings.append("Matplotlib is unavailable; numeric tables remain available.")
        return None


def _save_figure(plotter, figure, output, name):
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    with plotter.rc_context({"svg.hashsalt": "s01-aggregate-report", "svg.fonttype": "none"}):
        figure.savefig(directory / f"{name}.svg", bbox_inches="tight",
                       metadata={"Creator": "s01 aggregate report", "Date": None})
        figure.savefig(directory / f"{name}.png", bbox_inches="tight", dpi=160,
                       metadata={"Software": "s01 aggregate report"})
    plotter.close(figure)
    return directory / f"{name}.svg"


def _figure_html(path, report, caption):
    return f'<figure><img src="{_relative(path, report)}" alt="{_escape(caption)}"><figcaption>{_escape(caption)} · <a href="{_relative(path.with_suffix(".png"), report)}">PNG</a> · <a href="{_relative(path, report)}">SVG</a></figcaption></figure>'


def _learning(rows, output, report, warnings):
    section = '<section id="learning"><h2>Learning curves</h2>'
    if not rows:
        return section + '<p class="empty">No completed epoch history is available for this run. Fold histories are plotted separately.</p></section>'
    plotter = _plotter(output, warnings)
    figure_html = ""
    if plotter:
        figure, axes = plotter.subplots(2, 1, figsize=(7, 6.5), constrained_layout=True)
        for axis, series in zip(axes, (
            (("train_loss", "Training loss"), ("val_patient_loss", "Validation patient loss")),
            (("val_patient_macro_f1", "Validation patient macro F1"), ("val_patient_balanced_accuracy", "Validation patient balanced accuracy")),
        )):
            for key, label in series:
                values = [(row["epoch"], row[key]) for row in rows if row[key] is not None]
                if values:
                    axis.plot([value[0] for value in values], [value[1] for value in values], marker="o", label=label)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize=9)
        axes[0].set_ylabel("Cross-entropy loss")
        axes[1].set_ylabel("Patient score")
        axes[1].set_ylim(0, 1.03)
        path = _save_figure(plotter, figure, output, "learning_curves")
        figure_html = _figure_html(path, report, "Training and patient-validation progress for one run or fold")
    cells = "".join(f'<tr><td>{row["epoch"]}</td>' + "".join(
        f'<td>{_fmt(row[key])}</td>' for key in ("train_loss", "val_patient_loss", "val_patient_macro_f1", "val_patient_balanced_accuracy")) + '</tr>' for row in rows)
    return section + figure_html + '<div class="scroll"><table><thead><tr><th>Epoch</th><th>Train loss</th><th>Patient validation loss</th><th>Patient macro F1</th><th>Patient balanced accuracy</th></tr></thead><tbody>' + cells + '</tbody></table></div></section>'


def _evaluation(title, levels, context, output, report, warnings):
    if not levels:
        return ""
    blocks = []
    for level, values in levels.items():
        count = _fmt(values.get("count"), "count")
        caption = ("Primary assessment: one patient-bag prediction per patient."
                   if level == "patient" else "Supplementary single-instance assessment: each recording inherits its patient's diagnosis. Correlated recordings are not additional independent patients and do not select the variant or checkpoint.")
        cards = "".join(f'<div><span>{label}</span><strong>{_fmt(values.get(key), "score" if key == "loss" else "percent")}</strong></div>'
                        for key, label in (("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced accuracy"),
                                           ("accuracy", "Accuracy"), ("macro_ovr_auc", "Macro OvR ROC AUC"), ("loss", "Loss")))
        rows = "".join('<tr><th>' + label + '</th>' + "".join(
            f'<td>{_fmt(values["per_class"][label].get(key), "count" if key == "support" else "percent")}</td>'
            for key in ("precision", "recall", "f1", "auc", "support")) + '</tr>' for label in LABELS)
        matrix_html = '<p class="empty">No confusion matrix is available.</p>'
        matrix = values.get("confusion_matrix")
        if matrix is not None:
            matrix_rows = "".join('<tr><th>' + label + '</th>' + "".join(f'<td>{cell}</td>' for cell in row) + '</tr>' for label, row in zip(LABELS, matrix))
            matrix_html = '<table aria-label="' + _escape(f'{title} {level} confusion matrix') + '"><thead><tr><th>True / predicted</th><th>NM</th><th>AD</th><th>MCI</th></tr></thead><tbody>' + matrix_rows + '</tbody></table>'
            plotter = _plotter(output, warnings)
            if plotter:
                figure, axis = plotter.subplots(figsize=(4.6, 4), constrained_layout=True)
                image = axis.imshow(matrix, cmap="Blues", vmin=0)
                figure.colorbar(image, ax=axis, label="Count")
                axis.set(xticks=range(3), xticklabels=LABELS, yticks=range(3), yticklabels=LABELS,
                         xlabel="Predicted class", ylabel="True class", title=f'{title}: {level}')
                for row in range(3):
                    for column in range(3):
                        axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
                path = _save_figure(plotter, figure, output, f'{context}_{level}_confusion')
                matrix_html += _figure_html(path, report, f'{title}: {level} confusion counts')
        blocks.append(f'<article><h3>{level.capitalize()} level · n = {count}</h3><p>{caption}</p><div class="cards">{cards}</div><h4>Per-class performance</h4><div class="scroll"><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>OvR ROC AUC</th><th>Support</th></tr></thead><tbody>{rows}</tbody></table></div><h4>Confusion matrix</h4>{matrix_html}</article>')
    return f'<section><h2>{_escape(title)}</h2>{"".join(blocks)}<p class="small">ROC AUC uses saved class probabilities. Without both positive and negative examples, a class AUC is unavailable; full three-class macro AUC is unavailable if any class AUC is undefined.</p></section>'


def _cv_section(cv, output, report, warnings):
    variants = _dict(cv.get("variants"))
    if not any(variant in variants for variant in VARIANTS):
        return '<section id="cv"><h2>Cross-validation comparison</h2><p class="empty">No completed cross-validation summary is available.</p></section>'
    rows, fold_rows, plotted = [], [], []
    for variant in VARIANTS:
        if variant not in variants:
            continue
        values = _dict(variants[variant])
        means, deviations = _dict(values.get("mean")), _dict(values.get("std"))
        folds = values.get("folds", [])
        folds = folds if isinstance(folds, list) else []
        cells = "".join(f'<td>{_fmt(means.get(key))} ± {_fmt(deviations.get(key))}</td>' for key in ("macro_f1", "balanced_accuracy", "macro_ovr_auc", "loss"))
        rows.append(f'<tr><th>{variant}</th><td>{len(folds)}</td>{cells}</tr>')
        score, deviation = _number(means.get("macro_f1")), _number(deviations.get("macro_f1"))
        if score is not None:
            plotted.append((variant, score, deviation))
        for fold in folds:
            fold = _dict(fold)
            index = _count(fold.get("fold"))
            if index is None:
                continue
            metric = _metric(fold.get("patient"))
            fold_rows.append(f'<tr><th>{variant}</th><td>{index}</td><td>{_fmt(metric.get("count"), "count")}</td>' + "".join(f'<td>{_fmt(metric.get(key))}</td>' for key in ("macro_f1", "balanced_accuracy", "macro_ovr_auc", "loss")) + '</tr>')
    chart = ""
    if plotted:
        plotter = _plotter(output, warnings)
        if plotter:
            figure, axis = plotter.subplots(figsize=(7, 4), constrained_layout=True)
            for variant, score, deviation in plotted:
                axis.bar(variant, score, yerr=deviation if deviation is not None and deviation >= 0 else None,
                         capsize=5, color={"A": "#497c9a", "B": "#188b7d", "C": "#b17935"}[variant])
            axis.set_ylabel("CV patient macro F1")
            axis.set_ylim(0, max(1.02, *(score + (deviation or 0) + 0.05 for _, score, deviation in plotted)))
            axis.grid(axis="y", alpha=0.2)
            chart = _figure_html(_save_figure(plotter, figure, output, "cv_variants"), report,
                                 "CV variant means; error bars are the supplied fold standard deviations, not confidence intervals")
    selected = cv.get("selected_variant") if cv.get("selected_variant") in VARIANTS else "Not selected"
    return f'''<section id="cv"><h2>Cross-validation comparison</h2><p>Selected variant: <strong>{selected}</strong>. Selection uses highest mean CV patient macro F1, then lower mean patient loss, then stable A → B → C order. Original test data does not select the variant.</p>{chart}<div class="scroll"><table><thead><tr><th>Variant</th><th>Saved folds</th><th>Macro F1 mean ± SD</th><th>Balanced accuracy mean ± SD</th><th>Macro AUC mean ± SD</th><th>Loss mean ± SD</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div><h3>Saved fold patient metrics</h3><div class="scroll"><table><thead><tr><th>Variant</th><th>Fold</th><th>Patients</th><th>Macro F1</th><th>Balanced accuracy</th><th>Macro AUC</th><th>Loss</th></tr></thead><tbody>{"".join(fold_rows)}</tbody></table></div><p class="small">Values are read from cv_summary.json. Missing means, deviations or AUCs remain unavailable. Fold histories are never merged into one learning curve.</p></section>'''


def _domain_section(audit):
    if not audit:
        return '<section id="domain"><h2>Technical source-domain audit</h2><p class="empty">No aggregate source-header audit is available.</p></section>'
    groups = _dict(audit.get("by_diagnosis_and_original_split"))
    rows = []
    for label in (*LABELS, "unknown"):
        for split in SPLITS:
            values = _dict(_dict(groups.get(label)).get(split))
            if not values:
                continue
            durations = _dict(values.get("duration_seconds"))
            formats = values.get("audio_formats", [])
            formats = formats if isinstance(formats, list) else []
            format_text = []
            for audio_format in formats:
                audio_format = _dict(audio_format)
                # Values originate in header APIs; only short encoding identifiers enter HTML.
                encoding = audio_format.get("encoding", "")
                encoding = encoding if isinstance(encoding, str) and re.fullmatch(r"(?:PCM_(?:S8|U8|16|24|32)|FLOAT|DOUBLE|ULAW|ALAW|IMA_ADPCM|MS_ADPCM|GSM610|VOX_ADPCM|G721_32|G723_24|G723_40|DWVW_12|DWVW_16|DWVW_24|DPCM_8|DPCM_16|VORBIS|OPUS|MPEG_LAYER_III|ALAC_(?:16|20|24|32))", encoding) else "Other encoding"
                format_text.append(f'{_fmt(audio_format.get("sample_rate"), "count")} Hz / {_fmt(audio_format.get("channels"), "count")} ch / {encoding}: {_fmt(audio_format.get("recordings"), "count")}')
            duration_text = " / ".join(_fmt(durations.get(key)) for key in ("min", "median", "max"))
            rows.append(f'<tr><th>{label}</th><td>{split}</td><td>{_fmt(values.get("recordings"), "count")}</td><td>{_fmt(values.get("header_errors"), "count")}</td><td>{duration_text}</td><td>{_escape("; ".join(format_text) or "Unavailable")}</td></tr>')
    return f'''<section id="domain"><h2>Technical source-domain audit</h2><p>Source recordings: {_fmt(audit.get("total_recordings"), "count")} · headers inspected: {_fmt(audit.get("headers_read"), "count")} · header errors: {_fmt(audit.get("header_errors"), "count")}. The audit reads headers across the original train / validation / test partitions; it performs no model evaluation.</p><div class="scroll"><table><thead><tr><th>Diagnosis</th><th>Original split</th><th>Recordings</th><th>Header errors</th><th>Duration min / median / max (s)</th><th>Observed rate / channels / encoding: count</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div><p>These are available technical metadata, not device or site identities. A diagnosis is never used to infer a recording device. Resampling and channel transformations do not establish domain invariance; clinical labels are not verified by audio headers.</p></section>'''


def _summary_section(summary):
    rows = []
    for split in ("train", "val", "test"):
        values = _dict(_dict(summary.get("splits")).get(split))
        classes = _dict(values.get("classes"))
        rows.append(f'<tr><th>{split}</th><td>{_fmt(values.get("patients"), "count")}</td><td>{_fmt(values.get("recordings"), "count")}</td>' + "".join(f'<td>{_fmt(_dict(classes.get(label)).get("patients"), "count")}</td>' for label in LABELS) + '</tr>')
    return '<section><h2>Original patient partitions</h2><p>These are original frozen assignments. Five-fold development CV combines original training and validation patients, keeping all original test patients excluded and using the same folds for A/B/C. Final fitting retains the original train / validation / test roles.</p><div class="scroll"><table><thead><tr><th>Split</th><th>Patients</th><th>Recordings</th><th>NM patients</th><th>AD patients</th><th>MCI patients</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></section>'


def _comparison(baseline, current):
    if not baseline:
        return ""
    rows = []
    for name, values in (("s00", baseline), ("s01", current)):
        for split in ("validation", "test"):
            patient = _metric(_dict(_dict(values.get(split)).get("patient")))
            rows.append(f'<tr><th>{name}</th><td>{split}</td><td>{_fmt(patient.get("count"), "count")}</td>' + "".join(f'<td>{_fmt(patient.get(key))}</td>' for key in ("macro_f1", "balanced_accuracy", "macro_ovr_auc")) + '</tr>')
    return '<section><h2>Exploratory s00 / s01 comparison</h2><p>s00 test results were already known before s01 design. This is reuse of a known holdout, not a new blinded confirmatory comparison. Compare only when private patient memberships match; equal aggregate counts alone do not establish matched membership.</p><table><thead><tr><th>System</th><th>Split</th><th>Patients</th><th>Macro F1</th><th>Balanced accuracy</th><th>Macro AUC</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></section>'


CSS = '''
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#213442;font:15px/1.6 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:36px 24px}header{padding:8px 0 24px}h1{font-size:34px;line-height:1.2}h2{font-size:23px;margin-top:0}h3{margin-bottom:8px}h4{margin-bottom:5px}section{background:#fff;border:1px solid #dce4e9;border-radius:12px;padding:25px;margin-bottom:20px}article+article{border-top:1px solid #dce4e9;margin-top:24px;padding-top:20px}.badge{display:inline-block;background:#dcefe9;color:#126756;padding:4px 13px;border-radius:30px}.failed{background:#fbe9e7;color:#9a322d}.small,figcaption{font-size:12px;color:#526b7a}.empty{background:#f2f6f8;padding:16px;color:#526b7a}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:18px 0}.cards div{background:#eff5f7;border-radius:8px;padding:13px}.cards span{display:block;font-size:11px}.cards strong{font-size:24px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;border-bottom:1px solid #dae3e9;text-align:right}th:first-child{text-align:left}thead th{background:#f1f5f8}img{max-width:100%;height:auto}figure{margin:18px 0}figcaption{text-align:center}a{color:#167369}nav{display:flex;gap:18px;flex-wrap:wrap}ul{padding-left:23px}.methods{display:grid;grid-template-columns:1fr 1fr;gap:20px}.notice{padding:15px;background:#fff3dc;border-left:4px solid #bc8e32}footer{text-align:center;font-size:12px;color:#526b7a}@media(max-width:700px){main{padding:20px 12px}section{padding:18px}.cards{grid-template-columns:1fr 1fr}.methods{grid-template-columns:1fr}}@media print{body{background:white}main{padding:0}section{break-inside:avoid;box-shadow:none}nav{display:none}}
'''


def write_report(output_dir: Path, report_path: Path | None = None, *, status=None,
                 message=None, latest_validation=None, best_epoch=None, **progress) -> Path:
    """Regenerate the aggregate report and standalone figures without inference.

    Only status, numeric epoch/fold progress, A/B/C variant and whitelisted
    aggregate validation metrics are persisted. Free-form message is accepted
    for launcher compatibility but omitted to prevent identifiers leaking from
    exception text. No patient-row CSV or raw configuration string is rendered.
    """

    output = Path(output_dir).expanduser().resolve()
    report = Path(report_path).expanduser().resolve() if report_path is not None else output / "report.html"
    warnings = []
    config = _read(output / "config.json", warnings)
    summary = _read(output / "summary.json", warnings)
    audit = _read(output / "domain_audit.json", warnings)
    cv = _read(output / "cv_summary.json", warnings)
    if not cv:
        cv = _read(output / "cv_progress.json", warnings)
    final = _read(output / "metrics.json", warnings)
    baseline = _read(output / "s00_metrics.json", warnings)
    previous = _read(output / "report_state.json", warnings)
    state = {key: _count(previous.get(key)) for key in ("best_epoch", "epoch", "fold") if _count(previous.get(key)) is not None}
    if previous.get("variant") in VARIANTS:
        state["variant"] = previous["variant"]
    state["latest_validation"] = _levels(previous.get("latest_validation"))
    for key, value in (("best_epoch", best_epoch), ("epoch", progress.get("epoch")), ("fold", progress.get("fold"))):
        if _count(value) is not None:
            state[key] = _count(value)
    if progress.get("variant") in VARIANTS:
        state["variant"] = progress["variant"]
    elif "variant" not in state and config.get("variant") in VARIANTS:
        state["variant"] = config["variant"]
    if "fold" not in state and _count(config.get("fold")) is not None:
        state["fold"] = _count(config["fold"])
    if latest_validation is not None:
        state["latest_validation"] = _levels(latest_validation)
    inferred = "completed" if final else "cv" if cv else "prepared" if summary else "awaiting_data"
    current = status if status is not None else previous.get("status", inferred)
    current = ALIASES.get(current, current) if isinstance(current, str) else None
    if current not in STATUSES:
        raise ValueError("Unsupported report status; use awaiting_data, prepared, cv, training, completed, or failed.")
    state["status"] = current
    _atomic(output / "report_state.json", json.dumps(state, indent=2, allow_nan=False) + "\n")
    rows = _history(output / "history.csv", warnings)
    latest = state["latest_validation"]
    if not latest and rows:
        latest = {"patient": _metric({"macro_f1": rows[-1]["val_patient_macro_f1"],
                                       "balanced_accuracy": rows[-1]["val_patient_balanced_accuracy"],
                                       "loss": rows[-1]["val_patient_loss"]})}
    learning = _learning(rows, output, report, warnings)
    cross_validation = _cv_section(cv, output, report, warnings)
    latest_html = _evaluation("Latest completed epoch validation", latest, "latest", output, report, warnings)
    validation_html = _evaluation("Selected checkpoint validation", _levels(final.get("validation")), "validation", output, report, warnings)
    test_html = _evaluation("Final test evaluation", _levels(final.get("test")), "test", output, report, warnings)
    if not test_html:
        test_html = '<section><h2>Final test evaluation</h2><p class="empty">No final test results are available. Prepared splits, CV results and validation progress do not imply test evaluation.</p></section>'
    selected_epoch = _count(final.get("best_epoch")) or state.get("best_epoch")
    notes = {"awaiting_data": "No model results have been saved.", "prepared": "Inputs are prepared; final results remain unavailable.",
             "cv": "CV combines original training and validation patients; original test data does not select the variant.",
             "training": "Only saved completed epochs are shown. Final test results remain separate.",
             "completed": "Available saved aggregate evaluations are displayed below.",
             "failed": "The run stopped. Available progress is shown; missing results remain unavailable."}
    config_values = _dict(config.get("arguments")) or _dict(config.get("parameters")) or config
    projection = _count(config_values.get("projection_size")) or 256
    dropout = _number(config_values.get("dropout"))
    dropout = dropout if dropout is not None and 0 <= dropout < 1 else .3
    window = _number(config_values.get("clip_seconds"))
    window = window if window is not None and window > 0 else 5.0
    windows = _count(config_values.get("eval_windows")) or 5
    settings = "".join(f'<li>{title}: {_fmt(config_values.get(key), "count" if key in ("seed", "batch_size", "epochs") else "score")}</li>'
                       for key, title in (("seed", "Seed"), ("batch_size", "Patient batch size"), ("epochs", "Epoch cap"), ("learning_rate", "Learning rate"), ("weight_decay", "Weight decay")) if _number(config_values.get(key)) is not None)
    revision = config.get("model_revision", config_values.get("encoder_revision"))
    if isinstance(revision, str) and re.fullmatch(r"[a-fA-F0-9]{7,64}", revision):
        settings += f'<li>Model revision: {_escape(revision)}</li>'
    known_artifacts = ("config.json", "summary.json", "domain_audit.json", "cv_summary.json", "cv_progress.json", "selection.json",
                       "history.csv", "metrics.json", "s00_metrics.json", "best.pt")
    links = "".join(f'<li><a href="{_relative(output / name, report)}">{name}</a></li>' for name in known_artifacts if (output / name).is_file())
    warning_html = '<section><h2>Artifact reading notes</h2><ul>' + "".join(f'<li>{_escape(warning)}</li>' for warning in warnings) + '</ul></section>' if warnings else ""
    refreshed = '<meta http-equiv="refresh" content="20">' if current in ("cv", "training") else ""
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    progress_text = " · ".join(f'{key.capitalize()}: {_escape(state[key])}' for key in ("variant", "fold", "epoch") if key in state)
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refreshed}<title>s01 · XLS-R patient-bag MIL</title><style>{CSS}</style></head><body><main>
<header><p>s01 · NM / AD / MCI</p><h1>XLS-R patient-bag multiple-instance learning</h1><span class="badge {'failed' if current == 'failed' else ''}">{STATUSES[current]}</span><p>{notes[current]}</p><p class="small">Generated {timestamp} · Selected epoch: {_fmt(selected_epoch, "count")} · {progress_text}</p><nav><a href="#method">Method</a><a href="#domain">Domain audit</a><a href="#cv">CV comparison</a><a href="#learning">Learning curves</a><a href="#limitations">Interpretation</a></nav></header>
<section id="method"><h2>Method and evaluation protocol</h2><div class="methods"><article><h3>Frozen encoder, patient bags</h3><p>Audio is converted to mono at 16 kHz. Each recording supplies up to {windows} deterministic {_fmt(window)}-second waveform windows. Valid encoder frames are mean-pooled within each window, then window embeddings are averaged equally into one recording representation. Padding masks exclude padded waveform samples and encoder frames; the window cap can leave gaps in very long recordings.</p><p>A frozen multilingual XLS-R speech encoder provides recording representations. A trainable LayerNorm → Linear({projection}) → GELU → Dropout({dropout:g}) projection precedes patient-bag aggregation and a three-class linear diagnosis head. All available recording representations belong to that patient's bag.</p><p>Variant A uses mean MIL. Variant B uses learned attention MIL. Variant C uses the same attention MIL with train-only precomputed channel variants: each training recording can use its unaugmented base representation or an augmented representation. Validation and test use base representations.</p></article><article><h3>Patient-primary selection</h3><p>Five-fold CV combines original training and validation patients as the development pool. All recordings from one person stay together; A/B/C share identical folds and original test patients remain excluded. Mean CV patient macro F1 selects A/B/C; ties use lower mean patient loss, then A → B → C. The selected final model is fitted on original training patients; original validation patients select its checkpoint, followed by one original-test evaluation.</p><p>Training uses ordinary shuffled patient batches and inverse training-patient class-frequency weights in cross-entropy. Validation loss is unweighted patient cross-entropy. Single-instance recording results are supplemental and do not tune the bag model, select variants, or select checkpoints.</p><ul>{settings}</ul></article></div></section>
{_summary_section(summary)}{_domain_section(audit)}{cross_validation}{learning}{latest_html}{validation_html}{test_html}{_comparison(baseline, final)}
<section id="limitations"><h2>Interpretation and source confounding</h2><p>Test class supports can be very small; inspect the per-class patient counts. Recording observations within a patient are correlated. A high class AUC on this holdout does not establish clinical generalization.</p><p>s00 test results were already known before s01 design. Comparisons on this reused holdout are exploratory and are not a fresh blinded confirmatory test. Validation and CV results used for model selection are not independent test estimates.</p><p>Channel transforms do not establish domain invariance. Cross-device or cross-site generalization requires independent devices or sites with diagnosis overlap. Sample rate, duration and encoding are available technical metadata; diagnosis must never be treated as a measured device identity.</p></section>
{warning_html}<section><h2>Local aggregate artifacts</h2><ul>{links}</ul><p class="small">This report does not read patient-row prediction files, audio paths or patient identifiers. Only aggregate values and plots are displayed.</p></section><footer>s01 aggregate report · local figures only · no model inference during report generation</footer></main></body></html>\n'''
    _atomic(report, document)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)
    print(write_report(args.output_dir, args.report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
