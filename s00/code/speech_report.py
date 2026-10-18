"""Render an offline HTML method and training report without ML dependencies."""

from __future__ import annotations

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
STATUS_LABELS = {
    "awaiting_data": "Awaiting data",
    "prepared": "Splits prepared",
    "training": "Training in progress",
    "completed": "Training completed",
    "failed": "Run failed",
}
DEFAULT_FEATURES = {
    "sample_rate": 16000, "n_mels": 40, "n_fft": 400, "hop_length": 160,
    "clip_seconds": 5.0, "top_db": 80.0, "eval_windows": 5,
}


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _number(value) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _formatted(value, *, percent=False, integer=False) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if integer:
        return f"{number:,.0f}"
    return f"{number:.1%}" if percent else f"{number:.4f}"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_json(path: Path, warnings: list[str]) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, UnicodeError, ValueError):
        warnings.append(f"Could not read {path.name}; the file may be incomplete or invalid.")
        return {}


def _load_history(path: Path, warnings: list[str]) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    fields = (
        "epoch", "train_loss", "val_patient_loss", "val_patient_macro_f1",
        "val_patient_balanced_accuracy", "learning_rate",
    )
    rows = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                values = {key: _number(row.get(key)) for key in fields}
                if values["epoch"] is not None:
                    rows.append({key: value for key, value in values.items() if value is not None})
    except (OSError, UnicodeError, csv.Error):
        warnings.append("Could not read history.csv; the file may be incomplete or invalid.")
    return rows


def _safe_message(message) -> str:
    """Remove raw absolute file paths from error messages shown on the page."""

    text = str(message)
    return re.sub(r"(?:(?<!\S)/|[A-Za-z]:[\\/])[^\n]*", "[local path omitted]", text)


def _feature_values(features: dict) -> dict:
    result = dict(DEFAULT_FEATURES)
    for key in result:
        value = _number(features.get(key))
        if value is not None and value > 0:
            result[key] = value
    for key in ("sample_rate", "n_mels", "n_fft", "hop_length", "eval_windows"):
        result[key] = max(1, int(result[key]))
    result["frames"] = int(round(result["clip_seconds"] * result["sample_rate"] / result["hop_length"])) + 1
    return result


def _window_diagram(features: dict) -> str:
    duration = f'{features["clip_seconds"]:g}'
    maximum = str(features["eval_windows"])
    shape = f'1 × {features["n_mels"]} × {features["frames"]}'
    return f'''<figure class="diagram">
<svg viewBox="0 0 940 325" role="img" aria-labelledby="window-title window-description">
<title id="window-title">How variable-length recordings become fixed model inputs</title>
<desc id="window-description">Short recordings are right-padded at the log-mel floor. Training selects one random window from longer recordings. Evaluation averages evenly spaced windows up to the configured maximum.</desc>
<defs><pattern id="pad-pattern" patternUnits="userSpaceOnUse" width="8" height="8"><path d="M0 8L8 0" stroke="#c1cbd2" stroke-width="1"/></pattern></defs>
<text x="24" y="35" class="svg-label">Short clips</text>
<rect x="270" y="13" width="265" height="44" rx="6" fill="#e8eff0"/>
<rect x="270" y="13" width="115" height="44" rx="6" fill="#168576"/>
<rect x="385" y="13" width="150" height="44" fill="url(#pad-pattern)"/>
<text x="285" y="41" class="svg-on-color">audio</text><text x="403" y="41" class="svg-small">pad at −1</text>
<text x="270" y="82" class="svg-small">Right-pad to {_escape(duration)} seconds · {_escape(shape)}</text>
<text x="24" y="137" class="svg-label">Long clips during training</text>
<rect x="270" y="112" width="620" height="44" rx="6" fill="#e8eff0"/>
<rect x="462" y="109" width="118" height="50" rx="6" fill="#168576"/>
<text x="476" y="140" class="svg-on-color">random crop</text>
<text x="270" y="183" class="svg-small">One {_escape(duration)}-second window per sampled recording; the position can change each epoch</text>
<text x="24" y="239" class="svg-label">Validation and test</text>
<rect x="270" y="214" width="620" height="44" rx="6" fill="#e8eff0"/>
<g fill="#427bb2"><rect x="270" y="211" width="82" height="50" rx="6"/><rect x="404" y="211" width="82" height="50" rx="6"/><rect x="539" y="211" width="82" height="50" rx="6"/><rect x="673" y="211" width="82" height="50" rx="6"/><rect x="808" y="211" width="82" height="50" rx="6"/></g>
<text x="270" y="286" class="svg-small">Evenly spaced deterministic windows · maximum {_escape(maximum)} · mean class probabilities</text>
</svg><figcaption>Schematic only; the five blue windows illustrate the default cap. Actual window counts depend on clip length and configuration. Very long clips can have gaps between sampled windows.</figcaption></figure>'''


def _method_section(features: dict, config: dict) -> str:
    duration = f'{features["clip_seconds"]:g}'
    shape = f'1 × {features["n_mels"]} × {features["frames"]}'
    default_shape_note = (
        ' With the default 5-second configuration this is <strong>1 × 40 × 501</strong>.'
        if all(features[key] == value for key, value in DEFAULT_FEATURES.items()
               if key != "eval_windows")
        else ""
    )
    arguments = config.get("arguments", {})
    arguments = arguments if isinstance(arguments, dict) else {}
    val_fraction = _number(arguments.get("val_fraction"))
    test_fraction = _number(arguments.get("test_fraction"))
    val_fraction = 0.15 if val_fraction is None else val_fraction
    test_fraction = 0.15 if test_fraction is None else test_fraction
    ratios = f'{1 - val_fraction - test_fraction:.0%} / {val_fraction:.0%} / {test_fraction:.0%}'
    initialization = "ImageNet initialization (explicitly enabled)" if arguments.get("pretrained") is True else "Training from scratch (default)"
    return f'''<section id="method"><div class="section-heading"><h2>Method and clip-size handling</h2><span class="eyebrow">NM / AD / MCI</span></div>
<p>Every model input has the same <strong>{_escape(shape)}</strong> shape: one channel, {_escape(features["n_mels"])} mel bands, and {_escape(features["frames"])} time frames.{default_shape_note} The centered STFT contributes the extra endpoint frame. Recordings can have different lengths; window selection and padding make batching consistent.</p>
{_window_diagram(features)}
<div class="method-grid"><article><h3>Audio → log-mel</h3><p>Decode to mono and resample to {_escape(features["sample_rate"]):s} Hz (16 kHz by default). Compute a power mel spectrogram with {_escape(features["n_fft"])}-sample FFT (25 ms at the default rate), {_escape(features["hop_length"])}-sample hop (10 ms), and {_escape(features["n_mels"])} mel bands.</p><p>Use each recording’s own peak as the dB reference; clip to [−{_escape(f'{features["top_db"]:g}')}, 0] dB and linearly scale to [−1, 1]. The default range is [−80, 0] dB. This per-recording normalization fits no shared statistics on validation or test recordings.</p></article>
<article><h3>Fixed windows, preserved timing</h3><p>Short clips are padded on the right with <code>−1</code>, the log-mel floor. Padding does not inject white noise, repeat speech, or time-stretch a recording. Training samples one random {_escape(duration)}-second crop from a longer recording.</p><p>Validation and test use up to {_escape(features["eval_windows"])} evenly spaced windows. Their probabilities are averaged into one recording prediction, then recordings are averaged equally into one patient prediction. The cap limits evaluation cost; it does not guarantee complete coverage of very long clips.</p></article>
<article><h3>ResNet18</h3><p>A one-channel 7 × 7, stride-2 input convolution feeds the standard ResNet18 residual stages. Adaptive average pooling produces a fixed-size representation and the final layer predicts three classes. Adaptive pooling supports the feature dimensions; fixed windows keep batch shapes consistent.</p><p>{_escape(initialization)}. A clip’s original duration is not used as a classification target.</p></article>
<article><h3>Patient-disjoint validation</h3><p>Split unique patients approximately <strong>{_escape(ratios)}</strong> into train / validation / test, stratified by NM / AD / MCI. All recordings and windows from one patient stay together. The saved manifest freezes assignments.</p><p>Training balances classes and patients within each class before sampling recordings. Validation <strong>patient macro F1</strong> selects the checkpoint; ties use lower patient loss. Test is evaluated only after final checkpoint selection. Small patient holdouts warrant repeated patient-level cross-validation before stronger conclusions.</p></article></div></section>'''


def _split_table(summary: dict) -> str:
    splits = summary.get("splits", {})
    if not isinstance(splits, dict) or not splits:
        return '<section id="splits"><h2>Patient splits</h2><p class="empty">No validated split manifest yet. Counts will appear after preparation.</p></section>'
    rows = []
    for split, label in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
        values = splits.get(split, {})
        values = values if isinstance(values, dict) else {}
        classes = values.get("classes", {})
        classes = classes if isinstance(classes, dict) else {}
        cells = []
        for class_label in LABELS:
            class_values = classes.get(class_label, {})
            class_values = class_values if isinstance(class_values, dict) else {}
            cells.append(f'<td>{_formatted(class_values.get("patients"), integer=True)} / {_formatted(class_values.get("recordings"), integer=True)}</td>')
        rows.append(f'<tr><th scope="row">{label}</th><td>{_formatted(values.get("patients"), integer=True)}</td><td>{_formatted(values.get("recordings"), integer=True)}</td>{"".join(cells)}</tr>')
    return f'''<section id="splits"><h2>Patient splits</h2><p>Counts come from the saved split manifest summary. Class columns show patients / recordings.</p><div class="table-scroll"><table><thead><tr><th>Split</th><th>Patients</th><th>Recordings</th><th>NM</th><th>AD</th><th>MCI</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>'''


def _dataset_readiness(audit: dict) -> str:
    """Describe aggregate source availability and integrity, without patient rows."""

    if not audit:
        return '<section id="dataset"><h2>Dataset readiness</h2><p class="empty">No dataset audit yet. Aggregate source counts and audio checks appear after dataset inspection.</p></section>'
    classes = audit.get("classes", {})
    classes = classes if isinstance(classes, dict) else {}
    rows, format_rows = [], []
    rate_counts = {label: {} for label in LABELS}
    for label in LABELS:
        values = classes.get(label, {})
        values = values if isinstance(values, dict) else {}
        durations = values.get("duration_seconds", {})
        durations = durations if isinstance(durations, dict) else {}
        duration_range = " / ".join(_formatted(durations.get(key)) for key in ("min", "median", "max"))
        count_keys = (
            "source_patient_folders", "included_patients", "included_recordings",
            "excluded_patient_folders", "empty_patient_folders",
        )
        count_cells = "".join(f'<td>{_formatted(values.get(key), integer=True)}</td>' for key in count_keys)
        rows.append(f'<tr><th scope="row">{label}</th>{count_cells}<td>{duration_range}</td><td>{_formatted(values.get("shorter_than_window"), integer=True)}</td><td>{_formatted(values.get("longer_than_eval_span"), integer=True)}</td></tr>')
        formats = values.get("audio_formats", [])
        formats = formats if isinstance(formats, list) else []
        for audio_format in formats:
            if not isinstance(audio_format, dict):
                continue
            rate = _number(audio_format.get("sample_rate"))
            count = _number(audio_format.get("recordings"))
            if rate is not None and count is not None and rate > 0 and count >= 0:
                rate_counts[label][rate] = rate_counts[label].get(rate, 0) + count
            subtype = _escape(_safe_message(audio_format.get("subtype", "—")))
            format_rows.append(f'<tr><th scope="row">{label}</th><td>{_formatted(rate, integer=True)}</td><td>{_formatted(audio_format.get("channels"), integer=True)}</td><td>{subtype}</td><td>{_formatted(count, integer=True)}</td></tr>')
    format_body = (
        '<div class="table-scroll"><table><thead><tr><th>Class</th><th>Source sample rate (Hz)</th><th>Channels</th><th>Encoding</th><th>Clips</th></tr></thead><tbody>' + "".join(format_rows) + '</tbody></table></div>'
        if format_rows else '<p class="empty">Source audio formats have not been recorded in the audit.</p>'
    )
    def mostly(label, rate):
        counts = rate_counts[label]
        return counts.get(rate, 0) > sum(counts.values()) / 2
    acquisition = (
        'Source acquisition conditions differ: NM/AD recordings are mostly 96 kHz, while MCI recordings are mostly 16 kHz. '
        if mostly("NM", 96000) and mostly("AD", 96000) and mostly("MCI", 16000)
        else 'The source formats above describe acquisition conditions for each class. '
    )
    acquisition += 'Resampling standardizes model inputs but does not remove source, device, or cohort differences. Patient-disjoint evaluation can still reflect these differences alongside speech characteristics.'
    check_fields = [
        ("audio_header_errors", "Audio header errors"),
        ("complete_data_chunk_errors", "Incomplete audio data chunks"),
        ("duplicate_content_count", "SHA-256 byte duplicate copies found"),
        ("recovered_archive_recordings", "Recordings recovered from the archive"),
    ]
    check_fields.extend(
        (key, title) for key, title in (
            ("excluded_duplicate_recordings", "Duplicate clips excluded"),
            ("remaining_duplicate_content_count", "Remaining SHA-256 byte duplicate copies"),
            ("quarantined_patient_folders", "Patient folders excluded pending source review"),
            ("quarantined_recordings", "Recordings excluded with those folders"),
        ) if key in audit
    )
    checks = "".join(
        f'<li><strong>{title}:</strong> {_formatted(audit.get(key), integer=True)}</li>'
        for key, title in check_fields
    )
    excluded_duplicates = _number(audit.get("excluded_duplicate_recordings"))
    deduplication_note = (
        f'<p>Included clip counts reflect exclusion of {_formatted(excluded_duplicates, integer=True)} exact duplicate copies. The original source files are preserved.</p>'
        if excluded_duplicates is not None and excluded_duplicates > 0 else ""
    )
    audit_status = audit.get("status", "not recorded")
    has_issues = audit_status == "issues_found" or any(
        (_number(audit.get(key)) or 0) > 0 for key in
        ("audio_header_errors", "complete_data_chunk_errors", "remaining_duplicate_content_count")
    )
    issue_note = (
        '<div class="notice failed">The audio audit found unresolved issues. Review the integrity checks before training; prepared splits do not establish dataset readiness.</div>'
        if has_issues else ""
    )
    quarantine_note = (
        '<div class="notice">This report describes a prepared subset. Folders linked by '
        'identical audio with inconsistent participant metadata are excluded pending '
        'source review; both the complete input manifest and the review files are preserved.</div>'
        if (_number(audit.get("quarantined_patient_folders")) or 0) > 0 else ""
    )
    audit_badge = "failed" if has_issues else "" if audit_status == "verified" else "awaiting_data"
    notes = audit.get("notes", [])
    notes = notes if isinstance(notes, list) else []
    note_items = "".join(f'<li>{_escape(_safe_message(note))}</li>' for note in notes if isinstance(note, str))
    note_section = f'<h3>Audit notes</h3><ul>{note_items}</ul>' if note_items else ""
    window, span = _number(audit.get("window_seconds")), _number(audit.get("evaluation_span_seconds"))
    thresholds = (
        f'Length counts use the audit’s {_formatted(window)}-second window and {_formatted(span)}-second evaluation span.'
        if window is not None and span is not None and window > 0 and span > 0
        else 'Short and long clip counts use the length thresholds recorded during the audit.'
    )
    return f'''<section id="dataset"><h2>Dataset readiness</h2>
<p><strong>Audio audit status:</strong> <span class="badge {audit_badge}">{_escape(_safe_message(audit_status))}</span> · <strong>Included patients:</strong> {_formatted(audit.get("total_patients"), integer=True)} · <strong>Included clips:</strong> {_formatted(audit.get("total_recordings"), integer=True)}</p>{issue_note}{quarantine_note}
<p><strong>Selection:</strong> {_escape(_safe_message(audit.get("selection", "Not recorded in the audit.")))}</p>
<div class="table-scroll"><table aria-label="Dataset source availability"><thead><tr><th>Class</th><th>Source folders</th><th>Included patients</th><th>Included clips</th><th>Excluded folders</th><th>Empty folders</th><th>Duration min / median / max (s)</th><th>Shorter than window</th><th>Longer than eval span</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="small">Exclusions count source patient folders that contribute no selected recordings; empty-folder counts are reported separately. {_escape(thresholds)} These source counts are separate from the train, validation, and test assignments below.</p>
<h3>Source audio formats</h3>{format_body}<p>{_escape(acquisition)}</p>
<h3>Integrity checks</h3><ul class="updates">{checks}</ul>{deduplication_note}<p>These checks cover audio headers, declared audio-data completeness, and SHA-256 byte duplicates. They do not verify clinical labels; class labels come from the source metadata.</p>{note_section}</section>'''


def _dataset_copy_section(copy_summary: dict) -> str:
    """Describe the model's physical dataset using only aggregate copy metadata."""

    if not copy_summary:
        return ""
    copy_status = copy_summary.get("status")
    status_name = {
        "complete": "Complete", "copying": "Copying", "partial": "Partial",
        "failed": "Failed",
    }.get(copy_status, "Not recorded") if isinstance(copy_status, str) else "Not recorded"
    recordings = _number(copy_summary.get("total_recordings"))
    verified = _number(copy_summary.get("verified_copies"))
    complete = copy_status == "complete" and recordings is not None and recordings > 0 and verified == recordings
    total_bytes = _number(copy_summary.get("total_bytes"))
    size = (
        f'{total_bytes / (1024 ** 3):.2f} GiB · {_formatted(total_bytes, integer=True)} bytes'
        if total_bytes is not None and total_bytes >= 0 else "—"
    )
    destination = copy_summary.get("destination")
    # Display only the requested aggregate destination, never arbitrary source paths
    # or a directory name that could contain a participant identifier.
    display_root = (
        "./dataset/s00/"
        if isinstance(destination, str) and Path(destination).parts[-2:] == ("dataset", "s00")
        else "model dataset/"
    )
    splits = copy_summary.get("splits", {})
    splits = splits if isinstance(splits, dict) else {}
    tree = [display_root]
    for index, split in enumerate(("train", "val", "test")):
        last_split = index == 2
        tree.append(f'{"└──" if last_split else "├──"} {split}/')
        prefix = "    " if last_split else "│   "
        values = splits.get(split, {})
        values = values if isinstance(values, dict) else {}
        classes = values.get("classes", {})
        classes = classes if isinstance(classes, dict) else {}
        for class_index, label in enumerate(LABELS):
            last_class = class_index == len(LABELS) - 1
            class_values = classes.get(label, {})
            class_values = class_values if isinstance(class_values, dict) else {}
            patient_count = _formatted(class_values.get("patients"), integer=True)
            clip_count = _formatted(class_values.get("recordings"), integer=True)
            tree.append(f'{prefix}{"└──" if last_class else "├──"} {label}/  ({patient_count} patient folders · {clip_count} WAV files)')
            child_prefix = prefix + ("    " if last_class else "│   ")
            tree.append(f'{child_prefix}└── <patient folder>/  06-01.wav … 06-30.wav; 07-01.wav … 07-10.wav')
    preservation = (
        '<p>The model dataset contains verified copies of the selected raw WAV files. '
        'Raw WAV bytes, source sample rates, and encoding are preserved; original files remain intact. '
        'Canonical filenames normalize the task naming without changing audio content.</p>'
        '<p>The copied manifest preserves the current frozen train / validation / test patient partitions. '
        'Quarantined source folders are absent from this model dataset.</p>'
        if complete else
        '<div class="notice failed">Copy verification is not complete. The counts below do not establish '
        'that every selected recording has a verified model-dataset copy.</div>'
    )
    return f'''<section id="model-dataset"><h2>Model dataset copy</h2>
<p><strong>Copy status:</strong> {_escape(status_name)} · <strong>Patients:</strong> {_formatted(copy_summary.get("total_patients"), integer=True)} · <strong>WAV files:</strong> {_formatted(recordings, integer=True)} · <strong>Verified copies:</strong> {_formatted(verified, integer=True)}</p>
<p><strong>Copied audio size:</strong> {_escape(size)}</p>{preservation}
<p>Mono conversion and 16 kHz resampling happen when the baseline loads audio. Random training crops, deterministic evaluation windows, and short-clip log-mel padding are applied at runtime; the WAV copy is not pre-cropped, padded, or resampled.</p>
<h3>Split directory layout</h3><pre class="dataset-tree" aria-label="Model dataset directory layout">{_escape(chr(10).join(tree))}</pre>
<p class="small">This is a schematic of the copied layout; patient folder names are omitted. Filename ranges show the canonical tasks, and available tasks vary by patient. Counts are read from dataset_copy.json.</p></section>'''


def _curve(history: list[dict], series: list[tuple[str, str, str]], *, title: str, bounded=False) -> str:
    points = [(row["epoch"], row[key]) for key, _, _ in series for row in history if key in row]
    if not points:
        return f'<article class="chart"><h3>{_escape(title)}</h3><p class="empty">No values recorded yet.</p></article>'
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = (0.0, 1.0) if bounded else (0.0, max(y for _, y in points) * 1.1)
    ymax = max(ymax, 0.1)
    def coordinates(x, y):
        cx = 52 + (x - xmin) / max(xmax - xmin, 1) * 486
        cy = 215 - (y - ymin) / max(ymax - ymin, 1e-12) * 173
        return cx, cy
    content = []
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        _, cy = coordinates(xmin, value)
        content.append(f'<path d="M52 {cy:.1f}H538" stroke="#e5eaf0"/><text x="44" y="{cy + 4:.1f}" text-anchor="end" class="svg-tick">{value:.2f}</text>')
    legends = []
    for key, label, color in series:
        samples = [(row["epoch"], row[key]) for row in history if key in row]
        if not samples:
            continue
        plotted = [coordinates(x, y) for x, y in samples]
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in plotted)
        content.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for (epoch, value), (x, y) in zip(samples, plotted):
            content.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"><title>{_escape(label)} · epoch {_escape(epoch):s}: {_escape(f"{value:.4f}")}</title></circle>')
        legends.append(f'<span><i style="background:{color}"></i>{_escape(label)}</span>')
    content.extend([
        '<path d="M52 42V215H538" fill="none" stroke="#8493a2"/>',
        f'<text x="52" y="234" class="svg-tick">{xmin:g}</text>',
        f'<text x="538" y="234" text-anchor="end" class="svg-tick">{xmax:g}</text>',
        '<text x="294" y="256" text-anchor="middle" class="svg-tick">Epoch</text>',
    ])
    return f'<article class="chart"><h3>{_escape(title)}</h3><div class="legend">{"".join(legends)}</div><svg viewBox="0 0 570 265" role="img" aria-label="{_escape(title)} by epoch">{"".join(content)}</svg></article>'


def _learning_curves(history: list[dict]) -> str:
    if not history:
        return '<section id="learning"><h2>Learning curves</h2><p class="empty">No training results yet. Curves appear after the first completed epoch.</p></section>'
    loss = _curve(history, [("train_loss", "Train recording loss", "#168576"), ("val_patient_loss", "Validation patient loss", "#427bb2")], title="Training and validation loss")
    scores = _curve(history, [("val_patient_macro_f1", "Validation patient macro F1", "#168576"), ("val_patient_balanced_accuracy", "Validation patient balanced accuracy", "#b27035")], title="Validation patient scores", bounded=True)
    return f'<section id="learning"><h2>Learning curves</h2><p>Values are read from history.csv. Training loss is measured on sampled recording crops; validation loss is measured after averaging windows and recordings into patient probabilities.</p><div class="chart-grid">{loss}{scores}</div></section>'


def _metric_cards(metric: dict) -> str:
    specifications = [("macro_f1", "Macro F1", True), ("balanced_accuracy", "Balanced accuracy", True), ("accuracy", "Accuracy", True), ("loss", "Cross-entropy loss", False)]
    return '<div class="metric-grid">' + "".join(
        f'<div class="metric"><span>{title}</span><strong>{_formatted(metric.get(key), percent=percent)}</strong></div>'
        for key, title, percent in specifications
    ) + '</div>'


def _confusion_matrix(metric: dict, context: str) -> str:
    matrix = metric.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != len(LABELS) or any(not isinstance(row, list) or len(row) != len(LABELS) for row in matrix):
        return '<p class="empty">Confusion matrix not available.</p>'
    maximum = max((_number(value) or 0 for row in matrix for value in row), default=0)
    rows = []
    for label, values in zip(LABELS, matrix):
        cells = []
        for value in values:
            number = _number(value)
            alpha = 0.05 + 0.2 * max(0, number or 0) / max(1, maximum)
            cells.append(f'<td style="background:rgba(22,133,118,{alpha:.3f})">{_formatted(value, integer=True)}</td>')
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    return f'<div><h4>Confusion matrix</h4><p class="small">Rows: true class · columns: predicted class</p><table class="matrix" aria-label="{_escape(context)} confusion matrix"><thead><tr><th>True / predicted</th><th>NM</th><th>AD</th><th>MCI</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _per_class(metric: dict) -> str:
    classes = metric.get("per_class", {})
    if not isinstance(classes, dict) or not classes:
        return '<p class="empty">Per-class performance not available.</p>'
    rows = []
    for label in LABELS:
        values = classes.get(label, {})
        values = values if isinstance(values, dict) else {}
        rows.append(f'<tr><th scope="row">{label}</th><td>{_formatted(values.get("precision"), percent=True)}</td><td>{_formatted(values.get("recall"), percent=True)}</td><td>{_formatted(values.get("f1"), percent=True)}</td><td>{_formatted(values.get("support"), integer=True)}</td></tr>')
    return f'<div><h4>Per-class performance</h4><div class="table-scroll"><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>'


def _evaluation(title: str, metrics: dict, explanation: str) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return ""
    # A caller may supply only the patient metric dictionary for latest validation.
    if "patient" not in metrics and "macro_f1" in metrics:
        metrics = {"patient": metrics}
    blocks = []
    for level in ("patient", "recording"):
        values = metrics.get(level)
        if not isinstance(values, dict) or not values:
            continue
        name = f'{level.capitalize()}-level'
        support = _formatted(values.get("count"), integer=True)
        auc = _formatted(values.get("macro_ovr_auc"), percent=True)
        blocks.append(f'<article class="evaluation"><h3>{name}</h3><p class="small">Evaluated {level}s: {support} · Macro one-vs-rest AUC: {auc}</p>{_metric_cards(values)}<div class="evaluation-grid">{_per_class(values)}{_confusion_matrix(values, title + " " + name)}</div></article>')
    if not blocks:
        return ""
    return f'<section><h2>{_escape(title)}</h2><p>{_escape(explanation)}</p>{"".join(blocks)}</section>'


def _artifact_links(output_dir: Path, report_path: Path) -> str:
    artifacts = [
        ("split_manifest.csv", "Frozen split manifest"), ("split_summary.json", "Split counts"),
        ("dataset_copy.json", "Verified model dataset copy summary"),
        ("dataset_audit.json", "Dataset readiness audit"),
        ("dataset_audit_full.json", "Complete source audit before exclusions"),
        ("duplicate_audio.csv", "Duplicate audio review"),
        ("quarantined_patients.csv", "Patient folders pending source review"),
        ("full_input_manifest.csv", "Complete input manifest before exclusions"),
        ("config.json", "Run configuration"), ("history.csv", "Epoch history"),
        ("best.pt", "Selected checkpoint"), ("metrics.json", "Final metrics"),
        ("validation_patient_predictions.csv", "Validation patient predictions"),
        ("validation_recording_predictions.csv", "Validation recording predictions"),
        ("test_patient_predictions.csv", "Test patient predictions"),
        ("test_recording_predictions.csv", "Test recording predictions"),
    ]
    links = []
    for name, title in artifacts:
        artifact = output_dir / name
        if artifact.is_file():
            relative = os.path.relpath(artifact, report_path.parent).replace(os.sep, "/")
            links.append(f'<li><a href="{_escape(quote(relative, safe="/"))}">{_escape(title)}</a> <span class="small">{_escape(name)}</span></li>')
    body = f'<ul class="artifacts">{"".join(links)}</ul>' if links else '<p class="empty">Run artifacts will appear here when generated.</p>'
    return f'<section id="artifacts"><h2>Run artifacts</h2>{body}<p class="small">Links open local files. Prediction files and the manifest contain patient identifiers; this report displays only aggregate results.</p></section>'


CSS = '''
:root{color-scheme:light;--ink:#213341;--muted:#607282;--border:#dfe7ed;--accent:#168576}*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1160px;margin:auto;padding:44px 28px 64px}header{margin-bottom:26px}h1{font-size:clamp(30px,4vw,43px);line-height:1.2;letter-spacing:-.035em;margin:8px 0 15px}h2{font-size:23px;letter-spacing:-.02em;margin:0 0 12px}h3{font-size:17px;line-height:1.35;margin:0 0 10px}h4{font-size:15px;margin:0 0 6px}p{margin:0 0 14px}.subtitle{font-size:18px;color:var(--muted);max-width:820px}.eyebrow{font-size:12px;font-weight:750;text-transform:uppercase;letter-spacing:.12em;color:var(--accent)}.meta,.small,figcaption{font-size:13px;color:var(--muted)}.meta{display:flex;gap:18px;flex-wrap:wrap}.badge{display:inline-block;border-radius:99px;padding:4px 12px;background:#e0efeb;color:#116d61;font-size:13px;font-weight:700}.badge.failed{background:#fce9e8;color:#a53636}.badge.awaiting_data{background:#fff0d5;color:#92631b}section{background:#fff;border:1px solid var(--border);border-radius:14px;padding:28px;margin-bottom:22px;box-shadow:0 3px 12px rgba(31,51,65,.025)}.section-heading{display:flex;align-items:center;justify-content:space-between;gap:16px}.notice{border-left:4px solid var(--accent);padding:14px 18px;background:#f0f7f5;border-radius:6px;margin-top:20px}.notice.failed{border-color:#c6615b;background:#fff3f2}.overview,.method-grid,.chart-grid,.evaluation-grid{display:grid;gap:22px}.overview{grid-template-columns:repeat(4,1fr);margin:22px 0}.overview .metric{background:#fff;border:1px solid var(--border);border-radius:12px;padding:17px}.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:15px 0 24px}.metric{padding:12px;background:#f4f7f9;border-radius:8px}.metric span{display:block;color:var(--muted);font-size:12px}.metric strong{font-size:25px;font-weight:680;line-height:1.45}.method-grid,.chart-grid,.evaluation-grid{grid-template-columns:1fr 1fr}.method-grid article{padding:20px;background:#f5f8fa;border-radius:10px}.method-grid p{font-size:14px}.diagram{margin:22px 0 26px;border:1px solid var(--border);border-radius:10px;padding:14px;background:#fcfdfd}.diagram svg{width:100%;height:auto;display:block}.svg-label{font-size:16px;fill:#213341;font-weight:650}.svg-small{font-size:13px;fill:#607282}.svg-on-color{font-size:13px;fill:#fff;font-weight:600}.svg-tick{font-size:11px;fill:#607282}figcaption{margin:8px 8px 2px}.chart{border:1px solid var(--border);padding:18px;border-radius:10px}.chart svg{display:block;width:100%}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:var(--muted);margin:12px 0}.legend i{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:6px}.evaluation{border-top:1px solid var(--border);padding-top:20px;margin-top:20px}.evaluation-grid{align-items:start}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px 12px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child{text-align:left}thead th{background:#f4f7f9;font-size:12px;color:var(--muted);font-weight:650}tbody th{font-weight:650}.matrix th,.matrix td{border:1px solid #e5eaf0}.matrix{font-size:13px}.empty{color:var(--muted);padding:16px;background:#f5f8fa;border-radius:8px}.updates{display:grid;grid-template-columns:1fr 1fr;gap:14px 26px;padding-left:20px}.updates li{font-size:14px;padding-left:3px}.artifacts{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:10px 22px}.artifacts a{color:#16766c;text-decoration:underline;text-underline-offset:3px}.artifacts li{border-bottom:1px solid var(--border);padding:9px 0}.artifacts span{display:block}code{font:13px ui-monospace,SFMono-Regular,monospace;background:#e7eef2;padding:2px 5px;border-radius:4px}footer{font-size:12px;color:var(--muted);text-align:center;margin-top:32px}nav{display:flex;gap:20px;flex-wrap:wrap;margin-top:18px}nav a{font-size:13px;color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.warning-list{color:#986022;font-size:14px}@media(max-width:760px){main{padding:25px 14px}section{padding:20px}.method-grid,.chart-grid,.evaluation-grid,.artifacts,.updates{grid-template-columns:1fr}.overview,.metric-grid{grid-template-columns:1fr 1fr}.diagram{overflow:auto}.diagram svg{min-width:660px}.section-heading{align-items:start}.metric strong{font-size:22px}}@media print{body{background:white}main{padding:0}section{box-shadow:none;break-inside:avoid}nav{display:none}.overview{margin-top:12px}}
.dataset-tree{font:12px/1.7 ui-monospace,SFMono-Regular,monospace;overflow:auto;padding:18px;background:#f5f8fa;border:1px solid var(--border);border-radius:8px;margin:0 0 14px;white-space:pre}
'''


def write_report(
    output_dir: Path, *, status: str | None = None, features: dict | None = None,
    latest_validation: dict | None = None, best_epoch: int | None = None,
    message: str | None = None, report_path: Path | None = None,
) -> Path:
    """Atomically create an offline report from aggregate run artifacts.

    Optional progress fields are persisted in ``output_dir/report_state.json``
    so report-only regeneration preserves them. Final validation and test
    numbers come only from metrics.json. No audio, model, or patient row is read.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve() if report_path is not None else output_dir / "report.html"
    warnings: list[str] = []
    state = _load_json(output_dir / "report_state.json", warnings)
    config = _load_json(output_dir / "config.json", warnings)
    summary = _load_json(output_dir / "split_summary.json", warnings)
    audit = _load_json(output_dir / "dataset_audit.json", warnings)
    copy_summary = _load_json(output_dir / "dataset_copy.json", warnings)
    metrics = _load_json(output_dir / "metrics.json", warnings)
    history = _load_history(output_dir / "history.csv", warnings)
    for key, value in (("status", status), ("features", features), ("latest_validation", latest_validation), ("best_epoch", best_epoch), ("message", message)):
        if value is not None:
            state[key] = value
    inferred = "completed" if metrics else "training" if history else "prepared" if summary else "awaiting_data"
    current_status = state.get("status", inferred)
    if not isinstance(current_status, str) or current_status not in STATUS_LABELS:
        raise ValueError(f"Invalid report status; choose one of {', '.join(STATUS_LABELS)}.")
    state["status"] = current_status
    feature_input = dict(DEFAULT_FEATURES)
    for source in (config.get("features"), state.get("features")):
        if isinstance(source, dict):
            feature_input.update(source)
    feature_values = _feature_values(feature_input)
    state["features"] = {key: feature_values[key] for key in DEFAULT_FEATURES}
    _atomic_write(output_dir / "report_state.json", json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    final_best = metrics.get("best_epoch")
    selected_best = final_best if final_best is not None else state.get("best_epoch")
    latest_epoch = history[-1].get("epoch") if history else None
    state_messages = {
        "awaiting_data": "Waiting for a complete NM / AD / MCI dataset and a validated patient manifest. The method is documented below; no model has been evaluated.",
        "prepared": "Patient-disjoint splits are prepared. Training has not produced final test results. Finish dataset downloads before starting a run.",
        "training": "Training is running. Validation is updated after each completed epoch; the held-out test remains reserved for the selected checkpoint.",
        "completed": "Training and final evaluation have finished. Final test results below belong to the checkpoint selected using validation patients.",
        "failed": "The run stopped before completion. Any available history is shown for inspection; use the run message to resolve the failure before retrying.",
    }
    message_text = state.get("message") or state_messages[current_status]
    notice = f'<div class="notice {current_status}">{_escape(_safe_message(message_text))}</div>'
    overview = '<div class="overview">' + "".join([
        f'<div class="metric"><span>Classes</span><strong>NM / AD / MCI</strong></div>',
        f'<div class="metric"><span>Patients in manifest</span><strong>{_formatted(summary.get("total_patients"), integer=True)}</strong></div>',
        f'<div class="metric"><span>Latest completed epoch</span><strong>{_formatted(latest_epoch, integer=True)}</strong></div>',
        f'<div class="metric"><span>Selected best epoch</span><strong>{_formatted(selected_best, integer=True)}</strong></div>',
    ]) + '</div>'
    latest = state.get("latest_validation", {})
    latest = latest if isinstance(latest, dict) else {}
    if not latest and history:
        last = history[-1]
        latest = {"patient": {"macro_f1": last.get("val_patient_macro_f1"), "balanced_accuracy": last.get("val_patient_balanced_accuracy"), "loss": last.get("val_patient_loss")}}
    latest_section = _evaluation("Latest epoch validation", latest, "Progress metrics from the latest completed epoch. These are distinct from the final selected-checkpoint evaluation below.")
    validation_section = _evaluation("Selected checkpoint validation", metrics.get("validation", {}), "Final validation performance after restoring the selected checkpoint. This split guided checkpoint selection and is not an independent test.")
    test_section = _evaluation("Final held-out test", metrics.get("test", {}), "Final test performance is read only from metrics.json, after checkpoint selection. Patient-level results are the primary assessment; recording-level results are supplementary.")
    if not test_section:
        test_section = '<section><h2>Final held-out test</h2><p class="empty">No final test results yet. Test metrics appear only after final checkpoint selection and evaluation.</p></section>'
    warning_section = '<section><h2>Artifact reading notes</h2><ul class="warning-list">' + "".join(f'<li>{_escape(warning)}</li>' for warning in warnings) + '</ul></section>' if warnings else ""
    refresh = '<meta http-equiv="refresh" content="15">' if current_status in {"prepared", "training"} else ""
    refresh_note = " · Auto-refresh every 15 seconds while prepared or training" if refresh else ""
    copy_navigation = '<a href="#model-dataset">Model dataset copy</a>' if copy_summary else ""
    document = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>Speech classification · ResNet18 report</title><style>{CSS}</style></head>
<body><main><header><span class="eyebrow">Research baseline · three-class classification</span><h1>Speech classification with ResNet18</h1><p class="subtitle">A reproducible NM / AD / MCI baseline with patient-disjoint evaluation and fixed-size inputs from variable-length speech.</p><div class="meta"><span class="badge {current_status}">{STATUS_LABELS[current_status]}</span><span>Generated {_escape(timestamp)}{refresh_note}</span></div>{notice}<nav aria-label="Report sections"><a href="#dataset">Dataset readiness</a>{copy_navigation}<a href="#method">Method</a><a href="#splits">Patient splits</a><a href="#learning">Learning curves</a><a href="#artifacts">Artifacts</a></nav></header>
{overview}<section><h2>Implementation updates</h2><ul class="updates"><li><strong>Standalone three-class baseline.</strong> One ResNet18 predicts NM, AD, and MCI directly from speech log-mel inputs.</li><li><strong>Safe patient splits.</strong> Frozen manifests, duplicate checks, and patient-disjoint stratification prevent recordings from leaking between train, validation, and test.</li><li><strong>Variable-length handling.</strong> Short clips use floor padding; longer clips use random training crops and deterministic evaluation windows.</li><li><strong>Automatic reports.</strong> Preparation, epoch progress, final metrics, and failures update this offline HTML alongside machine-readable run artifacts.</li></ul></section>
{_dataset_readiness(audit)}{_dataset_copy_section(copy_summary)}{_method_section(feature_values, config)}{_split_table(summary)}{_learning_curves(history)}{latest_section}{validation_section}{test_section}{warning_section}{_artifact_links(output_dir, report_path)}
<footer>Report generated from local aggregate artifacts · no external assets or network requests · labels: NM / AD / MCI</footer></main></body></html>\n'''
    _atomic_write(report_path, document)
    return report_path
