"""Read source audio headers and return aggregate technical metadata only.

No waveform, model prediction, patient ID, filename, or inferred source-device
identity is included in the result. Original split assignments are descriptive
groups, not training fold assignments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import math
from statistics import median


LABELS = ("NM", "AD", "MCI")
SPLITS = ("train", "val", "test", "unassigned")


def _field(record, name, default=None):
    return record.get(name, default) if isinstance(record, Mapping) else getattr(record, name, default)


def _aggregate(items) -> dict:
    valid = [item for item in items if item is not None]
    durations = [item["duration"] for item in valid]
    formats = Counter((item["sample_rate"], item["channels"], item["encoding"], item["container"]) for item in valid)
    return {
        "recordings": len(items), "headers_read": len(valid), "header_errors": len(items) - len(valid),
        "sample_rates": [{"sample_rate": rate, "recordings": count} for rate, count in
                         sorted(Counter(item["sample_rate"] for item in valid).items())],
        "channels": [{"channels": channels, "recordings": count} for channels, count in
                     sorted(Counter(item["channels"] for item in valid).items())],
        "encodings": [{"encoding": encoding, "recordings": count} for encoding, count in
                      sorted(Counter(item["encoding"] for item in valid).items())],
        "audio_formats": [{"sample_rate": rate, "channels": channels, "encoding": encoding,
                           "container": container, "recordings": count}
                          for (rate, channels, encoding, container), count in sorted(formats.items())],
        "duration_seconds": {"min": min(durations) if durations else None,
                             "median": median(durations) if durations else None,
                             "max": max(durations) if durations else None,
                             "total": sum(durations) if durations else None},
    }


def domain_audit(records) -> dict:
    """Aggregate soundfile headers by diagnosis and original train/val/test split.

    ``records`` may be mappings or objects with path, label and split fields.
    An explicit original_split field takes precedence over split when present.
    Header failures are counted without serializing exceptions or source paths.
    """

    import soundfile as sf

    all_items = []
    diagnoses, splits, crossed = defaultdict(list), defaultdict(list), defaultdict(list)
    for record in records:
        label = _field(record, "label")
        label = label if label in LABELS else "unknown"
        split = _field(record, "original_split", _field(record, "split"))
        split = "val" if split == "validation" else split
        split = split if split in SPLITS else "unassigned"
        item = None
        try:
            info = sf.info(_field(record, "path"))
            rate, channels, frames = int(info.samplerate), int(info.channels), int(info.frames)
            duration = frames / rate if rate > 0 else float("nan")
            if rate <= 0 or channels <= 0 or frames <= 0 or not math.isfinite(duration):
                raise ValueError("Invalid audio header metadata")
            item = {"sample_rate": rate, "channels": channels, "duration": duration,
                    "encoding": str(info.subtype), "container": str(info.format)}
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
        all_items.append(item)
        diagnoses[label].append(item)
        splits[split].append(item)
        crossed[(label, split)].append(item)
    total = _aggregate(all_items)
    return {
        "schema_version": 1, "status": "complete" if not total["header_errors"] else "partial",
        "total_recordings": total["recordings"], "headers_read": total["headers_read"],
        "header_errors": total["header_errors"], "overall": total,
        "by_diagnosis": {label: _aggregate(items) for label, items in sorted(diagnoses.items())},
        "by_original_split": {split: _aggregate(items) for split, items in sorted(splits.items())},
        "by_diagnosis_and_original_split": {
            label: {split: _aggregate(crossed[(label, split)]) for split in SPLITS if (label, split) in crossed}
            for label in sorted(diagnoses)
        },
        "source_device_metadata_available": False,
        "clinical_labels_verified": False,
        "scope": "Source audio headers only; original partitions are descriptive. No waveform or model evaluation.",
        "limitation": "Sample rate, encoding, duration and channels do not identify recording devices or sites. "
                      "Diagnosis labels are source metadata and are not verified by this audit.",
    }
