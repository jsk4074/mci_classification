"""Discover speech recordings and make reproducible, patient-disjoint splits.

This module deliberately uses only the Python standard library. Preparing and
checking a manifest does not import a machine-learning library or decode audio.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import math
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


LABELS = ("NM", "AD", "MCI")
SPLITS = ("train", "val", "test")
DEFAULT_FILENAME_PATTERN = r"(?:06|07)-\d+\.wav"
RAW_MCI_FOLDER = "중기부 001-021 데이터"


@dataclass(frozen=True)
class Recording:
    """One audio recording, grouped by the original participant identifier."""

    path: str
    patient_id: str
    label: str
    split: str | None = None


def canonical_task_key(filename: str, label: str | None = None) -> str | None:
    """Identify a sentence task without copying or renaming its recording.

    Numbered 06/07 WAV names retain their task numbers. Canonical raw MCI
    names have exactly seven hyphens and no underscores: their final 61..90
    / 90 and 91..100 / 100 pairs map to tasks 06-01..30 and 07-01..10.
    Numbered prefixes may contain whitespace between the dot and ``MCI``.
    Other tasks, alternate takes, and malformed raw names return ``None``.
    """

    match = re.fullmatch(r"(06|07)-(\d+)\.wav", filename, re.IGNORECASE)
    if match:
        return f"{match[1]}-{int(match[2]):02d}"
    if label is not None and label.upper() != "MCI":
        return None
    if filename.count("-") != 7 or "_" in filename or "/" in filename or "\\" in filename:
        return None
    match = re.fullmatch(
        r"\d+\.\s*MCI-(?:[^-_]+-){5}(\d{2,3})-(90|100)\.wav",
        filename, re.IGNORECASE,
    )
    if match is None:
        return None
    sentence, endpoint = int(match[1]), match[2]
    if endpoint == "90" and 61 <= sentence <= 90:
        return f"06-{sentence - 60:02d}"
    if endpoint == "100" and 91 <= sentence <= 100:
        return f"07-{sentence - 90:02d}"
    return None


def _canonical_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Audio path does not exist or cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"Audio path must identify a regular file: {path}")
    return resolved


def load_manifest(manifest_path: Path) -> list[Recording]:
    """Read a CSV; relative audio paths are resolved beside the CSV file.

    Required columns are ``path``, ``patient_id`` and ``label``. A supplied
    ``split`` must be train, val (or validation), or test on every row.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    records: list[Recording] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "patient_id", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Manifest must contain CSV columns: path, patient_id, label "
                "(and optionally split)."
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Manifest row {line_number} has too many columns.")
            values = {key: (row.get(key) or "").strip() for key in required}
            if not all(values.values()):
                raise ValueError(
                    f"Manifest row {line_number} has a blank path, patient_id, or label."
                )
            audio_path = Path(values["path"]).expanduser()
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            split = (row.get("split") or "").strip().lower() or None
            if split == "validation":
                split = "val"
            records.append(
                Recording(
                    path=str(_canonical_path(audio_path)),
                    patient_id=values["patient_id"],
                    label=values["label"].upper(),
                    split=split,
                )
            )
    validate_recordings(records)
    return records


def _class_root_candidates(data_root: Path, label: str) -> list[Path]:
    """Support the current raw archive and simple NM/AD/MCI folder layouts."""

    bases = [data_root, data_root / "dataset"]
    candidates: list[Path] = []
    for base in bases:
        candidates.extend([base / label, base / label.lower()])
        if label in {"NM", "AD"}:
            folder = "nm_random_50" if label == "NM" else "ad_random_50"
            candidates.extend(
                [
                    base / folder,
                    base / "NM50_AD50" / folder,
                    base / "NM50_AD50_MCI20" / "NM50_AD50" / folder,
                ]
            )
        else:
            candidates.extend(
                [
                    base / "MCI20", base / "mci20", base / RAW_MCI_FOLDER,
                    base / "NM50_AD50_MCI20" / RAW_MCI_FOLDER,
                ]
            )
    if label == "MCI" and data_root.name in {"NM50_AD50_MCI20", "NM50_AD50"}:
        candidates.extend([data_root.parent / "MCI", data_root.parent / "mci"])
        if data_root.name == "NM50_AD50":
            candidates.extend(
                [
                    data_root.parent / RAW_MCI_FOLDER,
                    data_root.parent.parent / "MCI", data_root.parent.parent / "mci",
                ]
            )
    return candidates


def discover_recordings(
    data_root: Path,
    class_roots: dict[str, Path] | None = None,
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
) -> list[Recording]:
    """Find matching WAV files under class roots, grouping all clips by patient.

    The first directory below each class root is the patient identifier; keep
    that identifier unchanged so a patient present in two classes is rejected.
    Explicit roots may override any subset of classes and are resolved relative
    to the current working directory. Other classes use automatic discovery.
    The default selects numbered 06/07 sentence slices, excluding task-level
    files such as 06-SS.wav and alternative copies such as 06-01(clean).wav.
    Raw MCI names are matched through their canonical task key without
    altering source files. Duplicate patient/task recordings are rejected.
    """

    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"Data root is not a directory: {data_root}")
    try:
        matcher = re.compile(filename_pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid filename pattern: {exc}") from exc

    overrides: dict[str, Path] = {}
    for label, root in (class_roots or {}).items():
        normalized = label.upper()
        if normalized not in LABELS:
            raise ValueError(f"Unknown class root label {label!r}; use NM, AD, or MCI.")
        if normalized in overrides:
            raise ValueError(f"More than one root supplied for class {normalized}.")
        overrides[normalized] = Path(root).expanduser().resolve()

    records: list[Recording] = []
    task_paths: set[tuple[str, str, str]] = set()
    for label in LABELS:
        if label in overrides:
            root = overrides[label]
            if not root.is_dir():
                raise ValueError(f"The {label} class root is not a directory: {root}")
        else:
            root = next(
                (p for p in _class_root_candidates(data_root, label) if p.is_dir()),
                None,
            )
            if root is None:
                raise ValueError(
                    f"Cannot find a {label} class folder below {data_root}. "
                    "Wait for the dataset download to finish, supply an explicit "
                    "class root, or create a manifest with path, patient_id, label."
                )
        for audio_path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
            if not audio_path.is_file():
                continue
            task_key = canonical_task_key(audio_path.name, label=label)
            matches_raw_mci = (
                label == "MCI" and task_key is not None
                and matcher.fullmatch(f"{task_key}.wav") is not None
            )
            if not matcher.fullmatch(audio_path.name) and not matches_raw_mci:
                continue
            relative = audio_path.relative_to(root)
            if len(relative.parts) < 2:
                raise ValueError(
                    f"Matching audio found directly inside the {label} root. "
                    "Put recordings in one directory per patient or use a CSV manifest."
                )
            if task_key is not None:
                identity = (label, relative.parts[0], task_key)
                if identity in task_paths:
                    raise ValueError(
                        f"More than one {label} recording maps to the same patient/sentence task "
                        f"({task_key}). Remove duplicate inputs from discovery or use a manifest "
                        "that selects one canonical recording per patient/task."
                    )
                task_paths.add(identity)
            records.append(
                Recording(
                    path=str(_canonical_path(audio_path)),
                    patient_id=relative.parts[0],
                    label=label,
                )
            )
    validate_recordings(records)
    return records


def validate_recordings(records: Iterable[Recording]) -> None:
    """Reject missing files, duplicate audio, inconsistent labels, and leakage.

    Existing split assignments are accepted only if complete, patient-disjoint,
    and containing every class in train, val, and test. No files are changed.
    """

    records = list(records)
    if not records:
        raise ValueError("No recordings found; check data roots and the filename pattern.")
    paths: set[Path] = set()
    patients: dict[str, tuple[str, str | None]] = {}
    labels_found: set[str] = set()
    split_labels: dict[str, set[str]] = {split: set() for split in SPLITS}
    assigned = 0
    for index, recording in enumerate(records, start=1):
        if recording.label not in LABELS:
            raise ValueError(f"Recording {index} has an invalid label; use NM, AD, or MCI.")
        if not recording.patient_id or recording.patient_id != recording.patient_id.strip():
            raise ValueError(f"Recording {index} requires a nonblank patient_id without edge spaces.")
        if not recording.path or not Path(recording.path).is_absolute():
            raise ValueError(f"Recording {index} requires an absolute audio file path.")
        path = _canonical_path(Path(recording.path))
        if path in paths:
            raise ValueError(
                f"Recording {index} duplicates an audio path (including resolved symlinks). "
                "Keep each recording only once."
            )
        paths.add(path)
        if recording.split is not None:
            if recording.split not in SPLITS:
                raise ValueError(f"Recording {index} has an invalid split; use train, val, or test.")
            assigned += 1
            split_labels[recording.split].add(recording.label)
        existing = patients.get(recording.patient_id)
        if existing is not None:
            old_label, old_split = existing
            if old_label != recording.label:
                raise ValueError(
                    "A patient_id has recordings in more than one class. "
                    "Check the source labels; do not add class prefixes to patient IDs."
                )
            if old_split != recording.split:
                raise ValueError(
                    "A patient_id occurs in multiple splits or mixes assigned and "
                    "unassigned recordings. Assign every recording from that patient "
                    "to the same split."
                )
        else:
            patients[recording.patient_id] = (recording.label, recording.split)
        labels_found.add(recording.label)
    missing = set(LABELS) - labels_found
    if missing:
        raise ValueError(
            f"No matching recordings for classes: {', '.join(sorted(missing))}. "
            "Check that the download is complete and that the filename pattern matches."
        )
    if assigned and assigned != len(records):
        raise ValueError("Manifest mixes assigned and unassigned rows; provide split on every row or none.")
    if assigned:
        for split in SPLITS:
            missing = set(LABELS) - split_labels[split]
            if missing:
                raise ValueError(
                    f"The {split} split lacks classes: {', '.join(sorted(missing))}. "
                    "Each split must contain at least one patient from every class."
                )


def validate_cross_patient_content(records: Iterable[Recording]) -> None:
    """Reject byte-identical audio assigned to different patients, before splitting.

    Only files sharing a nonzero size need hashing. Empty placeholders supply
    no audio-content evidence; the training audio decoder rejects empty files.
    This check does not establish speaker identity or verify clinical labels.
    """

    records = list(records)
    validate_recordings(records)
    sizes: dict[int, list[Recording]] = defaultdict(list)
    for record in records:
        size = Path(record.path).stat().st_size
        if size:
            sizes[size].append(record)
    for candidates in sizes.values():
        if len({record.patient_id for record in candidates}) < 2:
            continue
        patients_by_digest: dict[str, str] = {}
        for record in candidates:
            digest = hashlib.sha256()
            with Path(record.path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            key = digest.hexdigest()
            if key in patients_by_digest and patients_by_digest[key] != record.patient_id:
                raise ValueError(
                    "Identical audio occurs under different patient IDs. Resolve the "
                    "participant metadata or exclude the affected folders before splitting."
                )
            patients_by_digest[key] = record.patient_id


def _split_counts(total: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    """Round requested fractions while retaining one patient in every split."""

    validation = max(1, int(math.floor(total * val_fraction + 0.5)))
    test = max(1, int(math.floor(total * test_fraction + 0.5)))
    while validation + test > total - 1:
        if validation > 1 and (validation / val_fraction >= test / test_fraction or test == 1):
            validation -= 1
        elif test > 1:
            test -= 1
        else:
            raise ValueError("At least three patients per class are needed for three splits.")
    return total - validation - test, validation, test


def assign_patient_splits(
    records: Iterable[Recording],
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> list[Recording]:
    """Stratify a unique patient table; keep every patient's clips together.

    Assignments depend on the sorted patient IDs and seed, never recording row
    order or the number of clips per patient. A complete existing assignment
    is validated and preserved even when a different seed is requested.
    """

    records = list(records)
    validate_recordings(records)
    if all(recording.split is not None for recording in records):
        return records
    if (
        not math.isfinite(val_fraction)
        or not math.isfinite(test_fraction)
        or val_fraction <= 0
        or test_fraction <= 0
        or val_fraction + test_fraction >= 1
    ):
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")
    by_class: dict[str, set[str]] = {label: set() for label in LABELS}
    for recording in records:
        by_class[recording.label].add(recording.patient_id)
    generator = random.Random(seed)
    assignments: dict[str, str] = {}
    for label in LABELS:
        patients = sorted(by_class[label])
        if len(patients) < 3:
            raise ValueError(
                f"Class {label} has {len(patients)} unique patients; at least 3 are "
                "required for patient-disjoint train/val/test splits. Wait for the "
                "dataset download to finish or check the patient IDs."
            )
        generator.shuffle(patients)
        train_count, val_count, _ = _split_counts(len(patients), val_fraction, test_fraction)
        for index, patient in enumerate(patients):
            if index < train_count:
                assignments[patient] = "train"
            elif index < train_count + val_count:
                assignments[patient] = "val"
            else:
                assignments[patient] = "test"
    assigned_records = [replace(recording, split=assignments[recording.patient_id]) for recording in records]
    validate_recordings(assigned_records)
    return assigned_records


def write_manifest(records: Iterable[Recording], path: Path) -> None:
    """Write a validated manifest with absolute paths and stable row ordering."""

    records = list(records)
    validate_recordings(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "patient_id", "label", "split"])
        writer.writeheader()
        for recording in sorted(records, key=lambda r: (LABELS.index(r.label), r.patient_id, r.path)):
            writer.writerow(
                {
                    "path": str(_canonical_path(Path(recording.path))),
                    "patient_id": recording.patient_id,
                    "label": recording.label,
                    "split": recording.split or "",
                }
            )


def manifest_summary(records: Iterable[Recording]) -> dict:
    """Return aggregate counts without exposing participant identifiers."""

    records = list(records)

    def counts(subset: list[Recording]) -> dict[str, int]:
        return {
            "recordings": len(subset),
            "patients": len({recording.patient_id for recording in subset}),
        }

    def class_counts(subset: list[Recording]) -> dict[str, dict[str, int]]:
        return {
            label: counts([recording for recording in subset if recording.label == label])
            for label in LABELS
        }

    result = {
        "total_recordings": len(records),
        "total_patients": len({recording.patient_id for recording in records}),
        "classes": class_counts(records),
        "splits": {},
    }
    for split in SPLITS:
        subset = [recording for recording in records if recording.split == split]
        result["splits"][split] = {**counts(subset), "classes": class_counts(subset)}
    return result
