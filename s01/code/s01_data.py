"""Frozen patient manifests and development folds for s01; standard library only.

The original test patients never enter development folds. Fingerprints contain
private paths and patient IDs and must stay in ignored local run directories.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable


LABELS = ("NM", "AD", "MCI")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Record:
    path: str
    patient_id: str
    label: str
    split: str


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_file(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("An audio path is missing or cannot be resolved.") from exc
    if not resolved.is_file():
        raise ValueError("Every audio path must identify a regular file.")
    return resolved


def group_patients(records: Iterable[Record]) -> list[dict]:
    """Return sorted patient bags with patient_id, label, target, split, records.

    ``records`` in each bag is a list[Record], sorted by absolute file path.
    This function also accepts a subset containing only one original split.
    """

    groups = defaultdict(list)
    for record in records:
        if record.label not in LABELS or record.split not in SPLITS:
            raise ValueError("Records require known labels and frozen train/val/test splits.")
        if not record.patient_id or record.patient_id != record.patient_id.strip():
            raise ValueError("Every record requires a nonblank patient_id without edge spaces.")
        groups[record.patient_id].append(record)
    bags = []
    for patient_id in sorted(groups):
        members = sorted(groups[patient_id], key=lambda record: record.path)
        if len({record.label for record in members}) != 1:
            raise ValueError("A patient has recordings in multiple classes.")
        if len({record.split for record in members}) != 1:
            raise ValueError("A patient has recordings in multiple frozen splits.")
        bags.append({
            "patient_id": patient_id, "label": members[0].label,
            "target": LABELS.index(members[0].label), "split": members[0].split,
            "records": members,
        })
    return bags


def load_frozen_manifest(path: Path, quarantine_path: Path | None = None) -> list[Record]:
    """Read and validate explicit split assignments without changing membership.

    Relative audio paths resolve beside the CSV. Duplicate resolved paths and
    identical audio bytes under different patient IDs are rejected. An optional
    quarantine CSV must have a patient_id column; included quarantined IDs cause
    an error, rather than an implicit change to the frozen cohort.
    """

    manifest_path = Path(path).expanduser().resolve()
    required = {"path", "patient_id", "label", "split"}
    records = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if (reader.fieldnames is None or not required.issubset(reader.fieldnames)
                or len(reader.fieldnames) != len(set(reader.fieldnames))):
            raise ValueError("Manifest requires unique path, patient_id, label, split columns.")
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Manifest row {line} has too many columns.")
            values = {key: (row.get(key) or "").strip() for key in required}
            if not all(values.values()):
                raise ValueError(f"Manifest row {line} requires all four fields, including split.")
            audio_path = Path(values["path"]).expanduser()
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            records.append(Record(
                str(_resolved_file(audio_path)), values["patient_id"],
                values["label"].upper(), values["split"].lower(),
            ))
    if not records:
        raise ValueError("The frozen manifest contains no recordings.")
    bags = group_patients(records)
    for split in SPLITS:
        if {bag["label"] for bag in bags if bag["split"] == split} != set(LABELS):
            raise ValueError(f"Every original split must contain NM, AD, MCI; check {split}.")
    paths = set()
    content_owners = {}
    for record in records:
        if record.path in paths:
            raise ValueError("Duplicate audio paths, including resolved symlinks, are forbidden.")
        paths.add(record.path)
        digest = _file_hash(Path(record.path))
        previous = content_owners.setdefault(digest, record.patient_id)
        if previous != record.patient_id:
            raise ValueError("Identical audio bytes occur under different patient IDs.")
    if quarantine_path is not None:
        with Path(quarantine_path).expanduser().open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "patient_id" not in reader.fieldnames:
                raise ValueError("Quarantine CSV requires a patient_id column.")
            quarantined = {(row.get("patient_id") or "").strip() for row in reader}
        if quarantined.intersection(bag["patient_id"] for bag in bags):
            raise ValueError("Quarantined patient IDs are included in the frozen manifest.")
    return records


def fingerprint(records: Iterable[Record]) -> dict:
    """Hash sorted private file metadata and SHA-256 content hashes, not row order."""

    rows = []
    for record in sorted(records, key=lambda record: (record.patient_id, record.path)):
        path = _resolved_file(Path(record.path))
        before = path.stat()
        digest = _file_hash(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("An audio file changed while its fingerprint was computed.")
        rows.append({**asdict(record), "path": str(path), "bytes": after.st_size,
                     "mtime_ns": after.st_mtime_ns, "content_sha256": digest})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": rows}


def summary(records: Iterable[Record]) -> dict:
    """Return aggregate class/split counts without patient identifiers or paths."""

    records = list(records)

    def counts(subset):
        return {"patients": len({record.patient_id for record in subset}),
                "recordings": len(subset)}

    def class_counts(subset):
        return {label: counts([record for record in subset if record.label == label])
                for label in LABELS}

    return {
        "total_patients": len({record.patient_id for record in records}),
        "total_recordings": len(records), "classes": class_counts(records),
        "splits": {split: {**counts(subset), "classes": class_counts(subset)}
                   for split in SPLITS
                   for subset in [[record for record in records if record.split == split]]},
    }


def make_development_folds(records: Iterable[Record], n_splits: int = 5,
                           seed: int = 42) -> list[dict]:
    """Stratify development patients, excluding the complete original test split.

    Within each class, sorted patient IDs are seeded/shuffled and allocated in
    round-robin order. A rotating offset balances overall fold sizes as well as
    each class. Every development patient validates exactly once.
    """

    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two.")
    development = [bag for bag in group_patients(records) if bag["split"] != "test"]
    by_class = {label: sorted(bag["patient_id"] for bag in development
                             if bag["label"] == label) for label in LABELS}
    if any(len(patients) < n_splits for patients in by_class.values()):
        raise ValueError("Every development class needs at least n_splits patients.")
    generator = random.Random(seed)
    validation = [[] for _ in range(n_splits)]
    offset = 0
    for label in LABELS:
        patients = by_class[label]
        generator.shuffle(patients)
        for index, patient_id in enumerate(patients):
            validation[(offset + index) % n_splits].append(patient_id)
        offset = (offset + len(patients)) % n_splits
    all_ids = {bag["patient_id"] for bag in development}
    return [{"fold": index + 1, "train_ids": sorted(all_ids - set(patient_ids)),
             "val_ids": sorted(patient_ids)} for index, patient_ids in enumerate(validation)]
