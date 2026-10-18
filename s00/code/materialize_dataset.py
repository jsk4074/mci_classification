#!/usr/bin/env python3
"""Copy an approved frozen speech manifest into a model-ready directory.

The source WAV files are preserved byte-for-byte. Audio decoding, resampling,
spectrograms, crops, and padding remain in resnet18_baseline.py at runtime.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from speech_data import (
    LABELS, Recording, canonical_task_key, load_manifest, manifest_summary,
    validate_cross_patient_content,
)


PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_FILES = ("split_manifest.csv", "source_manifest.csv", "copy_index.csv", "copy_summary.json")


@dataclass(frozen=True)
class CopyPlan:
    record: Recording
    target: Path
    relative: str
    task: str
    size: int
    digest: str
    fingerprint: tuple[int, ...]


def _fingerprint(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(path: Path) -> tuple[str, tuple[int, ...]]:
    before = _fingerprint(path)
    digest = _sha256(path)
    after = _fingerprint(path)
    if before != after:
        raise ValueError("An audio file changed while being hashed; finish all downloads before copying.")
    return digest, after


def _check_destination_path(path: Path, *, directory=False) -> None:
    """Reject symlinks and non-directory parents, including broken links."""

    for component in reversed(path.parents):
        if component.is_symlink():
            raise ValueError("Destination contains a symlink directory; use a regular directory for independent copies.")
        if component.exists() and not component.is_dir():
            raise ValueError("A destination parent is not a directory.")
    if path.is_symlink():
        raise ValueError("Destination is a symlink; use independent regular files and directories.")
    if path.exists() and (not path.is_dir() if directory else not path.is_file()):
        raise ValueError("Destination has an incompatible existing file or directory.")


def _safe_patient_id(patient_id: str) -> None:
    if patient_id in {".", ".."} or "/" in patient_id or "\\" in patient_id or "\x00" in patient_id:
        raise ValueError("Patient IDs must be safe directory components: reject '.', '..', slashes, backslashes, and NUL characters.")


def _verify_existing(plan: CopyPlan) -> None:
    _check_destination_path(plan.target)
    if not plan.target.exists():
        return
    source_stat, target_stat = Path(plan.record.path).stat(), plan.target.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (target_stat.st_dev, target_stat.st_ino):
        raise ValueError("An existing destination is a hardlink to its source; independent copies are required.")
    digest, _ = _stable_hash(plan.target)
    if digest != plan.digest:
        raise ValueError("An existing destination file differs from its source; it will not be overwritten. Choose a new destination or resolve the differing copy.")


def _preflight(records: list[Recording], destination: Path) -> list[CopyPlan]:
    """Validate the entire copy map and every existing target before writing."""

    _check_destination_path(destination, directory=True)
    for name in METADATA_FILES:
        _check_destination_path(destination / name)
    plans: list[CopyPlan] = []
    targets: set[Path] = set()
    for record in sorted(records, key=lambda r: (r.split or "", LABELS.index(r.label), r.patient_id, r.path)):
        _safe_patient_id(record.patient_id)
        if record.split is None:
            raise ValueError("The source manifest must contain frozen train/val/test assignments on every row; prepare the baseline splits first.")
        task = canonical_task_key(Path(record.path).name, label=record.label)
        if task is None:
            raise ValueError("A source recording has no canonical 06/07 sentence task. Use an approved manifest containing model-appropriate sentence recordings.")
        source = Path(record.path)
        if source.is_relative_to(destination):
            raise ValueError("A source recording is already inside the destination; choose a separate destination to preserve the original dataset.")
        relative = Path(record.split) / record.label / record.patient_id / f"{task}.wav"
        target = destination / relative
        if target in targets:
            raise ValueError("Multiple source recordings map to one patient/task destination; select one canonical recording per patient/task before copying.")
        targets.add(target)
        _check_destination_path(target)
        digest, fingerprint = _stable_hash(source)
        plan = CopyPlan(record, target, relative.as_posix(), task, fingerprint[2], digest, fingerprint)
        _verify_existing(plan)
        plans.append(plan)
    return plans


def _ensure_source_unchanged(plan: CopyPlan) -> None:
    digest, fingerprint = _stable_hash(Path(plan.record.path))
    if digest != plan.digest or fingerprint != plan.fingerprint:
        raise ValueError("A source recording changed after preflight; copying stopped without publishing the changed audio.")


def _publish_copy(temporary: Path, plan: CopyPlan) -> bool:
    """Reserve a previously absent target exclusively, then replace atomically.

    A reservation prevents replacement of a file created by another process
    after preflight. It contains no audio and is removed if publication fails.
    """

    _check_destination_path(plan.target)
    try:
        descriptor = os.open(plan.target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        _verify_existing(plan)
        return False
    reservation = os.fstat(descriptor)
    os.close(descriptor)
    try:
        os.replace(temporary, plan.target)
    except BaseException:
        if not plan.target.is_symlink() and plan.target.exists():
            current = plan.target.stat()
            if (current.st_dev, current.st_ino) == (reservation.st_dev, reservation.st_ino) and current.st_size == 0:
                plan.target.unlink()
        raise
    return True


def _copy_one(plan: CopyPlan) -> bool:
    _ensure_source_unchanged(plan)
    _check_destination_path(plan.target)
    if plan.target.exists():
        _verify_existing(plan)
        _ensure_source_unchanged(plan)
        return False
    plan.target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=plan.target.parent, prefix=f".{plan.target.name}.", suffix=".copytmp", delete=False) as handle:
            temporary = Path(handle.name)
        shutil.copy2(plan.record.path, temporary)
        _ensure_source_unchanged(plan)
        copied_digest, copied_fingerprint = _stable_hash(temporary)
        if copied_digest != plan.digest or copied_fingerprint[2] != plan.size:
            raise ValueError("A copied file failed byte verification; the invalid copy was not published.")
        return _publish_copy(temporary, plan)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _csv_text(fieldnames: list[str], rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _atomic_text(path: Path, text: str) -> None:
    _check_destination_path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def materialize_dataset(manifest_path: Path, destination: Path) -> dict:
    """Make verified independent WAV copies from an approved frozen manifest.

    Existing matching copies are verified and reused. A differing file,
    unsafe identifier, canonical task collision, or destination symlink fails
    before copying starts. The original source audio is never modified.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    _check_destination_path(destination, directory=True)
    destination = destination.resolve()
    records = load_manifest(manifest_path)
    validate_cross_patient_content(records)
    plans = _preflight(records, destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for completed, plan in enumerate(plans, start=1):
        copied += _copy_one(plan)
        if completed % 500 == 0 or completed == len(plans):
            print(f"Verified copies: {completed:,}/{len(plans):,}; new copies: {copied:,}", flush=True)

    manifest_rows, source_rows, index_rows = [], [], []
    for plan in plans:
        record = plan.record
        shared = {"patient_id": record.patient_id, "label": record.label, "split": record.split}
        manifest_rows.append({"path": plan.relative, **shared})
        source_rows.append({"path": record.path, **shared})
        index_rows.append({"source_path": record.path, "destination_path": plan.relative, **shared, "task": plan.task, "bytes": plan.size, "sha256": plan.digest})
    summary = {
        **manifest_summary(records), "total_bytes": sum(plan.size for plan in plans),
        "verified_copies": len(plans), "new_copies": copied, "reused_copies": len(plans) - copied,
        "layout": "[split]/[label]/[patient_id]/[task].wav",
        "source_manifest": str(manifest_path), "destination": str(destination), "status": "complete",
    }
    fields = ["path", "patient_id", "label", "split"]
    _atomic_text(destination / "split_manifest.csv", _csv_text(fields, manifest_rows))
    _atomic_text(destination / "source_manifest.csv", _csv_text(fields, source_rows))
    _atomic_text(destination / "copy_index.csv", _csv_text(["source_path", "destination_path", "patient_id", "label", "split", "task", "bytes", "sha256"], index_rows))
    _atomic_text(destination / "copy_summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "runs/resnet18/split_manifest.csv")
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "dataset/s00")
    args = parser.parse_args(argv)
    try:
        summary = materialize_dataset(args.manifest, args.destination)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Complete: {summary['total_recordings']:,} recordings from {summary['total_patients']:,} patients; {summary['total_bytes']:,} bytes verified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
