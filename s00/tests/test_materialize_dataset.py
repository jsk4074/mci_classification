"""Physical dataset copies preserve an approved manifest and original WAV bytes."""

import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT / "code"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "analysis"))

from speech_data import LABELS, SPLITS, Recording, canonical_task_key, load_manifest, write_manifest


def materialize(manifest, destination):
    from materialize_dataset import materialize_dataset
    return materialize_dataset(manifest, destination)


def approved_manifest(root):
    """Nonempty, unique placeholders suffice for a byte-preserving copy tool."""
    records = []
    for split in SPLITS:
        for label in LABELS:
            patient_id = f"{label.lower()}_{split}_participant"
            names = (
                ("1.MCI-001-032-M-77-114-61-90.wav", "1.MCI-001-032-M-77-114-91-100.wav")
                if label == "MCI" else ("06-01.wav",)
            )
            for name in names:
                source = root / "source" / label / patient_id / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(f"Unique source bytes: {split}|{label}|{patient_id}|{name}".encode())
                records.append(Recording(str(source.resolve()), patient_id, label, split))
    manifest = root / "approved.csv"
    write_manifest(records, manifest)
    return manifest, records


def copied_path(destination, record):
    task = canonical_task_key(Path(record.path).name, record.label)
    return destination / record.split / record.label / record.patient_id / f"{task}.wav"


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_tree(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_copies_only_approved_recordings_into_patient_disjoint_class_partitions(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    excluded = Path(records[0].path).parent / "06-30.wav"
    excluded.write_bytes(b"A valid task deliberately absent from the approved manifest")
    destination = tmp_path / "copied_dataset"
    summary = materialize(manifest, destination)
    expected = {copied_path(destination, record).relative_to(destination).as_posix() for record in records}
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*.wav")}
    assert actual == expected
    assert len(actual) == 12
    for split in SPLITS:
        assert {path.name for path in (destination / split).iterdir() if path.is_dir()} == set(LABELS)
        assert summary["splits"][split]["patients"] == 3
        assert summary["splits"][split]["recordings"] == 4
    assert summary["total_recordings"] == 12
    assert summary["total_patients"] == 9
    assert summary["verified_copies"] == 12
    assert summary["total_bytes"] == sum(Path(record.path).stat().st_size for record in records)
    assert summary["status"] == "complete"
    assert summary["source_manifest"] == str(manifest.resolve())
    assert summary["destination"] == str(destination.resolve())
    assert json.loads((destination / "copy_summary.json").read_text()) == summary
    assert excluded.is_file()
    assert not any(path.name == "06-30.wav" for path in destination.rglob("*.wav"))


def test_raw_mci_names_are_canonical_only_in_the_copy_and_sources_remain_unchanged(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    source_snapshot = {record.path: (Path(record.path).read_bytes(), Path(record.path).stat().st_mtime_ns)
                       for record in records}
    original_manifest = manifest.read_bytes()
    destination = tmp_path / "copied_dataset"
    materialize(manifest, destination)
    for record in records:
        source, copied = Path(record.path), copied_path(destination, record)
        assert source.read_bytes() == source_snapshot[record.path][0] == copied.read_bytes()
        assert source.stat().st_mtime_ns == source_snapshot[record.path][1]
        assert not copied.is_symlink()
        assert (source.stat().st_dev, source.stat().st_ino) != (copied.stat().st_dev, copied.stat().st_ino)
        if record.label == "MCI":
            assert source.name.startswith("1.MCI-")
            assert copied.name in {"06-01.wav", "07-01.wav"}
    assert manifest.read_bytes() == original_manifest


def test_relative_copy_manifest_roundtrips_and_remains_portable_after_directory_move(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    destination = tmp_path / "copied_dataset"
    materialize(manifest, destination)
    relative_rows = rows(destination / "split_manifest.csv")
    assert len(relative_rows) == len(records)
    assert all(not Path(row["path"]).is_absolute() for row in relative_rows)
    assert {record.path for record in load_manifest(destination / "split_manifest.csv")} == {
        str(copied_path(destination, record).resolve()) for record in records
    }
    moved = tmp_path / "moved_dataset"
    destination.rename(moved)
    relocated = load_manifest(moved / "split_manifest.csv")
    assert {record.path for record in relocated} == {
        str(copied_path(moved, record).resolve()) for record in records
    }
    assert {(record.patient_id, record.label, record.split) for record in relocated} == {
        (record.patient_id, record.label, record.split) for record in records
    }


def test_copy_index_hashes_match_both_files_and_source_manifest_preserves_provenance(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    destination = tmp_path / "copied_dataset"
    materialize(manifest, destination)
    index = rows(destination / "copy_index.csv")
    assert len(index) == len(records)
    by_source = {record.path: record for record in records}
    for row in index:
        record = by_source[row["source_path"]]
        copied = destination / row["destination_path"]
        assert not Path(row["destination_path"]).is_absolute()
        assert copied == copied_path(destination, record)
        assert row["sha256"] == digest(Path(record.path)) == digest(copied)
        assert int(row["bytes"]) == copied.stat().st_size
        assert (row["patient_id"], row["label"], row["split"]) == (
            record.patient_id, record.label, record.split,
        )
        assert row["task"] == canonical_task_key(Path(record.path).name, record.label)
    provenance = load_manifest(destination / "source_manifest.csv")
    assert {(record.path, record.patient_id, record.label, record.split) for record in provenance} == {
        (record.path, record.patient_id, record.label, record.split) for record in records
    }


def test_retry_reuses_verified_copies_without_altering_bytes_inodes_or_counts(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    destination = tmp_path / "copied_dataset"
    first = materialize(manifest, destination)
    original_files = file_tree(destination)
    stats = {path: (path.stat().st_ino, path.stat().st_mtime_ns)
             for path in (copied_path(destination, record) for record in records)}
    second = materialize(manifest, destination)
    counters = {"new_copies", "reused_copies"}
    assert {key: value for key, value in second.items() if key not in counters} == {
        key: value for key, value in first.items() if key not in counters
    }
    assert first["new_copies"] == second["reused_copies"] == len(records)
    assert second["new_copies"] == first["reused_copies"] == 0
    assert {path: contents for path, contents in file_tree(destination).items()
            if path != "copy_summary.json"} == {
        path: contents for path, contents in original_files.items() if path != "copy_summary.json"
    }
    assert {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in stats} == stats


def test_existing_different_copy_is_refused_before_any_other_recording_is_written(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    destination = tmp_path / "copied_dataset"
    conflicting = copied_path(destination, records[-1])
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"An existing file that must never be overwritten")
    before = file_tree(destination)
    with pytest.raises(ValueError):
        materialize(manifest, destination)
    assert file_tree(destination) == before
    assert conflicting.read_bytes() == b"An existing file that must never be overwritten"


@pytest.mark.parametrize("patient_id", [".", "..", "../escape", "nested/person", "nested\\person", "/absolute"])
def test_unsafe_patient_component_is_rejected_before_destination_creation(tmp_path, patient_id):
    manifest, records = approved_manifest(tmp_path)
    records[0] = replace(records[0], patient_id=patient_id)
    write_manifest(records, manifest)
    destination = tmp_path / "copied_dataset"
    with pytest.raises(ValueError):
        materialize(manifest, destination)
    assert not destination.exists()


def test_duplicate_canonical_patient_task_mapping_is_rejected_before_copying(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    alternate = Path(records[0].path).with_name("06-1.wav")
    alternate.write_bytes(b"Different bytes but the same patient and canonical task")
    records.append(replace(records[0], path=str(alternate.resolve())))
    write_manifest(records, manifest)
    destination = tmp_path / "copied_dataset"
    with pytest.raises(ValueError):
        materialize(manifest, destination)
    assert not destination.exists()


def test_recording_without_a_canonical_task_is_rejected_before_copying(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    unknown = Path(records[0].path).with_name("unknown_task.wav")
    unknown.write_bytes(b"This file has no approved 06 or 07 task mapping")
    records[0] = replace(records[0], path=str(unknown.resolve()))
    write_manifest(records, manifest)
    destination = tmp_path / "copied_dataset"
    with pytest.raises(ValueError):
        materialize(manifest, destination)
    assert not destination.exists()


def test_unassigned_manifest_is_not_split_implicitly_by_the_copy_tool(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    write_manifest([replace(record, split=None) for record in records], manifest)
    destination = tmp_path / "copied_dataset"
    with pytest.raises(ValueError):
        materialize(manifest, destination)
    assert not destination.exists()


def test_resolved_source_symlink_is_preserved_and_destination_is_an_independent_file(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    target = Path(records[0].path)
    alias = tmp_path / "source_alias" / target.name
    alias.parent.mkdir()
    alias.symlink_to(target)
    records[0] = replace(records[0], path=str(alias))
    raw_rows = rows(manifest)
    for row in raw_rows:
        if row["path"] == str(target):
            row["path"] = str(alias)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "patient_id", "label", "split"])
        writer.writeheader()
        writer.writerows(raw_rows)
    destination = tmp_path / "copied_dataset"
    materialize(manifest, destination)
    copied = copied_path(destination, records[0])
    assert alias.is_symlink() and alias.resolve() == target
    assert not copied.is_symlink()
    assert copied.read_bytes() == target.read_bytes()
    assert copied.stat().st_ino != target.stat().st_ino


def test_existing_destination_file_symlink_is_rejected_even_when_content_matches(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    destination = tmp_path / "copied_dataset"
    link = copied_path(destination, records[-1])
    link.parent.mkdir(parents=True)
    source = Path(records[-1].path)
    before = source.read_bytes()
    link.symlink_to(source)
    with pytest.raises(ValueError):
        materialize(manifest, destination)
    assert link.is_symlink() and source.read_bytes() == before
    assert not (destination / "copy_index.csv").exists()
    assert len(list(destination.rglob("*.wav"))) == 1


def test_destination_parent_directory_symlink_is_rejected_without_writing_through_it(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    destination = tmp_path / "copied_dataset"
    outside = tmp_path / "outside_destination"
    outside.mkdir()
    patient_directory = copied_path(destination, records[-1]).parent
    patient_directory.parent.mkdir(parents=True)
    patient_directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        materialize(manifest, destination)
    assert patient_directory.is_symlink()
    assert not list(outside.iterdir())
    assert not list(destination.rglob("*.wav"))


def test_destination_root_symlink_is_rejected_without_writing_to_its_target(tmp_path):
    manifest, _ = approved_manifest(tmp_path)
    outside = tmp_path / "outside_destination"
    outside.mkdir()
    destination = tmp_path / "copied_dataset"
    destination.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        materialize(manifest, destination)
    assert destination.is_symlink()
    assert not list(outside.iterdir())


def test_existing_destination_hardlink_is_rejected_instead_of_counted_as_an_independent_copy(tmp_path):
    manifest, records = approved_manifest(tmp_path)
    destination = tmp_path / "copied_dataset"
    linked = copied_path(destination, records[-1])
    linked.parent.mkdir(parents=True)
    source = Path(records[-1].path)
    source_bytes = source.read_bytes()
    os.link(source, linked)
    with pytest.raises(ValueError):
        materialize(manifest, destination)
    assert linked.stat().st_ino == source.stat().st_ino
    assert linked.read_bytes() == source_bytes
    assert not (destination / "copy_index.csv").exists()
