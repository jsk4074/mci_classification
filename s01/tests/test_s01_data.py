"""Synthetic manifest checks; no audio decoding or external models are used."""

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from s01_data import (
    LABELS, Record, fingerprint, group_patients, load_frozen_manifest,
    make_development_folds, summary,
)


def make_rows(root, patients_per_class_split=3, clips=2):
    rows = []
    for label in LABELS:
        for split in ("train", "val", "test"):
            for index in range(patients_per_class_split):
                patient_id = f"synthetic_{label}_{split}_{index}"
                for clip in range(clips):
                    path = root / "audio" / patient_id / f"clip-{clip}.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"unique synthetic audio: {patient_id}, {clip}".encode())
                    rows.append({"path": str(path.relative_to(root)), "patient_id": patient_id,
                                 "label": label, "split": split})
    return rows


def save_manifest(root, rows, columns=("path", "patient_id", "label", "split")):
    path = root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_resolves_relative_paths_and_preserves_frozen_assignments(tmp_path):
    rows = make_rows(tmp_path)
    loaded = load_frozen_manifest(save_manifest(tmp_path, rows))
    assert all(Path(record.path).is_absolute() for record in loaded)
    assert [(record.patient_id, record.label, record.split) for record in loaded] == [
        (row["patient_id"], row["label"], row["split"]) for row in rows]
    assert all(Path(record.path).is_file() for record in loaded)


@pytest.mark.parametrize("field,value", [
    ("split", ""), ("split", "unassigned"), ("split", "validation"),
    ("label", "unknown"), ("patient_id", " "), ("path", "missing.wav"),
])
def test_invalid_fields_are_rejected(tmp_path, field, value):
    rows = make_rows(tmp_path)
    rows[0][field] = value
    with pytest.raises(ValueError):
        load_frozen_manifest(save_manifest(tmp_path, rows))


def test_split_column_is_mandatory_and_no_split_is_assigned_implicitly(tmp_path):
    rows = make_rows(tmp_path)
    with pytest.raises(ValueError, match="split"):
        load_frozen_manifest(save_manifest(tmp_path, rows, ("path", "patient_id", "label")))


@pytest.mark.parametrize("field,value", [("split", "test"), ("label", "AD")])
def test_one_patient_cannot_have_multiple_classes_or_original_splits(tmp_path, field, value):
    rows = make_rows(tmp_path)
    rows[1][field] = value
    with pytest.raises(ValueError, match="multiple"):
        load_frozen_manifest(save_manifest(tmp_path, rows))


def test_every_original_split_requires_all_three_classes(tmp_path):
    rows = [row for row in make_rows(tmp_path)
            if not (row["label"] == "MCI" and row["split"] == "test")]
    with pytest.raises(ValueError, match="check test"):
        load_frozen_manifest(save_manifest(tmp_path, rows))


@pytest.mark.parametrize("use_symlink", [False, True])
def test_duplicate_paths_and_symlink_aliases_are_rejected(tmp_path, use_symlink):
    rows = make_rows(tmp_path)
    if use_symlink:
        alias = tmp_path / "alias.wav"
        alias.symlink_to(tmp_path / rows[0]["path"])
        rows[1]["path"] = alias.name
    else:
        rows[1]["path"] = rows[0]["path"]
    with pytest.raises(ValueError, match="Duplicate audio paths"):
        load_frozen_manifest(save_manifest(tmp_path, rows))


def test_identical_bytes_cannot_masquerade_as_a_different_patient(tmp_path):
    rows = make_rows(tmp_path)
    other = next(row for row in rows if row["patient_id"] != rows[0]["patient_id"])
    (tmp_path / other["path"]).write_bytes((tmp_path / rows[0]["path"]).read_bytes())
    with pytest.raises(ValueError, match="Identical audio bytes"):
        load_frozen_manifest(save_manifest(tmp_path, rows))


def test_same_patient_content_repetition_preserves_patient_membership(tmp_path):
    rows = make_rows(tmp_path)
    (tmp_path / rows[1]["path"]).write_bytes((tmp_path / rows[0]["path"]).read_bytes())
    assert len(load_frozen_manifest(save_manifest(tmp_path, rows))) == len(rows)


def test_quarantine_rejects_inclusion_without_silently_dropping_rows(tmp_path):
    rows = make_rows(tmp_path)
    manifest = save_manifest(tmp_path, rows)
    original = manifest.read_bytes()
    quarantine = tmp_path / "quarantine.csv"
    quarantine.write_text(f'patient_id,reason\n{rows[0]["patient_id"]},review\n')
    with pytest.raises(ValueError, match="Quarantined patient IDs"):
        load_frozen_manifest(manifest, quarantine)
    assert manifest.read_bytes() == original
    quarantine.write_text("patient_id,reason\nunrelated_synthetic_person,review\n")
    assert len(load_frozen_manifest(manifest, quarantine)) == len(rows)


def test_fingerprint_is_order_independent_and_hashes_actual_content(tmp_path):
    records = load_frozen_manifest(save_manifest(tmp_path, make_rows(tmp_path)))
    before = fingerprint(records)
    assert before == fingerprint(reversed(records))
    assert all(len(item["content_sha256"]) == 64 for item in before["files"])
    path = Path(records[0].path)
    stat = path.stat()
    path.write_bytes(b"X" * stat.st_size)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert fingerprint(records)["sha256"] != before["sha256"]


def test_summary_is_aggregate_and_grouping_is_deterministic(tmp_path):
    records = load_frozen_manifest(save_manifest(tmp_path, make_rows(tmp_path)))
    bags = group_patients(reversed(records))
    assert [bag["patient_id"] for bag in bags] == sorted({r.patient_id for r in records})
    assert all(bag["target"] == LABELS.index(bag["label"]) for bag in bags)
    assert all([record.path for record in bag["records"]] == sorted(record.path for record in bag["records"])
               for bag in bags)
    report = summary(records)
    assert report["total_patients"] == 27 and report["total_recordings"] == 54
    assert report["splits"]["test"]["classes"]["MCI"] == {"patients": 3, "recordings": 6}
    serialized = json.dumps(report)
    assert all(record.patient_id not in serialized and record.path not in serialized for record in records)


def test_development_folds_are_patient_stratified_and_never_include_original_test(tmp_path):
    records = load_frozen_manifest(save_manifest(tmp_path, make_rows(tmp_path)))
    folds = make_development_folds(records, n_splits=5, seed=42)
    assert folds == make_development_folds(reversed(records), n_splits=5, seed=42)
    assert folds != make_development_folds(records, n_splits=5, seed=43)
    development = {record.patient_id for record in records if record.split != "test"}
    original_test = {record.patient_id for record in records if record.split == "test"}
    labels = {record.patient_id: record.label for record in records}
    all_validation = []
    sizes = []
    class_sizes = {label: [] for label in LABELS}
    for index, fold in enumerate(folds, start=1):
        train, validation = set(fold["train_ids"]), set(fold["val_ids"])
        assert fold["fold"] == index
        assert fold["train_ids"] == sorted(train) and fold["val_ids"] == sorted(validation)
        assert train.isdisjoint(validation) and train | validation == development
        assert (train | validation).isdisjoint(original_test)
        assert {labels[patient] for patient in train} == set(LABELS)
        assert {labels[patient] for patient in validation} == set(LABELS)
        for label in LABELS:
            class_sizes[label].append(sum(labels[patient] == label for patient in validation))
        all_validation.extend(validation)
        sizes.append(len(validation))
    assert len(all_validation) == len(set(all_validation)) == len(development)
    assert max(sizes) - min(sizes) <= 1
    assert all(max(values) - min(values) <= 1 for values in class_sizes.values())
    extra = replace(records[0], path="/synthetic/another_recording.wav")
    assert make_development_folds([*records, extra]) == folds


@pytest.mark.parametrize("n_splits", [1, 7, True, 2.5])
def test_invalid_or_underpowered_development_folds_are_rejected(tmp_path, n_splits):
    records = load_frozen_manifest(save_manifest(tmp_path, make_rows(tmp_path)))
    with pytest.raises(ValueError):
        make_development_folds(records, n_splits=n_splits)
