"""Regression checks for patient isolation and the speech training pipeline.

Run from the project root, keeping fixtures inside the workspace:
  python -m pytest -q --basetemp=runs/test_tmp
Runtime checks skip when their optional training dependencies are unavailable.
"""

from collections import defaultdict
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import wave

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT / "code"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "analysis"))

import resnet18_baseline as baseline
from speech_data import (
    LABELS,
    Recording,
    assign_patient_splits,
    discover_recordings,
    load_manifest,
    validate_recordings,
    write_manifest,
)


def fake_recordings(root, patients=None):
    """Create distinct placeholder files; manifest validation must not decode them."""
    patients = patients or {label: [1, 1, 1] for label in LABELS}
    records = []
    for label, clip_counts in patients.items():
        for patient_index, clip_count in enumerate(clip_counts):
            patient_id = f"{label.lower()}_participant_{patient_index}"
            for clip_index in range(clip_count):
                path = root / label / patient_id / f"06-{clip_index + 1:02d}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                records.append(Recording(str(path.resolve()), patient_id, label))
    return records


def assignments(records):
    return {record.patient_id: record.split for record in records}


def write_csv(path, rows, columns=("path", "patient_id", "label", "split")):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_tone(path, seconds=0.5, frequency=220, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        int(12000 * math.sin(2 * math.pi * (frequency * time + 40 * time * time)))
        for time in (index / sample_rate for index in range(int(seconds * sample_rate)))
    ]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def require_runtime(*names):
    return [pytest.importorskip(name) for name in names]


def test_patient_splits_are_disjoint_and_independent_of_recording_counts(tmp_path):
    records = fake_recordings(tmp_path, {
        "NM": [1, 9, 2, 1, 3, 1, 5, 1],
        "AD": [3, 1, 7, 1, 2, 1, 1],
        "MCI": [1, 6, 2, 1, 1],
    })
    split_records = assign_patient_splits(records, seed=19)
    patient_splits = defaultdict(set)
    for record in split_records:
        patient_splits[record.patient_id].add(record.split)
    assert all(len(splits) == 1 for splits in patient_splits.values())
    assert {record.path for record in split_records} == {record.path for record in records}
    for label in LABELS:
        assert {record.split for record in split_records if record.label == label} == {
            "train", "val", "test",
        }

    reordered = records.copy()
    random.Random(91).shuffle(reordered)
    assert assignments(assign_patient_splits(reordered, seed=19)) == assignments(split_records)

    extra_path = tmp_path / "another_clip.wav"
    extra_path.touch()
    extra_record = replace(records[0], path=str(extra_path.resolve()))
    assert assignments(assign_patient_splits([*records, extra_record], seed=19)) == assignments(split_records)


def test_frozen_manifest_is_preserved_even_with_a_different_seed(tmp_path):
    records = assign_patient_splits(fake_recordings(tmp_path), seed=7)
    path = tmp_path / "frozen.csv"
    write_manifest(records, path)
    original = path.read_bytes()
    loaded = load_manifest(path)
    assert assignments(assign_patient_splits(loaded, seed=999)) == assignments(records)
    write_manifest(list(reversed(loaded)), path)
    assert path.read_bytes() == original


def test_prepare_reuses_saved_manifest_without_rediscovering_data(tmp_path):
    records = assign_patient_splits(fake_recordings(tmp_path / "audio"), seed=7)
    output = tmp_path / "run"
    manifest = output / "split_manifest.csv"
    write_manifest(records, manifest)
    original = manifest.read_bytes()
    assert baseline.main([
        "--prepare-only", "--output-dir", str(output),
        "--data-root", str(tmp_path / "missing_download_folder"), "--seed", "999",
    ]) == 0
    assert manifest.read_bytes() == original
    summary = json.loads((output / "split_summary.json").read_text())
    assert summary["total_patients"] == 9
    assert all(summary["splits"][split]["patients"] == 3 for split in ("train", "val", "test"))


def test_data_root_manifest_preserves_splits_for_another_output_directory(tmp_path):
    dataset = tmp_path / "copied_dataset"
    records = assign_patient_splits(fake_recordings(dataset / "audio"), seed=7)
    write_manifest(records, dataset / "split_manifest.csv")
    output = tmp_path / "new_run"
    assert baseline.main([
        "--prepare-only", "--data-root", str(dataset),
        "--output-dir", str(output), "--seed", "999",
    ]) == 0
    saved = load_manifest(output / "split_manifest.csv")
    assert assignments(saved) == assignments(records)
    assert {r.path for r in saved} == {r.path for r in records}


def test_default_run_uses_the_materialized_s00_manifest(tmp_path, monkeypatch):
    copied = tmp_path / "dataset/s00"
    records = assign_patient_splits(fake_recordings(copied / "audio"), seed=7)
    write_manifest(records, copied / "split_manifest.csv")
    monkeypatch.setattr(baseline, "PROJECT_ROOT", tmp_path)
    assert baseline.main(["--prepare-only"]) == 0
    prepared = load_manifest(tmp_path / "runs/s00/split_manifest.csv")
    assert assignments(prepared) == assignments(records)
    assert all(Path(r.path).is_relative_to(copied) for r in prepared)


def test_manifest_resolves_relative_paths_and_normalizes_labels_and_validation(tmp_path):
    records = fake_recordings(tmp_path)
    rows = []
    for index, record in enumerate(records):
        rows.append({
            "path": str(Path(record.path).relative_to(tmp_path)),
            "patient_id": record.patient_id,
            "label": record.label.lower(),
            "split": ("train", "validation", "test")[index % 3],
        })
    manifest = tmp_path / "input.csv"
    write_csv(manifest, rows)
    loaded = load_manifest(manifest)
    assert {record.path for record in loaded} == {record.path for record in records}
    assert {record.label for record in loaded} == set(LABELS)
    assert {record.split for record in loaded} == {"train", "val", "test"}


def test_duplicate_resolved_audio_paths_are_rejected(tmp_path):
    records = fake_recordings(tmp_path)
    alias = tmp_path / "alias.wav"
    alias.symlink_to(records[0].path)
    duplicate = Recording(str(alias), "another_participant", "NM")
    with pytest.raises(ValueError, match="duplicates an audio path"):
        validate_recordings([*records, duplicate])


def test_same_patient_in_different_classes_is_rejected(tmp_path):
    records = fake_recordings(tmp_path)
    ad_index = next(index for index, record in enumerate(records) if record.label == "AD")
    records[ad_index] = replace(records[ad_index], patient_id=records[0].patient_id)
    with pytest.raises(ValueError, match="more than one class"):
        validate_recordings(records)


def test_manually_overlapping_patient_split_is_rejected(tmp_path):
    records = assign_patient_splits(fake_recordings(tmp_path))
    path = tmp_path / "duplicate_patient_clip.wav"
    path.touch()
    extra = replace(records[0], path=str(path.resolve()),
                    split="test" if records[0].split == "train" else "train")
    with pytest.raises(ValueError, match="multiple splits"):
        validate_recordings([*records, extra])


def test_partial_manual_splits_are_rejected(tmp_path):
    records = assign_patient_splits(fake_recordings(tmp_path))
    records[-1] = replace(records[-1], split=None)
    with pytest.raises(ValueError, match="mixes assigned and unassigned"):
        validate_recordings(records)


def test_manual_split_without_a_class_is_rejected(tmp_path):
    records = assign_patient_splits(fake_recordings(tmp_path))
    records = [record for record in records if not (record.label == "MCI" and record.split == "test")]
    with pytest.raises(ValueError, match="test split lacks classes: MCI"):
        validate_recordings(records)


def test_absent_class_is_rejected(tmp_path):
    records = fake_recordings(tmp_path, {"NM": [1, 1, 1], "AD": [1, 1, 1]})
    with pytest.raises(ValueError, match="No matching recordings for classes: MCI"):
        assign_patient_splits(records)


def test_too_few_unique_patients_is_rejected_even_with_many_recordings(tmp_path):
    records = fake_recordings(tmp_path, {"NM": [1, 1, 1], "AD": [1, 1, 1], "MCI": [9, 12]})
    with pytest.raises(ValueError, match="Class MCI has 2 unique patients"):
        assign_patient_splits(records)


def test_discovery_includes_only_canonical_numbered_task_slices(tmp_path):
    allowed = ("06-01.wav", "07-20.WAV")
    excluded = (
        "06-SS.wav", "07-PS.wav", "06-01_1.wav", "07-20(clean).wav",
        "06-01_1(clean).wav", "05-01.wav", "06_01.wav", "prefix06-01.wav",
        "06-01.wav.bak",
    )
    for label in LABELS:
        folder = tmp_path / label / f"{label.lower()}_participant" / "SLICE"
        folder.mkdir(parents=True)
        for name in (*allowed, *excluded):
            (folder / name).touch()
    records = discover_recordings(tmp_path)
    assert len(records) == 6
    assert {Path(record.path).name for record in records} == set(allowed)
    assert {record.patient_id for record in records} == {f"{label.lower()}_participant" for label in LABELS}


def test_discovery_does_not_hide_identity_conflicts_with_class_prefixes(tmp_path):
    for label in LABELS:
        path = tmp_path / label / "same_participant" / "06-01.wav"
        path.parent.mkdir(parents=True)
        path.touch()
    with pytest.raises(ValueError, match="more than one class"):
        discover_recordings(tmp_path)


@pytest.mark.parametrize("total,window,limit,expected", [
    (40, 100, 5, [0]),
    (100, 100, 5, [0]),
    (101, 100, 5, [0, 1]),
    (600, 100, 3, [0, 250, 500]),
    (600, 100, 1, [250]),
])
def test_evaluation_windows_cover_the_recording_deterministically(total, window, limit, expected):
    assert baseline.evaluation_starts(total, window, limit) == expected


@pytest.mark.parametrize("arguments", [(0, 100, 5), (100, 0, 5), (100, 100, 0)])
def test_evaluation_windows_reject_nonpositive_sizes(arguments):
    with pytest.raises(ValueError, match="must be positive"):
        baseline.evaluation_starts(*arguments)


def test_sampling_gives_equal_class_mass_and_equal_patient_mass(tmp_path):
    records = fake_recordings(tmp_path, {"NM": [1, 9, 2], "AD": [3, 1], "MCI": [7]})
    weights = baseline.patient_sampling_weights(records)
    class_mass = defaultdict(float)
    patient_mass = defaultdict(float)
    for record, weight in zip(records, weights):
        class_mass[record.label] += weight
        patient_mass[record.patient_id] += weight
    assert all(mass == pytest.approx(1) for mass in class_mass.values())
    for label in LABELS:
        patients = {record.patient_id for record in records if record.label == label}
        assert all(patient_mass[patient] == pytest.approx(1 / len(patients)) for patient in patients)


def test_patient_aggregation_averages_recordings_equally():
    np, = require_runtime("numpy")
    records = [
        Recording("a.wav", "nm_person", "NM"),
        Recording("b.wav", "nm_person", "NM"),
        Recording("c.wav", "ad_person", "AD"),
    ]
    ids, targets, probabilities = baseline.aggregate_patients(records, np.array([
        [0.9, 0.05, 0.05], [0.1, 0.45, 0.45], [0.1, 0.8, 0.1],
    ]))
    assert ids == ["ad_person", "nm_person"]
    assert targets.tolist() == [1, 0]
    np.testing.assert_allclose(probabilities[1], [0.5, 0.25, 0.25])
    with pytest.raises(ValueError, match="rows must match"):
        baseline.aggregate_patients(records, probabilities)


def test_evaluation_averages_windows_before_recordings():
    np, torch, _ = require_runtime("numpy", "torch", "sklearn")
    records = [
        Recording("a.wav", "nm_person", "NM"),
        Recording("b.wav", "nm_person", "NM"),
        Recording("c.wav", "ad_person", "AD"),
        Recording("d.wav", "mci_person", "MCI"),
    ]
    window_probabilities = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45],
        [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
    ])
    loader = [(torch.tensor(np.log(window_probabilities)),
               torch.tensor([0, 1, 1, 1, 2, 3]), (0, 0, 1, 2), (0, 1, 2, 3))]
    result = baseline.evaluate(torch.nn.Identity(), loader, torch.device("cpu"), records)
    patient_ids, patient_probabilities = result[2], result[4]
    np.testing.assert_allclose(patient_probabilities[patient_ids.index("nm_person")], [0.5, 0.25, 0.25])
    assert result[0]["patient"]["macro_f1"] == pytest.approx(1)


def test_audio_features_padding_eval_windows_and_cache(tmp_path, monkeypatch):
    np, torch, librosa = require_runtime("numpy", "torch", "librosa")
    paths = [tmp_path / "short.wav", tmp_path / "long.wav"]
    write_tone(paths[0], seconds=0.2)
    write_tone(paths[1], seconds=1.5)
    records = [Recording(str(path), f"person_{index}", "NM") for index, path in enumerate(paths)]
    config = baseline.FeatureConfig(clip_seconds=0.5, eval_windows=3)
    cache = tmp_path / "features"
    training = baseline.AudioDataset(records, config, training=True, cache_dir=cache)
    short, label, index = training[0]
    assert short.shape == (1, 40, config.frames)
    assert (label, index) == (0, 0)
    assert short.dtype == torch.float32
    assert bool(torch.isfinite(short).all())
    assert short.min().item() >= -1 and short.max().item() <= 1
    assert bool((short[:, :, -20:] == -1).all())
    torch.manual_seed(19)
    long, _, _ = training[1]
    torch.manual_seed(19)
    assert torch.equal(long, training[1][0])
    assert len(list(cache.glob("*.npy"))) == 2

    def fail_to_decode(*args, **kwargs):
        raise AssertionError("Cached features should avoid audio decoding")

    monkeypatch.setattr(librosa, "load", fail_to_decode)
    evaluation = baseline.AudioDataset(records, config, training=False, cache_dir=cache)
    assert evaluation[0][0].shape == (1, 1, 40, config.frames)
    assert evaluation[1][0].shape == (3, 1, 40, config.frames)
    assert torch.equal(evaluation[1][0], evaluation[1][0])
    cached = [np.load(path, allow_pickle=False) for path in cache.glob("*.npy")]
    assert all(array.shape[0] == 40 and np.isfinite(array).all() for array in cached)


def test_changed_audio_invalidates_feature_cache(tmp_path):
    _, _, _ = require_runtime("numpy", "torch", "librosa")
    path = tmp_path / "audio.wav"
    write_tone(path, seconds=0.2)
    cache = tmp_path / "features"
    dataset = baseline.AudioDataset([Recording(str(path), "person", "NM")],
                                    baseline.FeatureConfig(clip_seconds=0.5), True, cache)
    dataset[0]
    write_tone(path, seconds=0.4, frequency=450)
    dataset[0]
    assert len(list(cache.glob("*.npy"))) == 2


def test_resnet18_accepts_one_audio_channel_and_outputs_three_logits():
    torch, _ = require_runtime("torch", "torchvision")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = baseline.build_model().eval()
        assert model.conv1.in_channels == 1
        assert model.fc.out_features == 3
        with torch.inference_mode():
            output = model(torch.zeros(2, 1, 40, 51))
        assert output.shape == (2, 3)
        assert bool(torch.isfinite(output).all())
    finally:
        torch.set_num_threads(old_threads)


def test_one_epoch_cpu_training_writes_complete_reviewable_outputs(tmp_path):
    require_runtime("numpy", "torch", "torchvision", "librosa", "sklearn")
    records = []
    for label_index, label in enumerate(LABELS):
        for patient_index in range(3):
            patient = f"{label.lower()}_{patient_index}"
            path = tmp_path / "audio" / patient / "06-01.wav"
            write_tone(path, seconds=0.5, frequency=180 + label_index * 150 + patient_index * 7)
            records.append(Recording(str(path), patient, label))
    manifest = tmp_path / "input.csv"
    write_manifest(records, manifest)
    output = tmp_path / "run"
    command = [
        sys.executable, str(PROJECT_ROOT / "resnet18_baseline.py"),
        "--manifest", str(manifest), "--output-dir", str(output),
        "--epochs", "1", "--batch-size", "3", "--workers", "0",
        "--device", "cpu", "--torch-threads", "1", "--clip-seconds", "0.5",
    ]
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = str(tmp_path / "numba")
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    required = {
        "split_manifest.csv", "split_summary.json", "config.json", "dataset_snapshot.json",
        "history.csv", "best.pt", "metrics.json", "validation_patient_predictions.csv",
        "validation_recording_predictions.csv", "test_patient_predictions.csv",
        "test_recording_predictions.csv", "report.html", "report_state.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["best_epoch"] == 1
    report_state = json.loads((output / "report_state.json").read_text())
    assert report_state["status"] == "completed"
    assert report_state["best_epoch"] == 1
    assert (output / "report.html").read_text().lstrip().lower().startswith("<!doctype html>")
    for split in ("validation", "test"):
        assert metrics[split]["patient"]["count"] == 3
        assert [metrics[split]["patient"]["per_class"][label]["support"] for label in LABELS] == [1, 1, 1]
        assert len(metrics[split]["patient"]["confusion_matrix"]) == 3
    with (output / "history.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    assert len(load_manifest(output / "split_manifest.csv")) == 9

    rerun = subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                           capture_output=True, text=True, timeout=30)
    assert rerun.returncode == 2
    assert "already contains a training run" in rerun.stderr
