"""Exact copied recordings must not masquerade as independent patients."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "code"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "analysis"))
from speech_data import Recording, validate_cross_patient_content


def records_at(root):
    records = []
    for label in ("NM", "AD", "MCI"):
        for index in range(2):
            path = root / f"{label}-{index}.wav"
            path.write_bytes(f"different audio {label}-{index}".encode())
            records.append(Recording(str(path), f"{label}-{index}", label))
    return records


def test_same_size_different_content_is_allowed(tmp_path):
    validate_cross_patient_content(records_at(tmp_path))


def test_identical_content_across_patients_is_rejected(tmp_path):
    records = records_at(tmp_path)
    Path(records[-1].path).write_bytes(Path(records[-2].path).read_bytes())
    with pytest.raises(ValueError, match="Identical audio occurs under different patient IDs"):
        validate_cross_patient_content(records)


def test_same_patient_repeated_content_cannot_cross_a_patient_split(tmp_path):
    records = records_at(tmp_path)
    Path(records[-1].path).write_bytes(Path(records[-2].path).read_bytes())
    records[-1] = replace(records[-1], patient_id=records[-2].patient_id)
    validate_cross_patient_content(records)


def test_content_guard_runs_in_prepare_mode_before_a_manifest_is_saved(tmp_path):
    from resnet18_baseline import main
    from speech_data import write_manifest

    records = records_at(tmp_path)
    Path(records[-1].path).write_bytes(Path(records[-2].path).read_bytes())
    manifest = tmp_path / "input.csv"
    write_manifest(records, manifest)
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        main(["--prepare-only", "--manifest", str(manifest), "--output-dir", str(output)])
    assert error.value.code == 2
    assert not (output / "split_manifest.csv").exists()
