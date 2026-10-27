"""Header-only technical audits must not infer or expose patient/device identities."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import wave

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from s01_audit import domain_audit


@dataclass
class Record:
    path: str
    patient_id: str
    label: str
    split: str


def wav(path, rate, channels, seconds):
    with wave.open(str(path), "wb") as handle:
        handle.setframerate(rate)
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.writeframes(b"\0\0" * channels * int(rate * seconds))


def test_header_aggregation_uses_observed_formats_and_original_splits_without_waveform_reads(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    values = [("NM", "train", 16000, 1, .2), ("NM", "test", 48000, 2, .1),
              ("AD", "train", 48000, 2, .1), ("MCI", "val", 16000, 1, .3)]
    records = []
    for index, (label, split, rate, channels, seconds) in enumerate(values):
        path = tmp_path / f"PRIVATE_ID_{index}.wav"
        wav(path, rate, channels, seconds)
        records.append(Record(str(path), f"PRIVATE_ID_{index}", label, split))
    original = {record.path: Path(record.path).read_bytes() for record in records}
    monkeypatch.setattr(sf, "read", lambda *args, **kwargs: pytest.fail("Audit must not decode waveform samples"))
    audit = domain_audit(records)
    assert audit["status"] == "complete" and audit["headers_read"] == 4
    assert audit["by_diagnosis_and_original_split"]["NM"]["test"]["audio_formats"] == [
        {"sample_rate": 48000, "channels": 2, "encoding": "PCM_16", "container": "WAV", "recordings": 1}]
    assert audit["by_diagnosis"]["NM"]["duration_seconds"]["median"] == pytest.approx(.15)
    assert audit["by_original_split"]["train"]["recordings"] == 2
    assert audit["source_device_metadata_available"] is False
    assert audit["clinical_labels_verified"] is False
    text = json.dumps(audit, allow_nan=False)
    assert "PRIVATE_ID" not in text and ".wav" not in text and str(tmp_path) not in text
    assert {record.path: Path(record.path).read_bytes() for record in records} == original


def test_header_failures_are_aggregate_only_and_do_not_leak_paths(tmp_path):
    pytest.importorskip("soundfile")
    path = tmp_path / "SENSITIVE_SOURCE_NAME.wav"
    path.write_bytes(b"invalid audio")
    audit = domain_audit([{"path": str(path), "patient_id": "SENSITIVE_PERSON", "label": "NM",
                           "split": "train", "original_split": "test"}])
    assert audit["status"] == "partial" and audit["header_errors"] == 1
    assert audit["by_original_split"]["test"]["header_errors"] == 1
    assert audit["overall"]["duration_seconds"]["min"] is None
    text = json.dumps(audit, allow_nan=False)
    assert "SENSITIVE" not in text and str(tmp_path) not in text
