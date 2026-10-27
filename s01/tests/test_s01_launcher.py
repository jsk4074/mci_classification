"""Exercise the published s01 registry and common launcher on synthetic WAVs."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_registered_s01_parameters_are_accepted_by_its_entrypoint():
    spec = importlib.util.spec_from_file_location("s01_common_launcher", ROOT / "run_experiment.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    entrypoint, fixed_arguments = launcher.load_experiment("s01")
    command = [sys.executable, str(entrypoint), "--help"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    for argument in fixed_arguments:
        if argument.startswith("--"):
            assert argument in result.stdout


def test_common_launcher_prepares_and_regenerates_report_without_encoder(tmp_path):
    pytest.importorskip("soundfile")
    manifest = tmp_path / "private_synthetic_manifest.csv"
    rows = []
    serial = 0
    for label in ("NM", "AD", "MCI"):
        for split, count in (("train", 3), ("val", 2), ("test", 1)):
            for index in range(count):
                serial += 1
                patient = f"synthetic_private_{label}_{split}_{index}"
                path = tmp_path / f"private_audio_{serial}.wav"
                with wave.open(str(path), "wb") as handle:
                    handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    handle.writeframes(struct.pack("<800h", *[(i * serial) % 16000 for i in range(800)]))
                rows.append({"path": path.name, "patient_id": patient, "label": label, "split": split})
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "patient_id", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "study"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "MPLCONFIGDIR": str(tmp_path / "mpl"),
           "XDG_CACHE_HOME": str(tmp_path / "cache")}
    base = [sys.executable, str(ROOT / "run_experiment.py"), "s01", "--output-dir", str(output)]
    subprocess.run([*base, "--manifest", str(manifest), "--prepare-only"],
                   cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["total_patients"] == 18
    folds = json.loads((output / "folds.json").read_text())
    assert len(folds) == 5
    assert all(len(fold["train_ids"]) == 12 and len(fold["val_ids"]) == 3 for fold in folds)
    assert not (output / "encoder_metadata.json").exists()
    assert not (output / "best.pt").exists()
    manifest.unlink()
    subprocess.run([*base, "--report-only"], cwd=ROOT, env=env,
                   text=True, capture_output=True, check=True)
    html = (output / "report.html").read_text()
    assert "synthetic_private_" not in html and "private_audio_" not in html
    assert "No final test results" in html
