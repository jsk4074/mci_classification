"""Synthetic CV→selection→final orchestration; never load a speech encoder."""

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import wave

import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "code"))
sys.path.insert(0, str(EXPERIMENT / "analysis"))

spec = importlib.util.spec_from_file_location("s01_workflow_entrypoint", EXPERIMENT / "code/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(root):
    records = []
    for split_index, split in enumerate(("train", "val", "test")):
        for label_index, label in enumerate(("NM", "AD", "MCI")):
            patient = f"synthetic_{split}_{label.lower()}"
            path = root / "audio" / patient / "06-01.wav"
            path.parent.mkdir(parents=True)
            frequency = 170 + 71 * split_index + 131 * label_index
            samples = [int(10000 * math.sin(2 * math.pi * frequency * index / 16000)) for index in range(800)]
            with wave.open(str(path), "wb") as handle:
                handle.setframerate(16000)
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            records.append({"path": str(path), "patient_id": patient, "label": label, "split": split})
    manifest = root / "private_synthetic.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "patient_id", "label", "split"])
        writer.writeheader()
        writer.writerows(records)
    return manifest, records


@pytest.fixture
def study(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    import numpy as np
    import s01_report
    import s01_training

    manifest, records = make_manifest(tmp_path)
    output = tmp_path / "study"
    # Other agents may be editing source concurrently; explicitly freeze mock
    # source identities, then test their integrity check separately below.
    source_identity = {"train.py": "synthetic-training-source", "analysis/s01_metrics.py": "synthetic-metrics-source"}
    monkeypatch.setattr(train, "source_hashes", lambda: dict(source_identity))

    def report_stub(directory, report_path=None, **kwargs):
        path = Path(report_path) if report_path is not None else Path(directory) / "report.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html><head><title>Synthetic workflow</title></head><body>No clinical results.</body></html>")
        return path

    monkeypatch.setattr(s01_report, "write_report", report_stub)
    calls, fitting, test_events, builds = [], [], [], []

    class FakeEmbedder:
        stats = {"kind": "synthetic_embeddings", "pretrained_model_loaded": False}

        def embed_records(self, selected, aug_index=0, progress=None):
            selected = list(selected)
            if any(record.split == "test" for record in selected):
                assert all(record.split == "test" for record in selected)
                assert read(output / "test_evaluation.json")["status"] == "started"
            calls.append({"patients": {record.patient_id for record in selected},
                          "splits": {record.split for record in selected}, "augmentation": aug_index})
            values = []
            for record in selected:
                seed = int(sha(Path(record.path))[:8], 16) + aug_index
                values.append(np.random.default_rng(seed).normal(size=6).astype(np.float32))
            if progress:
                progress(len(selected), len(selected))
            return np.stack(values)

    fake = FakeEmbedder()

    def build(args):
        builds.append(args.stage)
        identity = {"kind": "synthetic_encoder", "revision": "fixed", "hidden_size": 6}
        path = args.output_dir / "encoder_metadata.json"
        if path.exists() and read(path) != identity:
            raise ValueError("Synthetic encoder identity changed")
        s01_training.write_json(path, identity)
        return fake

    monkeypatch.setattr(train, "build_embedder", build)
    real_fit, real_evaluate = s01_training.fit_model, s01_training.evaluate

    def audited_fit(train_patients, val_patients, embeddings, augmented, args, variant, directory, seed, metadata):
        train_ids = {patient["patient_id"] for patient in train_patients}
        val_ids = {patient["patient_id"] for patient in val_patients}
        assert train_ids.isdisjoint(val_ids)
        assert all(patient["split"] != "test" for patient in [*train_patients, *val_patients])
        train_paths = {record.path for patient in train_patients for record in patient["records"]}
        val_paths = {record.path for patient in val_patients for record in patient["records"]}
        assert set(augmented) <= train_paths and not set(augmented) & val_paths
        fitting.append({"variant": variant, "train": train_ids, "val": val_ids,
                        "epochs": args.epochs, "augmented": set(augmented)})
        return real_fit(train_patients, val_patients, embeddings, augmented, args, variant, directory, seed, metadata)

    def audited_evaluate(model, patients, embeddings, device, batch_size=8):
        if any(patient["split"] == "test" for patient in patients):
            assert all(patient["split"] == "test" for patient in patients)
            assert read(output / "test_evaluation.json")["status"] == "started"
            test_events.append({patient["patient_id"] for patient in patients})
        return real_evaluate(model, patients, embeddings, device, batch_size)

    monkeypatch.setattr(s01_training, "fit_model", audited_fit)
    monkeypatch.setattr(s01_training, "evaluate", audited_evaluate)
    common = ["--manifest", str(manifest), "--output-dir", str(output), "--device", "cpu",
              "--torch-threads", "1", "--workers", "0", "--folds", "2", "--epochs", "2",
              "--patience", "2", "--batch-size", "3", "--projection-size", "4",
              "--attention-size", "4", "--dropout", "0", "--augmentation-variants", "1"]
    return {"root": tmp_path, "manifest": manifest, "records": records, "output": output,
            "common": common, "calls": calls, "fitting": fitting, "test_events": test_events,
            "builds": builds, "source_identity": source_identity}


def test_cv_then_final_respects_patient_boundaries_augmentation_and_single_test_guard(study):
    assert train.main([*study["common"], "--stage", "cv"]) == 0
    output = study["output"]
    development = {record["patient_id"] for record in study["records"] if record["split"] != "test"}
    original_test = {record["patient_id"] for record in study["records"] if record["split"] == "test"}
    assert all(not call["patients"] & original_test for call in study["calls"])
    assert study["test_events"] == []
    assert not (output / "test_evaluation.json").exists()
    assert not (output / "metrics.json").exists()
    folds = read(output / "folds.json")
    assert len(study["fitting"]) == 6
    for variant_index, variant in enumerate(("A", "B", "C")):
        for fold_index, fold in enumerate(folds):
            fitting = study["fitting"][variant_index * 2 + fold_index]
            assert fitting["variant"] == variant and fitting["epochs"] == 2
            assert fitting["train"] == set(fold["train_ids"])
            assert fitting["val"] == set(fold["val_ids"])
            assert fitting["train"] | fitting["val"] == development
    augmentations = [call for call in study["calls"] if call["augmentation"]]
    assert [call["patients"] for call in augmentations] == [set(fold["train_ids"]) for fold in folds]
    assert all(call["augmentation"] == 1 for call in augmentations)
    assert read(output / "selection.json")["selected_variant"] in ("A", "B", "C")

    assert train.main([*study["common"], "--stage", "final"]) == 0
    assert study["test_events"] == [original_test]
    assert read(output / "test_evaluation.json")["evaluations"] == 1
    assert read(output / "test_evaluation.json")["status"] == "completed"
    final_fit = study["fitting"][-1]
    assert final_fit["train"] == {record["patient_id"] for record in study["records"] if record["split"] == "train"}
    assert final_fit["val"] == {record["patient_id"] for record in study["records"] if record["split"] == "val"}
    protected = {name: sha(output / name) for name in ("selection.json", "metrics.json", "history.csv", "best.pt", "test_evaluation.json")}
    call_count, build_count = len(study["calls"]), len(study["builds"])
    with pytest.raises(SystemExit) as error:
        train.main([*study["common"], "--stage", "final"])
    assert error.value.code == 2
    assert len(study["calls"]) == call_count and len(study["builds"]) == build_count
    assert {name: sha(output / name) for name in protected} == protected


def test_selection_integrity_rejects_modified_encoder_config_source_and_data_before_final(study, monkeypatch):
    assert train.main([*study["common"], "--stage", "cv"]) == 0
    output = study["output"]
    selection_bytes = (output / "selection.json").read_bytes()
    calls = len(study["calls"])
    encoder_original = (output / "encoder_metadata.json").read_bytes()
    (output / "encoder_metadata.json").write_text('{"kind":"changed_encoder"}')
    with pytest.raises(SystemExit) as error:
        train.main([*study["common"], "--stage", "final"])
    assert error.value.code == 2 and len(study["calls"]) == calls
    (output / "encoder_metadata.json").write_bytes(encoder_original)
    with pytest.raises(SystemExit):
        train.main([*study["common"], "--stage", "final", "--learning-rate", "0.002"])
    with monkeypatch.context() as context:
        context.setattr(train, "source_hashes", lambda: {**study["source_identity"], "analysis/s01_metrics.py": "changed"})
        with pytest.raises(SystemExit):
            train.main([*study["common"], "--stage", "final"])
    source = Path(study["records"][0]["path"])
    source.write_bytes(source.read_bytes() + b"changed input content")
    with pytest.raises(SystemExit):
        train.main([*study["common"], "--stage", "final"])
    assert (output / "selection.json").read_bytes() == selection_bytes
    assert not (output / "test_evaluation.json").exists()
    assert len(study["calls"]) == calls


def test_prepare_report_and_synthetic_smoke_never_build_a_pretrained_encoder(study, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("This mode must not construct a speech encoder")
    monkeypatch.setattr(train, "build_embedder", forbidden)
    assert train.main([*study["common"], "--prepare-only"]) == 0
    assert (study["output"] / "domain_audit.json").is_file()
    assert not (study["output"] / "encoder_metadata.json").exists()
    with monkeypatch.context() as context:
        context.setattr(train, "load_frozen_manifest", forbidden)
        assert train.main(["--output-dir", str(study["root"] / "empty_report"), "--report-only"]) == 0
        assert train.main(["--output-dir", str(study["root"] / "smoke"), "--smoke-test", "--torch-threads", "1"]) == 0
    result = read(study["root"] / "smoke/smoke_check.json")
    assert result["clinical_training"] is False and result["pretrained_encoder_loaded"] is False
    assert result["finite_optimization"] is True and result["masked_padding_ignored"] is True
    assert study["calls"] == [] and study["builds"] == []


def test_cv_selection_uses_macro_f1_then_loss_then_stable_variant_order():
    def variants(scores):
        return {name: {"folds": [{"patient": {"macro_f1": score, "loss": loss,
                                               "accuracy": .5, "balanced_accuracy": .5, "macro_ovr_auc": .75}}
                                  for _ in range(2)]} for name, (score, loss) in scores.items()}
    result = train.summarize_cv(variants({"A": (.7, .4), "B": (.7, .3), "C": (.6, .1)}))
    assert result["selected_variant"] == "B"
    assert train.summarize_cv(variants({name: (.7, .3) for name in ("C", "B", "A")}))["selected_variant"] == "A"
    assert train.summarize_cv(variants({"A": (.7, .1), "B": (.7, .1), "C": (.8, 1.0)}))["selected_variant"] == "C"
