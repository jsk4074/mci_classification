"""Shared folds, frozen selection, and paired final-test boundaries."""

import csv
import importlib.util
import json
from pathlib import Path
import struct
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "s02/code"))
sys.path.insert(0, str(ROOT / "s01/analysis"))
spec = importlib.util.spec_from_file_location("s02_workflow_entry", ROOT / "s02/code/train.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
from s01_metrics import classification_metrics


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def study(tmp_path, monkeypatch):
    import s02_resnet
    manifest = tmp_path / "manifest.csv"
    rows = []
    serial = 0
    for label in ("NM", "AD", "MCI"):
        for split, count in (("train", 3), ("val", 2), ("test", 1)):
            for index in range(count):
                serial += 1
                audio = tmp_path / f"synthetic_{serial}.wav"
                with wave.open(str(audio), "wb") as out:
                    out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    out.writeframes(struct.pack("<800h", *[(i * serial) % 20000 for i in range(800)]))
                rows.append({"path": str(audio), "patient_id": f"{label}_{split}_{index}", "label": label, "split": split})
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "patient_id", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "out"
    common = ["--manifest", str(manifest), "--output-dir", str(output), "--device", "cpu", "--folds", "2"]
    calls = {"fit": [], "test": [], "xlsr": []}
    monkeypatch.setattr(entry, "source_hashes", lambda: {"synthetic": "unchanged"})
    monkeypatch.setattr(entry, "ensure_runtime", lambda args: None)
    monkeypatch.setattr(entry, "refresh", lambda *a, **kw: None)

    def metrics(records):
        labels = {"NM": 0, "AD": 1, "MCI": 2}
        targets = [labels[r.label] for r in records]
        probs = [[.8 if index == target else .1 for index in range(3)] for target in targets]
        m = classification_metrics(targets, probs)
        return {"patient": m, "recording": m}

    def fit(train, val, args, directory, metadata, seed):
        assert all(r.split != "test" for r in [*train, *val])
        assert not {r.patient_id for r in train} & {r.patient_id for r in val}
        calls["fit"].append((set(r.patient_id for r in train), set(r.patient_id for r in val)))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "history.csv").write_text("epoch,train_loss,val_patient_loss,val_patient_macro_f1,val_patient_balanced_accuracy,learning_rate\n1,.3,.3,1,1,.001\n")
        (directory / "best.pt").write_bytes(b"synthetic checkpoint, not a model")
        result = {"validation": metrics(val), "best_epoch": 1, "original_test_evaluated": False}
        entry.write_json(directory / "metrics.json", result)
        return object(), result

    def evaluate(model, records, args, cache_dir):
        assert read(output / "resnet18/test_evaluation.json")["status"] == "started"
        assert all(r.split == "test" for r in records)
        assert (output / "selection.json").is_file()
        calls["test"].append({r.patient_id for r in records})
        return (metrics(records), None, None, None, None)

    monkeypatch.setattr(s02_resnet, "fit_resnet", fit)
    monkeypatch.setattr(s02_resnet, "evaluate_records", evaluate)
    monkeypatch.setattr(s02_resnet, "write_predictions", lambda *a, **kw: None)
    real_xlsr = entry.run_xlsr

    def xlsr(args, stage, folds):
        directory = output / "xlsr_mil"
        directory.mkdir(exist_ok=True)
        if stage == "cv":
            if (directory / "cv_summary.json").exists():
                return
            records = entry.load_frozen_manifest(manifest)
            variants = {}
            for variant in ("A", "B", "C"):
                saved = []
                for fold in folds:
                    selected = [r for r in records if r.patient_id in fold["val_ids"]]
                    assert all(r.split != "test" for r in selected)
                    saved.append({"fold": fold["fold"], "best_epoch": 1, **metrics(selected)})
                variants[variant] = entry.mean_std(saved)
            entry.write_json(directory / "cv_summary.json", {"variants": variants, "selected_variant": "A"})
            calls["xlsr"].append(("cv", folds))
        else:
            guard = directory / "test_evaluation.json"
            if guard.exists():
                return
            assert read(output / "selection.json")["selected_xlsr_variant"] == "A"
            entry.write_json(guard, {"status": "started"})
            calls["xlsr"].append(("test", None))
            records = entry.load_frozen_manifest(manifest)
            result = {"validation": metrics([r for r in records if r.split == "val"]),
                      "test": metrics([r for r in records if r.split == "test"]),
                      "selected_variant": "A", "best_epoch": 1, "original_test_evaluated": True}
            entry.write_json(directory / "metrics.json", result)
            entry.write_json(guard, {"status": "completed", "evaluations": 1})
        entry.collect(args)
    monkeypatch.setattr(entry, "run_xlsr", xlsr)
    return dict(output=output, common=common, calls=calls, rows=rows, real_xlsr=real_xlsr, tmp=tmp_path)


def test_both_comparisons_share_folds_and_each_test_event_runs_once(study):
    assert entry.main(study["common"]) == 0
    output = study["output"]
    folds = read(output / "folds.json")
    for actual, fold in zip(study["calls"]["fit"][:2], folds):
        assert actual == (set(fold["train_ids"]), set(fold["val_ids"]))
    assert study["calls"]["xlsr"][0] == ("cv", folds)
    test_ids = {r["patient_id"] for r in study["rows"] if r["split"] == "test"}
    assert study["calls"]["test"] == [test_ids]
    assert read(output / "resnet18/test_evaluation.json")["evaluations"] == 1
    assert read(output / "xlsr_mil/test_evaluation.json")["evaluations"] == 1
    before = {key: len(value) for key, value in study["calls"].items()}
    assert entry.main(study["common"]) == 0
    assert {key: len(value) for key, value in study["calls"].items()} == before
    assert read(output / "metrics.json")["selected_variant"] == "A"


def test_final_is_blocked_until_both_development_comparisons_finish(study):
    assert entry.main([*study["common"], "--stage", "cv", "--system", "resnet18"]) == 0
    with pytest.raises(SystemExit) as error:
        entry.main([*study["common"], "--stage", "final"])
    assert error.value.code == 2
    assert not study["calls"]["test"]
    assert not (study["output"] / "selection.json").exists()


def test_changed_source_or_data_is_rejected_before_reusing_results(study, monkeypatch):
    assert entry.main([*study["common"], "--stage", "cv"]) == 0
    with monkeypatch.context() as patch:
        patch.setattr(entry, "source_hashes", lambda: {"synthetic": "changed"})
        with pytest.raises(SystemExit):
            entry.main([*study["common"], "--stage", "final"])
    path = Path(study["rows"][0]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(SystemExit):
        entry.main([*study["common"], "--stage", "final"])
    assert not study["calls"]["test"]


def test_started_xlsr_guard_blocks_before_subprocess_inference(study, monkeypatch):
    args = entry.parser().parse_args(study["common"])
    entry.validate_args(args)
    target = study["output"] / "xlsr_mil/test_evaluation.json"
    target.parent.mkdir(parents=True)
    entry.write_json(target, {"status": "started"})
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **kw: pytest.fail("Must not run an encoder after a started test guard"))
    with pytest.raises(ValueError, match="already started"):
        study["real_xlsr"](args, "final", [])


def test_registry_entry_accepts_all_committed_parameters():
    spec = importlib.util.spec_from_file_location("s02_launcher", ROOT / "run_experiment.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    path, flags = launcher.load_experiment("s02")
    assert path == ROOT / "s02/code/train.py"
    args = entry.parser().parse_args(flags)
    entry.validate_args(args)
    assert args.stage == "all" and args.system == "both" and args.export_pdf
