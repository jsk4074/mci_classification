"""Synthetic CV-safe ResNet orchestration, never private audio or real backbones."""

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import s02_resnet as adapter


@dataclass(frozen=True)
class SyntheticRecord:
    path: str
    patient_id: str
    label: str
    split: str


def records(root, split, patients=1, clips=2):
    result = []
    for label in adapter.LABELS:
        for patient in range(patients):
            patient_id = f"synthetic_{label}_{split}_{patient}"
            for clip in range(clips):
                path = root / patient_id / f"clip-{clip}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"synthetic bytes {patient_id} {clip}".encode())
                result.append(SyntheticRecord(str(path), patient_id, label, split))
    return result


@pytest.fixture
def components(monkeypatch):
    baseline = adapter._baseline()
    trace = {"datasets": [], "read_paths": [], "build_pretrained": [], "reports": []}

    class SyntheticDataset:
        def __init__(self, rows, config, training, cache_dir=None):
            self.rows, self.training = rows, training
            trace["datasets"].append({"paths": [row.path for row in rows],
                                      "training": training, "cache_dir": cache_dir,
                                      "frames": config.frames})

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            trace["read_paths"].append(row.path)
            target = adapter.LABELS.index(row.label)
            values = torch.zeros(1, 2, 2)
            values.flatten()[target] = 1 + (index % 2) / 10
            if not self.training:
                values = torch.stack([values, values * 1.2])
            return values, target, index

    def backbone(pretrained=False):
        trace["build_pretrained"].append(pretrained)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3))
        with torch.no_grad():
            model[1].weight.zero_()
            model[1].weight[:, :3].copy_(torch.eye(3))
            model[1].bias.zero_()
        return model

    monkeypatch.setattr(baseline, "AudioDataset", SyntheticDataset)
    monkeypatch.setattr(baseline, "build_model", backbone)
    args = SimpleNamespace(device="cpu", epochs=2, patience=2, batch_size=4,
                           workers=0, torch_threads=1, seed=42,
                           feature_cache_dir=None,
                           report_callback=lambda directory, **state: trace["reports"].append((directory, state)))
    return baseline, args, trace


def test_fit_selects_validation_checkpoint_and_never_reads_test(tmp_path, components, monkeypatch):
    baseline, args, trace = components
    train = records(tmp_path / "audio", "train", patients=2)
    validation = records(tmp_path / "audio", "val")
    original_test = records(tmp_path / "audio", "test")
    args.feature_cache_dir = tmp_path / "shared_features"
    sampling, losses, scheduling = [], [], []
    real_sampler = torch.utils.data.WeightedRandomSampler
    real_loss = torch.nn.CrossEntropyLoss
    real_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    def sampler(weights, **kwargs):
        sampling.append((weights, kwargs))
        return real_sampler(weights, **kwargs)

    def criterion(*values, **kwargs):
        loss = real_loss(*values, **kwargs)
        losses.append(loss.weight)
        return loss

    def scheduler(optimizer, **kwargs):
        scheduling.append(kwargs)
        return real_scheduler(optimizer, **kwargs)

    monkeypatch.setattr(torch.utils.data, "WeightedRandomSampler", sampler)
    monkeypatch.setattr(torch.nn, "CrossEntropyLoss", criterion)
    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", scheduler)
    output = tmp_path / "fold"
    _, result = adapter.fit_resnet(train, validation, args, output,
                                   {"run_name": "synthetic ResNet fold"}, seed=42)
    assert result["original_test_evaluated"] is False and "test" not in result
    assert result["best_epoch"] in {1, 2}
    assert result["validation"]["patient"]["count"] == 3
    assert all("auc" in value for value in result["validation"]["patient"]["per_class"].values())
    forbidden = {row.path for row in original_test}
    assert forbidden.isdisjoint(trace["read_paths"])
    assert all(forbidden.isdisjoint(dataset["paths"]) for dataset in trace["datasets"])
    assert all(dataset["cache_dir"] == args.feature_cache_dir for dataset in trace["datasets"])
    assert all(dataset["frames"] == 501 for dataset in trace["datasets"])
    assert trace["build_pretrained"] == [False]
    assert sampling[0][0] == baseline.patient_sampling_weights(adapter._records(train))
    assert sampling[0][1]["num_samples"] == len(train) and sampling[0][1]["replacement"] is True
    assert losses == [None]
    assert scheduling == [{"mode": "max", "factor": 0.5, "patience": 3}]
    assert trace["reports"][-1][1]["status"] == "completed"
    assert all("test" not in path.name for path in output.iterdir())
    for filename in ("best.pt", "history.csv", "config.json", "dataset_snapshot.json", "metrics.json",
                     "validation_patient_predictions.csv", "validation_recording_predictions.csv"):
        assert (output / filename).is_file()
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == result["best_epoch"]
    config = json.loads((output / "config.json").read_text())
    assert config["parameters"]["learning_rate"] == .001
    assert config["parameters"]["weight_decay"] == .0001
    assert config["loss"] == "plain unweighted cross entropy"
    with (output / "history.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    with pytest.raises(ValueError, match="already contains"):
        adapter.fit_resnet(train, validation, args, output, {}, seed=42)


def test_test_records_cannot_enter_either_fit_role(tmp_path, components):
    _, args, trace = components
    train, validation = records(tmp_path, "train"), records(tmp_path, "val")
    held_out = records(tmp_path, "test")
    for train_rows, val_rows in ((train, held_out), ([*train, held_out[0]], validation)):
        with pytest.raises(ValueError, match="test recordings are forbidden"):
            adapter.fit_resnet(train_rows, val_rows, args, tmp_path / "rejected", {}, seed=42)
    assert not trace["datasets"] and not trace["read_paths"]
    assert not (tmp_path / "rejected").exists()


def test_evaluation_preserves_s00_window_patient_averaging_and_input_alignment(tmp_path, components):
    baseline, args, trace = components
    rows = list(reversed(records(tmp_path, "val")))
    model = baseline.build_model()
    new = adapter.evaluate_records(model, rows, args, cache_dir=tmp_path / "cache")
    dataset = baseline.AudioDataset(adapter._records(rows), baseline.FeatureConfig(), training=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=baseline.collate_evaluation)
    old = baseline.evaluate(model, loader, torch.device("cpu"), adapter._records(rows))
    assert len(new) == 5
    torch.testing.assert_close(torch.as_tensor(new[1]), torch.as_tensor(old[1]))
    torch.testing.assert_close(torch.as_tensor(new[4]), torch.as_tensor(old[4]))
    assert new[2] == old[2]
    assert list(new[3]) == list(old[3])
    assert new[1].argmax(axis=1).tolist() == [adapter.LABELS.index(row.label) for row in rows]
    assert all(new[0]["patient"]["per_class"][label]["auc"] == 1 for label in adapter.LABELS)
    adapter.write_predictions(tmp_path, "validation", rows, new)
    with (tmp_path / "validation_recording_predictions.csv").open(newline="") as handle:
        assert [row["path"] for row in csv.DictReader(handle)] == [row.path for row in rows]


def test_missing_evaluation_classes_have_explicit_null_auc(tmp_path, components):
    baseline, args, _ = components
    rows = [row for row in records(tmp_path, "val") if row.label == "NM"]
    result = adapter.evaluate_records(baseline.build_model(), rows, args)
    assert result[0]["patient"]["macro_ovr_auc"] is None
    assert result[0]["patient"]["per_class"]["MCI"]["auc"] is None


def test_patient_overlap_is_rejected_before_feature_loading(tmp_path, components):
    _, args, trace = components
    train, validation = records(tmp_path, "train"), records(tmp_path, "val")
    validation[0] = replace(validation[0], patient_id=train[0].patient_id)
    with pytest.raises(ValueError, match="patients must be disjoint"):
        adapter.fit_resnet(train, validation, args, tmp_path / "overlap", {}, seed=42)
    assert not trace["datasets"]


def test_preserved_component_recipe_defaults_are_explicit():
    assert adapter.DEFAULTS["epochs"] == 50 and adapter.DEFAULTS["patience"] == 10
    assert adapter.DEFAULTS["batch_size"] == 16
    assert adapter.DEFAULTS["learning_rate"] == .001 and adapter.DEFAULTS["weight_decay"] == .0001
    assert adapter.DEFAULTS["pretrained"] is False
