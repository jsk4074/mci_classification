"""Validation-only s02 adapter around the unchanged s00 ResNet18 components.

The controller owns the approved cohort, shared folds, and final-test guard.
``fit_resnet`` never accepts original-test recordings or evaluates a test set.
``evaluate_records`` is a separate explicit inference boundary and returns the
five-item s00 prediction tuple, with common s01 metrics (including class AUCs).
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LABELS = ("NM", "AD", "MCI")
DEFAULTS = {
    "epochs": 50, "patience": 10, "batch_size": 16, "learning_rate": 0.001,
    "weight_decay": 0.0001, "workers": 0, "torch_threads": 16,
    "device": "auto", "clip_seconds": 5.0, "eval_windows": 5,
    "pretrained": False, "no_cache": False,
}


@lru_cache(maxsize=1)
def _baseline():
    """Import the preserved implementation without copying or editing its code."""
    directory = REPOSITORY_ROOT / "s00/code"
    source = directory / "resnet18_baseline.py"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("_s02_preserved_s00_resnet", source)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import the preserved s00 implementation.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))


@lru_cache(maxsize=1)
def _metrics():
    source = REPOSITORY_ROOT / "s01/analysis/s01_metrics.py"
    spec = importlib.util.spec_from_file_location("_s02_common_patient_metrics", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the common patient metrics.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classification_metrics


def _option(args, name):
    return getattr(args, name, DEFAULTS[name])


def _device(args):
    import torch
    name = str(_option(args, "device"))
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def _feature_config(args):
    seconds, windows = _option(args, "clip_seconds"), _option(args, "eval_windows")
    if not math.isfinite(seconds) or seconds < 0.1 or not isinstance(windows, int) or windows < 1:
        raise ValueError("Clip seconds must be at least 0.1 and evaluation windows positive.")
    return _baseline().FeatureConfig(clip_seconds=seconds, eval_windows=windows)


def _records(records):
    """Convert compatible s01 Record objects to the preserved s00 dataclass."""
    baseline = _baseline()
    return [baseline.Recording(str(Path(record.path).expanduser().resolve()),
                               record.patient_id, record.label, record.split)
            for record in records]


def _cache_dir(args, output_dir):
    if _option(args, "no_cache"):
        return None
    explicit = getattr(args, "feature_cache_dir", None)
    return Path(explicit).expanduser().resolve() if explicit is not None else Path(output_dir) / "features"


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def _report(args, output_dir, **state):
    callback = getattr(args, "report_callback", None)
    if callback is not None:
        callback(Path(output_dir), **state)
        return
    # Reporting is optional and never falls back to the old s00 study renderer.
    try:
        from s02_report import write_report
    except ModuleNotFoundError as exc:
        if exc.name != "s02_report":
            raise
    else:
        write_report(Path(output_dir), **state)


def evaluate_records(model, records, args, cache_dir=None):
    """Evaluate exactly the supplied rows; return metrics/probs/IDs/targets/probs.

    This function has no split discovery. The controller must acquire its final
    test-evaluation guard before supplying any approved test recordings here.
    Recording predictions align with the input row order.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    records = _records(records)
    if not records:
        raise ValueError("Evaluation needs at least one recording.")
    baseline, device = _baseline(), _device(args)
    dataset = baseline.AudioDataset(records, _feature_config(args), training=False,
                                    cache_dir=Path(cache_dir) if cache_dir is not None else None)
    loader = DataLoader(dataset, batch_size=_option(args, "batch_size"),
                        num_workers=_option(args, "workers"), pin_memory=device.type == "cuda",
                        worker_init_fn=baseline.seed_worker, collate_fn=baseline.collate_evaluation,
                        shuffle=False, generator=torch.Generator().manual_seed(getattr(args, "seed", 42) + 1))
    model.eval()
    probabilities = np.zeros((len(records), len(LABELS)), dtype=np.float64)
    seen = set()
    with torch.inference_mode():
        for inputs, owners, _, indices in loader:
            window_probs = torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy()
            owner_array = owners.numpy()
            for local_index, recording_index in enumerate(indices):
                if recording_index in seen:
                    raise ValueError("An evaluation recording was predicted twice.")
                probabilities[recording_index] = window_probs[owner_array == local_index].mean(axis=0)
                seen.add(recording_index)
    if len(seen) != len(records):
        raise ValueError("Evaluation did not cover all supplied recordings.")
    ids, patient_targets, patient_probs = baseline.aggregate_patients(records, probabilities)
    recording_targets = [LABELS.index(record.label) for record in records]
    calculate = _metrics()
    metrics = {"recording": calculate(recording_targets, probabilities),
               "patient": calculate(patient_targets, patient_probs)}
    return metrics, probabilities, ids, patient_targets, patient_probs


def write_predictions(output_dir, split, records, result):
    """Write aligned local prediction CSVs using the preserved s00 writer."""
    _baseline().write_predictions(Path(output_dir), split, _records(records), result)


def _validate_training(train_records, val_records, args, seed):
    for name in ("epochs", "patience", "batch_size"):
        value = _option(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Epochs, patience, and batch size must be positive integers.")
    if any(_option(args, name) < 0 for name in ("workers", "torch_threads")):
        raise ValueError("Workers and thread count cannot be negative.")
    for name in ("learning_rate", "weight_decay"):
        value = _option(args, name)
        if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
            raise ValueError("Optimizer settings must be finite and valid.")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between zero and 2**32 - 1.")
    if _option(args, "pretrained"):
        raise ValueError("The s02 ResNet comparison preserves s00 random initialization.")
    if not train_records or not val_records:
        raise ValueError("Training and validation recording lists must both be nonempty.")
    if any(record.split == "test" for record in [*train_records, *val_records]):
        raise ValueError("Original test recordings are forbidden in fit_resnet.")
    if any(record.split not in {"train", "val"} or record.label not in LABELS
           for record in [*train_records, *val_records]):
        raise ValueError("Fit records require known labels and original train/val assignments.")
    train_ids = {record.patient_id for record in train_records}
    if train_ids.intersection(record.patient_id for record in val_records):
        raise ValueError("Training and validation patients must be disjoint.")
    if {record.label for record in train_records} != set(LABELS):
        raise ValueError("Training must contain all three diagnosis classes.")
    paths = [record.path for record in [*train_records, *val_records]]
    if len(paths) != len(set(paths)):
        raise ValueError("Training and validation must have distinct recording paths.")
    assignments = {}
    for record in [*train_records, *val_records]:
        if not record.patient_id or record.patient_id != record.patient_id.strip():
            raise ValueError("Records require nonblank patient identifiers.")
        old = assignments.setdefault(record.patient_id, (record.label, record.split))
        if old != (record.label, record.split):
            raise ValueError("A patient has inconsistent diagnosis or original split metadata.")


def fit_resnet(train_records, val_records, args, output_dir, metadata, seed):
    """Fit the s00 ResNet recipe, select on validation, and return without test.

    Optional ``args.feature_cache_dir`` shares unchanged log-mels across folds.
    ``args.report_callback(output_dir, **state)`` overrides optional s02 reports.
    The controller remains responsible for full-cohort content/identity checks.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_records, val_records = _records(train_records), _records(val_records)
    _validate_training(train_records, val_records, args, seed)
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in ("config.json", "history.csv", "best.pt", "metrics.json")):
        raise ValueError("ResNet output already contains a run; choose a fresh directory.")
    baseline, features, device = _baseline(), _feature_config(args), _device(args)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    baseline.seed_everything(seed)
    if _option(args, "torch_threads"):
        torch.set_num_threads(_option(args, "torch_threads"))
    order = lambda record: (LABELS.index(record.label), record.patient_id, record.path)
    train_records, val_records = sorted(train_records, key=order), sorted(val_records, key=order)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _cache_dir(args, output_dir)
    train_dataset = baseline.AudioDataset(train_records, features, training=True, cache_dir=cache_dir)
    sampler = WeightedRandomSampler(baseline.patient_sampling_weights(train_records),
                                    num_samples=len(train_dataset), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_dataset, sampler=sampler, batch_size=_option(args, "batch_size"),
                        num_workers=_option(args, "workers"), pin_memory=device.type == "cuda",
                        worker_init_fn=baseline.seed_worker,
                        generator=torch.Generator().manual_seed(seed))
    model = baseline.build_model(pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_option(args, "learning_rate"),
                                  weight_decay=_option(args, "weight_decay"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    config = {"experiment": "s02", "model_family": "resnet18", "metadata": metadata,
              "parameters": {key: _option(args, key) for key in DEFAULTS}, "seed": seed,
              "features": asdict(features), "labels": {label: index for index, label in enumerate(LABELS)},
              "feature_cache_dir": str(cache_dir) if cache_dir is not None else None,
              "checkpoint_selection": "validation patient macro F1; ties use patient loss",
              "sampling": "equal classes, equal patients within class; weighted recording sampler",
              "loss": "plain unweighted cross entropy", "device": str(device),
              "preserved_source_sha256": hashlib.sha256((REPOSITORY_ROOT / "s00/code/resnet18_baseline.py").read_bytes()).hexdigest(),
              "resume_supported": False}
    snapshot = baseline.dataset_fingerprint([*train_records, *val_records])
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "dataset_snapshot.json", snapshot)
    _report(args, output_dir, status="training", message="ResNet training uses development patients only.")
    fields = ("epoch", "train_loss", "val_patient_loss", "val_patient_macro_f1",
              "val_patient_balanced_accuracy", "learning_rate")
    best_score, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, _option(args, "epochs") + 1):
            model.train()
            loss_total, sample_count = 0.0, 0
            learning_rate = optimizer.param_groups[0]["lr"]
            for inputs, targets, _ in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise ValueError("ResNet training produced nonfinite loss.")
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach()) * len(targets)
                sample_count += len(targets)
            result = evaluate_records(model, val_records, args, cache_dir)
            patient = result[0]["patient"]
            score, val_loss = patient["macro_f1"], patient["loss"]
            writer.writerow({"epoch": epoch, "train_loss": loss_total / sample_count,
                             "val_patient_loss": val_loss, "val_patient_macro_f1": score,
                             "val_patient_balanced_accuracy": patient["balanced_accuracy"],
                             "learning_rate": learning_rate})
            handle.flush()
            improved = score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and val_loss < best_loss)
            if improved:
                best_score, best_loss, best_epoch, stale = score, val_loss, epoch, 0
                torch.save({"model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch, "validation": result[0], "config": config}, output_dir / "best.pt")
            else:
                stale += 1
            scheduler.step(score)
            _report(args, output_dir, status="training", latest_validation=result[0], best_epoch=best_epoch,
                    message=f"Completed ResNet epoch {epoch}; no test evaluation.")
            print(f"s02 ResNet epoch {epoch:03d}: train_loss={loss_total/sample_count:.4f}, "
                  f"val_patient_macro_f1={score:.4f}, val_patient_loss={val_loss:.4f}", flush=True)
            if stale >= _option(args, "patience"):
                break
    if baseline.dataset_fingerprint([*train_records, *val_records])["sha256"] != snapshot["sha256"]:
        raise ValueError("Development recordings changed during ResNet fitting.")
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate_records(model, val_records, args, cache_dir)
    if baseline.dataset_fingerprint([*train_records, *val_records])["sha256"] != snapshot["sha256"]:
        raise ValueError("Development recordings changed during final validation.")
    write_predictions(output_dir, "validation", val_records, validation)
    metrics = {"best_epoch": best_epoch, "labels": list(LABELS),
               "validation": validation[0], "original_test_evaluated": False}
    _write_json(output_dir / "metrics.json", metrics)
    _report(args, output_dir, status="completed", best_epoch=best_epoch,
            message="Validation-selected ResNet checkpoint saved; test remains reserved for the controller.")
    return model, metrics
