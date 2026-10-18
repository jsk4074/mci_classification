#!/usr/bin/env python3
"""Patient-disjoint, three-class speech baseline (NM=0, AD=1, MCI=2).

Install dependencies from requirements-resnet18.txt. Examples from this folder:
  python resnet18_baseline.py --prepare-only --report-path report.html
  python resnet18_baseline.py --report-path report.html
  python resnet18_baseline.py --report-only --report-path report.html

Automatic discovery includes numbered 06-/07- WAV clips and maps the downloaded
raw MCI filenames to the same tasks. Each class root must contain patient
directories. For other layouts/names, supply a CSV with
path,patient_id,label and optional split (train/val/test) columns. Relative audio
paths in a CSV are resolved against that CSV's directory. Patient IDs must be
globally unique and must group all visits/recordings from the same person.
When dataset/s00/split_manifest.csv exists, defaults use that copied dataset
and runs/s00. A data root with split_manifest.csv can be used directly, retaining
its patient assignments without rediscovering or splitting its directories.

The default 70/15/15 split is stratified by PATIENT, never by recording/window.
Validation patient macro F1 selects the checkpoint. Test is evaluated only after
training. A saved manifest is reused so subsequent downloads cannot change the
split silently. Final metrics include both recording and patient predictions.
Each run writes report.html after preparation, after every epoch, and after final
evaluation. --report-only regenerates a report from saved artifacts without
loading audio or the model; --report-path also writes a copy at your chosen path.
Small patient holdouts have high uncertainty; this is a reproducible held-out
baseline, not a substitute for repeated patient-level cross-validation.

References:
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
https://scikit-learn.org/stable/modules/cross_validation.html
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys

from speech_data import (
    DEFAULT_FILENAME_PATTERN,
    LABELS,
    Recording,
    assign_patient_splits,
    discover_recordings,
    load_manifest,
    manifest_summary,
    validate_cross_patient_content,
    write_manifest,
)
from speech_report import write_report

PROJECT_ROOT = Path(__file__).resolve().parent
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 40
    n_fft: int = 400
    hop_length: int = 160
    clip_seconds: float = 5.0
    top_db: float = 80.0
    eval_windows: int = 5

    @property
    def frames(self) -> int:
        return int(round(self.clip_seconds * self.sample_rate / self.hop_length)) + 1


def evaluation_starts(total_frames: int, window_frames: int, max_windows: int) -> list[int]:
    """Deterministic windows spanning a recording; short recordings are padded."""
    if min(total_frames, window_frames, max_windows) < 1:
        raise ValueError("Frame lengths and evaluation window count must be positive.")
    last = max(0, total_frames - window_frames)
    count = min(max_windows, math.ceil(total_frames / window_frames))
    if count == 1:
        return [last // 2]
    return sorted({round(i * last / (count - 1)) for i in range(count)})


def patient_sampling_weights(records: list[Recording]) -> list[float]:
    """Give each class, then each patient within that class, equal probability."""
    clips = Counter(r.patient_id for r in records)
    patients = {label: {r.patient_id for r in records if r.label == label} for label in LABELS}
    return [1.0 / (len(patients[r.label]) * clips[r.patient_id]) for r in records]


class AudioDataset:
    """Cache full log-mels; crop training inputs and span evaluation recordings."""

    def __init__(self, records: list[Recording], config: FeatureConfig,
                 training: bool, cache_dir: Path | None = None):
        self.records = records
        self.config = config
        self.training = training
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def _features(self, record: Recording):
        import librosa
        import numpy as np

        path = Path(record.path)
        stat = path.stat()
        cache_path = None
        if self.cache_dir:
            identity = json.dumps({
                "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "features": asdict(self.config), "pipeline": "logmel-v1",
                "librosa": librosa.__version__,
            }, sort_keys=True)
            key = hashlib.sha256(identity.encode()).hexdigest()
            cache_path = self.cache_dir / f"{key}.npy"
            if cache_path.exists():
                feature = np.load(cache_path, allow_pickle=False)
                if (feature.ndim == 2 and feature.shape[0] == self.config.n_mels
                        and feature.shape[1] > 0 and np.isfinite(feature).all()):
                    return feature
                raise ValueError(f"Invalid feature cache: {cache_path}")
        try:
            audio, _ = librosa.load(path, sr=self.config.sample_rate, mono=True)
        except Exception as exc:
            raise ValueError(f"Cannot decode audio (check that download is complete): {path}") from exc
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Audio must be nonempty and finite: {path}")
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.config.sample_rate, n_fft=self.config.n_fft,
            hop_length=self.config.hop_length, n_mels=self.config.n_mels,
            fmin=20, fmax=self.config.sample_rate / 2, power=2.0,
        )
        # Each recording uses its own peak; no fitted statistics from validation/test.
        db = librosa.power_to_db(mel, ref=np.max, top_db=self.config.top_db)
        feature = (np.clip(db, -self.config.top_db, 0) / self.config.top_db * 2 + 1).astype(np.float32)
        if cache_path:
            temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, feature, allow_pickle=False)
            temporary.replace(cache_path)
        return feature

    def __getitem__(self, index):
        import numpy as np
        import torch

        record = self.records[index]
        feature = self._features(record)
        frames = self.config.frames
        if feature.shape[1] < frames:
            feature = np.pad(feature, ((0, 0), (0, frames - feature.shape[1])),
                             constant_values=-1)
        if self.training:
            start = int(torch.randint(feature.shape[1] - frames + 1, (1,)).item())
            tensor = torch.from_numpy(feature[:, start:start + frames].copy()).unsqueeze(0)
        else:
            starts = evaluation_starts(feature.shape[1], frames, self.config.eval_windows)
            windows = np.stack([feature[:, start:start + frames] for start in starts])
            tensor = torch.from_numpy(windows).unsqueeze(1)
        return tensor, LABEL_TO_INDEX[record.label], index


def collate_evaluation(batch):
    import torch

    inputs, labels, indices = zip(*batch)
    owners = torch.cat([torch.full((len(x),), i, dtype=torch.long) for i, x in enumerate(inputs)])
    return torch.cat(inputs), owners, labels, indices


def seed_worker(worker_id):
    import numpy as np
    import torch

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def seed_everything(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_model(pretrained: bool = False):
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    if pretrained:
        torch.hub.set_dir(str(PROJECT_ROOT / ".cache/torch"))
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    else:
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model


def classification_metrics(targets, probabilities):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix,
        precision_recall_fscore_support, roc_auc_score,
    )

    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, labels=list(range(len(LABELS))), zero_division=0,
    )
    return {
        "count": len(targets),
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_f1": float(f1.mean()),
        "loss": float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1)).mean()),
        "macro_ovr_auc": float(roc_auc_score(targets, probabilities, multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(targets, predicted, labels=list(range(len(LABELS)))).tolist(),
        "confusion_matrix_labels": list(LABELS),
        "per_class": {label: {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        } for i, label in enumerate(LABELS)},
    }


def aggregate_patients(records: list[Recording], recording_probabilities):
    """Average recordings equally per patient after averaging their windows."""
    import numpy as np

    if len(records) != len(recording_probabilities):
        raise ValueError("Prediction rows must match the recording manifest exactly.")
    grouped = defaultdict(list)
    targets = {}
    for record, probability in zip(records, recording_probabilities):
        grouped[record.patient_id].append(probability)
        label = LABEL_TO_INDEX[record.label]
        if record.patient_id in targets and targets[record.patient_id] != label:
            raise ValueError("Patient has inconsistent labels.")
        targets[record.patient_id] = label
    patient_ids = sorted(grouped)
    labels = np.array([targets[p] for p in patient_ids])
    probabilities = np.stack([np.mean(grouped[p], axis=0) for p in patient_ids])
    return patient_ids, labels, probabilities


def evaluate(model, loader, device, records: list[Recording]):
    import numpy as np
    import torch

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
        raise ValueError("Evaluation did not cover all recordings.")
    ids, patient_targets, patient_probs = aggregate_patients(records, probabilities)
    recording_targets = np.array([LABEL_TO_INDEX[r.label] for r in records])
    metrics = {
        "recording": classification_metrics(recording_targets, probabilities),
        "patient": classification_metrics(patient_targets, patient_probs),
    }
    return metrics, probabilities, ids, patient_targets, patient_probs


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def update_html_report(args, **state):
    """Keep the run report and an optional project report in sync."""
    report = write_report(args.output_dir, **state)
    if args.report_path is not None:
        report = write_report(args.output_dir, report_path=args.report_path)
    return report


def write_predictions(output_dir: Path, split: str, records: list[Recording], result):
    _, probabilities, patient_ids, patient_targets, patient_probs = result
    probability_columns = [f"p_{label}" for label in LABELS]
    with (output_dir / f"{split}_recording_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "patient_id", "true_label", "predicted_label", *probability_columns])
        for record, probs in zip(records, probabilities):
            writer.writerow([record.path, record.patient_id, record.label,
                             LABELS[int(probs.argmax())], *map(float, probs)])
    with (output_dir / f"{split}_patient_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "true_label", "predicted_label", *probability_columns])
        for patient_id, label, probs in zip(patient_ids, patient_targets, patient_probs):
            writer.writerow([patient_id, LABELS[int(label)], LABELS[int(probs.argmax())], *map(float, probs)])


def dataset_fingerprint(records: list[Recording]):
    rows = []
    for record in records:
        stat = Path(record.path).stat()
        rows.append({**asdict(record), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": rows}


def train(args, records: list[Recording], features: FeatureConfig):
    try:
        import librosa
        import numpy as np
        import sklearn
        import torch
        import torchvision
        from torch import nn
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:
        raise RuntimeError("Missing training dependencies. Install requirements-resnet18.txt; "
                           "--prepare-only works without them.") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu or auto.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    subsets = {split: [r for r in records if r.split == split] for split in ("train", "val", "test")}
    cache_dir = None if args.no_cache else args.output_dir / "features"
    train_dataset = AudioDataset(subsets["train"], features, training=True, cache_dir=cache_dir)
    sampler = WeightedRandomSampler(
        patient_sampling_weights(subsets["train"]), num_samples=len(train_dataset),
        replacement=True, generator=torch.Generator().manual_seed(args.seed),
    )
    common = {"batch_size": args.batch_size, "num_workers": args.workers,
              "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
    train_loader = DataLoader(train_dataset, sampler=sampler,
                              generator=torch.Generator().manual_seed(args.seed), **common)
    # The test loader is only constructed after checkpoint selection is finished.
    val_loader = DataLoader(
        AudioDataset(subsets["val"], features, training=False, cache_dir=cache_dir),
        collate_fn=collate_evaluation, shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 1), **common,
    )
    model = build_model(args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    config = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "features": asdict(features), "labels": LABEL_TO_INDEX,
        "checkpoint_selection": "validation patient macro F1; ties use patient loss",
        "sampling": "equal classes, equal patients within each class",
        "device": str(device), "python": sys.version,
        "versions": {name: version(name) for name in
                     ("torch", "torchvision", "numpy", "librosa", "scikit-learn")},
    }
    fingerprint = dataset_fingerprint(records)
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "dataset_snapshot.json", fingerprint)
    update_html_report(args, status="training", features=asdict(features),
                       message="Training has started. Test results remain pending until checkpoint selection finishes.")
    best_score, best_loss, stale, best_epoch = -1.0, float("inf"), 0, 0
    checkpoint_path = args.output_dir / "best.pt"
    print(f"Training ResNet18 on {device}; checkpoint metric: validation patient macro F1.", flush=True)
    with (args.output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_patient_loss",
                               "val_patient_macro_f1", "val_patient_balanced_accuracy", "learning_rate"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_total, sample_count = 0.0, 0
            learning_rate = optimizer.param_groups[0]["lr"]
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise ValueError("Training produced nonfinite loss.")
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach()) * len(targets)
                sample_count += len(targets)
            result = evaluate(model, val_loader, device, subsets["val"])
            validation = result[0]["patient"]
            score, val_loss = validation["macro_f1"], validation["loss"]
            writer.writerow({"epoch": epoch, "train_loss": loss_total / sample_count,
                             "val_patient_loss": val_loss, "val_patient_macro_f1": score,
                             "val_patient_balanced_accuracy": validation["balanced_accuracy"],
                             "learning_rate": learning_rate})
            handle.flush()
            improved = score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and val_loss < best_loss)
            if improved:
                best_score, best_loss, best_epoch, stale = score, val_loss, epoch, 0
                torch.save({
                    "model_state_dict": {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "validation": result[0], "config": config,
                }, checkpoint_path)
            else:
                stale += 1
            scheduler.step(score)
            update_html_report(args, status="training", latest_validation=result[0],
                               best_epoch=best_epoch, message=f"Completed epoch {epoch} of {args.epochs}.")
            print(f"Epoch {epoch:03d}: train loss={loss_total / sample_count:.4f}, "
                  f"val patient macro F1={score:.4f}, balanced accuracy={validation['balanced_accuracy']:.4f}", flush=True)
            if stale >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.", flush=True)
                break

    # Refuse to report results if a download changed any included recording mid-run.
    if dataset_fingerprint(records)["sha256"] != fingerprint["sha256"]:
        raise ValueError("Dataset files changed during training. Finish the download before running again.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    update_html_report(args, status="training", best_epoch=best_epoch,
                       message="Training finished. Evaluating the selected checkpoint on validation and test patients.")
    validation_result = evaluate(model, val_loader, device, subsets["val"])
    test_loader = DataLoader(
        AudioDataset(subsets["test"], features, training=False, cache_dir=cache_dir),
        collate_fn=collate_evaluation, shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 2), **common,
    )
    test_result = evaluate(model, test_loader, device, subsets["test"])
    if dataset_fingerprint(records)["sha256"] != fingerprint["sha256"]:
        raise ValueError("Dataset files changed during final evaluation; results were not saved.")
    write_predictions(args.output_dir, "validation", subsets["val"], validation_result)
    write_predictions(args.output_dir, "test", subsets["test"], test_result)
    metrics = {"best_epoch": best_epoch, "labels": list(LABELS),
               "validation": validation_result[0], "test": test_result[0]}
    write_json(args.output_dir / "metrics.json", metrics)
    report = update_html_report(args, status="completed", best_epoch=best_epoch,
                                message="Final results use the selected checkpoint. Test was evaluated after training.")
    print(f"Best epoch: {best_epoch}; test patient macro F1={test_result[0]['patient']['macro_f1']:.4f}, "
          f"balanced accuracy={test_result[0]['patient']['balanced_accuracy']:.4f}", flush=True)
    print(f"Saved checkpoint, metrics, history, and predictions to {args.output_dir}", flush=True)
    print(f"HTML report: {report}", flush=True)


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    copied_root = PROJECT_ROOT / "dataset/s00"
    copied_ready = (copied_root / "split_manifest.csv").is_file()
    default_data_root = copied_root if copied_ready else PROJECT_ROOT / "dataset/NM50_AD50_MCI20"
    default_output_dir = PROJECT_ROOT / ("runs/s00" if copied_ready else "runs/resnet18")
    argument_parser.add_argument("--data-root", type=Path, default=default_data_root)
    argument_parser.add_argument("--manifest", type=Path, help="CSV of paths, globally unique patient IDs, and NM/AD/MCI labels")
    for label in LABELS:
        argument_parser.add_argument(f"--{label.lower()}-root", type=Path, help=f"{label} folder containing patient subdirectories")
    argument_parser.add_argument("--filename-pattern", default=DEFAULT_FILENAME_PATTERN, help="Full-match, case-insensitive regex for auto-discovery")
    argument_parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    argument_parser.add_argument("--prepare-only", action="store_true", help="Validate and save patient splits without ML dependencies/training")
    argument_parser.add_argument("--report-only", action="store_true", help="Regenerate HTML from saved artifacts without training or loading audio")
    argument_parser.add_argument("--report-path", type=Path, help="Also write the HTML report at this path, e.g. report.html in the project root")
    argument_parser.add_argument("--seed", type=int, default=42)
    argument_parser.add_argument("--val-fraction", type=float, default=0.15)
    argument_parser.add_argument("--test-fraction", type=float, default=0.15)
    argument_parser.add_argument("--epochs", type=int, default=50)
    argument_parser.add_argument("--patience", type=int, default=10)
    argument_parser.add_argument("--batch-size", type=int, default=16)
    argument_parser.add_argument("--learning-rate", type=float, default=1e-3)
    argument_parser.add_argument("--weight-decay", type=float, default=1e-4)
    argument_parser.add_argument("--workers", type=int, default=0)
    argument_parser.add_argument("--torch-threads", type=int, default=0, help="CPU threads; 0 keeps PyTorch default")
    argument_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    argument_parser.add_argument("--pretrained", action="store_true", help="Opt in to downloading ImageNet weights; default trains from scratch")
    argument_parser.add_argument("--no-cache", action="store_true", help="Do not cache log-mel features in output-dir/features")
    argument_parser.add_argument("--clip-seconds", type=float, default=5.0)
    argument_parser.add_argument("--eval-windows", type=int, default=5, help="Maximum deterministic windows per evaluation recording")
    return argument_parser


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    report_started = False
    try:
        if min(args.epochs, args.patience, args.batch_size, args.eval_windows) < 1:
            raise ValueError("Epochs, patience, batch size, and evaluation windows must be positive.")
        if args.workers < 0 or args.torch_threads < 0:
            raise ValueError("Workers and CPU thread count cannot be negative.")
        if not 0 <= args.seed < 2**32:
            raise ValueError("Seed must be between 0 and 2**32 - 1.")
        if (not math.isfinite(args.val_fraction) or not math.isfinite(args.test_fraction)
                or min(args.val_fraction, args.test_fraction) <= 0
                or args.val_fraction + args.test_fraction >= 1):
            raise ValueError("Validation and test fractions must be positive and sum to less than 1.")
        if not math.isfinite(args.clip_seconds) or args.clip_seconds < 0.1:
            raise ValueError("Clip duration must be finite and at least 0.1 seconds.")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative.")
        args.output_dir = args.output_dir.resolve()
        if args.report_only:
            if args.prepare_only:
                raise ValueError("Choose either --report-only or --prepare-only.")
            features = None if (args.output_dir / "config.json").exists() else asdict(
                FeatureConfig(clip_seconds=args.clip_seconds, eval_windows=args.eval_windows))
            report = update_html_report(args, features=features)
            print(f"HTML report: {report}", flush=True)
            return 0
        for name in ("config.json", "best.pt", "history.csv", "metrics.json"):
            if (args.output_dir / name).exists():
                raise ValueError("This output directory already contains a training run. Choose a new --output-dir "
                                 "and reuse its split_manifest.csv via --manifest.")
        saved_manifest = args.output_dir / "split_manifest.csv"
        dataset_manifest = args.data_root / "split_manifest.csv"
        manifest = args.manifest or (saved_manifest if saved_manifest.exists() else
                                     dataset_manifest if dataset_manifest.is_file() else None)
        if manifest:
            records = load_manifest(manifest)
            print(f"Using frozen manifest: {manifest}", flush=True)
        else:
            roots = {label: getattr(args, f"{label.lower()}_root") for label in LABELS
                     if getattr(args, f"{label.lower()}_root") is not None}
            records = discover_recordings(args.data_root, roots, args.filename_pattern)
        validate_cross_patient_content(records)
        records = assign_patient_splits(records, args.seed, args.val_fraction, args.test_fraction)
        # Dataset indices must match the saved manifest for seeded sampling to reproduce.
        records = sorted(records, key=lambda r: (LABELS.index(r.label), r.patient_id, r.path))
        summary = manifest_summary(records)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(records, saved_manifest)
        write_json(args.output_dir / "split_summary.json", summary)
        features = FeatureConfig(clip_seconds=args.clip_seconds, eval_windows=args.eval_windows)
        report = update_html_report(args, status="prepared", features=asdict(features),
                                    message="Patient splits are prepared. No training results are available yet.")
        report_started = True
        for split in ("train", "val", "test"):
            counts = summary["splits"][split]
            classes = ", ".join(f"{label}={counts['classes'][label]['patients']}" for label in LABELS)
            print(f"{split}: {counts['patients']} patients ({classes}); {counts['recordings']} recordings", flush=True)
        if args.prepare_only:
            print(f"Prepared {saved_manifest}.", flush=True)
            print(f"HTML report: {report}", flush=True)
            return 0
        train(args, records, features)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        if report_started:
            update_html_report(args, status="failed",
                               message="The run stopped before final results were saved. See the terminal for the error details.")
        argument_parser.exit(2, f"Error: {exc}\n")
    except KeyboardInterrupt:
        if report_started:
            update_html_report(args, status="failed", message="The run was interrupted before completion.")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
