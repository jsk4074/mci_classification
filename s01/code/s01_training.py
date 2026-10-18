"""Patient-only optimization and evaluation; the speech encoder is never trained."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
import tempfile


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id: int) -> None:
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class BagDataset:
    """All recordings per patient; augmentation is supplied only for training."""

    def __init__(self, patients, embeddings, augmented=None):
        self.patients = list(patients)
        self.embeddings = embeddings
        self.augmented = augmented or {}

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        import numpy as np
        import torch
        patient = self.patients[index]
        members = []
        for record in patient["records"]:
            choices = [self.embeddings[record.path], *self.augmented.get(record.path, [])]
            selected = int(torch.randint(len(choices), (1,)).item()) if len(choices) > 1 else 0
            members.append(choices[selected])
        return {"embeddings": torch.from_numpy(np.stack(members).astype(np.float32)),
                "target": patient["target"], "patient_id": patient["patient_id"]}


def evaluate(model, patients, embeddings, device, batch_size=8):
    """One evaluation event: true MIL patient predictions plus singleton clips.

    Clip labels are inherited from their patients. Singleton-bag predictions
    are supplementary and do not select checkpoints or architectures.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from s01_data import LABELS
    from s01_mil import collate_bags
    from s01_metrics import classification_metrics

    model.eval()
    patient_probs, patient_targets, patient_ids = [], [], []
    loader = DataLoader(BagDataset(patients, embeddings), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_bags,
                        generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["embeddings"].to(device), batch["mask"].to(device))
            patient_probs.extend(torch.softmax(logits, dim=-1).cpu().numpy())
            patient_targets.extend(batch["targets"].tolist())
            patient_ids.extend(batch["patient_ids"])
        records = [record for patient in patients for record in patient["records"]]
        recording_probs = []
        for start in range(0, len(records), max(batch_size, 32)):
            subset = records[start:start + max(batch_size, 32)]
            values = torch.from_numpy(np.stack([embeddings[r.path] for r in subset]).astype(np.float32)).to(device)
            logits = model(values.unsqueeze(1), torch.ones((len(subset), 1), dtype=torch.bool, device=device))
            recording_probs.extend(torch.softmax(logits, dim=-1).cpu().numpy())
    patient_probs = np.asarray(patient_probs, dtype=np.float64)
    recording_probs = np.asarray(recording_probs, dtype=np.float64)
    metrics = {
        "patient": classification_metrics(patient_targets, patient_probs),
        "recording": classification_metrics([LABELS.index(r.label) for r in records], recording_probs),
    }
    predictions = {"patient_ids": patient_ids, "patient_targets": patient_targets,
                   "patient_probabilities": patient_probs, "records": records,
                   "recording_probabilities": recording_probs}
    return metrics, predictions


def write_predictions(output_dir: Path, split: str, predictions: dict) -> None:
    from s01_data import LABELS
    output_dir = Path(output_dir)
    with (output_dir / f"{split}_patient_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "true_label", "predicted_label", *[f"p_{x}" for x in LABELS]])
        for patient_id, target, probs in zip(predictions["patient_ids"], predictions["patient_targets"], predictions["patient_probabilities"]):
            writer.writerow([patient_id, LABELS[target], LABELS[int(probs.argmax())], *map(float, probs)])
    with (output_dir / f"{split}_recording_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "patient_id", "true_label", "predicted_label", *[f"p_{x}" for x in LABELS]])
        for record, probs in zip(predictions["records"], predictions["recording_probabilities"]):
            writer.writerow([record.path, record.patient_id, record.label, LABELS[int(probs.argmax())], *map(float, probs)])


def fit_model(train_patients, val_patients, embeddings, augmented, args,
              variant: str, output_dir: Path, seed: int, metadata: dict):
    """Fit one head; selection is validation macro F1 then patient loss."""
    import torch
    from torch.utils.data import DataLoader
    from s01_mil import PatientMIL, collate_bags, patient_class_weights
    from s01_report import write_report

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("history.csv", "best.pt", "metrics.json")):
        raise ValueError("A head-training directory already contains a run; use a fresh study directory.")
    seed_everything(seed)
    hidden_size = len(next(iter(embeddings.values())))
    model_config = {"hidden_size": hidden_size, "pooling": "mean" if variant == "A" else "attention",
                    "projection_size": args.projection_size, "dropout": args.dropout,
                    "attention_size": args.attention_size}
    model = PatientMIL(**model_config).to(args.device)
    weights = patient_class_weights([p["target"] for p in train_patients]).to(args.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(BagDataset(train_patients, embeddings, augmented if variant == "C" else None),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_bags, worker_init_fn=seed_worker, generator=generator)
    config = {**metadata, "variant": variant, "model": model_config, "seed": seed,
              "class_weights": weights.detach().cpu().tolist(),
              "class_balancing": "inverse training-patient frequency in cross-entropy; ordinary shuffle, no balanced sampling",
              "checkpoint_selection": "validation patient macro F1; ties use unweighted patient cross-entropy",
              "recording_metrics": "supplementary singleton-bag predictions with inherited patient labels"}
    write_json(output_dir / "config.json", config)
    write_report(output_dir, status="training", message="Training a patient-level MIL head; no original test patients are used here.")
    best_score, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    fields = ["epoch", "train_loss", "val_patient_loss", "val_patient_macro_f1", "val_patient_balanced_accuracy", "learning_rate"]
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_total, count = 0.0, 0
            for batch in loader:
                targets = batch["targets"].to(args.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["embeddings"].to(args.device), batch["mask"].to(args.device))
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise ValueError("MIL training produced nonfinite loss.")
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach()) * len(targets)
                count += len(targets)
            result, _ = evaluate(model, val_patients, embeddings, args.device, args.batch_size)
            patient = result["patient"]
            score, val_loss = patient["macro_f1"], patient["loss"]
            writer.writerow({"epoch": epoch, "train_loss": loss_total / count,
                             "val_patient_loss": val_loss, "val_patient_macro_f1": score,
                             "val_patient_balanced_accuracy": patient["balanced_accuracy"],
                             "learning_rate": optimizer.param_groups[0]["lr"]})
            handle.flush()
            improved = score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and val_loss < best_loss)
            if improved:
                best_score, best_loss, best_epoch, stale = score, val_loss, epoch, 0
                torch.save({"model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                            "validation": result, "config": config}, output_dir / "best.pt")
            else:
                stale += 1
            print(f"{metadata.get('run_name', variant)} epoch {epoch:03d}: train_loss={loss_total/count:.4f}, "
                  f"val_patient_macro_f1={score:.4f}, val_patient_loss={val_loss:.4f}", flush=True)
            if stale >= args.patience:
                break
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation, predictions = evaluate(model, val_patients, embeddings, args.device, args.batch_size)
    write_predictions(output_dir, "validation", predictions)
    metrics = {"best_epoch": best_epoch, "validation": validation, "original_test_evaluated": False}
    write_json(output_dir / "metrics.json", metrics)
    write_report(output_dir, status="completed", message="Selected checkpoint validation is complete; the original test is not evaluated by this head-training step.")
    return model, metrics
