#!/usr/bin/env python3
"""Frozen XLS-R patient MIL: development CV, frozen selection, then one test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent / "analysis"
REPOSITORY_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))

from s01_data import LABELS, fingerprint, group_patients, load_frozen_manifest, make_development_folds, summary
from s01_training import write_json

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
SELECTION_RULE = "highest mean development-fold patient macro F1; ties: lower mean patient loss, then A/B/C order"
KNOWN_TEST_CAVEAT = (
    "The original s00 test scores are already known and can indirectly influence subsequent design decisions. "
    "This is a new model's single planned evaluation on that same holdout, not a pristine unseen study."
)


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes() -> dict:
    paths = [*CODE_DIR.glob("*.py"), ANALYSIS_DIR / "s01_metrics.py"]
    return {str(path.relative_to(CODE_DIR.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "huggingface-hub", "numpy", "scipy", "soundfile", "scikit-learn"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


def study_parameters(args) -> dict:
    keys = ("seed", "epochs", "patience", "batch_size", "learning_rate", "weight_decay",
            "workers", "torch_threads", "encoder_batch_size",
            "folds", "projection_size", "attention_size", "dropout", "encoder_name", "encoder_revision",
            "clip_seconds", "eval_windows", "augmentation_variants", "augmentation_version",
            "cutoff_min", "cutoff_max", "snr_min", "snr_max")
    return {**{key: getattr(args, key) for key in keys}, "variants": list(args.variant_list),
            "sample_rate": 16000, "encoder_frozen": True, "embedding_precision": "float32"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--quarantine-manifest", type=Path)
    p.add_argument("--s00-metrics", type=Path, help="Optional local aggregate baseline metrics for reporting only")
    p.add_argument("--data-root", type=Path, help="Accepted by the common launcher; never used to rediscover audio")
    p.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "runs/s01")
    p.add_argument("--report-path", type=Path)
    p.add_argument("--stage", choices=("cv", "final"), default="cv")
    p.add_argument("--selection", type=Path, help="Frozen CV selection; defaults to output-dir/selection.json")
    p.add_argument("--variants", default="A,B,C")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=16)
    p.add_argument("--encoder-batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8, help="Patients per batch; every available recording is retained")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--projection-size", type=int, default=256)
    p.add_argument("--attention-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=.3)
    p.add_argument("--encoder-name", default=MODEL_NAME)
    p.add_argument("--encoder-revision", default=MODEL_REVISION)
    p.add_argument("--clip-seconds", type=float, default=5)
    p.add_argument("--eval-windows", type=int, default=5)
    p.add_argument("--augmentation-variants", type=int, default=1)
    p.add_argument("--augmentation-version", default="channel-v1")
    p.add_argument("--cutoff-min", type=float, default=4000)
    p.add_argument("--cutoff-max", type=float, default=7500)
    p.add_argument("--snr-min", type=float, default=20)
    p.add_argument("--snr-max", type=float, default=35)
    p.add_argument("--encoder-cache-dir", type=Path,
                   default=Path(os.environ.get("HF_HUB_CACHE", str(REPOSITORY_ROOT / ".cache/huggingface"))))
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--export-pdf", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true", help="Synthetic CPU checks only; no speech model or patient data")
    return p


def validate_args(args):
    args.output_dir = args.output_dir.expanduser().resolve()
    values = [x.strip().upper() for x in args.variants.split(",")]
    if not values or len(set(values)) != len(values) or any(x not in {"A", "B", "C"} for x in values):
        raise ValueError("Variants must be a nonempty comma-separated subset of A,B,C without repeats.")
    args.variant_list = sorted(values)
    if min(args.epochs, args.patience, args.batch_size, args.encoder_batch_size,
           args.eval_windows, args.projection_size, args.attention_size) < 1 or args.folds < 2:
        raise ValueError("Training sizes/patience must be positive and at least two folds are required.")
    if args.workers < 0 or args.torch_threads < 0 or args.augmentation_variants < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid worker/thread/augmentation count or seed.")
    numeric = (args.learning_rate, args.weight_decay, args.dropout, args.clip_seconds,
               args.cutoff_min, args.cutoff_max, args.snr_min, args.snr_max)
    if not all(math.isfinite(x) for x in numeric):
        raise ValueError("Numeric settings must be finite.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or args.clip_seconds <= 0:
        raise ValueError("Invalid optimizer, dropout, or window setting.")
    if not 0 < args.cutoff_min <= args.cutoff_max < 8000 or not 0 < args.snr_min <= args.snr_max:
        raise ValueError("Channel perturbations require valid cutoff and positive SNR ranges.")


def refresh_reports(args, status, message, *, pdf=False):
    from s01_report import write_report
    write_report(args.output_dir, status=status, message=message)
    if args.report_path:
        write_report(args.output_dir, report_path=args.report_path, status=status, message=message)
    if pdf and args.export_pdf:
        from s01_export_pdf import export_pdf
        export_pdf(args.output_dir / "report.html", args.output_dir / "report.pdf")


def prepare_study(args):
    from s01_audit import domain_audit
    if args.manifest is None:
        raise ValueError("A frozen --manifest is required; s01 never discovers or reassigns patients.")
    records = load_frozen_manifest(args.manifest, args.quarantine_manifest)
    current = fingerprint(records)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "manifest_fingerprint.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text())["sha256"] != current["sha256"]:
        raise ValueError("The approved manifest or recording content changed; use a fresh study directory.")
    parameters = study_parameters(args)
    config_path = output / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing.get("parameters") != parameters or existing.get("source_hashes") != source_hashes():
            raise ValueError("Study settings or training source differ from the frozen configuration; use a fresh study directory.")
    folds = make_development_folds(records, args.folds, args.seed)
    fold_path = output / "folds.json"
    if fold_path.exists() and json.loads(fold_path.read_text()) != folds:
        raise ValueError("Existing development folds differ; do not overwrite patient memberships.")
    config = {"experiment": "s01", "parameters": parameters, "versions": versions(),
              "source_hashes": source_hashes(), "manifest_sha256": current["sha256"],
              "selection_rule": SELECTION_RULE, "known_test_caveat": KNOWN_TEST_CAVEAT,
              "augmentation_scope": "C training-fold patients only; validation and test always use base embeddings",
              "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}
    write_json(snapshot_path, current)
    if not config_path.exists():
        write_json(config_path, config)
    write_json(output / "summary.json", summary(records))
    write_json(fold_path, folds)
    audit = domain_audit(records)
    write_json(output / "domain_audit.json", audit)
    if audit.get("header_errors", 0):
        raise ValueError("Source audio header audit failed; inspect the local aggregate audit before training.")
    if args.s00_metrics:
        baseline = json.loads(args.s00_metrics.read_text())
        write_json(output / "s00_metrics.json", {key: baseline[key] for key in ("validation", "test") if key in baseline})
    if args.prepare_only or not (output / "cv_summary.json").exists():
        refresh_reports(args, "prepared", "Frozen patient memberships and development folds validated. No original test inference has occurred.")
    return records, folds, current, config


def resolve_device(args):
    import torch
    from s01_training import seed_everything
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but cannot initialize; repair the host or choose CPU explicitly.")
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)


def build_embedder(args):
    from s01_audio import AudioConfig, AugmentationConfig
    from s01_encoder import EmbeddingCache, FrozenEncoder, RecordingEmbedder
    audio = AudioConfig(sample_rate=16000, clip_seconds=args.clip_seconds, max_windows=args.eval_windows)
    augmentation = AugmentationConfig(enabled=args.augmentation_variants > 0, seed=args.seed,
                                      version=args.augmentation_version, cutoff_min=args.cutoff_min,
                                      cutoff_max=args.cutoff_max, snr_min=args.snr_min, snr_max=args.snr_max)
    encoder = FrozenEncoder(args.encoder_name, args.encoder_revision, device=args.device,
                            batch_size=args.encoder_batch_size, cache_dir=args.encoder_cache_dir,
                            local_files_only=args.local_files_only)
    cache = EmbeddingCache(args.output_dir / "embeddings", encoder.identity, audio,
                           augmentation_version=args.augmentation_version)
    metadata_path = args.output_dir / "encoder_metadata.json"
    identity = json.loads(json.dumps(encoder.identity, allow_nan=False))
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != identity:
            raise ValueError("Loaded encoder identity, software, batching, or backend differs from development CV; use the frozen environment.")
    else:
        write_json(metadata_path, identity)
    return RecordingEmbedder(encoder, cache, audio_config=audio, augmentation=augmentation)


def extract_map(embedder, records, aug_index=0):
    def progress(done, total):
        if done == total or done % 100 == 0:
            print(f"Embedding {'base' if aug_index == 0 else 'training augmentation '+str(aug_index)}: {done}/{total}", flush=True)
    values = embedder.embed_records(records, aug_index=aug_index, progress=progress)
    return {record.path: value for record, value in zip(records, values)}


def summarize_cv(variants: dict) -> dict:
    import statistics
    scalar_keys = ("macro_f1", "balanced_accuracy", "macro_ovr_auc", "accuracy", "loss")
    for values in variants.values():
        values["mean"], values["std"] = {}, {}
        for key in scalar_keys:
            numbers = [fold["patient"][key] for fold in values["folds"]]
            if any(x is None for x in numbers):
                values["mean"][key] = values["std"][key] = None
            else:
                values["mean"][key] = statistics.mean(numbers)
                values["std"][key] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
    chosen = min(variants, key=lambda name: (-variants[name]["mean"]["macro_f1"], variants[name]["mean"]["loss"], name))
    return {"variants": variants, "selected_variant": chosen, "selection_rule": SELECTION_RULE,
            "std_definition": "sample standard deviation across development folds",
            "original_test_evaluated": False, "known_test_caveat": KNOWN_TEST_CAVEAT}


def run_cv(args, records, folds, snapshot, config, embedder):
    from s01_training import fit_model
    if (args.output_dir / "cv_summary.json").exists():
        raise ValueError("Development CV already finished. Use --stage final with its frozen selection or a fresh study directory.")
    patients = {p["patient_id"]: p for p in group_patients(records)}
    development = [r for r in records if r.split in {"train", "val"}]
    embeddings = extract_map(embedder, development)
    results = {}
    for variant in args.variant_list:
        results[variant] = {"folds": []}
        for fold in folds:
            train_patients = [patients[key] for key in fold["train_ids"]]
            val_patients = [patients[key] for key in fold["val_ids"]]
            augmented = {}
            if variant == "C":
                training_records = [r for p in train_patients for r in p["records"]]
                for index in range(1, args.augmentation_variants + 1):
                    for path, value in extract_map(embedder, training_records, index).items():
                        augmented.setdefault(path, []).append(value)
            directory = args.output_dir / "cv" / variant / f"fold_{fold['fold']:02d}"
            name = f"s01-{variant} fold {fold['fold']}/{len(folds)}"
            refresh_reports(args, "cv", f"Development CV: {name}. Original test patients remain excluded.")
            _, metrics = fit_model(train_patients, val_patients, embeddings, augmented, args,
                                   variant, directory, args.seed + fold["fold"],
                                   {**config, "run_name": name, "fold": fold["fold"]})
            results[variant]["folds"].append({"fold": fold["fold"], "patient": metrics["validation"]["patient"],
                                             "recording": metrics["validation"]["recording"], "best_epoch": metrics["best_epoch"]})
            write_json(args.output_dir / "cv_progress.json", {"variants": results})
    if fingerprint(records)["sha256"] != snapshot["sha256"]:
        raise ValueError("Audio files changed during CV; architecture selection was not saved.")
    result = summarize_cv(results)
    write_json(args.output_dir / "cv_summary.json", result)
    selection = {"selected_variant": result["selected_variant"], "selection_rule": SELECTION_RULE,
                 "parameters": study_parameters(args), "manifest_sha256": snapshot["sha256"],
                 "folds_sha256": digest_json(folds), "cv_summary_sha256": digest_json(result),
                 "encoder_identity_sha256": digest_json(json.loads((args.output_dir / "encoder_metadata.json").read_text())),
                 "training_source_hashes": source_hashes(), "known_test_caveat": KNOWN_TEST_CAVEAT}
    selection["selection_sha256"] = digest_json(selection)
    write_json(args.output_dir / "selection.json", selection)
    refresh_reports(args, "completed", f"Development CV complete. s01-{result['selected_variant']} selected and configuration frozen; original test remains unevaluated.", pdf=True)
    return result


def load_selection(args, snapshot, folds):
    path = args.selection or args.output_dir / "selection.json"
    if not path.is_file() or not (args.output_dir / "cv_summary.json").is_file():
        raise ValueError("Final training requires completed development CV and its frozen selection.json.")
    selected = json.loads(path.read_text())
    signature = selected.get("selection_sha256")
    unsigned = {key: value for key, value in selected.items() if key != "selection_sha256"}
    if signature != digest_json(unsigned):
        raise ValueError("Frozen selection integrity check failed.")
    cv = json.loads((args.output_dir / "cv_summary.json").read_text())
    encoder_path = args.output_dir / "encoder_metadata.json"
    if not encoder_path.is_file() or selected.get("encoder_identity_sha256") != digest_json(json.loads(encoder_path.read_text())):
        raise ValueError("Frozen encoder provenance differs from development selection.")
    if (selected.get("parameters") != study_parameters(args)
            or selected.get("manifest_sha256") != snapshot["sha256"]
            or selected.get("folds_sha256") != digest_json(folds)
            or selected.get("cv_summary_sha256") != digest_json(cv)
            or selected.get("training_source_hashes") != source_hashes()
            or selected.get("selected_variant") != cv.get("selected_variant")):
        raise ValueError("Data, code, folds, or settings differ from the frozen development selection.")
    return selected


def run_final(args, records, folds, snapshot, config, embedder):
    from s01_training import evaluate, fit_model, write_predictions
    selected = load_selection(args, snapshot, folds)
    guard = args.output_dir / "test_evaluation.json"
    if guard.exists():
        raise ValueError("Original test evaluation has already started in this study; it cannot be repeated automatically.")
    patients = group_patients(records)
    train_patients = [p for p in patients if p["split"] == "train"]
    val_patients = [p for p in patients if p["split"] == "val"]
    test_patients = [p for p in patients if p["split"] == "test"]
    embeddings = extract_map(embedder, [r for r in records if r.split != "test"])
    augmented = {}
    variant = selected["selected_variant"]
    if variant == "C":
        training_records = [r for p in train_patients for r in p["records"]]
        for index in range(1, args.augmentation_variants + 1):
            for path, value in extract_map(embedder, training_records, index).items():
                augmented.setdefault(path, []).append(value)
    directory = args.output_dir / "final"
    refresh_reports(args, "training", f"Final s01-{variant}: original training patients only; original validation selects the checkpoint.")
    model, metrics = fit_model(train_patients, val_patients, embeddings, augmented, args, variant,
                               directory, args.seed, {**config, "run_name": f"s01-{variant} final", "selection": selected})
    if fingerprint(records)["sha256"] != snapshot["sha256"]:
        raise ValueError("Audio changed during final training; original test evaluation was not started.")
    # Exclusive creation is the boundary before any original-test encoder/model inference.
    with guard.open("x", encoding="utf-8") as handle:
        json.dump({"status": "started", "selection_sha256": selected["selection_sha256"],
                   "manifest_sha256": snapshot["sha256"], "known_test_caveat": KNOWN_TEST_CAVEAT}, handle, indent=2)
    embeddings.update(extract_map(embedder, [r for r in records if r.split == "test"]))
    test_metrics, predictions = evaluate(model, test_patients, embeddings, args.device, args.batch_size)
    if fingerprint(records)["sha256"] != snapshot["sha256"]:
        raise ValueError("Audio changed during test evaluation; final results were not published.")
    write_predictions(directory, "test", predictions)
    metrics.update(test=test_metrics, selected_variant=variant, original_test_evaluated=True,
                   selection_sha256=selected["selection_sha256"], known_test_caveat=KNOWN_TEST_CAVEAT)
    write_json(directory / "metrics.json", metrics)
    write_json(args.output_dir / "metrics.json", metrics)
    shutil.copy2(directory / "history.csv", args.output_dir / "history.csv")
    shutil.copy2(directory / "best.pt", args.output_dir / "best.pt")
    write_json(guard, {"status": "completed", "selection_sha256": selected["selection_sha256"],
                       "test_patient_count": len(test_patients), "evaluations": 1})
    from s01_report import write_report
    write_report(directory, status="completed", message="Final selected checkpoint evaluated once on the original test patients.")
    refresh_reports(args, "completed", "CV-selected configuration and validation-selected checkpoint evaluated once on the original test. " + KNOWN_TEST_CAVEAT, pdf=True)
    return metrics


def synthetic_smoke(args):
    import torch
    from s01_mil import PatientMIL, patient_class_weights
    from s01_training import seed_everything
    seed_everything(args.seed)
    torch.set_num_threads(1)
    values = torch.randn(6, 4, 12)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]] * 2, dtype=torch.bool)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    for pooling in ("mean", "attention"):
        model = PatientMIL(12, pooling=pooling, dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss(weight=patient_class_weights(targets.tolist()))
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values, mask), targets)
            assert bool(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        model.eval()
        altered = values.clone()
        altered[~mask] = 1e6
        with torch.inference_mode():
            assert torch.allclose(model(values, mask), model(altered, mask), atol=1e-6)
    result = {"kind": "synthetic_cpu_smoke", "clinical_training": False, "device": "cpu",
              "finite_optimization": True, "masked_padding_ignored": True,
              "optimizer_steps": 4, "pretrained_encoder_loaded": False}
    write_json(args.output_dir / "smoke_check.json", result)
    print("Synthetic CPU smoke passed: mean/attention MIL, finite optimization, masked padding. No clinical results.")
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    started = False
    try:
        validate_args(args)
        os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".cache/matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir / ".cache"))
        if args.report_only:
            from s01_report import write_report
            write_report(args.output_dir)
            if args.report_path:
                write_report(args.output_dir, report_path=args.report_path)
            if args.export_pdf:
                from s01_export_pdf import export_pdf
                export_pdf(args.output_dir / "report.html", args.output_dir / "report.pdf")
            return 0
        if args.smoke_test:
            synthetic_smoke(args)
            return 0
        if not args.prepare_only:
            if args.stage == "final" and (args.output_dir / "test_evaluation.json").exists():
                raise ValueError("Original test evaluation already started; use --report-only to view saved results.")
            if args.stage == "cv" and (args.output_dir / "cv_summary.json").exists():
                raise ValueError("CV is complete; use --stage final with the frozen selection or --report-only.")
        records, folds, snapshot, config = prepare_study(args)
        if args.prepare_only:
            print("Prepared frozen patient splits, development folds, and acquisition audit. No encoder/model inference.")
            return 0
        started = True
        if args.stage == "final":
            load_selection(args, snapshot, folds)
            if (args.output_dir / "test_evaluation.json").exists():
                raise ValueError("Original test evaluation already started; use --report-only to view saved results.")
        elif (args.output_dir / "cv_summary.json").exists():
            raise ValueError("CV is complete; use --stage final with the frozen selection or --report-only.")
        resolve_device(args)
        embedder = build_embedder(args)
        if args.stage == "cv":
            run_cv(args, records, folds, snapshot, config, embedder)
        else:
            run_final(args, records, folds, snapshot, config, embedder)
        write_json(args.output_dir / f"embedding_cache_stats_{args.stage}.json", embedder.stats)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        if started and args.output_dir.exists():
            try:
                refresh_reports(args, "failed", "The run stopped before the requested stage completed. Inspect the local log; no missing results are inferred.")
            except Exception:
                pass
        p.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
