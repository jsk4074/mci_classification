#!/usr/bin/env python3
"""Matched full-cohort comparison of s00 ResNet18 and s01 frozen XLS-R MIL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
for directory in (CODE, REPO / "s01/code", REPO / "s01/analysis", CODE.parent / "analysis"):
    sys.path.insert(0, str(directory))

from s01_data import fingerprint, group_patients, load_frozen_manifest, make_development_folds, summary
from s01_training import write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = [*CODE.glob("*.py"), * (REPO / "s01/code").glob("*.py"),
             REPO / "s01/analysis/s01_metrics.py", REPO / "s00/code/resnet18_baseline.py",
             REPO / "s00/code/speech_data.py"]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--baseline-manifest", type=Path)
    p.add_argument("--full-data-root", type=Path)
    p.add_argument("--quarantine-manifest", type=Path)
    p.add_argument("--data-root", type=Path, help="Common-launcher compatibility; never used to rediscover assigned patients")
    p.add_argument("--output-dir", type=Path, default=REPO / "runs/s02")
    p.add_argument("--report-path", type=Path)
    p.add_argument("--stage", choices=("all", "cv", "final"), default="all")
    p.add_argument("--system", choices=("both", "resnet18", "xlsr_mil"), default="both")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--eval-windows", type=int, default=5)
    p.add_argument("--resnet-epochs", type=int, default=50)
    p.add_argument("--resnet-patience", type=int, default=10)
    p.add_argument("--resnet-batch-size", type=int, default=16)
    p.add_argument("--resnet-learning-rate", type=float, default=.001)
    p.add_argument("--resnet-weight-decay", type=float, default=.0001)
    p.add_argument("--xlsr-epochs", type=int, default=100)
    p.add_argument("--xlsr-patience", type=int, default=12)
    p.add_argument("--xlsr-batch-size", type=int, default=8)
    p.add_argument("--xlsr-learning-rate", type=float, default=.001)
    p.add_argument("--xlsr-weight-decay", type=float, default=.001)
    p.add_argument("--encoder-batch-size", type=int, default=4)
    p.add_argument("--encoder-cache-dir", type=Path,
                   default=Path(os.environ.get("HF_HUB_CACHE", str(REPO / ".cache/huggingface"))))
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--export-pdf", action="store_true")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    return p


def validate_args(args):
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.workers < 0 or args.torch_threads < 0 or args.folds < 2 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid workers, CPU threads, folds, or seed.")
    positive = (args.clip_seconds, args.eval_windows, args.resnet_epochs, args.resnet_patience,
                args.resnet_batch_size, args.resnet_learning_rate, args.xlsr_epochs,
                args.xlsr_patience, args.xlsr_batch_size, args.xlsr_learning_rate, args.encoder_batch_size)
    import math
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("Training sizes, rates, patience, and windows must be finite and positive.")
    if any(not math.isfinite(value) or value < 0 for value in (args.resnet_weight_decay, args.xlsr_weight_decay)):
        raise ValueError("Weight decay must be finite and nonnegative.")
    if bool(args.baseline_manifest) != bool(args.full_data_root):
        raise ValueError("Cohort construction needs both --baseline-manifest and --full-data-root.")


def settings(args):
    names = ("seed", "folds", "workers", "torch_threads", "clip_seconds", "eval_windows",
             "resnet_epochs", "resnet_patience", "resnet_batch_size", "resnet_learning_rate",
             "resnet_weight_decay", "xlsr_epochs", "xlsr_patience", "xlsr_batch_size",
             "xlsr_learning_rate", "xlsr_weight_decay", "encoder_batch_size")
    return {**{key: getattr(args, key) for key in names}, "xlsr_variants": ["A", "B", "C"],
            "encoder_revision": "1a640f32ac3e39899438a2931f9924c02f080a54",
            "resnet_pretrained": False, "cohort_tasks": ["06", "07"],
            "within_patient_distinct_takes": "retain"}


def refresh(args, status=None, pdf=False):
    from s02_report import write_report, export_report
    write_report(args.output_dir, report_path=args.report_path, status=status)
    if args.report_path:
        write_report(args.output_dir, status=status)
    if pdf and args.export_pdf:
        export_report(args.output_dir, status=status)
        if args.report_path:
            export_report(args.output_dir, report_path=args.report_path,
                          pdf_path=args.report_path.with_suffix(".pdf"), status=status)


def prepare_study(args):
    from s02_data import prepare_cohort
    if args.baseline_manifest:
        prepare_cohort(args.baseline_manifest, args.full_data_root, args.output_dir,
                       seed=args.seed, folds=args.folds, old_quarantine=args.quarantine_manifest)
    manifest = args.manifest or args.output_dir / "split_manifest.csv"
    if not manifest.is_file():
        raise ValueError("Provide the shared assigned --manifest or build it with --baseline-manifest and --full-data-root.")
    records = load_frozen_manifest(manifest, args.quarantine_manifest)
    snap = fingerprint(records)
    folds = make_development_folds(records, args.folds, args.seed)
    args.manifest = manifest.resolve()
    config = {"experiment": "s02", "parameters": settings(args), "source_hashes": source_hashes(),
              "manifest_sha256": snap["sha256"], "folds_sha256": digest(folds),
              "protocol": "Shared patient development folds for ResNet18 and XLS-R A/B/C; freeze both before final test.",
              "known_holdout_caveat": "Old NM/AD holdout outcomes from s00/s01 are already known; this is exploratory reuse.",
              "source_confounding": "Existing NM/AD and new MCI come from different cohorts; task matching and resampling do not establish domain invariance."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Frozen s02 data, folds, training source, or settings changed; use a fresh study.")
    for name, value in (("manifest_fingerprint.json", snap), ("folds.json", folds)):
        target = args.output_dir / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError("Existing cohort fingerprint or fold membership differs; refusing overwrite.")
        if not target.exists():
            write_json(target, value)
    if not path.exists():
        write_json(path, config)
    write_json(args.output_dir / "summary.json", summary(records))
    from s01_audit import domain_audit
    audit = domain_audit(records)
    if audit.get("header_errors"):
        raise ValueError("Source audio header audit failed.")
    write_json(args.output_dir / "domain_audit.json", audit)
    return records, folds, snap, config


def assert_unchanged(args, records, snap):
    if fingerprint(records)["sha256"] != snap["sha256"]:
        raise ValueError("Selected source recordings changed during s02; results cannot be published.")
    if (args.output_dir / "cohort_config.json").exists():
        from s02_data import verify_cohort
        verify_cohort(args.output_dir)


def ensure_runtime(args):
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    runtime = {"device": device, "python": sys.version.split()[0],
               "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
               "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               "versions": {name: version(name) for name in
                            ("torch", "torchvision", "numpy", "librosa", "transformers", "scipy", "soundfile")}}
    if device == "cuda":
        runtime.update(gpu=torch.cuda.get_device_name(), cuda_runtime=torch.version.cuda,
                       capability=list(torch.cuda.get_device_capability()))
    target = args.output_dir / "execution_environment.json"
    if target.exists() and json.loads(target.read_text()) != runtime:
        raise ValueError("The frozen s02 runtime, software, or GPU changed; use the original environment.")
    if not target.exists():
        write_json(target, runtime)


def mean_std(folds):
    keys = ("macro_f1", "balanced_accuracy", "macro_ovr_auc", "accuracy", "loss")
    means, stds = {}, {}
    for key in keys:
        values = [item["patient"][key] for item in folds]
        means[key] = statistics.mean(values) if all(x is not None for x in values) else None
        stds[key] = statistics.stdev(values) if means[key] is not None and len(values) > 1 else 0.0 if means[key] is not None else None
    return {"folds": folds, "mean": means, "std": stds,
            "std_definition": "sample standard deviation across matched development folds"}


def collect(args):
    cv, metrics = {}, {}
    for method in ("resnet18", "xlsr_mil"):
        directory = args.output_dir / method
        for filename in ("cv_summary.json", "cv_progress.json"):
            if (directory / filename).exists():
                cv[method] = json.loads((directory / filename).read_text())
                break
        if (directory / "metrics.json").exists():
            metrics[method] = json.loads((directory / "metrics.json").read_text())
    if cv:
        write_json(args.output_dir / "cv_summary.json", cv)
    if metrics:
        if "xlsr_mil" in metrics:
            metrics["selected_variant"] = metrics["xlsr_mil"].get("selected_variant")
        write_json(args.output_dir / "metrics.json", metrics)
    return cv, metrics


def resnet_args(args, phase=None):
    result = argparse.Namespace(device=args.device, workers=args.workers, torch_threads=args.torch_threads,
                                epochs=args.resnet_epochs, patience=args.resnet_patience,
                                batch_size=args.resnet_batch_size, learning_rate=args.resnet_learning_rate,
                                weight_decay=args.resnet_weight_decay, clip_seconds=args.clip_seconds,
                                eval_windows=args.eval_windows, pretrained=False,
                                feature_cache_dir=args.output_dir / "resnet18/features")
    if phase:
        def report_callback(directory, **state):
            if phase == "training" and (directory / "history.csv").exists():
                shutil.copy2(directory / "history.csv", args.output_dir / "resnet18/history.csv")
            collect(args)
            refresh(args, phase)
        result.report_callback = report_callback
    return result


def run_resnet_cv(args, records, folds, snap, config):
    from s02_resnet import fit_resnet
    directory = args.output_dir / "resnet18"
    if (directory / "cv_summary.json").exists():
        return
    results = []
    for fold in folds:
        destination = directory / "cv" / f"fold_{fold['fold']:02d}"
        train = [r for r in records if r.patient_id in set(fold["train_ids"])]
        val = [r for r in records if r.patient_id in set(fold["val_ids"])]
        refresh(args, "cv")
        if (destination / "metrics.json").exists():
            metric = json.loads((destination / "metrics.json").read_text())
        else:
            print(f"Starting s02 ResNet development fold {fold['fold']}/{len(folds)}", flush=True)
            model, metric = fit_resnet(train, val, resnet_args(args, "cv"), destination,
                                      {"study": config, "fold": fold["fold"], "run_name": f"s02 ResNet fold {fold['fold']}",
                                       "train_patient_ids": fold["train_ids"], "validation_patient_ids": fold["val_ids"]},
                                      args.seed + fold["fold"])
            del model
        results.append({"fold": fold["fold"], "best_epoch": metric["best_epoch"], **metric["validation"]})
        write_json(directory / "cv_progress.json", {"folds": results})
        collect(args)
    assert_unchanged(args, records, snap)
    write_json(directory / "cv_summary.json", mean_std(results))
    collect(args)


def xlsr_command(args, stage):
    command = [sys.executable, "-u", str(REPO / "s01/code/train.py"),
               "--manifest", str(args.manifest), "--output-dir", str(args.output_dir / "xlsr_mil"),
               "--stage", stage, "--device", args.device, "--workers", str(args.workers),
               "--torch-threads", str(args.torch_threads), "--seed", str(args.seed),
               "--folds", str(args.folds), "--epochs", str(args.xlsr_epochs),
               "--patience", str(args.xlsr_patience), "--batch-size", str(args.xlsr_batch_size),
               "--learning-rate", str(args.xlsr_learning_rate), "--weight-decay", str(args.xlsr_weight_decay),
               "--clip-seconds", str(float(args.clip_seconds)), "--eval-windows", str(args.eval_windows),
               "--encoder-batch-size", str(args.encoder_batch_size), "--encoder-cache-dir", str(args.encoder_cache_dir),
               "--variants", "A,B,C", "--cutoff-min", "4000.0", "--cutoff-max", "7500.0",
               "--snr-min", "20.0", "--snr-max", "35.0"]
    if args.local_files_only:
        command.append("--local-files-only")
    if args.export_pdf:
        command.append("--export-pdf")
    if args.quarantine_manifest:
        command.extend(["--quarantine-manifest", str(args.quarantine_manifest)])
    if stage == "final" and (args.output_dir / "resnet18/metrics.json").exists():
        command.extend(["--s00-metrics", str(args.output_dir / "resnet18/metrics.json")])
    return command


def run_xlsr(args, stage, folds):
    output = args.output_dir / "xlsr_mil"
    if stage == "cv" and (output / "cv_summary.json").exists():
        return
    if stage == "final" and (output / "test_evaluation.json").exists():
        completed = json.loads((output / "test_evaluation.json").read_text())
        if completed.get("status") == "completed" and (output / "metrics.json").exists():
            return
        raise ValueError("XLS-R test evaluation already started; automatic repetition is forbidden.")
    refresh(args, "cv" if stage == "cv" else "training")
    subprocess.run(xlsr_command(args, stage), check=True, cwd=REPO)
    if json.loads((output / "folds.json").read_text()) != folds:
        raise ValueError("XLS-R patient folds differ from the shared s02 folds.")
    collect(args)


def freeze_selection(args, config):
    cv, _ = collect(args)
    if not all((args.output_dir / name / "cv_summary.json").exists() for name in ("resnet18", "xlsr_mil")):
        raise ValueError("Both matched development comparisons must finish before either final test.")
    if len(cv["resnet18"].get("folds", [])) != args.folds:
        raise ValueError("ResNet development folds are incomplete.")
    for variant in ("A", "B", "C"):
        if len(cv["xlsr_mil"]["variants"][variant]["folds"]) != args.folds:
            raise ValueError("XLS-R variant development folds are incomplete.")
    value = {"experiment": "s02", "study_sha256": digest(config), "cv_summary_sha256": digest(cv),
             "resnet_architecture": "single-channel ResNet18, random initialization",
             "selected_xlsr_variant": cv["xlsr_mil"]["selected_variant"],
             "selection_rule": "XLS-R: mean CV patient macro F1; ties lower mean loss then A/B/C"}
    value["selection_sha256"] = digest(value)
    target = args.output_dir / "selection.json"
    if target.exists() and json.loads(target.read_text()) != value:
        raise ValueError("The frozen paired selection differs from current data/code/config/CV.")
    if not target.exists():
        write_json(target, value)
    return value


def run_resnet_final(args, records, snap, config, selection):
    from s02_resnet import evaluate_records, fit_resnet, write_predictions
    directory = args.output_dir / "resnet18"
    final = directory / "final"
    guard = directory / "test_evaluation.json"
    if guard.exists():
        completed = json.loads(guard.read_text())
        if completed.get("status") == "completed" and (directory / "metrics.json").exists():
            return
        raise ValueError("ResNet original-test evaluation already started; automatic repetition is forbidden.")
    train, val, test = ([r for r in records if r.split == split] for split in ("train", "val", "test"))
    refresh(args, "training")
    model, metrics = fit_resnet(train, val, resnet_args(args, "training"), final,
                               {"study": config, "selection": selection, "run_name": "s02 ResNet final"}, args.seed)
    assert_unchanged(args, records, snap)
    with guard.open("x", encoding="utf-8") as handle:
        json.dump({"status": "started", "selection_sha256": selection["selection_sha256"]}, handle)
    result = evaluate_records(model, test, resnet_args(args), directory / "features")
    assert_unchanged(args, records, snap)
    write_predictions(final, "test", test, result)
    metrics.update(test=result[0], original_test_evaluated=True,
                   selection_sha256=selection["selection_sha256"])
    write_json(final / "metrics.json", metrics)
    write_json(directory / "metrics.json", metrics)
    for name in ("history.csv", "best.pt"):
        shutil.copy2(final / name, directory / name)
    write_json(guard, {"status": "completed", "evaluations": 1,
                       "test_patient_count": len({r.patient_id for r in test}),
                       "selection_sha256": selection["selection_sha256"]})
    del model
    collect(args)


def synthetic_smoke(args):
    import torch
    sys.path.insert(0, str(REPO / "s00/code"))
    from resnet18_baseline import build_model
    from s01_training import seed_everything
    seed_everything(args.seed)
    torch.set_num_threads(2)
    model = build_model(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(3, 1, 40, 501)), torch.tensor([0, 1, 2]))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    spec = importlib.util.spec_from_file_location("s02_synthetic_mil", REPO / "s01/code/train.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.synthetic_smoke(argparse.Namespace(seed=args.seed, output_dir=args.output_dir / "mil"))
    write_json(args.output_dir / "smoke_check.json", {"clinical_training": False, "device": "cpu",
                                                      "resnet_finite_optimization": True, "mil_smoke_passed": True})
    print("s02 synthetic CPU smoke passed; no clinical data or pretrained encoder.", flush=True)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        validate_args(args)
        os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".cache/matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir / ".cache"))
        os.environ.setdefault("NUMBA_CACHE_DIR", str(args.output_dir / ".cache/numba"))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        if args.report_only:
            collect(args)
            refresh(args, pdf=True)
            return 0
        if args.smoke_test:
            synthetic_smoke(args)
            return 0
        records, folds, snap, config = prepare_study(args)
        if args.prepare_only:
            refresh(args, "prepared", pdf=True)
            print("Prepared the shared s02 cohort and patient folds; no clinical model inference.", flush=True)
            return 0
        ensure_runtime(args)
        if args.stage in {"cv", "all"}:
            if args.system in {"both", "resnet18"}:
                run_resnet_cv(args, records, folds, snap, config)
            if args.system in {"both", "xlsr_mil"}:
                run_xlsr(args, "cv", folds)
            collect(args)
        if all((args.output_dir / name / "cv_summary.json").exists() for name in ("resnet18", "xlsr_mil")):
            selection = freeze_selection(args, config)
        elif args.stage in {"all", "final"}:
            raise ValueError("Final evaluation requires both completed shared development comparisons.")
        if args.stage in {"all", "final"}:
            assert_unchanged(args, records, snap)
            if args.system in {"both", "resnet18"}:
                run_resnet_final(args, records, snap, config, selection)
            if args.system in {"both", "xlsr_mil"}:
                run_xlsr(args, "final", folds)
            assert_unchanged(args, records, snap)
        collect(args)
        refresh(args, "completed", pdf=True)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        try:
            collect(args)
            refresh(args, "failed")
        except Exception:
            pass
        p.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
