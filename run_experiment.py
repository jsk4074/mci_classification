#!/usr/bin/env python3
"""Run a numbered experiment using its saved configuration and source code."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
RUNTIME_OPTIONS = {
    "manifest", "data_root", "output_dir", "report_path", "device", "workers",
    "torch_threads", "prepare_only", "report_only",
}


def load_experiment(experiment_id: str) -> tuple[Path, list[str]]:
    """Return the entrypoint and fixed arguments from experiment.json.

    Schema: {"schema_version": 1, "entrypoint": "code/example.py",
    "parameters": {"epochs": 50, "learning_rate": 0.001,
    "pretrained": false}}. The entrypoint is relative to its experiment
    directory. Parameter keys become --hyphenated-options; scalar values
    become arguments, true booleans become flags, and false booleans are
    omitted. Runtime/path options are reserved for the launcher CLI. New
    numbered experiments can choose their own entrypoint and parameters.
    """

    if not re.fullmatch(r"s[0-9]{2,}", experiment_id):
        raise ValueError("Experiment ID must match s[0-9]{2,}, for example s00 or s01.")
    experiment_dir = (REPOSITORY_ROOT / experiment_id).resolve()
    if not experiment_dir.is_relative_to(REPOSITORY_ROOT.resolve()) or not experiment_dir.is_dir():
        raise ValueError("Experiment directory must exist inside this repository.")
    config_path = (experiment_dir / "experiment.json").resolve()
    if not config_path.is_relative_to(experiment_dir) or not config_path.is_file():
        raise ValueError("experiment.json must be a regular file inside the selected experiment.")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("experiment.json must contain valid UTF-8 JSON.") from exc
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("experiment.json must use schema_version 1.")
    relative = config.get("entrypoint")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("The experiment entrypoint must be a relative file path.")
    entrypoint = (experiment_dir / relative).resolve()
    if not entrypoint.is_relative_to(experiment_dir) or not entrypoint.is_file():
        raise ValueError("The experiment entrypoint must be a regular file inside its experiment directory.")
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("experiment.json parameters must be an object.")
    arguments = []
    for key, value in parameters.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key in RUNTIME_OPTIONS:
            raise ValueError("Experiment parameters must use option names and cannot override launcher runtime/path options.")
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(option)
        elif isinstance(value, (str, int, float)) and not (isinstance(value, float) and not math.isfinite(value)):
            arguments.extend([option, str(value)])
        else:
            raise ValueError("Experiment parameter values must be strings, finite numbers, or booleans.")
    return entrypoint, arguments


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("experiment", help="Experiment directory, for example s00")
    result.add_argument("--manifest", type=Path, help="Existing split manifest; required for training and preparation")
    result.add_argument("--data-root", type=Path, help="Audio root; defaults to the manifest directory")
    result.add_argument("--output-dir", type=Path, help="Run artifacts; defaults to runs/<experiment> in this repository")
    result.add_argument("--report-path", type=Path, help="Optional additional HTML report destination")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--torch-threads", type=int, default=16)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    return result


def main(argv=None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.workers < 0 or args.torch_threads < 0:
            raise ValueError("Workers and CPU thread count must be nonnegative.")
        entrypoint, fixed_arguments = load_experiment(args.experiment)
        manifest = args.manifest.expanduser().resolve() if args.manifest is not None else None
        if not args.report_only and (manifest is None or not manifest.is_file()):
            raise ValueError("--manifest must identify an existing regular file for training or --prepare-only.")
        output_dir = (args.output_dir.expanduser().resolve() if args.output_dir is not None
                      else (REPOSITORY_ROOT / "runs" / args.experiment).resolve())
        data_root = (args.data_root.expanduser().resolve() if args.data_root is not None
                     else manifest.parent if manifest is not None
                     else (REPOSITORY_ROOT / "dataset" / args.experiment).resolve())
        command = [sys.executable, "-u", str(entrypoint), *fixed_arguments,
                   "--data-root", str(data_root), "--output-dir", str(output_dir),
                   "--device", args.device, "--workers", str(args.workers),
                   "--torch-threads", str(args.torch_threads)]
        if manifest is not None:
            command.extend(["--manifest", str(manifest)])
        if args.report_path is not None:
            command.extend(["--report-path", str(args.report_path.expanduser().resolve())])
        if args.prepare_only:
            command.append("--prepare-only")
        if args.report_only:
            command.append("--report-only")
        cache_dir = output_dir / "numba_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(NUMBA_CACHE_DIR=str(cache_dir), OPENBLAS_NUM_THREADS="1")
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=False)
    except (ValueError, OSError) as exc:
        argument_parser.error(str(exc))
    if completed.returncode:
        print(f"Experiment {args.experiment} exited with status {completed.returncode}.", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
