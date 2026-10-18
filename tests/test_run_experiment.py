"""Check experiment routing without opening audio or importing training code."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("published_launcher", ROOT / "run_experiment.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    directory = repository / "s00"
    entrypoint = directory / "code" / "resnet18_baseline.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# A mocked child: never executed.\n")
    config = json.loads((ROOT / "s00" / "experiment.json").read_text())
    config_path = directory / "experiment.json"
    config_path.write_text(json.dumps(config))
    manifest = tmp_path / "inputs" / "split_manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("path,patient_id,label,split\n")
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0))
    return repository, config_path, manifest, calls


def option(command, name):
    return command[command.index(name) + 1]


def test_routes_absolute_paths_and_frozen_s00_parameters(experiment, monkeypatch):
    repository, _, manifest, calls = experiment
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "37")
    assert launcher.main(["s00", "--manifest", "inputs/split_manifest.csv", "--report-path", "reports/summary.html", "--device", "cuda"]) == 0
    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, "-u", str(repository / "s00/code/resnet18_baseline.py")]
    assert option(command, "--manifest") == str(manifest)
    assert option(command, "--data-root") == str(manifest.parent)
    assert option(command, "--output-dir") == str(repository / "runs/s00")
    assert Path(option(command, "--report-path")).is_absolute()
    expected = {"seed": "42", "epochs": "50", "patience": "10", "batch-size": "16", "learning-rate": "0.001", "weight-decay": "0.0001", "clip-seconds": "5", "eval-windows": "5", "device": "cuda", "workers": "0", "torch-threads": "16"}
    assert all(option(command, "--" + key) == value for key, value in expected.items())
    assert "--pretrained" not in command
    assert kwargs["cwd"] == repository and kwargs["check"] is False
    assert kwargs["env"]["NUMBA_CACHE_DIR"] == str(repository / "runs/s00/numba_cache")
    assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert launcher.os.environ["OPENBLAS_NUM_THREADS"] == "37"


def test_report_only_needs_no_manifest_and_routes_custom_runtime_paths(experiment):
    repository, _, _, calls = experiment
    assert launcher.main(["s00", "--report-only", "--output-dir", "saved/run", "--data-root", "audio", "--workers", "2", "--torch-threads", "4"]) == 0
    command, kwargs = calls[0]
    assert "--manifest" not in command and "--report-only" in command
    assert option(command, "--output-dir") == str(repository.parent / "saved/run")
    assert option(command, "--data-root") == str(repository.parent / "audio")
    assert option(command, "--workers") == "2"
    assert option(command, "--torch-threads") == "4"
    assert kwargs["env"]["NUMBA_CACHE_DIR"] == str(repository.parent / "saved/run/numba_cache")


@pytest.mark.parametrize("arguments", [
    ["../s00", "--report-only"], ["s0", "--report-only"],
    ["s00"], ["s00", "--prepare-only"],
    ["s00", "--manifest", "inputs"], ["s00", "--manifest", "missing.csv"],
    ["s00", "--report-only", "--prepare-only"],
    ["s00", "--report-only", "--workers", "-1"],
    ["s00", "--report-only", "--epochs", "1"],
])
def test_invalid_requests_never_launch(experiment, arguments):
    repository, _, _, calls = experiment
    with pytest.raises(SystemExit) as error:
        launcher.main(arguments)
    assert error.value.code == 2 and calls == []
    assert not (repository / "runs").exists()


@pytest.mark.parametrize("escape", ["entrypoint", "config", "directory"])
def test_experiment_paths_cannot_escape(experiment, escape):
    repository, config_path, _, calls = experiment
    outside = repository.parent / "outside.py"
    outside.write_text("# outside\n")
    if escape == "entrypoint":
        config = json.loads(config_path.read_text())
        config["entrypoint"] = "../outside.py"
        (repository / "outside.py").write_text("# outside experiment\n")
        config_path.write_text(json.dumps(config))
    elif escape == "config":
        copied = repository.parent / "external.json"
        copied.write_bytes(config_path.read_bytes())
        config_path.unlink()
        config_path.symlink_to(copied)
    else:
        (repository / "s01").symlink_to(repository.parent, target_is_directory=True)
    with pytest.raises(SystemExit) as error:
        launcher.main(["s01" if escape == "directory" else "s00", "--report-only"])
    assert error.value.code == 2 and calls == []


def test_child_failure_status_is_preserved(experiment, monkeypatch, capsys):
    _, _, manifest, _ = experiment
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=7))
    assert launcher.main(["s00", "--manifest", str(manifest), "--prepare-only"]) == 7
    assert "exited with status 7" in capsys.readouterr().err
