"""Encoder identity is frozen across stages after JSON key normalization."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
spec = importlib.util.spec_from_file_location("s01_provenance_train", CODE_DIR / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


@pytest.fixture
def encoder_setup(tmp_path, monkeypatch):
    import s01_encoder

    identity = {
        "model_name": s01_encoder.MODEL_NAME,
        "resolved_revision": s01_encoder.PINNED_REVISION,
        "hidden_size": 4,
        "model_config": {"id2label": {0: "NM", 1: "AD", 2: "MCI"}},
        "software": {"transformers": "synthetic-version-1"},
        "device_type": "cpu",
    }
    instances = []

    class FakeEncoder:
        hidden_size, batch_size = 4, 4

        def __init__(self, *args, **kwargs):
            self.identity = deepcopy(identity)
            instances.append(self)

        def encode_windows(self, *args, **kwargs):
            pytest.fail("Provenance validation must not run encoder inference.")

    monkeypatch.setattr(s01_encoder, "FrozenEncoder", FakeEncoder)
    args = SimpleNamespace(
        output_dir=tmp_path / "run", clip_seconds=5, eval_windows=5,
        augmentation_variants=1, seed=42, augmentation_version="channel-v1",
        cutoff_min=4000, cutoff_max=7500, snr_min=20, snr_max=35,
        encoder_name=s01_encoder.MODEL_NAME,
        encoder_revision=s01_encoder.PINNED_REVISION,
        device="cpu", encoder_batch_size=4,
        encoder_cache_dir=tmp_path / "hf-cache", local_files_only=True,
    )
    return args, identity, instances


def snapshot(path):
    stat = path.stat()
    return path.read_bytes(), stat.st_ino, stat.st_mtime_ns


def test_unchanged_encoder_identity_with_integer_label_keys_is_reusable(encoder_setup):
    args, _, instances = encoder_setup
    train.build_embedder(args)
    metadata = args.output_dir / "encoder_metadata.json"
    original = snapshot(metadata)
    assert json.loads(original[0])["model_config"]["id2label"] == {"0": "NM", "1": "AD", "2": "MCI"}
    assert all(type(key) is int for key in instances[0].identity["model_config"]["id2label"])

    train.build_embedder(args)

    assert len(instances) == 2
    assert snapshot(metadata) == original
    assert not (args.encoder_cache_dir).exists()  # The fake never loads/downloads weights.


@pytest.mark.parametrize("change", ["software", "backend"])
def test_changed_encoder_identity_is_rejected_without_overwriting_metadata(encoder_setup, change):
    args, identity, _ = encoder_setup
    train.build_embedder(args)
    metadata = args.output_dir / "encoder_metadata.json"
    original = snapshot(metadata)
    if change == "software":
        identity["software"]["transformers"] = "synthetic-version-2"
    else:
        identity["device_type"] = "cuda"

    with pytest.raises(ValueError, match="encoder identity"):
        train.build_embedder(args)

    assert snapshot(metadata) == original
