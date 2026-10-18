"""Synthetic waveform, frozen-encoder, and cache checks; no weight downloads."""

from dataclasses import replace
import importlib.machinery
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from s01_audio import AudioConfig, AugmentationConfig, augment_waveform, load_audio, make_windows, pad_normalize, window_starts
import s01_encoder as encoder_module
from s01_encoder import EmbeddingCache, FrozenEncoder, PINNED_REVISION, RecordingEmbedder, convolution_output_lengths


def test_short_audio_normalizes_valid_samples_then_zero_pads_with_explicit_mask():
    config = AudioConfig(clip_seconds=0.001)
    raw = np.array([2, 4, 6, 8], dtype=np.float32)
    windows = make_windows(raw, config)
    assert windows[0].shape == (4,)  # Padding is deferred until batching.
    values, mask = pad_normalize([*windows, np.zeros(3, dtype=np.float32)], config)
    expected = (raw - raw.mean()) / np.sqrt(raw.var() + 1e-7)
    np.testing.assert_allclose(values[0, :4], expected, atol=1e-6)
    assert values.dtype == np.float32 and mask.dtype == np.int64
    np.testing.assert_array_equal(mask.sum(axis=1), [4, 3])
    assert np.count_nonzero(values[0, 4:]) == 0
    assert np.count_nonzero(values[1]) == 0  # Valid silence remains masked in, unlike padding.
    np.testing.assert_array_equal(mask[1, :4], [1, 1, 1, 0])
    np.testing.assert_array_equal(raw, [2, 4, 6, 8])


def test_window_cap_deterministically_spans_long_audio_and_exposes_possible_gaps():
    config = AudioConfig(clip_seconds=0.002, max_windows=5)
    waveform = np.arange(320, dtype=np.float32)
    assert window_starts(len(waveform), config) == [0, 72, 144, 216, 288]
    windows = make_windows(waveform, config)
    assert all(window.shape == (32,) for window in windows)
    assert [int(window[0]) for window in windows] == [0, 72, 144, 216, 288]
    assert window_starts(320, replace(config, max_windows=1)) == [144]
    assert window_starts(5, config) == [0]


def test_stereo_audio_is_averaged_and_resampled_without_changing_source(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    from scipy.signal import resample_poly
    tone = np.sin(np.arange(800, dtype=np.float32) * (2 * np.pi * 200 / 8000))
    source = tmp_path / "synthetic.wav"
    soundfile.write(source, np.column_stack([tone * 0.2, tone * 0.6]), 8000, subtype="FLOAT")
    original = source.read_bytes()
    result = load_audio(source)
    np.testing.assert_allclose(result, resample_poly(tone * 0.4, 2, 1), atol=1e-6)
    assert result.shape == (1600,) and result.dtype == np.float32
    assert source.read_bytes() == original


def test_channel_augmentation_is_repeatable_and_never_adds_noise_to_padding():
    waveform = np.sin(np.arange(1200, dtype=np.float32) * (2 * np.pi * 220 / 16000))
    config = AugmentationConfig(enabled=True)
    first = augment_waveform(waveform, config, 1, "synthetic-content-hash")
    np.testing.assert_array_equal(first, augment_waveform(waveform, config, 1, "synthetic-content-hash"))
    assert not np.array_equal(first, waveform)
    assert not np.array_equal(first, augment_waveform(waveform, config, 2, "synthetic-content-hash"))
    assert not np.array_equal(first, augment_waveform(waveform, replace(config, seed=43), 1, "synthetic-content-hash"))
    np.testing.assert_array_equal(augment_waveform(waveform, config, 0, "ignored"), waveform)
    values, mask = pad_normalize([first], AudioConfig(clip_seconds=0.1))
    assert first.shape == waveform.shape and np.isfinite(first).all()
    assert np.count_nonzero(values[0, len(first):]) == 0
    assert np.count_nonzero(mask[0, len(first):]) == 0
    with pytest.raises(ValueError, match="enabled=True"):
        augment_waveform(waveform, AugmentationConfig(), 1, "synthetic")


@pytest.mark.parametrize("waveform", [[], [1, np.nan], [[1, 2], [3, 4]]])
def test_invalid_waveforms_are_rejected(waveform):
    with pytest.raises(ValueError, match="nonempty, finite, mono"):
        make_windows(waveform)


def test_default_convolution_lengths_count_only_fully_valid_frames():
    kernels, strides = (10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)
    np.testing.assert_array_equal(convolution_output_lengths([80000, 400, 399, 0], kernels, strides), [249, 1, 0, 0])


def cache_at(tmp_path, *, model=None, audio=None, version="channel-v1"):
    return EmbeddingCache(tmp_path, model or {"hidden_size": 3, "resolved_revision": PINNED_REVISION}, audio or AudioConfig(), version)


@pytest.mark.parametrize("changes", [
    {"enabled": False}, {"seed": 43}, {"cutoff_min": 4500},
    {"cutoff_max": 7000}, {"snr_min": 22}, {"snr_max": 40},
])
def test_every_augmentation_field_changes_the_embedding_cache_identity(tmp_path, changes):
    cache = cache_at(tmp_path)
    source = {"sha256": "a" * 64, "bytes": 100}
    augmentation = AugmentationConfig(enabled=True)
    baseline, _ = cache.request(source, augmentation, 0)
    changed, _ = cache.request(source, replace(augmentation, **changes), 0)
    assert changed != baseline
    augmented, _ = cache.request(source, augmentation, 1)
    assert augmented != baseline


def test_content_revision_audio_and_augmentation_version_are_cache_inputs(tmp_path):
    source = {"sha256": "a" * 64, "bytes": 100}
    baseline, _ = cache_at(tmp_path).request(source, AugmentationConfig(), 0)
    variants = [
        cache_at(tmp_path).request({"sha256": "b" * 64, "bytes": 100}, AugmentationConfig(), 0)[0],
        cache_at(tmp_path, model={"hidden_size": 3, "resolved_revision": "b" * 40}).request(source, AugmentationConfig(), 0)[0],
        cache_at(tmp_path, audio=AudioConfig(clip_seconds=4)).request(source, AugmentationConfig(), 0)[0],
        cache_at(tmp_path, version="channel-v2").request(source, AugmentationConfig(version="channel-v2"), 0)[0],
    ]
    assert all(key != baseline for key in variants)


def test_cache_roundtrip_is_atomic_and_validates_metadata_and_finite_shape(tmp_path):
    cache = cache_at(tmp_path)
    key, metadata = cache.request({"sha256": "a" * 64, "bytes": 100}, AugmentationConfig(), 0)
    assert cache.load(key, metadata) is None
    value = np.array([0.2, -0.3, 0.4], dtype=np.float32)
    cache.store(key, metadata, value)
    np.testing.assert_array_equal(cache.load(key, metadata), value)
    assert (cache.hits, cache.misses, cache.writes) == (1, 1, 1)
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".npz"]
    with pytest.raises(ValueError, match="nonfinite"):
        cache.store(key, metadata, np.array([0, 1, np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        cache.store(key, metadata, np.ones(2, dtype=np.float32))
    with (tmp_path / f"{key}.npz").open("wb") as handle:
        np.savez(handle, embedding=value, metadata=np.asarray(json.dumps({"wrong": "metadata"})))
    with pytest.raises(ValueError, match="Invalid embedding cache"):
        cache.load(key, metadata)


@pytest.fixture
def tiny_loader(monkeypatch):
    torch = pytest.importorskip("torch")

    class TinyConfig:
        _commit_hash = PINNED_REVISION
        hidden_size = 2
        conv_kernel, conv_stride = (3,), (2,)
        feat_extract_norm, add_adapter = "layer", False
        _attn_implementation = "synthetic"

        def to_dict(self):
            return {"hidden_size": self.hidden_size, "conv_kernel": [3], "conv_stride": [2], "feat_extract_norm": "layer", "add_adapter": False}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TinyConfig()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []

        def forward(self, input_values, attention_mask, return_dict):
            assert not self.training and not torch.is_grad_enabled() and return_dict
            assert input_values.dtype == torch.float32
            self.calls.append((input_values.clone(), attention_mask.clone()))
            frames = (input_values.shape[1] - 3) // 2 + 1
            positions = torch.arange(frames, device=input_values.device, dtype=torch.float32)
            hidden = torch.stack((positions, positions + 10), dim=-1)[None].repeat(len(input_values), 1, 1)
            lengths = (attention_mask.sum(dim=1) - 3) // 2 + 1
            valid = torch.arange(frames, device=hidden.device)[None] < lengths[:, None]
            hidden[~valid] = float("nan")  # Invalid padding must never enter pooling.
            return SimpleNamespace(last_hidden_state=hidden)

    state = SimpleNamespace(model=TinyModel(), loading={"missing_keys": [], "mismatched_keys": [], "error_msgs": []}, requests=[])

    class TinyFactory:
        @staticmethod
        def from_pretrained(name, **kwargs):
            state.requests.append((name, kwargs))
            return state.model, state.loading

    module = ModuleType("transformers")
    module.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    module.__version__ = "4.57.synthetic-test"
    module.Wav2Vec2Model = TinyFactory
    monkeypatch.setitem(sys.modules, "transformers", module)
    return state


def test_frozen_tiny_encoder_excludes_padded_frames_and_records_verified_identity(tmp_path, tiny_loader):
    encoder = FrozenEncoder(cache_dir=tmp_path / "hf", batch_size=2)
    assert not encoder.model.training and not any(parameter.requires_grad for parameter in encoder.model.parameters())
    values = encoder.encode_windows([np.arange(8, dtype=np.float32), np.arange(16, dtype=np.float32)], AudioConfig(clip_seconds=0.001))
    np.testing.assert_array_equal(values, [[1, 11], [3, 13]])
    assert values.dtype == np.float32 and np.isfinite(values).all()
    inputs, mask = tiny_loader.model.calls[0]
    assert inputs[0, 8:].count_nonzero().item() == 0
    assert mask.sum(dim=1).tolist() == [8, 16]
    assert encoder.identity["resolved_revision"] == PINNED_REVISION
    assert encoder.identity["model_class"] == "Wav2Vec2Model"
    assert "torch" in encoder.identity["software"]
    assert tiny_loader.requests[0][1]["revision"] == PINNED_REVISION
    with pytest.raises(ValueError, match="receptive field"):
        encoder.encode_windows([np.ones(2, dtype=np.float32)], AudioConfig(clip_seconds=0.001))


def test_unpinned_or_incomplete_encoder_cannot_fall_back_to_random_weights(tmp_path, tiny_loader):
    with pytest.raises(ValueError, match="pinned"):
        FrozenEncoder(revision="main", cache_dir=tmp_path)
    assert tiny_loader.requests == []
    tiny_loader.loading["missing_keys"] = ["encoder.some_weight"]
    with pytest.raises(RuntimeError, match="refusing random"):
        FrozenEncoder(cache_dir=tmp_path)
    tiny_loader.loading["missing_keys"] = []
    tiny_loader.model.config._commit_hash = "b" * 40
    with pytest.raises(RuntimeError, match="pinned commit"):
        FrozenEncoder(cache_dir=tmp_path)


def test_embedder_batches_recordings_preserves_order_and_reuses_content_cache(tmp_path, monkeypatch):
    soundfile = pytest.importorskip("soundfile")

    class FakeEncoder:
        hidden_size, batch_size = 2, 4
        identity = {"hidden_size": 2, "resolved_revision": PINNED_REVISION}

        def __init__(self):
            self.calls = []

        def encode_windows(self, windows, config):
            self.calls.append([len(window) for window in windows])
            return np.asarray([[window.mean(), len(window)] for window in windows], dtype=np.float32)

    paths = []
    for index, length in enumerate((64, 32, 16)):
        path = tmp_path / f"synthetic-{index}.wav"
        soundfile.write(path, np.linspace(0, 0.1 * (index + 1), length, dtype=np.float32), 16000, subtype="FLOAT")
        paths.append(path)
    audio, augmentation = AudioConfig(clip_seconds=0.002), AugmentationConfig(enabled=True)
    encoder = FakeEncoder()
    cache = EmbeddingCache(tmp_path / "cache", encoder.identity, audio)
    embedder = RecordingEmbedder(encoder, cache, audio, augmentation)
    progress = []
    result = embedder.embed_records([SimpleNamespace(path=path) for path in paths], progress=lambda done, total: progress.append((done, total)))
    assert encoder.calls == [[32, 32, 32, 16]]  # Windows from all three recordings share one call.
    np.testing.assert_allclose(result[:, 0], [0.05, 0.1, 0.15], atol=1e-7)
    np.testing.assert_array_equal(result[:, 1], [32, 32, 16])
    assert progress == [(3, 3)]
    monkeypatch.setattr(encoder_module, "load_audio", lambda *args: pytest.fail("A cache hit decoded audio."))
    repeated = embedder.embed_records(list(reversed(paths)))
    np.testing.assert_array_equal(repeated, result[::-1])
    assert len(encoder.calls) == 1 and embedder.stats["cache_hits"] == 3
