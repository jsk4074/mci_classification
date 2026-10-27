"""Frozen, pinned XLS-R embeddings with valid-frame pooling and atomic caches.

Transformers is imported only when FrozenEncoder is constructed. No model is
created or downloaded when this module is imported, during input preparation,
or while a report is regenerated.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import tempfile

import numpy as np

from s01_audio import (
    AudioConfig, AugmentationConfig, NORMALIZATION, augment_waveform,
    check_augmentation_index, load_audio, make_windows, pad_normalize,
)


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
PINNED_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
POOLING = "last-hidden-state-valid-conv-frame-mean_then-equal-window-mean-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@lru_cache(maxsize=1)
def _software() -> dict:
    values = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("torch", "transformers", "scipy", "soundfile"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "not-installed"
    return values


def convolution_output_lengths(sample_lengths, kernels, strides) -> np.ndarray:
    """Exact unpadded Conv1d lengths, independently of model private helpers."""

    lengths = np.asarray(sample_lengths, dtype=np.int64)
    if lengths.ndim != 1 or np.any(lengths < 0):
        raise ValueError("Sample lengths must be a vector of nonnegative integers.")
    if not kernels or len(kernels) != len(strides) or any(int(value) < 1 for value in (*kernels, *strides)):
        raise ValueError("Encoder convolution kernels and strides must be positive and aligned.")
    for kernel, stride in zip(kernels, strides):
        lengths = (lengths - int(kernel)) // int(stride) + 1
    return np.maximum(lengths, 0)


def masked_mean_pool(hidden_states, valid_frame_mask):
    """Pool valid encoder frames only, discarding even nonfinite padded tails."""

    if hidden_states.ndim != 3 or valid_frame_mask.shape != hidden_states.shape[:2]:
        raise ValueError("Hidden states and their encoder-frame mask have incompatible shapes.")
    mask = valid_frame_mask.bool()
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("A waveform window is shorter than the encoder's convolutional receptive field.")
    return hidden_states.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / counts.unsqueeze(-1)


class FrozenEncoder:
    """Load the pinned base Wav2Vec2Model; never create a random fallback."""

    def __init__(
        self, model_name: str = MODEL_NAME, revision: str = PINNED_REVISION,
        device: str = "cpu", batch_size: int = 4, cache_dir: Path | None = None,
        local_files_only: bool = False,
    ):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("The encoder revision must be a full pinned 40-character commit SHA.")
        if model_name != MODEL_NAME:
            raise ValueError(f"s01 requires the base encoder checkpoint {MODEL_NAME}.")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Encoder batch size must be a positive integer.")
        import torch
        import transformers
        from transformers import Wav2Vec2Model

        device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA encoder inference was requested but CUDA is unavailable.")
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir or PROJECT_ROOT / ".cache/huggingface").expanduser().resolve()
        try:
            model, loading = Wav2Vec2Model.from_pretrained(
                model_name, revision=revision.lower(), cache_dir=str(self.cache_dir),
                local_files_only=local_files_only, dtype=torch.float32,
                output_loading_info=True,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError("Cannot load the pinned XLS-R base checkpoint; no random or CTC fallback is permitted.") from exc
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("The pinned checkpoint has missing or incompatible base weights; refusing random initialization.")
        config = model.config
        resolved = getattr(config, "_commit_hash", None)
        if not isinstance(resolved, str) or resolved.lower() != revision.lower():
            raise RuntimeError("Loaded checkpoint metadata does not verify the requested pinned commit.")
        if getattr(config, "feat_extract_norm", None) != "layer" or getattr(config, "add_adapter", False):
            raise ValueError("s01 expects a layer-normalized XLS-R frontend without convolutional adapters.")
        self.hidden_size = int(config.hidden_size)
        self.kernels, self.strides = tuple(config.conv_kernel), tuple(config.conv_stride)
        convolution_output_lengths([80000], self.kernels, self.strides)
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.model.requires_grad_(False)
        self.model.eval()
        self.identity = {
            "model_name": model_name, "requested_revision": revision.lower(),
            "resolved_revision": resolved.lower(), "model_class": "Wav2Vec2Model",
            "hidden_size": self.hidden_size, "dtype": "float32",
            "model_config": config.to_dict(), "normalization": dict(NORMALIZATION),
            "pooling": POOLING, "attention_mask": "valid-raw-samples-and-exact-conv-frames",
            "batch_size": batch_size, "device_type": self.device.type,
            "attention_implementation": getattr(config, "_attn_implementation", None),
            "software": {**_software(), "torch": torch.__version__, "transformers": transformers.__version__},
            "cuda_runtime": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if self.device.type == "cuda":
            self.identity.update(
                cuda_capability=list(torch.cuda.get_device_capability(self.device)),
                cudnn_version=torch.backends.cudnn.version(),
                matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
            )
        # Validate that all identity fields can be represented in cache metadata.
        _json(self.identity)

    def encode_windows(self, windows: list[np.ndarray], config: AudioConfig = AudioConfig()) -> np.ndarray:
        """Return one float32 [hidden] embedding per valid raw waveform window."""

        import torch

        if not windows:
            return np.empty((0, self.hidden_size), dtype=np.float32)
        embeddings = []
        self.model.eval()
        for start in range(0, len(windows), self.batch_size):
            values, raw_mask = pad_normalize(windows[start:start + self.batch_size], config)
            valid_lengths = convolution_output_lengths(raw_mask.sum(axis=1), self.kernels, self.strides)
            if np.any(valid_lengths < 1):
                raise ValueError("A waveform window is shorter than the encoder's convolutional receptive field.")
            inputs = torch.from_numpy(values).to(self.device)
            attention = torch.from_numpy(raw_mask).to(self.device)
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
                output = self.model(input_values=inputs, attention_mask=attention, return_dict=True)
                hidden = output.last_hidden_state
                if hidden.ndim != 3 or hidden.shape[0] != len(values) or hidden.shape[2] != self.hidden_size:
                    raise ValueError("The frozen encoder returned an unexpected hidden-state shape.")
                expected = int(convolution_output_lengths([config.window_samples], self.kernels, self.strides)[0])
                if hidden.shape[1] != expected or np.any(valid_lengths > hidden.shape[1]):
                    raise ValueError("Encoder frame counts disagree with the configured convolution lengths.")
                lengths = torch.as_tensor(valid_lengths, device=hidden.device)
                frame_mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
                pooled = masked_mean_pool(hidden, frame_mask).float().cpu().numpy()
            if not np.isfinite(pooled).all():
                raise ValueError("The frozen encoder produced a nonfinite recording feature.")
            embeddings.append(pooled)
        return np.concatenate(embeddings).astype(np.float32, copy=False)

    def encode(self, waveform, config: AudioConfig = AudioConfig()) -> np.ndarray:
        """Average valid-frame window embeddings equally into one recording."""

        return self.encode_windows(make_windows(waveform, config), config).mean(axis=0, dtype=np.float64).astype(np.float32)


class EmbeddingCache:
    """Content-addressed, metadata-checked float embeddings; never pickle objects."""

    def __init__(self, directory: Path, model_identity: dict, audio_config: AudioConfig, augmentation_version: str = "channel-v1"):
        self.directory = Path(directory).expanduser().resolve()
        self.model_identity = json.loads(_json(model_identity))
        self.audio_config = audio_config
        self.augmentation_version = augmentation_version
        self.hits = self.misses = self.writes = 0

    def request(self, source: dict, augmentation: AugmentationConfig, index: int) -> tuple[str, dict]:
        check_augmentation_index(index, augmentation)
        if augmentation.version != self.augmentation_version:
            raise ValueError("Cache and augmentation versions disagree.")
        metadata = {
            "schema": "s01-recording-embedding-v1", "source": source,
            "encoder": self.model_identity, "audio": asdict(self.audio_config),
            "normalization": NORMALIZATION, "pooling": POOLING,
            "augmentation": asdict(augmentation), "augmentation_index": index,
            "software": _software(),
        }
        key = hashlib.sha256(_json(metadata).encode("utf-8")).hexdigest()
        return key, metadata

    def _path(self, key: str, metadata: dict) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key) or hashlib.sha256(_json(metadata).encode("utf-8")).hexdigest() != key:
            raise ValueError("Cache key does not match its complete embedding metadata.")
        return self.directory / f"{key}.npz"

    def _embedding(self, embedding) -> np.ndarray:
        values = np.asarray(embedding)
        dimension = self.model_identity.get("hidden_size")
        if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError("A cached embedding must be a nonempty floating-point vector.")
        if dimension is not None and values.shape != (int(dimension),):
            raise ValueError("A cached embedding has the wrong encoder dimension.")
        values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("A cached embedding contains nonfinite values.")
        return values

    def load(self, key: str, metadata: dict) -> np.ndarray | None:
        path = self._path(key, metadata)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as cached:
                if set(cached.files) != {"embedding", "metadata"} or cached["metadata"].ndim != 0:
                    raise ValueError("Cache fields are invalid.")
                if str(cached["metadata"].item()) != _json(metadata):
                    raise ValueError("Cache metadata does not match the requested embedding.")
                values = self._embedding(cached["embedding"])
        except (OSError, ValueError, KeyError, EOFError) as exc:
            raise ValueError("Invalid embedding cache entry; remove that generated cache entry and rerun.") from exc
        self.hits += 1
        return values

    def store(self, key: str, metadata: dict, embedding) -> None:
        path = self._path(key, metadata)
        values = self._embedding(embedding)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=f".{key}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, embedding=values, metadata=np.asarray(_json(metadata)))
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self.writes += 1


def _source(path: Path) -> tuple[dict, tuple[int, ...]]:
    def fingerprint():
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    before = fingerprint()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if before != fingerprint():
        raise ValueError("A source recording changed while its content identity was being computed.")
    return {"sha256": digest.hexdigest(), "bytes": before[2]}, before


def _unchanged(path: Path, expected: tuple[int, ...]) -> None:
    stat = path.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != expected:
        raise ValueError("A source recording changed while its embedding was being computed.")


class RecordingEmbedder:
    """Batch windows across recordings, cache them, and preserve record order.

    ``embed_records(records, aug_index=0, progress=None)`` accepts objects with
    a ``path`` attribute or paths directly. ``progress(done, total)`` receives
    aggregate counts once per small record batch. Only root/controller-selected
    training fold records may use positive augmentation indices.
    """

    def __init__(self, encoder: FrozenEncoder, cache: EmbeddingCache, audio_config: AudioConfig = AudioConfig(), augmentation: AugmentationConfig = AugmentationConfig()):
        if cache.audio_config != audio_config or cache.model_identity != json.loads(_json(encoder.identity)):
            raise ValueError("Embedder audio/model settings do not match the embedding cache identity.")
        if cache.augmentation_version != augmentation.version:
            raise ValueError("Embedder and cache augmentation versions disagree.")
        self.encoder, self.cache = encoder, cache
        self.audio_config, self.augmentation = audio_config, augmentation

    @property
    def stats(self) -> dict:
        return {"cache_hits": self.cache.hits, "cache_misses": self.cache.misses, "cache_writes": self.cache.writes}

    def get(self, path: Path | str, augmentation_index: int = 0) -> np.ndarray:
        return self.embed_records([path], aug_index=augmentation_index)[0]

    def embed_records(self, records, aug_index: int = 0, progress=None) -> np.ndarray:
        check_augmentation_index(aug_index, self.augmentation)
        records = list(records)
        if not records:
            return np.empty((0, self.encoder.hidden_size), dtype=np.float32)
        results = [None] * len(records)
        chunk_size = max(1, self.encoder.batch_size * 4)
        for start in range(0, len(records), chunk_size):
            pending, windows = [], []
            for index in range(start, min(start + chunk_size, len(records))):
                path = Path(getattr(records[index], "path", records[index])).expanduser().resolve(strict=True)
                if not path.is_file():
                    raise ValueError("A recording path must identify an existing regular audio file.")
                source, fingerprint = _source(path)
                key, metadata = self.cache.request(source, self.augmentation, aug_index)
                hit = self.cache.load(key, metadata)
                if hit is not None:
                    _unchanged(path, fingerprint)
                    results[index] = hit
                    continue
                waveform = load_audio(path, self.audio_config.sample_rate)
                waveform = augment_waveform(waveform, self.augmentation, aug_index, source["sha256"], self.audio_config.sample_rate)
                selected = make_windows(waveform, self.audio_config)
                lower = len(windows)
                windows.extend(selected)
                pending.append((index, path, fingerprint, key, metadata, lower, len(windows)))
            if windows:
                encoded = self.encoder.encode_windows(windows, self.audio_config)
                if encoded.shape != (len(windows), self.encoder.hidden_size) or not np.isfinite(encoded).all():
                    raise ValueError("Encoder window embeddings have invalid dimensions or values.")
                for index, path, fingerprint, key, metadata, lower, upper in pending:
                    value = encoded[lower:upper].mean(axis=0, dtype=np.float64).astype(np.float32)
                    _unchanged(path, fingerprint)
                    self.cache.store(key, metadata, value)
                    results[index] = value
            if progress is not None:
                progress(min(start + chunk_size, len(records)), len(records))
        return np.stack(results).astype(np.float32, copy=False)
