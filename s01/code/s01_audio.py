"""Valid-sample waveform preprocessing for frozen XLS-R recording embeddings.

Only the waveform reaches the encoder. Labels, patient IDs, and source metadata
are not features. Augmentation indices above zero are for training folds only;
the experiment controller is responsible for selecting those recordings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


NORMALIZATION = {
    "version": "valid-window-zero-mean-unit-variance-v1",
    "epsilon": 1e-7,
    "statistics_dtype": "float64",
    "padding_value": 0.0,
    "return_attention_mask": True,
}


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    clip_seconds: float = 5.0
    max_windows: int = 5

    def __post_init__(self):
        if self.sample_rate != 16000:
            raise ValueError("The pinned XLS-R encoder requires 16,000 Hz audio.")
        if not math.isfinite(self.clip_seconds) or self.clip_seconds <= 0 or self.window_samples < 1:
            raise ValueError("Clip duration must be finite and contain at least one sample.")
        if type(self.max_windows) is not int or self.max_windows < 1:
            raise ValueError("Maximum window count must be a positive integer.")

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    seed: int = 42
    version: str = "channel-v1"
    cutoff_min: float = 4000.0
    cutoff_max: float = 7500.0
    snr_min: float = 20.0
    snr_max: float = 35.0

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Augmentation needs a boolean enabled flag and a nonnegative integer seed.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Augmentation version must be a nonblank string.")
        values = (self.cutoff_min, self.cutoff_max, self.snr_min, self.snr_max)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Augmentation cutoff and SNR ranges must be finite.")
        if not 0 < self.cutoff_min <= self.cutoff_max < 8000:
            raise ValueError("Low-pass cutoffs must be ordered and strictly between 0 and 8,000 Hz.")
        if not 0 < self.snr_min <= self.snr_max:
            raise ValueError("Noise SNR bounds must be positive and ordered.")


def _waveform(waveform) -> np.ndarray:
    result = np.asarray(waveform, dtype=np.float32)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("Audio must be a nonempty, finite, mono waveform.")
    return np.ascontiguousarray(result)


def load_audio(path: Path | str, sample_rate: int = 16000) -> np.ndarray:
    """Decode floating-point audio, average channels, and resample to 16 kHz.

    No clipping, duration change, silence removal, or corpus-fitted statistics
    are applied. The returned array contains only real samples, without padding.
    """

    import soundfile as sf
    from scipy.signal import resample_poly

    if sample_rate != 16000:
        raise ValueError("XLS-R waveform loading requires a 16,000 Hz target rate.")
    try:
        samples, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Cannot decode a recording; check that the audio download is complete.") from exc
    if samples.shape[0] == 0 or samples.shape[1] == 0 or not np.isfinite(samples).all():
        raise ValueError("Decoded audio must be nonempty and finite.")
    waveform = samples.mean(axis=1, dtype=np.float64).astype(np.float32)
    if source_rate != sample_rate:
        divisor = math.gcd(int(source_rate), sample_rate)
        waveform = resample_poly(waveform, sample_rate // divisor, int(source_rate) // divisor)
    return _waveform(waveform)


def window_starts(sample_count: int, config: AudioConfig) -> list[int]:
    """Deterministically span a recording with at most the configured cap.

    Short recordings yield one unpadded window. Very long recordings can have
    gaps because the cap limits the number of windows rather than their stride.
    """

    if sample_count < 1:
        raise ValueError("A recording must contain at least one sample.")
    last = max(0, sample_count - config.window_samples)
    count = min(config.max_windows, math.ceil(sample_count / config.window_samples))
    if count == 1:
        return [last // 2]
    return sorted({round(index * last / (count - 1)) for index in range(count)})


def make_windows(waveform, config: AudioConfig = AudioConfig()) -> list[np.ndarray]:
    """Return copies of valid raw samples; padding happens only at batching."""

    waveform = _waveform(waveform)
    return [waveform[start:start + config.window_samples].copy() for start in window_starts(len(waveform), config)]


def pad_normalize(windows: list[np.ndarray], config: AudioConfig = AudioConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 [windows,samples] and int64 masks; padded samples are zero.

    Mean and variance use only a window's valid samples. This follows the
    Wav2Vec2 feature extractor's valid-sample normalization, using float64
    accumulation for stable statistics and the same 1e-7 variance epsilon.
    """

    if not windows:
        raise ValueError("At least one waveform window is required.")
    values = np.zeros((len(windows), config.window_samples), dtype=np.float32)
    mask = np.zeros(values.shape, dtype=np.int64)
    for index, raw in enumerate(windows):
        raw = _waveform(raw)
        if len(raw) > config.window_samples:
            raise ValueError("A waveform window exceeds the configured clip duration.")
        precise = raw.astype(np.float64)
        normalized = (precise - precise.mean()) / np.sqrt(precise.var() + NORMALIZATION["epsilon"])
        values[index, :len(raw)] = normalized.astype(np.float32)
        mask[index, :len(raw)] = 1
    if not np.isfinite(values).all():
        raise ValueError("Waveform normalization produced nonfinite values.")
    return values, mask


def check_augmentation_index(index: int, config: AugmentationConfig) -> None:
    if type(index) is not int or index < 0:
        raise ValueError("Augmentation index must be a nonnegative integer.")
    if index and not config.enabled:
        raise ValueError("Positive augmentation indices require augmentation.enabled=True and training-only records.")


def augment_waveform(
    waveform, config: AugmentationConfig, index: int, recording_identity: str,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Apply a repeatable mild channel perturbation before window padding.

    A fourth-order low-pass magnitude response, a random ±2 dB spectral tilt,
    and additive noise at the configured SNR operate only on valid waveform
    samples. Content identity, configuration, and index determine the RNG seed;
    no class, patient, acquisition source, or held-out statistics are used.
    Index zero always returns the original waveform unchanged.
    """

    waveform = _waveform(waveform)
    check_augmentation_index(index, config)
    if index == 0:
        return waveform.copy()
    if sample_rate != 16000:
        raise ValueError("Channel augmentation expects 16,000 Hz audio.")
    material = json.dumps({"identity": recording_identity, "config": asdict(config), "index": index}, sort_keys=True, allow_nan=False)
    seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    cutoff = generator.uniform(config.cutoff_min, config.cutoff_max)
    tilt_db = generator.uniform(-2.0, 2.0)
    frequencies = np.fft.rfftfreq(len(waveform), d=1 / sample_rate)
    response = 1 / np.sqrt(1 + (frequencies / cutoff) ** 8)
    response *= 10 ** (tilt_db * (frequencies / (sample_rate / 2) - 0.5) / 20)
    filtered = np.fft.irfft(np.fft.rfft(waveform.astype(np.float64)) * response, n=len(waveform))
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    snr = generator.uniform(config.snr_min, config.snr_max)
    noise = generator.standard_normal(len(filtered)) * rms * 10 ** (-snr / 20)
    return _waveform((filtered + noise).astype(np.float32))
