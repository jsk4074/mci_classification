# s01: frozen XLS-R with patient-level MIL

`s01` uses a frozen multilingual speech encoder and a trainable patient-bag
multiple-instance learning (MIL) head. Its implementation is independent of
`s00`; the completed ResNet18 source remains unchanged. The completed run is documented in the saved aggregate report below.

## Model and audio handling

The base encoder is `facebook/wav2vec2-xls-r-300m`, pinned to commit
`1a640f32ac3e39899438a2931f9924c02f080a54`. Loading must verify that revision and
the base weights. There is no random-initialization or CTC-model fallback.
The encoder stays in evaluation mode with gradients disabled and float32
inference; only the projection, MIL aggregator, and diagnosis head are trained.

Audio is decoded to mono and resampled to 16 kHz. Each recording supplies up to
five deterministic 5-second waveform windows distributed across its duration.
Short windows are normalized using valid samples only, then zero-padded with a
raw-sample attention mask. Exact convolution output lengths define the encoder
frame mask. Mean pooling excludes padded frames, and window embeddings are
averaged equally into one recording embedding. Very long recordings can have
gaps between sampled windows. Unlike s00 training, window positions are fixed.

Each patient bag retains all of that patient's available recordings. Variable
bag sizes use a separate recording mask. A shared LayerNorm → Linear(256) →
GELU → Dropout(0.3) projection precedes one of these aggregators and a three-class
linear output (`NM=0`, `AD=1`, `MCI=2`):

| Variant | Aggregation | Training recordings |
| --- | --- | --- |
| A | Equal-weight mean MIL | Base embeddings |
| B | Learned masked attention MIL, attention width 128 | Base embeddings |
| C | Same attention MIL as B | Random choice of base or cached channel-augmented embedding per recording |

C has one cached augmentation per recording by default. `channel-v1` applies a
random low-pass magnitude response with a 4–7.5 kHz cutoff, spectral tilt within
±2 dB, and additive noise at 20–35 dB SNR. Source content hash, augmentation
configuration, seed, and index determine each perturbation. Augmentation uses
only valid waveform samples before padding. Only the current training-fold
patients receive augmented representations; validation and test always use
base embeddings. There is no encoder fine-tuning or label-derived channel
transformation.

Embeddings are cached under the local run's `embeddings/` directory. Cache
identities include audio content, pinned/resolved encoder identity, software,
precision, normalization, windowing, pooling, and augmentation settings. Numeric
NPZ entries are checked against their metadata and loaded without pickle.
Changing relevant inputs creates different keys. The shared frozen base
representations can be reused across folds and variants.

`encoder_metadata.json` records the resolved encoder identity, encoder-side
software versions, float32 settings, execution backend, and encoder batch size.
Its digest is frozen in the CV selection. Final training must reproduce that
identity; switching CPU/CUDA, backend settings, software, or encoder batching
within the study is rejected rather than silently replacing cached features.

## Patient protocol

Use the same approved private manifest as s00, with explicit
`path,patient_id,label,split` columns. Relative audio paths resolve beside the
CSV. Every patient must have one diagnosis and one original split; each original
split must contain all three classes. Missing/duplicate paths and identical
audio under different patient IDs are rejected. The two MCI folders pending
source review remain excluded. An optional `--quarantine-manifest` CSV with a
`patient_id` column rejects accidental inclusion; it never silently removes
patients or changes their assignments.

For the approved 98-patient, 3,615-recording cohort:

1. Combine the original **68 training + 15 validation patients** into an
   **83-patient development pool**. Create five deterministic, diagnosis-
   stratified patient folds, shared by A/B/C. The original **15 test patients**
   are excluded from all folds and development encoder/model inference.
2. Train each variant on each fold. Checkpoint selection uses validation
   **patient macro F1**, breaking ties with lower unweighted patient
   cross-entropy. Compare mean and sample standard deviation across folds.
3. Select A/B/C by highest mean fold patient macro F1; ties use lower mean
   patient loss, then A → B → C. `selection.json` freezes the choice, study
   parameters, data/fold hashes, and training-source hashes.
4. Train the selected variant from scratch on the **original 68 training
   patients**. The **original 15 validation patients** select its checkpoint.
   Evaluate that checkpoint once in a guarded final event on the **original
   15 test patients**. Test embedding extraction begins only at that boundary.

Defaults are seed 42, 100 maximum epochs, early-stopping patience 12, eight
patients per batch, and AdamW with learning rate 0.001 and weight decay 0.001.
Training uses ordinary shuffled patient batches and cross-entropy weights
`n / (3 * class_count)` computed only from the current training patients.
There is no balanced sampler. Recording-level metrics use singleton bags with
inherited patient labels and are supplementary; they do not select models.

The source audio headers are audited by diagnosis and original split. This
describes sample rates, channels, encoding, and duration, including test header
metadata; it performs no test encoder/model inference and verifies no clinical
labels or recording-device identities.

## Environment and commands

From the repository root, using Python 3.10 or newer:

```bash
python -m venv .venv-s01
source .venv-s01/bin/activate
python -m pip install -r s01/requirements.txt
python -m pip install -r requirements-dev.txt
```

Choose a PyTorch installation matching the intended CPU/CUDA device. The
requirements pin Transformers to 4.57.6 and require PyTorch 2.6 or newer.
The pinned encoder weights download on first inference to
`.cache/huggingface/` by default. Direct entrypoint options
`--encoder-cache-dir` and `--local-files-only` support another local cache and
offline loading of already cached weights. No weights download occurs during
preparation, report regeneration, or synthetic smoke checks.

```bash
# Validate frozen memberships, save folds/fingerprints, and audit audio headers.
python run_experiment.py s01 --manifest /path/to/private/split_manifest.csv --prepare-only

# Development CV for A/B/C. Output defaults to runs/s01.
python run_experiment.py s01 --manifest /path/to/private/split_manifest.csv --device cuda

# After CV completes, fit/evaluate its frozen selection in the SAME study folder.
python s01/code/train.py --manifest /path/to/private/split_manifest.csv --output-dir runs/s01 --stage final --device cuda
```

The common launcher fixes `--stage cv` through `experiment.json`; use the direct
entrypoint for `--stage final` or additional experiment-specific options. The
final command above uses the same default study parameters as the published CV
configuration. If CV used other settings, pass those same settings for final
training. This includes the same `--workers` and `--torch-threads` settings
(defaults 0 and 16), as well as `--encoder-batch-size` (default 4). Data, folds,
settings, training/metric source, and the encoder identity/software/backend/
batching digest must match the saved selection.
Use the same output directory for preparation, CV, and final training; after a
partial or completed training attempt, an independent study needs a fresh
directory. Exact interrupted-run resumption is not implemented.

To enforce a private quarantine list, use the direct entrypoint with
`--quarantine-manifest /path/to/private/quarantine.csv` during preparation, CV,
and final training. For example:

```bash
python s01/code/train.py --manifest /path/to/private/split_manifest.csv --quarantine-manifest /path/to/private/quarantine.csv --output-dir runs/s01 --prepare-only
```

An optional direct `--s00-metrics /path/to/private/s00/metrics.json` copies
aggregate baseline metrics into the report for comparison; it does not select
variants or checkpoints.

```bash
# Synthetic CPU-only head check; no patient data or pretrained model.
python s01/code/train.py --smoke-test --output-dir runs/s01-smoke
python -m pytest -q s01/tests

# Regenerate saved aggregate figures/HTML without inference.
python run_experiment.py s01 --report-only

# Optional A4 PDF export.
python -m pip install -r s01/requirements-report.txt
python s01/analysis/s01_export_pdf.py --html runs/s01/report.html --output runs/s01/report.pdf
```

Direct `train.py --report-only --export-pdf --output-dir runs/s01` combines
regeneration and PDF export. `--export-pdf` on direct CV/final commands exports
at the end of the requested stage. Matplotlib writes standalone PNG/SVG figures;
HTML uses local assets only. WeasyPrint 70 is optional and may need its platform
text/font rendering libraries.

## Local outputs and interpretation

`runs/s01/` contains the private fingerprint, patient folds, aggregate summaries
and domain audit, encoder metadata/cache, per-fold `cv/A|B|C/fold_XX/` histories,
checkpoints and predictions, CV summary, and frozen selection. Final outputs
include `final/`, the run-level selected checkpoint/metrics/history,
`test_evaluation.json`, aggregate HTML, figures, and optional PDF. Private paths
and patient identifiers occur in manifests, fold memberships, fingerprints,
and prediction CSVs; keep the entire run directory out of Git. The report reads
aggregate artifacts and excludes patient rows.

The original s00 test results were already known when s01 was designed. This
reused holdout can be indirectly influenced by subsequent design choices, so
the comparison is exploratory rather than a fresh blinded confirmation. The
15-patient test includes only three MCI patients. Correlated clips, source
acquisition differences, and limited patient counts constrain interpretation.
Channel augmentation does not establish cross-device/site robustness; that
requires appropriate independent cohorts with diagnosis overlap.

## Saved aggregate report

The completed s01 evaluation is available as [PDF](../reports/s01/report.pdf) and [HTML](../reports/s01/report.html). See the [report index](../reports/README.md) for study sizes, result summaries, provenance, and instructions for opening the HTML with its figures.
