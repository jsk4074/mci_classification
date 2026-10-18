# s00: ResNet18 speech baseline

The first numbered system predicts `NM`, `AD`, and `MCI` directly with one
single-channel ResNet18, initialized from scratch.

`code/` preserves the source used by the completed baseline. `analysis/`
contains subsequent AUC and report updates, including the original-clipping
comparison and the A4 PDF exporter. The launcher passes explicit paths so
moving the original scripts into a numbered directory does not change input
or output discovery.
`code/SHA256SUMS` records the preserved source and dependency file hashes.

## Method

| Setting | s00 |
| --- | --- |
| Audio | Mono, 16 kHz |
| Features | 40-band log-mel; FFT/window 400 samples; hop 160 samples |
| Normalization | Each recording's own dB peak; clip to [-80, 0] dB and scale to [-1, 1] |
| Model input | 5 seconds: 1 x 40 x 501 |
| Short recordings | Right padding at the log-mel floor |
| Training windows | One random crop per sampled recording |
| Evaluation windows | Up to five evenly spaced windows; mean probabilities per recording, then per patient |
| Sampling | Equal expected class mass and equal patient mass within each class |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Training | Batch 16; maximum 50 epochs; early-stopping patience 10; seed 42 |
| Checkpoint | Highest validation patient macro F1; ties use lower patient loss |
| Test | Evaluated after checkpoint selection |

The fixed windows and mel settings differ from the original CNN notebook, which
used task-specific mean durations, beginning truncation, and waveform noise
padding. This system is a new baseline with documented preprocessing changes.
Small patient holdouts and differences in source acquisition conditions limit
interpretation; this single holdout does not replace repeated patient-level
cross-validation.

## Commands

Run from the repository root after installing the dependencies:

```bash
python run_experiment.py s00 --manifest /path/to/private/split_manifest.csv --device cuda
python s00/analysis/calculate_auc.py --output-dir runs/s00
python s00/analysis/export_report_pdf.py --html runs/s00/report.html --output runs/s00/report.pdf
```

The default output folder is `runs/s00`, which Git ignores. The private manifest
must retain the approved patient assignments and source-review exclusions.
This repository does not contain the data or patient identities needed to
reproduce those memberships from scratch.

Exact interrupted-run resumption is not implemented. Start another run in a
fresh output folder instead of overwriting existing training artifacts.

## Dependencies

- `requirements.txt`: general training and AUC dependencies.
- `code/requirements.lock.txt`: exact original CUDA package versions.
- `requirements-report.txt`: optional PDF dependency; native Pango is also required.
- `../requirements-dev.txt`: synthetic test dependencies.

The generated reports, predictions, checkpoints, and measured run results are
kept in the local study workspace rather than this code repository.
