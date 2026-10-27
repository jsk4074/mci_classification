# s02: matched full-cohort ResNet18 / XLS-R comparison

s02 reruns the s00 ResNet18 recipe and the s01 frozen XLS-R A/B/C comparison
on one shared cohort. The previous experiment implementations and results are
preserved. Training uses new model initialization, caches, and run directories.

## Cohort and split policy

The new archive contains MCI cases only. At the user's direction, s02 combines
the existing approved **48 NM and 32 AD patients** with **154 new MCI patients**.
The previous 18 MCI patients are replaced by the new cohort. The two unresolved
old MCI participant folders remain outside this study.

Both models use the same canonical **06/07 sentence tasks** as the original
baseline. The full new archive contains additional tasks, which are excluded
from this matched comparison. Eligible distinct repeated takes remain in the
same participant bag; they are not treated as new patients. There are 30 repeated
patient/task slots containing 61 nonidentical recordings (31 additional takes).
One very short recording in the full archive is outside the eligible tasks.
Raw WAVs, JSON annotations, and original ZIPs are read only.

The expected cohort has **234 patients and 9,080 recordings**: 1,839 NM, 1,065 AD,
and 6,176 MCI clips. JSON recording IDs are not patient IDs. New MCI grouping uses
the preserved participant archive folder; metadata must agree within each folder.
Diagnosis, age, gender, filenames, and source metadata are not numerical inputs.

Existing NM/AD patient assignments are retained exactly. Only the new MCI patients
are shuffled deterministically with seed 42 and assigned 108 train / 23 validation
/ 23 test. The combined original roles are:

| Split | NM | AD | MCI | Total patients |
| --- | ---: | ---: | ---: | ---: |
| Train | 34 | 22 | 108 | 164 |
| Validation | 7 | 5 | 23 | 35 |
| Test | 7 | 5 | 23 | 35 |

Five patient-stratified development folds combine train and validation patients;
all 35 test patients are excluded. Every method uses exactly the same folds.
ResNet18 has one fixed architecture. XLS-R A/B/C is selected by mean fold patient
macro F1, then lower mean patient loss, then stable A/B/C order. Both comparisons
must finish and their configurations are frozen before final test evaluation.
Each final model fits the original s02 training patients and selects its checkpoint
on the s02 validation patients. Each method's selected checkpoint is evaluated
once on the same test patients, behind its own exclusive evaluation guard.

## Methods

- **ResNet18:** unchanged single-channel, randomly initialized s00 model and
  log-mel preprocessing; 16 kHz, five-second inputs, random training crop,
  floor padding, up to five deterministic evaluation windows. Sampling balances
  classes and patients; ordinary cross-entropy avoids a second class correction.
  AdamW LR 0.001, weight decay 0.0001, batch 16, cap 50 epochs, patience 10,
  and the original plateau scheduler. Patient predictions average windows and
  recordings equally.
- **XLS-R:** unchanged pinned s01 frozen base encoder, valid-frame pooling and
  equal window means; all recordings retained in patient bags. A uses mean MIL,
  B attention MIL, C attention plus training-fold-only cached channel variants.
  Patient-frequency-weighted cross-entropy with ordinary patient shuffling;
  AdamW LR 0.001, weight decay 0.001, patient batch 8, cap 100, patience 12.
  The encoder revision, backend, software, and batching are frozen across stages.

Both select checkpoints with validation patient macro F1 and unweighted patient
loss as tiebreaker. Metrics include patient and supplementary recording accuracy,
balanced accuracy, macro F1, cross-entropy, macro OvR AUC, per-class precision,
recall, F1, AUC, supports, and confusion matrices. CV means and sample standard
deviations describe development comparisons used for selection.

## Commands

From a normal repository checkout, install the matching CUDA-enabled PyTorch
build and the project dependencies:

```bash
python -m pip install -r s02/requirements.txt -r requirements-dev.txt
python s02/code/train.py --smoke-test --output-dir runs/s02-smoke
python -m pytest -q s02/tests

# Build the cohort, freeze memberships, and prepare the aggregate report.
python s02/code/prepare_dataset.py --baseline-manifest /path/to/dataset/s00/split_manifest.csv --full-data-root /path/to/dataset/full_dataset/unziped --quarantine-manifest /path/to/private/quarantined_patients.csv --output-dir runs/s02 --export-pdf

# Run both shared five-fold comparisons and both guarded final evaluations.
python run_experiment.py s02 --manifest runs/s02/split_manifest.csv --device cuda

# Regenerate the saved HTML and A4 PDF without model inference.
python s02/code/train.py --report-only --output-dir runs/s02 --export-pdf
```

For a predownloaded encoder, set `HF_HUB_CACHE` or pass `--encoder-cache-dir` to
the direct entrypoint. `--local-files-only` prevents downloading weights. The
direct entrypoint also supports `--stage cv|final|all`,
`--system resnet18|xlsr_mil|both`, and method-specific epoch, patience, batch-size,
learning-rate and weight-decay options. The common launcher uses the committed
defaults in experiment.json. Keep all frozen settings, worker/thread counts,
encoder batching, software, and execution device consistent across stages.

Completed stages can be skipped in the same frozen study. Interrupted individual
fits are not resumed automatically; use a new study directory if a fit is partial.
A started test guard always prevents automatic repeat inference. Saved prediction
probabilities can be audited and reports regenerated without evaluating a model.

## Outputs and interpretation

Private artifacts belong under ignored `runs/s02/`: shared manifest, content
fingerprint and source inventory, folds, source/task/domain audit, config and
runtime, CV comparisons, frozen paired selection, `resnet18/` and `xlsr_mil/`
checkpoints/histories/metrics/predictions, and aggregate HTML/A4 PDF/figures.
Reports omit timestamps and participant identities. Git contains code, tests,
and documentation only. Method subreports retain the original pipeline names;
the s02 aggregate report identifies the paired full-cohort results.

NM/AD and new MCI come from different cohorts. Task matching, resampling, and
channel perturbations cannot establish clinical or device/site invariance.
The original NM/AD test patients have already been evaluated in s00 and s01;
their reused holdout results are exploratory. The enlarged MCI class also changes
class balance. Validation/CV scores used for selection are not independent test
estimates, and five-fold comparisons are not nested-CV estimates of the entire
selection procedure. Inspect class supports and NM/AD recalls alongside macro
scores; strong MCI discrimination can reflect source/task differences.

## Saved aggregate report

The completed s02 evaluation is available as [PDF](../reports/s02/report.pdf) and [HTML](../reports/s02/report.html). See the [report index](../reports/README.md) for study sizes, result summaries, provenance, and instructions for opening the HTML with its figures.
