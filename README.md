# Speech classification: NM / AD / MCI

Code for three-class speech classification, organized by numbered experiments.
Each experiment owns its model implementation and settings. `s00` is the first
ResNet18 baseline; subsequent systems use `s01`, `s02`, and so on.

This repository contains source, configuration, tests, and documentation.
Keep audio, patient manifests, predictions, checkpoints, caches, and generated
HTML/PDF reports in local ignored directories.

## Experiments

| ID | Implementation | Evaluation |
| --- | --- | --- |
| [s00](s00/README.md) | Single-channel ResNet18 on log-mel speech features | Frozen patient-disjoint train/validation/test splits |
| [s01](s01/README.md) | Frozen pinned XLS-R embeddings with mean/attention patient-bag MIL; A/B/C channel-augmentation comparison | Shared five-fold CV on original train + validation patients, then frozen selection and one original-test evaluation |
| [s02](s02/README.md) | Matched s00 ResNet18 and s01 XLS-R comparison with the enlarged MCI cohort | Shared frozen patient splits and five development folds; each final model tested once |

Subsequent experiments will be added as their implementations become available.

```text
run_experiment.py          # Common launcher for numbered experiments
s00/
  experiment.json          # Model/training settings for this experiment
  code/                    # Preserved training, data, and original report source
  analysis/                # AUC calculation, updated HTML report, A4 PDF export
  tests/                   # Synthetic unit and integration checks
  requirements.txt
  requirements-report.txt
dataset/                   # Private local input; ignored by Git
runs/s00/                  # Local training and analysis output; ignored by Git
```

## Environment

Use Python 3.10 or newer. Create an environment and install a matching PyTorch
and torchvision build for your CPU/CUDA device before the remaining dependencies.
The [PyTorch installation selector](https://pytorch.org/get-started/locally/)
provides platform-specific commands.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r s00/requirements.txt
python -m pip install -r requirements-dev.txt
```

For the exact original CUDA environment, see `s00/code/requirements.lock.txt`.
That lock includes `torch==2.14.1+cu130` and `torchvision==0.29.1+cu130`; installing
those builds requires the matching official PyTorch `cu130` wheel index. The
general requirements file supports other matching CPU/CUDA installations.

## Private input and training

Supply a private CSV with these columns:

```text
path,patient_id,label,split
```

Labels are `NM`, `AD`, and `MCI`; splits are `train`, `val`, and `test`. Relative
audio paths resolve against the CSV directory. A patient ID must identify the
same person across every recording and visit. All recordings from that person
must share one split. Use the previously approved frozen manifest to reproduce
the existing study; the seed alone cannot reconstruct omitted participant IDs
or exclusions.

From this repository root:

```bash
# Validate inputs and write split/report artifacts without training.
python run_experiment.py s00 --manifest /path/to/private/split_manifest.csv --prepare-only

# Train using the same private manifest. The output folder must be fresh.
python run_experiment.py s00 --manifest /path/to/private/split_manifest.csv --device cuda
```

The default output is `runs/s00`. Preparation may write into that directory;
a completed or interrupted training run must use a different fresh output for
another training attempt. Use `--output-dir runs/s00-repeat` for a new run.
The launcher always supplies explicit paths to the preserved training source.

The supplied frozen manifest determines membership. Source-data issues such as
unresolved duplicate participant identities must be resolved or excluded in that
private manifest before training. No participant review files are published.

For s01's separate environment and complete protocol, see [s01/README.md](s01/README.md).
Its common-launcher default performs development CV; final training uses the
direct entrypoint with the saved selection in the same local study directory:

```bash
python -m pip install -r s01/requirements.txt
python run_experiment.py s01 --manifest /path/to/private/split_manifest.csv --prepare-only
python run_experiment.py s01 --manifest /path/to/private/split_manifest.csv --device cuda
python s01/code/train.py --manifest /path/to/private/split_manifest.csv --output-dir runs/s01 --stage final --device cuda
```

## AUC and reports

Training generates a method/results HTML report in its output directory.
Reports omit generation and update timestamps. Use this convention for future
numbered experiments.
Postprocessing uses saved prediction probabilities and does not rerun the model.

```bash
python s00/analysis/calculate_auc.py --output-dir runs/s00

# Optional PDF dependencies and A4 export.
python -m pip install -r s00/requirements-report.txt
python s00/analysis/export_report_pdf.py --html runs/s00/report.html --output runs/s00/report.pdf
```

The analysis renderer includes the clipping/padding comparison with the original
CNN pipeline and class-specific one-vs-rest ROC AUC. A4 styling applies only to
the exported PDF. Training and analysis outputs remain local.

## Tests

```bash
python -m pytest -q tests
python -m pytest -q s00/tests
```

Tests use synthetic inputs and do not require the private dataset or a GPU.
Run each experiment's tests in a separate process (for example,
`python -m pytest -q s01/tests`) to isolate its module imports and test names.

## Adding the next system

Create the next unused directory (`s01`, then `s02`, etc.) with its own
`experiment.json`, `code/`, dependency files, and tests. Copy a previous
experiment into the new directory if it is a suitable starting point, then
change only the new experiment. Record its method and intended validation in
its README. The common launcher reads that experiment's entrypoint and settings;
outputs default to the matching `runs/<experiment>/` folder.

Keep the completed `s00` training implementation available for comparison.
When comparing later systems, reuse the same approved patient memberships and
select models on validation data. Repeated decisions based on the held-out test
would change its role as an independent evaluation.
