# Completed speech classification reports

These snapshots document the saved NM / AD / MCI evaluations. All results come
from completed runs; publication changes presentation and links only.

| Experiment | Cohort | Reports | Method |
| --- | --- | --- | --- |
| s01 | 98 patients; 3,615 recordings | [PDF](s01/report.pdf) · [HTML](s01/report.html) | [Frozen XLS-R patient-bag MIL](../s01/README.md) |
| s02 | 234 patients; 9,080 recordings | [PDF](s02/report.pdf) · [HTML](s02/report.html) | [Matched ResNet18 / XLS-R comparison](../s02/README.md) |

## Saved test results

| Method | Test patients | Macro F1 | Balanced accuracy | Accuracy | Macro OvR AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| s01 XLS-R mean MIL (A) | 15 | 0.6639 | 0.6857 | 0.6667 | 0.8174 |
| s02 ResNet18 | 35 | 0.7203 | 0.7238 | 0.8571 | 0.9686 |
| s02 XLS-R attention MIL (B) | 35 | 0.6571 | 0.6571 | 0.8286 | 0.9166 |

s01 selected variant A by shared five-fold development CV. s02 selected variant
B for XLS-R and evaluated both final methods on the same 35 test patients.
The cohorts differ: s01 has three MCI test patients, while s02 has 23. These
tables do not establish a paired improvement between experiments. NM/AD test
participants were reused, and source acquisition differs between cohorts.
The reports provide full class supports, confusion matrices, development CV,
learning curves, and interpretation of these exploratory results.

## Opening the reports

GitHub previews the PDFs. To view an HTML report with its figures, download the
repository and open `reports/s01/report.html` or `reports/s02/report.html` in a
browser, keeping each adjacent `figures/` directory. PNG and SVG downloads are
included with each report.

The published snapshots contain aggregate results and figures. Local run artifact
download sections are omitted. Audio, patient identities, split memberships,
prediction rows, embedding caches, and model checkpoints remain outside this
directory. Original experiment implementations and saved evaluations are retained.

## Provenance

The included source snapshot is based on commit
`369cf8f118d46a7139386d264c179a5b799eab94` from `s02-full-dataset`, which also
contains s01. The prior s01 branch head is
`197d70e159a2d51d6033894114fcab257410cc0f`.

Each `publication.json` records source report hashes, published file hashes, and
presentation adjustments. Published PDFs replace server-local figure hyperlinks
with GitHub links. One s01 chart title is shortened to avoid its colorbar.
