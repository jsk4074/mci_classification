"""Regressions for selecting raw MCI sentence clips without changing sources."""

from pathlib import Path
import sys

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT / "code"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "analysis"))

from speech_data import canonical_task_key, discover_recordings  # noqa: E402


def raw_name(sentence: int, endpoint: int, *, whitespace: str = "", suffix: str = "") -> str:
    """Build an anonymized filename with the source archive's token layout."""

    return f"1.{whitespace}MCI-000000-001-X-X-00-{sentence}-{endpoint}{suffix}.wav"


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def raw_layout(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "NM50_AD50_MCI20"
    touch(data_root / "NM50_AD50" / "nm_random_50" / "synthetic-a" / "slice" / "06-01.wav")
    touch(data_root / "NM50_AD50" / "ad_random_50" / "synthetic-b" / "slice" / "07-01.wav")
    mci_patient = data_root / "중기부 001-021 데이터" / "synthetic-c"
    mci_patient.mkdir(parents=True)
    return data_root, mci_patient


@pytest.mark.parametrize(
    "sentence,endpoint,expected",
    [(61, 90, "06-01"), (90, 90, "06-30"), (91, 100, "07-01"), (100, 100, "07-10")],
)
def test_raw_mci_shared_task_boundaries(sentence, endpoint, expected):
    assert canonical_task_key(raw_name(sentence, endpoint), label="MCI") == expected
    assert canonical_task_key(raw_name(sentence, endpoint)) == expected


@pytest.mark.parametrize("whitespace", ["", " ", "   ", "\t"])
def test_numbered_mci_prefix_accepts_optional_whitespace(whitespace):
    assert canonical_task_key(raw_name(61, 90, whitespace=whitespace), label="mci") == "06-01"


@pytest.mark.parametrize("sentence,endpoint", [(60, 90), (91, 90), (90, 100), (101, 100)])
def test_raw_mci_outside_shared_sentence_ranges_is_excluded(sentence, endpoint):
    assert canonical_task_key(raw_name(sentence, endpoint), label="MCI") is None


@pytest.mark.parametrize("suffix", ["_1", "_2", "-1", "-2", "(clean)"])
def test_raw_mci_alternate_takes_are_excluded(suffix):
    assert canonical_task_key(raw_name(61, 90, suffix=suffix), label="MCI") is None


@pytest.mark.parametrize(
    "filename",
    [
        raw_name(61, 90).replace("1.MCI", "MCI"),
        raw_name(61, 90).replace("1.MCI", "1MCI"),
        raw_name(61, 90).replace("1.MCI", "1.NM"),
        raw_name(61, 90).replace("-001-", "-001-extra-"),
        raw_name(61, 90).replace("-001-", "-001_extra-"),
    ],
)
def test_raw_mci_malformed_prefix_or_extra_delimiters_are_excluded(filename):
    assert canonical_task_key(filename, label="MCI") is None


def test_task_helper_is_case_insensitive_and_does_not_apply_raw_mapping_to_other_classes():
    filename = raw_name(100, 100).replace("MCI", "mci").replace(".wav", ".WAV")
    assert canonical_task_key(filename, label="MCI") == "07-10"
    assert canonical_task_key(filename, label="NM") is None
    assert canonical_task_key(filename, label="AD") is None
    assert canonical_task_key("07-20.WAV", label="MCI") == "07-20"
    assert canonical_task_key("06-001.wav") == "06-01"


def test_korean_raw_folder_is_discovered_and_files_are_not_renamed(tmp_path):
    data_root, mci_patient = raw_layout(tmp_path)
    expected = {
        touch(mci_patient / raw_name(61, 90)),
        touch(mci_patient / raw_name(91, 100, whitespace=" ")),
    }
    touch(mci_patient / raw_name(60, 90))
    touch(mci_patient / raw_name(61, 90, suffix="_1"))
    touch(mci_patient / raw_name(91, 100, suffix="-1"))
    before = {path.relative_to(data_root) for path in data_root.rglob("*") if path.is_file()}

    records = discover_recordings(data_root)

    mci = [record for record in records if record.label == "MCI"]
    assert {Path(record.path) for record in mci} == {path.resolve() for path in expected}
    assert {record.patient_id for record in mci} == {"synthetic-c"}
    assert len(records) == 4
    assert {path.relative_to(data_root) for path in data_root.rglob("*") if path.is_file()} == before


def test_raw_mci_matcher_filters_by_mapped_task_not_original_filename(tmp_path):
    data_root, mci_patient = raw_layout(tmp_path)
    selected = touch(mci_patient / raw_name(61, 90))
    touch(mci_patient / raw_name(62, 90))
    records = discover_recordings(data_root, filename_pattern=r"(?:06|07)-01\.wav")
    assert [Path(record.path) for record in records if record.label == "MCI"] == [selected.resolve()]


@pytest.mark.parametrize("duplicate_kind", ["canonical_name", "second_raw_prefix"])
def test_duplicate_paths_for_one_patient_sentence_are_rejected(tmp_path, duplicate_kind):
    data_root, mci_patient = raw_layout(tmp_path)
    touch(mci_patient / raw_name(61, 90))
    duplicate = "06-01.wav" if duplicate_kind == "canonical_name" else raw_name(61, 90).replace("1.", "2.", 1)
    touch(mci_patient / duplicate)

    with pytest.raises(ValueError, match="same patient/sentence task"):
        discover_recordings(data_root)


def test_project_root_also_finds_nested_raw_archive(tmp_path):
    data_root, mci_patient = raw_layout(tmp_path / "dataset")
    touch(mci_patient / raw_name(100, 100))
    records = discover_recordings(tmp_path)
    assert len(records) == 3
    assert {record.label for record in records} == {"NM", "AD", "MCI"}
