"""Aggregate dataset readiness is distinct from model evaluation and patient rows."""

from copy import deepcopy
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT / "code"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "analysis"))

from speech_report import write_report


class ReportHTML(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.elements = []
        self.tables = []
        self.table = self.row = self.cell = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        if tag == "table":
            self.table = {"attributes": attributes, "rows": []}
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"th", "td"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table["rows"].append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None

    @property
    def visible_text(self):
        return " ".join(" ".join(self.text).split())


def audit_fixture():
    classes = {}
    counts = {
        "NM": (50, 48, 1839, 2, 0, 96000),
        "AD": (50, 32, 1065, 18, 3, 96000),
        "MCI": (20, 20, 790, 0, 0, 16000),
    }
    for label, (source, included, clips, excluded, empty, rate) in counts.items():
        classes[label] = {
            "source_patient_folders": source,
            "included_patients": included,
            "included_recordings": clips,
            "excluded_patient_folders": excluded,
            "empty_patient_folders": empty,
            "audio_formats": [{"sample_rate": rate, "channels": 1,
                               "subtype": "PCM_16", "recordings": clips}],
            "duration_seconds": {"min": 0.25, "median": 2.5, "max": 58},
            "shorter_than_window": 42,
            "longer_than_eval_span": 7,
        }
    return {
        "status": "verified",
        "selection": "Canonical 06-01..30 and 07-01..10; original MCI filename mapping",
        "classes": classes,
        "total_recordings": 3694,
        "total_patients": 100,
        "recovered_archive_recordings": 3,
        "audio_header_errors": 0,
        "complete_data_chunk_errors": 0,
        "duplicate_content_count": 0,
        "window_seconds": 5,
        "evaluation_span_seconds": 25,
        "notes": ["Canonical slices were selected without modifying source recordings."],
    }


def save_audit(output, audit):
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_audit.json").write_text(json.dumps(audit), encoding="utf-8")


def availability_table(document):
    return next(table["rows"] for table in document.tables
                if table["attributes"].get("aria-label") == "Dataset source availability")


def test_dataset_source_counts_and_exclusions_are_separate_from_patient_splits(tmp_path):
    save_audit(tmp_path, audit_fixture())
    report = write_report(tmp_path, status="prepared")
    document = ReportHTML(report.read_text())
    rows = availability_table(document)
    assert rows[1][:6] == ["NM", "50", "48", "1,839", "2", "0"]
    assert rows[2][:6] == ["AD", "50", "32", "1,065", "18", "3"]
    assert rows[3][:6] == ["MCI", "20", "20", "790", "0", "0"]
    assert rows[1][6:] == ["0.2500 / 2.5000 / 58.0000", "42", "7"]
    text = document.visible_text
    assert "Included patients: 100" in text
    assert "Included clips: 3,694" in text
    assert "No validated split manifest yet" in text
    assert "No final test results yet" in text
    assert not (tmp_path / "metrics.json").exists()


def test_source_formats_and_audit_scope_are_explained_without_claiming_label_verification(tmp_path):
    save_audit(tmp_path, audit_fixture())
    document = ReportHTML(write_report(tmp_path).read_text())
    text = document.visible_text
    assert "Source audio formats" in text
    assert "96,000" in text and "16,000" in text and "PCM_16" in text
    assert "NM/AD recordings are mostly 96 kHz" in text
    assert "MCI recordings are mostly 16 kHz" in text
    assert "does not remove source, device, or cohort differences" in text
    assert "SHA-256 byte duplicate copies found: 0" in text
    assert "Audio header errors: 0" in text
    assert "Incomplete audio data chunks: 0" in text
    assert "Recordings recovered from the archive: 3" in text
    assert "They do not verify clinical labels" in text


def test_missing_audit_has_no_fabricated_readiness_counts_or_results(tmp_path):
    report = write_report(tmp_path, status="prepared")
    document = ReportHTML(report.read_text())
    assert "No dataset audit yet" in document.visible_text
    assert not any(table["attributes"].get("aria-label") == "Dataset source availability"
                   for table in document.tables)
    assert "No final test results yet" in document.visible_text
    assert not (tmp_path / "dataset_audit.json").exists()


def test_unavailable_audit_fields_remain_unknown_instead_of_becoming_zero(tmp_path):
    save_audit(tmp_path, {"status": "partial", "classes": {"NM": {"source_patient_folders": 50}}})
    document = ReportHTML(write_report(tmp_path).read_text())
    assert availability_table(document)[1][:6] == ["NM", "50", "—", "—", "—", "—"]
    assert "SHA-256 byte duplicate copies found: —" in document.visible_text
    assert "Audio audit status: partial" in document.visible_text


def test_audit_notes_and_format_strings_are_escaped_and_patient_rows_are_ignored(tmp_path):
    audit = audit_fixture()
    payload = '<script>alert("source");</script><img src=x onerror="alert(1)">'
    identifier = "PRIVATE_PATIENT_249af"
    audit["selection"] = payload
    audit["notes"] = [payload, f"Recovered archive entry /private/audio/{identifier}/06-01.wav"]
    audit["classes"]["NM"]["audio_formats"][0]["subtype"] = payload
    audit["patient_rows"] = [{"patient_id": identifier, "path": f"/private/audio/{identifier}"}]
    audit["classes"]["AD"]["excluded_patient_ids"] = [identifier]
    save_audit(tmp_path, audit)
    source = write_report(tmp_path).read_text()
    assert payload in unescape(source)
    assert "&lt;script&gt;" in source
    assert "<script>alert" not in source and "<img src=x" not in source
    assert identifier not in source and "/private/audio/" not in source
    assert "[local path omitted]" in source
    assert not any(tag == "img" and attrs.get("src") == "x"
                   for tag, attrs in ReportHTML(source).elements)


def test_audit_artifact_link_resolves_from_a_custom_report_location(tmp_path):
    output = tmp_path / "run #1"
    save_audit(output, audit_fixture())
    report = write_report(output, report_path=tmp_path / "reports" / "status.html")
    document = ReportHTML(report.read_text())
    hrefs = [attrs["href"] for tag, attrs in document.elements if tag == "a" and "href" in attrs]
    audit_link = next(href for href in hrefs if urlsplit(href).path.endswith("dataset_audit.json"))
    parsed = urlsplit(audit_link)
    assert not parsed.scheme and not parsed.netloc and not parsed.fragment
    assert "%20" in audit_link and "%23" in audit_link
    assert (report.parent / unquote(parsed.path)).resolve() == (output / "dataset_audit.json").resolve()


def test_regeneration_reads_new_audit_counts_without_changing_run_status(tmp_path):
    audit = audit_fixture()
    save_audit(tmp_path, audit)
    write_report(tmp_path, status="prepared")
    updated = deepcopy(audit)
    updated["classes"]["NM"]["included_patients"] = 45
    updated["total_patients"] = 97
    save_audit(tmp_path, updated)
    document = ReportHTML(write_report(tmp_path).read_text())
    assert availability_table(document)[1][2] == "45"
    assert "Included patients: 97" in document.visible_text
    assert json.loads((tmp_path / "report_state.json").read_text())["status"] == "prepared"
    assert "No final test results yet" in document.visible_text


def test_incomplete_audit_file_reports_missing_information_without_inventing_counts(tmp_path):
    (tmp_path / "dataset_audit.json").write_text('{"status":')
    document = ReportHTML(write_report(tmp_path).read_text())
    assert "No dataset audit yet" in document.visible_text
    assert "Could not read dataset_audit.json" in document.visible_text
    assert "No final test results yet" in document.visible_text


def test_unresolved_duplicates_remain_visible_when_splits_are_prepared(tmp_path):
    audit = audit_fixture()
    audit["status"] = "issues_found"
    audit["duplicate_content_count"] = 20
    save_audit(tmp_path, audit)
    document = ReportHTML(write_report(tmp_path, status="prepared").read_text())
    assert "Audio audit status: issues_found" in document.visible_text
    assert "SHA-256 byte duplicate copies found: 20" in document.visible_text
    assert "prepared splits do not establish dataset readiness" in document.visible_text
    assert "No final test results yet" in document.visible_text


def test_excluded_duplicate_copies_are_distinct_from_remaining_duplicates(tmp_path):
    audit = audit_fixture()
    audit.update(duplicate_content_count=20, excluded_duplicate_recordings=20,
                 remaining_duplicate_content_count=0, total_recordings=3674)
    audit["classes"]["NM"]["included_recordings"] = 1819
    save_audit(tmp_path, audit)
    document = ReportHTML(write_report(tmp_path, status="prepared").read_text())
    text = document.visible_text
    assert "Audio audit status: verified" in text
    assert "Included clips: 3,674" in text
    assert "SHA-256 byte duplicate copies found: 20" in text
    assert "Duplicate clips excluded: 20" in text
    assert "Remaining SHA-256 byte duplicate copies: 0" in text
    assert "Included clip counts reflect exclusion of 20 exact duplicate copies" in text
    assert "original source files are preserved" in text
    assert "unresolved issues" not in text


def test_quarantined_folders_are_distinct_from_duplicate_copies(tmp_path):
    audit = audit_fixture()
    audit.update(status="verified_subset", duplicate_content_count=20,
                 remaining_duplicate_content_count=0,
                 quarantined_patient_folders=2, quarantined_recordings=79,
                 total_patients=98, total_recordings=3615)
    save_audit(tmp_path, audit)
    document = ReportHTML(write_report(tmp_path, status="prepared").read_text())
    text = document.visible_text
    assert "Included patients: 98" in text
    assert "Included clips: 3,615" in text
    assert "Patient folders excluded pending source review: 2" in text
    assert "Recordings excluded with those folders: 79" in text
    assert "Remaining SHA-256 byte duplicate copies: 0" in text
    assert "inconsistent participant metadata" in text
    assert "20 exact duplicate copies" not in text


def test_copy_report_uses_verified_aggregates_without_source_identifiers(tmp_path):
    private_identifier = "PRIVATE_SOURCE_PATIENT_48"
    summary = {
        "status": "complete", "total_recordings": 3615, "total_patients": 98,
        "verified_copies": 3615, "total_bytes": 2969891964,
        "destination": str(tmp_path / "dataset/s00"),
        "source_manifest": f"/private/{private_identifier}/original.csv",
        "patient_rows": [{"patient_id": private_identifier}],
        "splits": {},
    }
    (tmp_path / "dataset_copy.json").write_text(json.dumps(summary), encoding="utf-8")
    source = write_report(tmp_path, status="prepared").read_text()
    document = ReportHTML(source)
    text = document.visible_text
    assert "Model dataset copy" in text
    assert "Verified copies: 3,615" in text
    assert "original files remain intact" in text
    assert "./dataset/s00/" in text
    assert "06-01.wav" in text and "07-10.wav" in text
    assert "applied at runtime" in text
    assert "No final test results yet" in text
    assert private_identifier not in source
    assert any(tag == "a" and attrs.get("href") == "dataset_copy.json"
               for tag, attrs in document.elements)


def test_copy_report_does_not_claim_completion_from_unverified_counts(tmp_path):
    summary = {"status": "complete", "total_recordings": 9,
               "total_patients": 9, "verified_copies": 8}
    (tmp_path / "dataset_copy.json").write_text(json.dumps(summary), encoding="utf-8")
    text = ReportHTML(write_report(tmp_path, status="prepared").read_text()).visible_text
    assert "Copy verification is not complete" in text
    assert "original files remain intact" not in text
