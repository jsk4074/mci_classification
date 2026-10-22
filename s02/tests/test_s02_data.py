"""Synthetic cohort freezing checks; no clinical data or model inference."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import wave

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from s02_data import prepare_cohort, verify_cohort
import s02_data


def audio(path, value, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack('<h', value) * frames)


def annotation(path, age=70):
    data = {'id': path.stem,
            'metadata': {'age': age, 'gender': 'M', 'state': 'M',
                         'record_date': '2020-01-01', 'bitratio': 16},
            'utterance': []}
    path.with_suffix('.json').write_text(json.dumps(data), encoding='utf-8')


def fixture(tmp_path, new_patients=9):
    old_root = tmp_path / 'dataset/s00'
    old_root.mkdir(parents=True)
    rows = []
    value = 1000
    for label in ('NM', 'AD', 'MCI'):
        for split in ('train', 'val', 'test'):
            patient = f'old_{label}_{split}'
            path = old_root / split / label / patient / '06-01.wav'
            audio(path, value)
            value += 1
            rows.append({'path': str(path.relative_to(old_root)),
                         'patient_id': patient, 'label': label, 'split': split})
    manifest = old_root / 'split_manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'patient_id', 'label', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    full = tmp_path / 'full'
    for index in range(new_patients):
        patient = f'new_MCI_{index:03d}'
        name = f'1.MCI-{index:03d}-000-M-70-001-61-90.wav'
        path = full / 'outer_archive' / patient / name
        audio(path, 2000 + index)
        annotation(path)
    return manifest, full, tmp_path / 'output', rows


def prepared_rows(output):
    with (output / 'split_manifest.csv').open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def test_original_assignments_new_patient_splits_and_identical_shared_folds(tmp_path):
    manifest, full, output, old = fixture(tmp_path)
    audit = prepare_cohort(manifest, full, output, folds=2)
    rows = prepared_rows(output)
    kept = {row['patient_id']: row['split'] for row in rows if row['label'] != 'MCI'}
    assert kept == {row['patient_id']: row['split'] for row in old if row['label'] != 'MCI'}
    assert not any(row['patient_id'].startswith('old_MCI') for row in rows)
    new = [row for row in rows if row['label'] == 'MCI']
    assert {split: sum(row['split'] == split for row in new)
            for split in ('train', 'val', 'test')} == {'train': 7, 'val': 1, 'test': 1}
    assert audit['excluded_old_mci'] == {'patients': 3, 'recordings': 3}
    assert audit['total_patients'] == 15
    assert audit['new_mci']['metadata_used_as_model_features'] is False
    folds = json.loads((output / 'folds.json').read_text())
    test_ids = {row['patient_id'] for row in rows if row['split'] == 'test'}
    dev_ids = {row['patient_id'] for row in rows if row['split'] != 'test'}
    validated = []
    for fold in folds:
        assert set(fold['train_ids']).isdisjoint(fold['val_ids'])
        assert set(fold['train_ids']) | set(fold['val_ids']) == dev_ids
        assert test_ids.isdisjoint(fold['train_ids'] + fold['val_ids'])
        validated.extend(fold['val_ids'])
    assert sorted(validated) == sorted(dev_ids)
    assert json.loads((output / 'summary.json').read_text())['total_recordings'] == len(rows)
    assert 'new_MCI_' not in (output / 'cohort_audit.json').read_text()


def test_distinct_takes_retained_variants_and_noncanonical_tiny_clip_excluded(tmp_path):
    manifest, full, output, _ = fixture(tmp_path)
    original = sorted(full.rglob('*.wav'))[0]
    take = original.with_name(original.name.replace('1.', '2.', 1))
    audio(take, 3000)
    annotation(take)
    variant = original.with_name(original.stem + '_variant.wav')
    audio(variant, 3001)
    annotation(variant)
    outside = original.with_name(original.name.replace('1.', '3.', 1).replace('-61-90', '-20-27'))
    audio(outside, 3002, frames=1)
    annotation(outside)
    # An unchanged participant grouping survives redundant inner wrappers.
    last = sorted(full.rglob('*.wav'))[-1]
    wrapper = last.parent / last.parent.name
    wrapper.mkdir()
    last.rename(wrapper / last.name)
    last.with_suffix('.json').rename(wrapper / last.with_suffix('.json').name)
    audit = prepare_cohort(manifest, full, output, folds=2)
    assert audit['new_mci']['selected_recordings'] == 10
    assert audit['new_mci']['duplicate_task_slots'] == 1
    assert audit['new_mci']['recordings_in_duplicate_task_slots'] == 2
    assert audit['new_mci']['additional_distinct_takes'] == 1
    assert audit['new_mci']['excluded_recordings'] == {'outside_canonical_06_07': 2}
    rows = prepared_rows(output)
    assert sum(row['path'] in (str(original), str(take)) for row in rows) == 2
    assert all('_variant' not in row['path'] and '-20-27' not in row['path'] for row in rows)


@pytest.mark.parametrize('problem', ['state', 'age', 'missing_pair', 'short', 'same_bytes'])
def test_invalid_metadata_pair_short_task_and_cross_patient_content_rejected(tmp_path, problem):
    manifest, full, output, _ = fixture(tmp_path)
    files = sorted(full.rglob('*.wav'))
    first = files[0]
    if problem in ('state', 'age'):
        second = first.with_name(first.name.replace('1.', '2.', 1).replace('-61-90', '-62-90'))
        audio(second, 4000)
        annotation(second)
        path = second.with_suffix('.json')
        data = json.loads(path.read_text())
        data['metadata'][problem] = 'AD' if problem == 'state' else 71
        path.write_text(json.dumps(data))
    elif problem == 'missing_pair':
        first.with_suffix('.json').unlink()
    elif problem == 'short':
        audio(first, 4000, frames=1)
    else:
        files[1].write_bytes(first.read_bytes())
    with pytest.raises(ValueError):
        prepare_cohort(manifest, full, output, folds=2)
    assert not output.exists()


def test_frozen_rerun_preserves_outputs_and_rejects_data_settings_and_tampering(tmp_path):
    manifest, full, output, _ = fixture(tmp_path)
    prepare_cohort(manifest, full, output, folds=2)
    original = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in output.iterdir()}
    verify_cohort(output)
    assert original == {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                        for path in output.iterdir()}
    with pytest.raises(ValueError, match='Frozen cohort'):
        prepare_cohort(manifest, full, output, folds=2, seed=43)
    first = sorted(full.rglob('*.wav'))[0]
    audio(first, 5000)
    with pytest.raises(ValueError, match='Frozen cohort'):
        verify_cohort(output)
    assert original == {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                        for path in output.iterdir()}


def test_frozen_source_fingerprint_and_nonselected_annotation_changes_rejected(tmp_path):
    manifest, full, output, _ = fixture(tmp_path)
    original = sorted(full.rglob('*.wav'))[0]
    outside = original.with_name(original.name.replace('-61-90', '-20-27'))
    audio(outside, 5001)
    annotation(outside)
    prepare_cohort(manifest, full, output, folds=2)
    payload = json.loads(outside.with_suffix('.json').read_text())
    payload['utterance'].append({'synthetic': True})
    outside.with_suffix('.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Frozen cohort'):
        verify_cohort(output)


def test_copy_provenance_mismatch_and_quarantined_old_patient_rejected(tmp_path):
    manifest, full, output, rows = fixture(tmp_path)
    index = manifest.parent / 'copy_index.csv'
    with index.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['destination_path', 'sha256'])
        writer.writeheader()
        writer.writerow({'destination_path': rows[0]['path'], 'sha256': '0' * 64})
    with pytest.raises(ValueError, match='SHA-256'):
        prepare_cohort(manifest, full, output, folds=2)
    index.unlink()
    quarantine = tmp_path / 'quarantine.csv'
    quarantine.write_text('patient_id\n' + rows[0]['patient_id'] + '\n')
    with pytest.raises(ValueError, match='quarantined'):
        prepare_cohort(manifest, full, output, folds=2, old_quarantine=quarantine)


def test_production_patient_count_has_108_23_23_new_assignments(tmp_path):
    manifest, full, output, _ = fixture(tmp_path, new_patients=154)
    audit = prepare_cohort(manifest, full, output, folds=2)
    assert {split: audit['splits'][split]['classes']['MCI']['patients']
            for split in ('train', 'val', 'test')} == {'train': 108, 'val': 23, 'test': 23}


def test_old_absolute_paths_rebased_and_checked_against_copy_provenance(tmp_path):
    manifest, full, output, rows = fixture(tmp_path)
    with (manifest.parent / 'copy_index.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['destination_path', 'sha256'])
        writer.writeheader()
        for row in rows:
            path = manifest.parent / row['path']
            writer.writerow({'destination_path': row['path'],
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'patient_id', 'label', 'split'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'path': str(Path('/obsolete/synthetic-project/dataset/s00') / row['path'])})
    audit = prepare_cohort(manifest, full, output, folds=2)
    assert audit['old_paths_rebased'] == 6
    assert audit['old_copy_hashes_verified'] == 6
    assert all(Path(row['path']).is_file() for row in prepared_rows(output))


def test_ambiguous_old_dataset_suffix_requires_portable_manifest(tmp_path):
    manifest, full, output, rows = fixture(tmp_path)
    rows[0]['path'] = str(Path('/obsolete/dataset/project/dataset/s00') / rows[0]['path'])
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'patient_id', 'label', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='portable manifest'):
        prepare_cohort(manifest, full, output, folds=2)
    assert not output.exists()


def test_changed_ingestion_source_and_modified_frozen_outputs_are_rejected(tmp_path, monkeypatch):
    manifest, full, output, _ = fixture(tmp_path)
    repository = tmp_path / 'ingestion_sources'
    for name in ('s02/code/s02_data.py', 's00/code/speech_data.py', 's01/code/s01_data.py'):
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s02_data.REPOSITORY / name, target)
    monkeypatch.setattr(s02_data, 'REPOSITORY', repository)
    prepare_cohort(manifest, full, output, folds=2)
    source = repository / 's00/code/speech_data.py'
    original = source.read_bytes()
    source.write_bytes(original + b'\n# synthetic dependency change\n')
    with pytest.raises(ValueError, match='Frozen cohort'):
        verify_cohort(output)
    source.write_bytes(original)
    (output / 'summary.json').write_text('{}\n')
    with pytest.raises(ValueError, match='Frozen cohort'):
        verify_cohort(output)
