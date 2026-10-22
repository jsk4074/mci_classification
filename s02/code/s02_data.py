"""Prepare the frozen s02 comparison cohort without altering source recordings.

``prepare_cohort(old_manifest, full_root, output_dir, seed=42, folds=5,
old_quarantine=None)`` retains approved NM/AD assignments, replaces old MCI with
all new participants, and returns an aggregate audit. The new corpus may contain
one enclosing archive directory; its immediate participant directories supply
the patient IDs, including when an inner redundant directory is present.

Outputs are split_manifest.csv, manifest_fingerprint.json, folds.json,
summary.json, cohort_audit.json, and cohort_config.json. These contain private
provenance except the aggregate summary/audit and belong in ignored run folders.
An identical rerun verifies and preserves every output byte; changed inputs,
ingestion source, settings, or frozen outputs require a new output directory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import wave


REPOSITORY = Path(__file__).resolve().parents[2]
LABELS = ('NM', 'AD', 'MCI')
SPLITS = ('train', 'val', 'test')
MIN_SECONDS = 0.025


def _module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _helpers():
    return (
        _module('_s02_original_speech_data', REPOSITORY / 's00/code/speech_data.py'),
        _module('_s02_original_s01_data', REPOSITORY / 's01/code/s01_data.py'),
    )


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _hash(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('A source file changed during cohort preparation.')
    return digest.hexdigest(), after


def _patient_component(value):
    if (not value or value != value.strip() or value in ('.', '..')
            or any(ch in value for ch in ('/', '\\', '\x00', '\n', '\r'))):
        raise ValueError('Patient IDs must be unchanged safe directory components.')
    return value


def _existing_records(manifest, quarantine, canonical, data):
    required = {'path', 'patient_id', 'label', 'split'}
    rows = []
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if (reader.fieldnames is None or not required.issubset(reader.fieldnames)
                or len(reader.fieldnames) != len(set(reader.fieldnames))):
            raise ValueError('The approved manifest requires unique path/patient_id/label/split columns.')
        for row in reader:
            if None in row or any(not (row.get(key) or '').strip() for key in required):
                raise ValueError('Every approved manifest row requires four complete fields.')
            rows.append({key: row[key].strip() for key in required})
    if not rows:
        raise ValueError('The approved manifest is empty.')
    quarantined = set()
    if quarantine is not None:
        with quarantine.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or 'patient_id' not in reader.fieldnames:
                raise ValueError('The optional quarantine CSV requires patient_id.')
            quarantined = {row['patient_id'].strip() for row in reader}
    index = {}
    copy_index = manifest.parent / 'copy_index.csv'
    if copy_index.is_file():
        with copy_index.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and {'destination_path', 'sha256'} <= set(reader.fieldnames):
                for row in reader:
                    digest = row['sha256'].strip()
                    if not re.fullmatch('[0-9a-f]{64}', digest):
                        raise ValueError('The copy provenance index has an invalid SHA-256.')
                    index[str((manifest.parent / row['destination_path']).resolve())] = digest
    records = []
    excluded = []
    rebased = 0
    verified = 0
    for row in rows:
        label, split = row['label'].upper(), row['split'].lower()
        patient = _patient_component(row['patient_id'])
        if label not in LABELS or split not in SPLITS:
            raise ValueError('The approved manifest contains an unknown label or original split.')
        if label == 'MCI':
            excluded.append((patient, split))
            continue
        if patient in quarantined:
            raise ValueError('The approved NM/AD cohort includes a quarantined patient.')
        path = Path(row['path']).expanduser()
        was_absolute = path.is_absolute()
        was_rebased = False
        if not was_absolute:
            path = manifest.parent / path
        if not path.is_file():
            dataset_root = manifest.parent.parent
            positions = [index for index, part in enumerate(path.parts) if part == 'dataset']
            if (not was_absolute or dataset_root.name != 'dataset'
                    or len(positions) != 1):
                raise ValueError('An approved NM/AD recording is missing; supply a portable manifest.') from None
            relative = Path(*path.parts[positions[0] + 1:])
            candidate = (dataset_root / relative).resolve()
            if not candidate.is_relative_to(dataset_root) or not candidate.is_file():
                raise ValueError('An old absolute path cannot be rebased to an existing recording.')
            path = candidate
            rebased += 1
            was_rebased = True
        path = path.resolve(strict=True)
        if was_rebased and copy_index.is_file() and str(path) not in index:
            raise ValueError('A rebased recording is missing its available copy provenance SHA-256.')
        if canonical(path.name, label=label) is None:
            raise ValueError('The approved NM/AD manifest contains a recording outside canonical 06/07 tasks.')
        if str(path) in index:
            if _hash(path)[0] != index[str(path)]:
                raise ValueError('An approved recording differs from its saved copy SHA-256.')
            verified += 1
        records.append(data.Record(str(path), patient, label, split))
    bags = data.group_patients(records)
    if any({bag['label'] for bag in bags if bag['split'] == split} != {'NM', 'AD'}
           for split in SPLITS):
        raise ValueError('Every original split must retain both NM and AD patients.')
    return records, {'patients': len({patient for patient, _ in excluded}),
                     'recordings': len(excluded)}, rebased, verified


def _new_records(full_root, canonical, data, seed):
    all_files = sorted((p for p in full_root.rglob('*') if p.is_file()),
                       key=lambda p: p.relative_to(full_root).as_posix())
    wavs = [p for p in all_files if p.suffix.lower() == '.wav']
    jsons = [p for p in all_files if p.suffix.lower() == '.json']
    if not wavs:
        raise ValueError('The full dataset contains no WAV recordings.')
    relative_parts = [p.relative_to(full_root).parts for p in wavs]
    wrapped = (len({parts[0] for parts in relative_parts}) == 1
               and all(len(parts) >= 3 for parts in relative_parts))
    patient_level = 1 if wrapped else 0
    pairs = {}
    for path in wavs + jsons:
        key = path.relative_to(full_root).with_suffix('').as_posix().casefold()
        kind = path.suffix.lower()
        if (key, kind) in pairs:
            raise ValueError('The full dataset has duplicate case-insensitive WAV/JSON paths.')
        pairs[key, kind] = path
    if ({key for key, kind in pairs if kind == '.wav'}
            != {key for key, kind in pairs if kind == '.json'}):
        raise ValueError('The full dataset requires exactly one JSON beside every WAV.')
    metadata_by_patient = {}
    identity_by_patient = defaultdict(set)
    ids = set()
    inventory = []
    selected = []
    rejected = Counter()
    input_owners = {}
    slots = defaultdict(list)
    durations = []
    for wav in wavs:
        relative = wav.relative_to(full_root)
        if len(relative.parts) < patient_level + 2:
            raise ValueError('WAV files must remain beneath their original participant folder.')
        patient = _patient_component(relative.parts[patient_level])
        key = relative.with_suffix('').as_posix().casefold()
        annotation = pairs[key, '.json']
        try:
            payload = json.loads(annotation.read_text(encoding='utf-8-sig'))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError('A full-dataset JSON annotation is invalid.') from exc
        if not isinstance(payload, dict):
            raise ValueError('A full-dataset JSON annotation must contain an object.')
        meta = payload.get('metadata', {})
        required = ('age', 'gender', 'state', 'record_date', 'bitratio')
        if (not isinstance(meta, dict) or any(field not in meta or meta[field] is None for field in required)
                or meta['state'] != 'M' or not isinstance(payload.get('utterance'), list)):
            raise ValueError('Every full-dataset annotation must have consistent MCI state M and required metadata.')
        identity = payload.get('id')
        if not isinstance(identity, str) or identity != wav.stem or identity in ids:
            raise ValueError('Annotation IDs must uniquely match recording stems, not define patient grouping.')
        ids.add(identity)
        signature = {field: meta[field] for field in required}
        if metadata_by_patient.setdefault(patient, signature) != signature:
            raise ValueError('A participant folder has inconsistent age, gender, state, date, or bitrate metadata.')
        task = canonical(wav.name, label='MCI')
        match = re.fullmatch(r'\d+\.\s*(MCI-(?:[^-_]+-){5})\d{2,3}-(90|100)\.wav', wav.name, re.I)
        if task and match:
            identity_by_patient[patient].add(match[1])
        for source in (wav, annotation):
            digest, details = _hash(source)
            inventory.append({'path': source.relative_to(full_root).as_posix(),
                              'sha256': digest, 'bytes': details.st_size,
                              'mtime_ns': details.st_mtime_ns})
            if source == wav:
                previous = input_owners.setdefault(digest, patient)
                if previous != patient:
                    raise ValueError('Identical full-dataset audio bytes occur under different patients.')
                content_hash = digest
        if task is None:
            rejected['outside_canonical_06_07'] += 1
            continue
        try:
            with wave.open(str(wav), 'rb') as audio:
                duration = audio.getnframes() / audio.getframerate()
                if audio.getcomptype() != 'NONE' or audio.getnchannels() < 1:
                    raise ValueError('Canonical recordings require valid uncompressed PCM WAV audio.')
        except (wave.Error, EOFError, ZeroDivisionError) as exc:
            raise ValueError('A canonical full-dataset WAV header is invalid.') from exc
        if duration < MIN_SECONDS:
            raise ValueError('A canonical recording is shorter than the 25-ms encoder minimum.')
        durations.append(duration)
        slots[patient, task].append(content_hash)
        selected.append((wav.resolve(strict=True), patient, task))
    if any(len(prefixes) > 1 for prefixes in identity_by_patient.values()):
        raise ValueError('A participant folder mixes canonical recording identity prefixes.')
    for hashes in slots.values():
        if len(hashes) != len(set(hashes)):
            raise ValueError('Duplicate identical takes would double-weight a patient/task slot.')
    patients = sorted(metadata_by_patient)
    selected_patients = {patient for _, patient, _ in selected}
    if set(patients) != selected_patients:
        raise ValueError('Every new participant must have at least one canonical 06/07 recording.')
    if len(patients) < 3:
        raise ValueError('At least three new MCI participants are needed for train/val/test.')
    shuffled = patients.copy()
    random.Random(seed).shuffle(shuffled)
    validation = max(1, round(len(patients) * .15))
    test = max(1, round(len(patients) * .15))
    training = len(patients) - validation - test
    if training < 1:
        raise ValueError('The new MCI cohort cannot populate every split.')
    assignments = {patient: split
                   for split, members in [('train', shuffled[:training]),
                                          ('val', shuffled[training:training + validation]),
                                          ('test', shuffled[training + validation:])]
                   for patient in members}
    records = [data.Record(str(path), patient, 'MCI', assignments[patient])
               for path, patient, _ in selected]
    repeated = {key: hashes for key, hashes in slots.items() if len(hashes) > 1}
    audit = {'all_participants': len(patients), 'all_recordings': len(wavs),
             'annotation_state_counts': {'M': len(wavs)},
             'excluded_recordings': dict(rejected),
             'selected_recordings': len(records),
             'selected_min_seconds': min(durations), 'selected_max_seconds': max(durations),
             'duplicate_task_slots': len(repeated),
             'recordings_in_duplicate_task_slots': sum(map(len, repeated.values())),
             'additional_distinct_takes': sum(len(values) - 1 for values in repeated.values()),
             'participants_with_multiple_takes': len({patient for patient, _ in repeated}),
             'multiple_takes_policy': 'Retain every distinct recording in repeated patient/task slots; all identity fields agree.',
             'patient_identity': 'Preserved participant directory; annotation id identifies a recording.',
             'metadata_used_as_model_features': False,
             'participant_directory_level': patient_level}
    # Also detect ordinary source mutation after its individual hash was read.
    for row in inventory:
        current = (full_root / row['path']).stat()
        if (current.st_size, current.st_mtime_ns) != (row['bytes'], row['mtime_ns']):
            raise ValueError('A full-dataset source changed during cohort preparation.')
    return records, audit, _digest(inventory)


def _manifest_text(records):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=['path', 'patient_id', 'label', 'split'],
                            lineterminator='\n')
    writer.writeheader()
    writer.writerows(asdict(record) for record in records)
    return buffer.getvalue()


def prepare_cohort(old_manifest: Path, full_root: Path, output_dir: Path, *,
                   seed: int = 42, folds: int = 5,
                   old_quarantine: Path | None = None) -> dict:
    """Validate and freeze a shared s00/s01 cohort; return only aggregate audit.

    Both methods must read the emitted manifest and folds unchanged. Old NM/AD
    patient IDs and split roles remain exact; sorted new MCI IDs are shuffled
    with ``random.Random(seed)`` and allocated about 70/15/15 (108/23/23 for154).
    Clinical metadata validates identity only and never enters model features.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer.')
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError('folds must be an integer of at least two.')
    old_manifest = Path(old_manifest).expanduser().resolve(strict=True)
    full_root = Path(full_root).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().absolute()
    if output_dir.is_symlink():
        raise ValueError('The cohort output directory cannot be a symlink.')
    if not old_manifest.is_file() or not full_root.is_dir():
        raise ValueError('Provide an approved manifest file and extracted full dataset directory.')
    quarantine = (Path(old_quarantine).expanduser().resolve(strict=True)
                  if old_quarantine is not None else None)
    original, data = _helpers()
    old, excluded, rebased, verified = _existing_records(
        old_manifest, quarantine, original.canonical_task_key, data)
    new, new_audit, inventory_sha = _new_records(
        full_root, original.canonical_task_key, data, seed)
    if {record.patient_id for record in old} & {record.patient_id for record in new}:
        raise ValueError('Old NM/AD and new MCI patient identifiers overlap.')
    records = sorted(old + new, key=lambda record: (record.patient_id, record.path))
    data.group_patients(records)
    if len({record.path for record in records}) != len(records):
        raise ValueError('The comparison cohort contains duplicate resolved recording paths.')
    snapshot = data.fingerprint(records)
    content_owners = {}
    for row in snapshot['files']:
        if content_owners.setdefault(row['content_sha256'], row['patient_id']) != row['patient_id']:
            raise ValueError('Identical audio bytes cross patient boundaries in the combined cohort.')
    development_folds = data.make_development_folds(records, n_splits=folds, seed=seed)
    aggregate = data.summary(records)
    source_tasks = {}
    for source, source_records in [('existing_approved_NM_AD', old), ('full_dataset_MCI', new)]:
        source_tasks[source] = {}
        for split in SPLITS:
            selected = [record for record in source_records if record.split == split]
            source_tasks[source][split] = {
                task: {'patients': len({record.patient_id for record in selected
                                        if original.canonical_task_key(Path(record.path).name, record.label).startswith(task + '-')}),
                       'recordings': sum(original.canonical_task_key(Path(record.path).name, record.label).startswith(task + '-')
                                         for record in selected)}
                for task in ('06', '07')}
    audit = {**aggregate, 'new_mci': new_audit, 'excluded_old_mci': excluded,
             'old_NM_AD_assignments_preserved': True,
             'old_paths_rebased': rebased, 'old_copy_hashes_verified': verified,
             'source_split_task_counts': source_tasks,
             'manifest_sha256': snapshot['sha256'], 'development_fold_count': folds,
             'domain_caveat': 'NM/AD come from the existing approved source and MCI from the new archive; diagnosis and source are confounded. Original NM/AD test patients were evaluated in earlier experiments.',
             'eligibility': 'Same original canonical 06/07 parser for both methods; no underscore variants; all eligible distinct takes retained; minimum25ms.'}
    config = {'schema': 's02-cohort-v1', 'seed': seed, 'folds': folds,
              'minimum_clip_seconds': MIN_SECONDS, 'new_mci_split_ratios': [.70, .15, .15],
              'old_manifest': str(old_manifest), 'old_manifest_sha256': _hash(old_manifest)[0],
              'full_root': str(full_root), 'full_inventory_sha256': inventory_sha,
              'old_quarantine': str(quarantine) if quarantine is not None else None,
              'old_quarantine_sha256': _hash(quarantine)[0] if quarantine is not None else None,
              'source_hashes': {name: _hash(REPOSITORY / name)[0] for name in
                                ('s02/code/s02_data.py', 's00/code/speech_data.py', 's01/code/s01_data.py')}}
    payloads = {'split_manifest.csv': _manifest_text(records),
                'manifest_fingerprint.json': json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                'folds.json': json.dumps(development_folds, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                'summary.json': json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                'cohort_audit.json': json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                'cohort_config.json': json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + '\n'}
    present = [name for name in payloads if (output_dir / name).exists() or (output_dir / name).is_symlink()]
    if present:
        if len(present) != len(payloads):
            raise ValueError('The frozen cohort is incomplete; use a new output directory.')
        for name, text in payloads.items():
            path = output_dir / name
            if path.is_symlink() or not path.is_file() or path.read_text(encoding='utf-8') != text:
                raise ValueError('Frozen cohort settings, sources, data, or outputs changed; use a new output directory.')
        return audit
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in payloads.items():
        fd, temporary = tempfile.mkstemp(prefix='.cohort-', suffix='.tmp', dir=output_dir)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle:
                handle.write(text)
            os.replace(temporary, output_dir / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return audit


def verify_cohort(output_dir: Path) -> None:
    """Recheck all frozen sources, inventory, settings and outputs without writes."""
    output_dir = Path(output_dir)
    config = json.loads((output_dir / 'cohort_config.json').read_text(encoding='utf-8'))
    prepare_cohort(Path(config['old_manifest']), Path(config['full_root']), output_dir,
                   seed=config['seed'], folds=config['folds'],
                   old_quarantine=(Path(config['old_quarantine'])
                                   if config.get('old_quarantine') else None))
