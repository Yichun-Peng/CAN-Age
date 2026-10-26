"""Standard-library job control for disjoint, resumable anatomical ROI workers."""
import argparse
import csv
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

EXPECTED_IMAGES = 1143
CHANNELS = ['red', 'green', 'blue', 'hue', 'saturation', 'value', 'lab_l', 'lab_a', 'lab_b']
ARTIFACT_DIRS = ['beak_masks', 'feather_masks', 'aligned_foreground_masks',
                 'aligned_beak_masks', 'aligned_feather_masks', 'aligned_rgba']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None,
                        help='Exclusive upper image index, retained from the single-process CLI.')
    parser.add_argument('--start', type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--records-only', action='store_true',
                       help='Write only assigned image artifacts and worker-specific job files.')
    modes.add_argument('--aggregate-only', action='store_true',
                       help='Validate completed workers and export all records without model imports.')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-id', type=int, default=None)
    args = parser.parse_args(argv)
    if args.shards < 1 or args.start < 0 or (args.limit is not None and args.limit < 0):
        parser.error('shards must be positive; start and limit must be nonnegative')
    if args.shard_id is not None and not 0 <= args.shard_id < args.shards:
        parser.error('shard-id must satisfy 0 <= shard-id < shards')
    if args.aggregate_only:
        if args.shard_id is not None or args.start != 0 or args.limit is not None:
            parser.error('aggregate-only does not accept shard-id, start or limit')
    elif args.records_only:
        if args.shard_id is None:
            if args.shards != 1:
                parser.error('records-only with multiple shards requires shard-id')
            args.shard_id = 0
    elif args.shards != 1 or args.shard_id is not None:
        parser.error('sharded inference requires records-only')
    return args


def assigned_indices(n_images, shards, shard_id):
    if n_images < 0 or shards < 1 or not 0 <= shard_id < shards:
        raise ValueError('Invalid image count or shard assignment')
    return list(range(shard_id, n_images, shards))


def worker_directory(base, shards, shard_id):
    assigned_indices(0, shards, shard_id)
    return Path(base) / 'worker_jobs' / f'shard_{shard_id:04d}_of_{shards:04d}'


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.part')
    try:
        with temporary.open('w') as stream:
            json.dump(obj, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def record_files_hash(base, indices):
    digest = hashlib.sha256()
    for index in sorted(indices):
        path = Path(base) / 'feature_records' / f'{index:04d}.json'
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_metadata(path):
    with Path(path).open(newline='') as stream:
        metadata = list(csv.DictReader(stream))
    if len(metadata) != EXPECTED_IMAGES:
        raise ValueError(f'Expected {EXPECTED_IMAGES} metadata rows; found {len(metadata)}')
    for index, row in enumerate(metadata):
        if int(row['image_index']) != index:
            raise ValueError('Metadata image indices must be ordered and contiguous')
    return metadata


def load_records(base, metadata, indices=None, require_complete=False, require_artifacts=False):
    base = Path(base)
    expected = set(range(len(metadata))) if indices is None else set(indices)
    records = []
    seen = set()
    for path in sorted((base / 'feature_records').glob('*.json')):
        # A worker never reads another worker's record while that worker is active.
        if indices is not None and path.stem.isdigit() and int(path.stem) not in expected:
            continue
        record = json.loads(path.read_text())
        index = record.get('image_index')
        if type(index) is not int or not 0 <= index < len(metadata):
            raise ValueError(f'Invalid image_index in {path}')
        if path.name != f'{index:04d}.json' or index in seen or index not in expected:
            raise ValueError(f'Duplicate, misnamed or unassigned image record: {path}')
        row = metadata[index]
        for key in ['image_path', 'day_group']:
            if record.get(key) != row[key]:
                raise ValueError(f'Metadata mismatch for {index}: {key}')
        if record.get('age_label') != int(row['age_label']):
            raise ValueError(f'Metadata mismatch for {index}: age_label')
        for key in ['width', 'height']:
            if record.get(key) != int(row[key]):
                raise ValueError(f'Native image size mismatch for {index}: {key}')
        required = ['foreground_ok', 'beak_ok', 'feather_ok', 'orientation_flip',
                    'orientation_confidence', 'head_to_beak_dx', 'foreground_gray_std',
                    'foreground_gray_mean', 'feather_gray_std']
        required += ['beak_' + channel + '_mean' for channel in CHANNELS]
        required += ['feather_' + channel + '_std' for channel in CHANNELS]
        if any(key not in record for key in required):
            raise ValueError(f'Incomplete numerical record: {path}')
        for key in ['foreground_ok', 'beak_ok', 'feather_ok', 'orientation_flip']:
            if record[key] not in [0, 1]:
                raise ValueError(f'Invalid QC flag for {index}: {key}')
        if require_artifacts:
            outputs = [base / directory / f'{index:04d}.png' for directory in ARTIFACT_DIRS]
            outputs.append(base / 'detections' / f'{index:04d}.json')
            missing = [str(output) for output in outputs if not output.is_file() or output.stat().st_size == 0]
            if missing:
                raise ValueError(f'Missing completed image artifacts for {index}: {missing}')
        records.append(record)
        seen.add(index)
    if require_complete and seen != expected:
        raise ValueError(f'Incomplete record set: expected {len(expected)}, found {len(seen)}; '
                         f'missing indices {sorted(expected - seen)}')
    return sorted(records, key=lambda record: record['image_index'])


@contextmanager
def partition_lock(base, indices):
    """Lock one byte per owned image; incompatible jobs fail before model loading."""
    import fcntl
    path = Path(base) / 'worker_jobs' / 'image_partition.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        for index in sorted(set(indices)):
            try:
                fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, index, os.SEEK_SET)
            except BlockingIOError as exc:
                raise RuntimeError(f'Image {index} is owned by an active ROI job') from exc
        yield
    finally:
        os.close(descriptor)


def validate_worker_summaries(base, metadata, shards, config_hash, script_hash):
    summaries = []
    for shard_id in range(shards):
        directory = worker_directory(base, shards, shard_id)
        path = directory / 'processing_summary.json'
        if not path.is_file():
            raise ValueError('Missing completed worker summary: ' + str(path))
        summary = json.loads(path.read_text())
        assigned = assigned_indices(len(metadata), shards, shard_id)
        for key, expected in [('complete', True), ('shards', shards), ('shard_id', shard_id),
                              ('n_expected', len(assigned)), ('n_processed', len(assigned)),
                              ('completed_indices', assigned), ('config_sha256', config_hash),
                              ('source_script_sha256', script_hash)]:
            if summary.get(key) != expected:
                raise ValueError(f'Incompatible worker summary {path}: {key}')
        config_path = directory / 'processing_config.json'
        if not config_path.is_file() or canonical_hash(json.loads(config_path.read_text())) != config_hash:
            raise ValueError('Missing or incompatible worker configuration: ' + str(config_path))
        if summary.get('records_sha256') != record_files_hash(base, assigned):
            raise ValueError('Worker records changed after completion: ' + str(path))
        summaries.append(summary)
    return summaries


def validate_resume_provenance(directory, config_hash, script_hash, indices):
    """Existing worker records can only resume under their original execution design."""
    directory = Path(directory)
    config_path = directory / 'processing_config.json'
    provenance_path = directory / 'resume_provenance.jsonl'
    if not config_path.is_file() or not provenance_path.is_file():
        raise ValueError('Existing worker records have no matching worker provenance: ' + str(directory))
    if canonical_hash(json.loads(config_path.read_text())) != config_hash:
        raise ValueError('Worker configuration changed while checkpoint records exist: ' + str(directory))
    entries = [json.loads(line) for line in provenance_path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError('Empty worker resume provenance: ' + str(directory))
    for entry in entries:
        for key, expected in [('config_sha256', config_hash), ('source_script_sha256', script_hash),
                              ('assigned_indices', indices)]:
            if entry.get(key) != expected:
                raise ValueError(f'Incompatible worker resume provenance {directory}: {key}')
