#!/usr/bin/env python3
"""Download the four pinned official ImageNet encoders and check full SHA-256."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='show URLs and hashes without downloading')
    args = parser.parse_args()
    source = json.loads((ROOT / 'code/model_sources.json').read_text())['files']
    cache = ROOT / 'code/model_cache/torch/hub/checkpoints'
    records = []
    for name, record in source.items():
        if not name.endswith('.pth'):
            continue
        url = 'https://download.pytorch.org/models/' + name
        records.append({'filename': name, 'url': url, 'sha256': record['sha256'], 'bytes': record['bytes']})
        if args.list:
            continue
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / name
        if not target.exists():
            temporary = target.with_suffix('.download')
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(target)
        digest = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        assert target.stat().st_size == record['bytes'] and digest.hexdigest() == record['sha256'], name
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
