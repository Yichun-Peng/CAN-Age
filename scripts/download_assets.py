#!/usr/bin/env python3
"""Download and verify the versioned complete CAN-Age data/code release."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TAG = 'v2026.10.27'
NAME = 'CAN_Age_data_code_20261027.zip'
ASSET_SHA256 = '50daf95fda9423384f15e95dd625256721f08549ba1e7a52b58aa3769adab8ce'
URL = f'https://github.com/Yichun-Peng/CAN-Age/releases/download/{TAG}/{NAME}'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='use an already downloaded ZIP')
    args = parser.parse_args()
    assert len(ASSET_SHA256) == 64, 'The release checksum must be finalized before publication'
    if args.archive:
        archive = args.archive.expanduser().resolve()
    else:
        directory = ROOT / 'outputs/downloads'
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / NAME
        if not archive.exists() or digest(archive) != ASSET_SHA256:
            temporary = archive.with_suffix('.download')
            urllib.request.urlretrieve(URL, temporary)
            temporary.replace(archive)
    assert digest(archive) == ASSET_SHA256, 'Release archive SHA-256 mismatch'
    restored = 0
    with zipfile.ZipFile(archive) as source:
        assert source.testzip() is None, 'Release ZIP CRC mismatch'
        names = source.namelist()
        assert len(names) == len(set(names))
        manifest = json.loads(source.read('CAN-Age/release_manifest.json'))['files']
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == 'CAN-Age'
            if name.endswith('/'):
                continue
            relative = Path(*path.parts[1:])
            target = (ROOT / relative).resolve()
            assert target.is_relative_to(ROOT) and '.git' not in relative.parts
            if name == 'CAN-Age/release_manifest.json':
                continue
            assert relative.as_posix() in manifest
            content = source.read(name)
            assert hashlib.sha256(content).hexdigest() == manifest[relative.as_posix()]['sha256']
            # Keep the clone's download-oriented README; the archive has its own run-oriented README.
            if relative.as_posix() == 'README.md':
                continue
            if target.exists() and digest(target) == manifest[relative.as_posix()]['sha256']:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            restored += 1
        (ROOT / 'release_manifest.json').write_text(json.dumps({'files': manifest}, indent=2) + '\n')
    print(f'Verified {len(manifest)} release files; restored {restored} files from {TAG}.')


if __name__ == '__main__':
    main()
