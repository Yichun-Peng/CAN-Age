#!/usr/bin/env python3
"""Verify and restore the frozen CAN-Age base release plus its aligned-figure patch."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE_TAG = 'v2026.10.27'
BASE_NAME = 'CAN_Age_data_code_20261027.zip'
BASE_SHA256 = '50daf95fda9423384f15e95dd625256721f08549ba1e7a52b58aa3769adab8ce'
PATCH_TAG = 'v2026.10.27.1'
PATCH_NAME = 'CAN_Age_figure_alignment_20261027.zip'
PATCH_SHA256 = '3295af3e18e2b6cf076657447dfe0387770a2fc9ec9c269f994c42e90929a368'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def obtain(given, tag, name, expected):
    assert len(expected) == 64, 'The release checksum must be finalized before publication'
    if given:
        path = given.expanduser().resolve()
    else:
        directory = ROOT / 'outputs/downloads'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        if not path.exists() or digest(path) != expected:
            temporary = path.with_suffix('.download')
            url = f'https://github.com/Yichun-Peng/CAN-Age/releases/download/{tag}/{name}'
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
    assert digest(path) == expected, name + ': archive SHA-256 mismatch'
    return path


def restore(path, expected_tag):
    restored = 0
    with zipfile.ZipFile(path) as source:
        assert source.testzip() is None, 'Release ZIP CRC mismatch'
        names = source.namelist()
        assert len(names) == len(set(names))
        manifest = json.loads(source.read('CAN-Age/release_manifest.json'))
        assert manifest['release_tag'] == expected_tag
        for name in names:
            member = PurePosixPath(name)
            assert not member.is_absolute() and '..' not in member.parts
            assert member.parts[0] == 'CAN-Age' and '.git' not in member.parts
            if name.endswith('/'):
                continue
            relative = Path(*member.parts[1:])
            target = (ROOT / relative).resolve()
            assert target.is_relative_to(ROOT)
            if relative.as_posix() == 'release_manifest.json':
                continue
            assert relative.as_posix() in manifest['files']
            content = source.read(name)
            expected = manifest['files'][relative.as_posix()]['sha256']
            assert hashlib.sha256(content).hexdigest() == expected
            # The clone keeps its download-oriented README and this two-stage downloader.
            if relative.as_posix() in ('README.md', 'scripts/download_assets.py'):
                continue
            if target.exists() and digest(target) == expected:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            restored += 1
    return manifest, restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='already downloaded complete base ZIP')
    parser.add_argument('--patch-archive', type=Path, help='already downloaded figure-alignment patch ZIP')
    args = parser.parse_args()
    base = obtain(args.archive, BASE_TAG, BASE_NAME, BASE_SHA256)
    patch = obtain(args.patch_archive, PATCH_TAG, PATCH_NAME, PATCH_SHA256)
    base_manifest, base_count = restore(base, BASE_TAG)
    final_manifest, patch_count = restore(patch, PATCH_TAG)
    assert final_manifest['base_release_tag'] == BASE_TAG
    assert final_manifest['base_archive_sha256'] == BASE_SHA256
    for key, item in final_manifest['files'].items():
        relative = PurePosixPath(key)
        assert not relative.is_absolute() and '..' not in relative.parts and '.git' not in relative.parts
        if key in ('README.md', 'scripts/download_assets.py'):
            continue
        target = (ROOT / Path(*relative.parts)).resolve()
        assert target.is_relative_to(ROOT) and target.is_file()
        assert digest(target) == item['sha256'], 'Final restored file mismatch: ' + key
    (ROOT / 'release_manifest.json').write_text(json.dumps(final_manifest, indent=2) + '\n')
    print(f'Verified {len(final_manifest["files"])} final files; restored {base_count} base files and {patch_count} patch files from {BASE_TAG} + {PATCH_TAG}.')


if __name__ == '__main__':
    main()
