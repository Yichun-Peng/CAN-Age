#!/usr/bin/env python3
"""Restore model input PNGs from the published original pixels and native masks."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--limit', type=int, default=1143)
    args = parser.parse_args()
    processing = ROOT / 'code/image_processing_cpu_semantic_v1'
    with (ROOT / 'data/metadata.csv').open(newline='', encoding='utf-8') as stream:
        metadata = list(csv.DictReader(stream))
    with (processing / 'features.csv').open(newline='', encoding='utf-8') as stream:
        features = list(csv.DictReader(stream))
    directories = ['aligned_rgba', 'aligned_foreground_masks', 'aligned_beak_masks', 'aligned_feather_masks']
    if not args.check_only:
        for name in directories:
            (processing / name).mkdir(exist_ok=True)
    checked = 0
    for meta, feature in zip(metadata[:args.limit], features[:args.limit]):
        index = int(meta['image_index'])
        assert index == int(feature['image_index'])
        source = ROOT / 'data' / meta['image_path']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['sha256']
        with Image.open(source) as image:
            rgb = np.asarray(image.convert('RGB'))
        masks = {}
        for name in ['foreground', 'beak', 'feather']:
            with Image.open(processing / (name + '_masks') / f'{index:04d}.png') as image:
                mask = np.asarray(image) > 0
            assert mask.shape == rgb.shape[:2]
            assert int(mask.sum()) == int(feature[name + '_pixels'])
            masks[name] = mask
        primary = masks['foreground'] | (masks['beak'] if int(feature['beak_ok']) else False)
        flip = bool(int(feature['orientation_flip']))
        alpha = np.fliplr(primary) if flip else primary
        aligned_rgb = np.fliplr(rgb) if flip else rgb
        if not args.check_only:
            rgba = np.dstack([aligned_rgb, alpha.astype(np.uint8) * 255])
            Image.fromarray(rgba).save(processing / 'aligned_rgba' / f'{index:04d}.png', compress_level=1)
            for region, mask in [('foreground', primary), ('beak', masks['beak']), ('feather', masks['feather'])]:
                aligned = np.fliplr(mask) if flip else mask
                Image.fromarray(aligned.astype(np.uint8) * 255).save(
                    processing / ('aligned_' + region + '_masks') / f'{index:04d}.png', compress_level=1)
        checked += 1
    print(json.dumps({'status': 'passed', 'checked_images': checked,
                      'native_candidate_pixels_verified': 3 * checked,
                      'derived_pngs_written': not args.check_only,
                      'training_executed': False}))


if __name__ == '__main__':
    main()
