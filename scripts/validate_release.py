#!/usr/bin/env python3
"""Independently validate public images, metadata, grouped folds, and saved OOFs."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ['mobilenet_v3_small', 'vgg11', 'resnet18', 'densenet121']
VARIANTS = ['baseline', 'color_attention', 'grayscale']
EXPECTED_COUNTS = {1: 400, 2: 227, 3: 161, 4: 138, 5: 217}


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def metrics(labels, probability):
    predicted = probability.argmax(1)
    matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    precision_denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * matrix.diagonal(), precision_denominator,
                   out=np.zeros(5, float), where=precision_denominator > 0)
    recall = matrix.diagonal() / matrix.sum(1)
    distance = np.abs(predicted - labels)
    wrong = distance > 0
    return {'n': len(labels), 'correct': int(matrix.trace()),
            'accuracy': float(matrix.trace() / len(labels)), 'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall.mean()), 'ordinal_mae': float(distance.mean()),
            'adjacent_error_fraction': float((distance[wrong] == 1).mean()) if wrong.any() else 0.0,
            'confusion_matrix': matrix.tolist()}


def compare_metrics(actual, expected):
    for key, value in actual.items():
        if key == 'confusion_matrix':
            assert value == expected[key], key
        else:
            assert abs(value - expected[key]) < 1e-12, (key, value, expected[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    metadata = csv_rows(root / 'data/metadata.csv')
    assert len(metadata) == 1143
    assert [int(row['image_index']) for row in metadata] == list(range(1143))
    assert len({row['image_id'] for row in metadata}) == 1143
    assert len({row['image_path'] for row in metadata}) == 1143
    assert len({row['original_image_path'] for row in metadata}) == 1143
    assert Counter(int(row['age_label']) for row in metadata) == EXPECTED_COUNTS
    labels = np.array([int(row['age_label']) - 1 for row in metadata])
    file_hashes, pixel_hashes = set(), set()
    for row in metadata:
        assert row['image_id'] == f"{int(row['image_index']):04d}"
        assert row['image_path'] == 'images/' + row['image_id'] + Path(row['original_filename']).suffix.lower()
        assert int(row['label']) == int(row['age_label']) - 1
        path = (root / 'data' / row['image_path']).resolve()
        assert path.is_relative_to(root / 'data/images')
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        assert digest == row['sha256'] and len(content) == int(row['bytes']), row['image_id']
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            assert rgb.size == (int(row['width']), int(row['height']))
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        assert pixels == row['decoded_pixels_sha256'], row['image_id']
        file_hashes.add(digest); pixel_hashes.add(pixels)
    assert len(file_hashes) == len(pixel_hashes) == 1143
    inventory = csv_rows(root / 'data/inventory.csv')
    assert len(inventory) == 1143
    for row, expected in zip(inventory, metadata):
        for field in ['image_path', 'age_label', 'width', 'height', 'sha256', 'decoded_pixels_sha256', 'source_group', 'capture_timestamp']:
            assert row[field] == expected[field], (field, expected['image_id'])
    folds = csv_rows(root / 'data/folds.csv')
    assert len(folds) == 5715
    original_folds = csv_rows(root / 'code/data_audit/fold_assignments.csv')
    assert folds == original_folds
    test_counts = np.zeros(1143, dtype=np.int64)
    partition_indices = {}
    for fold in range(5):
        partition_indices[fold] = {}
        for part in ['train', 'val', 'test']:
            rows = [row for row in folds if int(row['fold']) == fold and row['partition'] == part]
            ids = {int(row['image_index']) for row in rows}
            assert len(ids) == len(rows)
            partition_indices[fold][part] = ids
            for row in rows:
                expected = metadata[int(row['image_index'])]
                for field in ['image_path', 'age_label', 'label', 'day_group', 'source_group', 'capture_timestamp']:
                    assert row[field] == expected[field]
        assert set.union(*partition_indices[fold].values()) == set(range(1143))
        for left, right in itertools.combinations(['train', 'val', 'test'], 2):
            assert not partition_indices[fold][left] & partition_indices[fold][right]
            for group in ['day_group', 'source_group']:
                a = {metadata[i][group] for i in partition_indices[fold][left]}
                b = {metadata[i][group] for i in partition_indices[fold][right]}
                assert not a & b, (fold, left, right, group)
        for index in partition_indices[fold]['test']:
            test_counts[index] += 1
            assert int(metadata[index]['outer_fold']) == fold
    assert (test_counts == 1).all()
    summary = json.loads((root / 'results/primary/summary.json').read_text())
    assert set(summary) == set(BACKBONES)
    oof = csv_rows(root / 'results/primary/oof_predictions.csv')
    groups = defaultdict(list)
    for row in oof:
        groups[(row['backbone'], row['variant'])].append(row)
    assert set(groups) == set(itertools.product(BACKBONES, VARIANTS))
    verified_models = []
    checked_folds = 0
    for (backbone, variant), rows in groups.items():
        rows.sort(key=lambda row: int(row['image_index']))
        assert [int(row['image_index']) for row in rows] == list(range(1143))
        probability = np.array([[float(row[f'prob_age_{age}']) for age in range(1, 6)] for row in rows])
        assert probability.shape == (1143, 5) and np.isfinite(probability).all()
        assert (probability >= 0).all() and (probability <= 1).all()
        assert np.allclose(probability.sum(1), 1, atol=1e-6, rtol=0)
        for index, row in enumerate(rows):
            expected = metadata[index]
            for field in ['image_path', 'age_label', 'day_group', 'outer_fold']:
                assert row[field] == expected[field]
            predicted = int(probability[index].argmax()) + 1
            assert int(row['predicted_age_label']) == predicted
            assert int(row['correct']) == int(predicted == int(expected['age_label']))
        recomputed = metrics(labels, probability)
        compare_metrics(recomputed, summary[backbone][variant]['pooled'])
        for fold in range(5):
            log_path = root / 'results/primary/results' / f'{backbone}_{variant}_fold{fold}.json'
            record = json.loads(log_path.read_text())
            assert (record['backbone'], record['variant'], record['fold']) == (backbone, variant, fold)
            assert record['seed'] == 20261026 + BACKBONES.index(backbone) * 1000 + fold * 10
            for name, key in [('training_n', 'train'), ('validation_n', 'val'), ('test_n', 'test')]:
                assert record[name] == len(partition_indices[fold][key])
            train_indices = sorted(partition_indices[fold]['train'])
            counts = np.bincount(labels[train_indices], minlength=5)
            weights = len(train_indices) / (5 * counts)
            assert np.allclose(weights, record['class_weights'], atol=1e-6, rtol=0)
            test_indices = sorted(partition_indices[fold]['test'])
            compare_metrics(metrics(labels[test_indices], probability[test_indices]), record['test_metrics'])
            assert 1 <= record['best_epoch'] <= record['epochs_ran'] <= 50
            assert len(record['history']) == record['epochs_ran']
            checked_folds += 1
        verified_models.append({'backbone': backbone, 'variant': variant,
                                'accuracy': recomputed['accuracy'], 'macro_f1': recomputed['macro_f1']})
    assert checked_folds == 60
    log_files = list((root / 'results/primary/results').glob('*_fold*.json'))
    assert len(log_files) == 60
    feature_path = root / 'code/image_processing_cpu_semantic_v1/features.csv'
    features = csv_rows(feature_path)
    assert len(features) == 1143
    channels = ['red', 'green', 'blue', 'hue', 'saturation', 'value', 'lab_l', 'lab_a', 'lab_b']
    with np.load(root / 'code/image_processing_cpu_semantic_v1/raw_roi_features.npz') as archive:
        raw_roi = archive['features']
        assert raw_roi.shape == (1143, 18) and raw_roi.dtype == np.float32
        for index, row in enumerate(features):
            assert row['image_path'] == metadata[index]['image_path']
            for region, start, statistic in [('beak', 0, 'mean'), ('feather', 9, 'std')]:
                reference = np.array([float(row[f'{region}_{channel}_{statistic}']) for channel in channels], dtype=np.float32)
                if not int(row[region + '_ok']):
                    assert np.isnan(reference).all()
                assert np.array_equal(reference, raw_roi[index, start:start + 9], equal_nan=True)
    attention = [summary[name]['color_attention']['pooled']['accuracy'] for name in BACKBONES]
    assert f'{100 * float(np.mean(attention)):.2f}' == '80.14'
    assert f'{100 * max(attention):.2f}' == '82.33'
    result = {'status': 'passed', 'images': 1143, 'source_sha256_verified': 1143,
              'decoded_pixels_sha256_verified': 1143, 'class_counts': EXPECTED_COUNTS,
              'grouped_folds': 5, 'date_group_overlap': False, 'source_group_overlap': False,
              'every_image_tested_once': True, 'primary_model_conditions': len(verified_models),
              'primary_fold_logs_verified': checked_folds, 'primary_oof_rows': len(oof),
              'roi_float32_entries_verified': int(raw_roi.size),
              'mean_primary_color_attention_accuracy': float(np.mean(attention)),
              'best_primary_color_attention_accuracy': max(attention), 'models': verified_models}
    report = args.report or root / 'outputs/release_validation.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
