#!/usr/bin/env python3
"""Run the published protocol with portable paths and a fresh output workspace."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['primary', 'whole-photograph', 'components', 'brightness', 'roi-analysis'])
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/reproduction')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    destination = args.run_dir.resolve()
    code = destination / 'code'
    environment = dict(os.environ, EAGLE_DATA_DIR=str(ROOT / 'data'),
                       EAGLE_PROCESSING_DIR=str(ROOT / 'code/image_processing_cpu_semantic_v1'),
                       EAGLE_SOURCE_PROCESSING_DIR=str(ROOT / 'code/image_processing'),
                       EAGLE_DEVICE=args.device, EAGLE_NUM_THREADS='16', EAGLE_NUM_THREADS_HEAD='4',
                       TORCH_HOME=str(ROOT / 'code/model_cache/torch'),
                       HF_HOME=str(ROOT / 'code/image_processing/huggingface'),
                       PYTHONDONTWRITEBYTECODE='1', BWHD_ROOT=str(ROOT),
                       BWHD_AUDIT_ROOT=str(ROOT / 'code/data_audit'),
                       BWHD_NEURAL_RESULTS_ROOT=str(ROOT / 'results/primary'),
                       BWHD_WHOLE_RESULTS_ROOT=str(ROOT / 'results/supplementary/whole_photograph'),
                       BWHD_SUPPLEMENT_OUT=str(destination / 'outputs/supplementary'))
    commands = {
        'primary': [('experiments_foreground/run_foreground.py', 'all', '--device', args.device)],
        'whole-photograph': [('experiments/run_experiments.py', 'all', '--device', args.device)],
        'components': [('component_ablations/run_component_ablations.py',)],
        'brightness': [('experiments_foreground/brightness_robustness_foreground.py',)],
        'roi-analysis': [('exploratory/recompute_native_texture.py',), ('exploratory/roi_color_correlations.py',),
                         ('exploratory/roi_exploration.py', '--classify-only')],
    }[args.stage]
    if args.dry_run:
        for command in commands:
            print(' '.join([sys.executable, '-B', str(code / command[0]), *command[1:]]))
        return
    if not code.exists():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / 'code', code, ignore=shutil.ignore_patterns(
            'cache', 'model_cache', 'huggingface', 'models', 'aligned_rgba', 'aligned_foreground_masks',
            'aligned_beak_masks', 'aligned_feather_masks', '__pycache__'))
    # Primary runs create their own fold signatures, caches, checkpoints, and predictions.
    for command in commands:
        subprocess.run([sys.executable, '-B', '-u', str(code / command[0]), *command[1:]],
                       cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
