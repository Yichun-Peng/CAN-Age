#!/usr/bin/env python3
"""Plot actual primary held-out results without fitting or synthetic data."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'mobilenet_v3_small': 'MobileNetV3-Small', 'vgg11': 'VGG11',
         'resnet18': 'ResNet18', 'densenet121': 'DenseNet121'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/primary_results')
    args = parser.parse_args()
    summary = json.loads((ROOT / 'results/primary/summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    variants = [('baseline', 'RGB baseline', '#617d99'), ('color_attention', 'Color attention', '#b36d39'),
                ('grayscale', 'Grayscale', '#8d9299')]
    x = np.arange(4)
    for offset, (key, label, color) in enumerate(variants):
        for ax, field, scale in [(axes[0], 'accuracy', 100), (axes[1], 'macro_f1', 1)]:
            values = [summary[name][key]['pooled'][field] * scale for name in NAMES]
            ax.bar(x + (offset - 1) * .24, values, width=.23, color=color, label=label)
            ax.set_xticks(x, NAMES.values(), rotation=22, ha='right')
            ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Out-of-fold accuracy (%)'); axes[1].set_ylabel('Macro-F1')
    axes[0].legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(args.output.with_suffix('.' + extension), dpi=300)
    print(args.output)


if __name__ == '__main__':
    main()
