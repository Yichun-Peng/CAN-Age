# CAN-Age

CAN-Age tests how foreground color and anatomical beak/plumage descriptors contribute to winter-stage classification of white-tailed eagles. Frozen ImageNet encoders are compared using RGB, color attention, and grayscale inputs under the same five grouped folds.

The dataset contains **1,143 original photographs** in five annotated winter-stage classes: 1, 2, 3, 4, and **5+ (fifth winter or older)**. The class counts are 400, 227, 161, 138, and 217. `data/metadata.csv` records original filenames, capture dates/source groups, labels, source-file SHA-256, and decoded-pixel SHA-256. `data/folds.csv` contains all fixed training/validation/test memberships.

## Results

The primary analysis uses background-removed, direction-aligned images and anatomical color descriptors. Mean color-attention accuracy across four encoders is **80.14%**; the best color-attention model reaches **82.33%**. `results/primary/` contains 12 model configurations, their 60 fold logs/checkpoints, and complete held-out probabilities. The whole-photograph comparison is separately labeled in `results/supplementary/whole_photograph/` (mean color-attention accuracy: **79.09%**).

## Download and validate

Python 3.12 is the reference runtime. Clone the browsable code, install dependencies, and download the versioned complete release:

```bash
git clone https://github.com/Yichun-Peng/CAN-Age.git
cd CAN-Age
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_assets.py
python scripts/validate_release.py
python scripts/plot_saved_results.py
```

The downloader fetches `CAN_Age_data_code_20261027.zip` from release **v2026.10.27**, verifies its fixed SHA-256, and restores the complete directory tree. Original photographs, native masks, measurements, OOF probabilities, and checkpoints are in the release asset. The validator checks every original/decoded image hash, label/fold membership, date/source isolation, all 60 primary fold logs, and metrics recomputed from saved probabilities. It writes `outputs/release_validation.json`.

An already downloaded archive works offline:

```bash
python scripts/download_assets.py --archive CAN_Age_data_code_20261027.zip
```

## Reproduce

Restore aligned model inputs from the original pixels and published native masks, download the four official ImageNet encoders, and run the fixed protocol:

```bash
python scripts/prepare_inputs.py
python scripts/download_weights.py
python scripts/reproduce.py primary
python scripts/reproduce.py whole-photograph
python scripts/reproduce.py components
python scripts/reproduce.py brightness
python scripts/reproduce.py roi-analysis
```

Fresh caches, checkpoints, and predictions are written under `runs/reproduction/`. Published reference results remain in `results/`. `code/model_sources.json` records official sources, the GroundingDINO revision, and weight hashes. Reference package versions are recorded in `reference_environment.json`.

## Verify and redraw figures

The complete release includes the final renderer and its 41 fixed input files or original-photo mappings. Verify them without rendering, or redraw all eight figures into a new directory:

```bash
python figure_source/replay.py --verify-only
python figure_source/replay.py --out outputs/redrawn_figures
```

`figure_source/README.md` records the input mapping and path conversions. Each figure exports PDF, SVG, and PNG.

## Files

- `code/`: detector/segmentation, semantic ROI, training, bootstrap, robustness, and plotting implementations.
- `data/`: source images, metadata, fixed folds, and file/pixel hashes.
- `results/primary/`: foreground-based OOF results and 60 fold records.
- `results/supplementary/`: whole-photograph (60 folds), component controls (60 folds), and preprocessing controls (15 folds).
- `scripts/`: download, validation, plotting, input reconstruction, and reproduction commands.
- `CAN-Age.ipynb`: plots/tables read from actual held-out results.
- `figure_source/`: the final eight-figure renderer, fixed inputs, and checksum verification.
- `release_provenance.json`: source hashes and portable-path conversions.
- `paper/` and `figures/`: revised manuscript and final figures in the complete release.

## License

MIT. See [LICENSE](LICENSE).

## Release archive checksum

`CAN_Age_data_code_20261027.zip` (v2026.10.27):

```text
50daf95fda9423384f15e95dd625256721f08549ba1e7a52b58aa3769adab8ce
```
