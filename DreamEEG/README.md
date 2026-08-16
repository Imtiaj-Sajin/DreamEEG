# DreamEEG - Visual Imagery EEG Decoding

Working code + data for the DreamEEG project (visual-imagery BCI for ALS /
locked-in users). Everything is self-contained inside this folder on the
**G: drive** - no dependence on the C: drive.

## Folder layout

```
DreamEEG/
├── code/
│   ├── dreameeg.py            # main pipeline (download, preprocess, train, CV)
│   ├── requirements.txt       # Python packages
│   └── authors_reference/     # the dataset authors' original code + metadata
│       ├── Preprocess.py      #   their preprocessing (ICA + ICLabel)
│       ├── EEGNet.py          #   their EEGNet training script
│       ├── Machine learning.py#   their CSP+KNN baseline
│       ├── README             #   dataset documentation
│       └── participants.tsv   #   subject metadata (age, gender, VVIQ)
├── data/
│   ├── sub-09.zip             # raw subject archive (from figshare)
│   ├── sub-09/                # extracted BIDS (.bdf + events)
│   ├── cache/                 # preprocessed .npy tensors (fast reload)
│   └── fig.json               # figshare file listing (all 22 subjects)
└── results/
    └── results.csv            # every run appended here
```

## Dataset

Gao et al. (2026), *An EEG Dataset for Visual Imagery-Based BCI*, Scientific
Data. figshare article `30227503` (~11.9 GB total, one zip per subject).
- 22 subjects, 32 EEG ch @ 1000 Hz, 10-20 layout, 2 sessions (19 did both).
- 3 tasks / categories: **AVI** animals (dog/bird/fish), **FVI** figures
  (circle/square/pentagram), **OVI** objects (cup/chair/watch/scissors).
- Decode the 4 s imagery window. Chance = 33% (3-class).
- Authors' within-subject EEGNet baseline: figures 75.1%, animals 75.8%,
  objects 62.0%.

We only download subjects as we need them (one is ~280-530 MB), never the
whole 12 GB up front.

## How to run

```bash
cd DreamEEG/code
pip install -r requirements.txt          # one-time

# reproduce the authors' baseline on one subject/category (uses GPU if present)
python dreameeg.py --subject sub-09 --task AVI

# faster check, skipping the slow ICA stage
python dreameeg.py --subject sub-09 --task AVI --no-ica

# sliding-window augmentation instead of the full 4 s window
python dreameeg.py --subject sub-09 --task AVI --window slide

# just fetch another subject's data into data/
python dreameeg.py --download sub-04
```

Results append to `results/results.csv`. Preprocessed tensors are cached in
`data/cache/`, so re-runs skip the expensive ICA step.

## Status / notes

- Sanity check on sub-09 (animals) confirmed the pipeline end to end: data
  loads, 120 triggers extract, epochs form `(120, 32, 1000)`, and EEGNet
  decodes **above the 33% chance level**.
- Reproducing the authors' exact per-category numbers is Stage 0 of the
  project and is in progress; see `results/results.csv`.
- GPU: the training auto-selects CUDA (tested on an RTX 3060 Ti).
