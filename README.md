# DreamEEG

A Visual-Imagery Brain-Computer Interface for people with ALS and locked-in
syndrome. This repository collects the full arc of the project so far: the
initial proposal, the midterm IEEE report, and the first working research
code with a feasibility sanity-check on real EEG data.

## What is here

### 1. Proposal work
- `DreamEEG_Midterm_Proposal_v2.pptx` - latest midterm proposal slides.
- `DreamEEG_Midterm_Proposal.pptx` - earlier version.
- `VI-EEG_Research_Proposal_Draft_updated-2.docx` - written proposal draft
  (background reading, staged plan, references).
- `s41597-025-06512-5.pdf` - the dataset descriptor paper (Gao et al., 2026).
- `Inner_Speech_Decoding_..._.pdf` - related reference paper.

### 2. Midterm IEEE report  (`report_ieee/mid_project/`)
LaTeX source (`main.tex`) and compiled `main.pdf` of the midterm report in
IEEE conference format, plus a ready-to-upload Overleaf zip.

### 3. Research code  (`DreamEEG/`)
A working, GPU-enabled pipeline that downloads the EEG dataset on demand,
preprocesses it, and trains an EEGNet decoder with 5-fold cross-validation.
See `DreamEEG/README.md` for details and commands.

```
DreamEEG/
├── code/
│   ├── dreameeg.py            # download, preprocess, train, cross-validate
│   ├── control_perception.py  # perception-vs-imagery control experiment
│   ├── requirements.txt
│   └── authors_reference/     # the dataset authors' original code (CC BY 4.0)
├── results/results.csv        # logged runs
└── README.md
```

> The EEG dataset itself (several GB) is **not** committed. It is openly
> available on figshare (article `30227503`) and `code/dreameeg.py`
> re-downloads any subject on demand into `DreamEEG/data/` (git-ignored).

## The problem

Most EEG brain-computer interfaces rely on motor imagery, but that routes
through the motor system, which is exactly what ALS destroys. Visual imagery
(picturing an object) is a motor-free alternative. On the Gao et al. (2026)
dataset the published EEGNet baseline reaches ~75% on figures and animals but
only ~62% on objects, and only within a single subject. The goal of DreamEEG
is a stronger, lighter decoder that closes that gap.

## Feasibility check (what the code confirms so far)

Running the pipeline on one subject verified the whole chain end to end (data
loads, triggers align to the official event files within 1 ms, EEGNet trains
on GPU) and confirmed the imagery signal decodes above chance (perception
~63%, imagery up to ~61% on the animals category vs 33% chance). It also
surfaced two realities for the next phase: single-subject accuracy is noisy
(only ~120 trials per category), and preprocessing choices matter a lot. A
proper baseline reproduction therefore needs several subjects and multiple
seeds. Per-run numbers are logged in `DreamEEG/results/results.csv`.

## Quick start

```bash
cd DreamEEG/code
pip install -r requirements.txt
python dreameeg.py --subject sub-09 --task AVI   # auto-downloads that subject
```

## Authors
Md. Imtiaj Alam Sajin and Md. Samiul Islam Saif.
