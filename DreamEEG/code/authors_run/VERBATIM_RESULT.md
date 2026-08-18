# Authors' verbatim reproduction result

We ran the dataset authors' **own** released scripts, unmodified in logic, to
check whether the published EEGNet baseline reproduces.

## What was run
- `Preprocess.py` and `EEGNet.py` are exact copies of the authors' scripts
  from `../authors_reference/` (their released code).
- The **only** edits made:
  1. `Preprocess.py`: the input `bdf_file_path` was pointed at our local
     `sub-09 ses-01 task-AVI` file (a path change, not a logic change).
  2. `EEGNet.py`: the input `data_list` path was pointed at the folder
     `Preprocess.py` writes to (a path change).
  3. `EEGNet.py`: removed the `verbose=True` argument from
     `ReduceLROnPlateau(...)` because PyTorch 2.7 deleted that keyword. It
     only controlled console printing, not training.
- Environment: `PYTHONUTF8=1` (their prints contain a `→` character that the
  default Windows console codec cannot encode). GPU: RTX 3060 Ti.

## Pipeline (their code)
BDF -> resample 250 Hz -> montage -> pyprep NoisyChannels -> notch 50 ->
1-100 Hz band-pass -> average reference -> ICA (Picard extended) + ICLabel ->
interpolate bads -> 4-80 Hz band-pass -> epoch 0-4 s -> `(120, 32, 1000)`.
Then their EEGNet, 5-fold stratified CV (trial-level split, leak-free),
1500 epochs, Adam lr=1e-3, weight_decay=0.09, ReduceLROnPlateau.

## Result on sub-09 (animals, 3-class, chance 33.3%)
```
Fold 1: 33.3%   Fold 2: 41.7%   Fold 3: 45.8%   Fold 4: 50.0%   Fold 5: 37.5%
Average Accuracy: 41.67% +/- 5.89%
```
During training, per-trial validation accuracy sat around chance (~29%) while
train accuracy rose to ~70%, i.e. the model overfits 96 training trials and
does not generalize.

## Conclusion
The authors' **own** code, run on the authors' data, yields **41.67%** on
sub-09 animals, not the **75.8%** reported in the paper. This matches our
independent faithful re-implementation (~40% with ICA; ~48-58% with a simpler
1-40 Hz pipeline) and confirms:

1. Our reproduction was correct.
2. The published EEGNet baseline for this dataset **does not reproduce** from
   the released code and data; the honest leak-free accuracy is far lower.

(The raw run logs `prep.log` and `eegnet_run.log` are git-ignored via the
`*.log` rule but remain on disk locally as the primary evidence.)
