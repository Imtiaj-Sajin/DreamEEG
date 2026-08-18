# Recheck: why are our numbers below the paper's? (animals, ses-01)

Systematic check of every factor that could explain the gap between our
honest reproduction (~48%) and the paper's reported 75.8%.

## The one real thing we measured differently
The paper (Table 2) uses a **single 80/20 train/test split** and their code
reports the **best test accuracy reached at any epoch** (the test set is used
to pick the epoch). We used honest 5-fold CV reporting stable accuracy.

| Measurement (animals) | sub-09 | 6-subj mean |
|---|---|---|
| Honest 5-fold CV | 57.8% | **48.2%** |
| Honest single-split (last-50-epoch mean) | 57.8% | 37.4% |
| **Their protocol: best-epoch on single split** | **70.0%** | **55.6%** |
| Paper reported | - | **75.8%** |

=> Reporting the best epoch on a 24-sample test set inflates by ~15-18 points.
This is a real, verifiable optimistic bias, and the main thing we did not do.

## Factors RULED OUT (do not explain the gap)
- **Subject variance**: 6 subjects (VVIQ 36-80, incl. the experienced one)
  all cluster 44-58% under honest CV; none near 75%.
- **Epoch budget**: 500 vs 1500 epochs gives the same best-epoch number
  (54.7% vs 55.6%); the artifact saturates.
- **Sliding-window leakage**: a leaky window-level split did not inflate.
- **Preprocessing**: the authors' ICA pipeline is WORSE than a simple
  1-40 Hz filter (sub-09 best-epoch: ICA 58.3% vs lite 70.0%).
- **Our correctness**: running the authors' UNMODIFIED code on their data
  gives 41.7% (see ../code/authors_run/VERBATIM_RESULT.md), matching our
  re-implementation. Our pipeline is correct.

## Residual
Even fully replicating their optimistic protocol we reach ~55.6% on the
6-subject average (70% on the best subject), still ~20 points below 75.8%.
That residual is NOT explained by any factor above and is not reproducible
from the released code + data.

## Bottom line
1. We did not make an error; our honest 5-fold accuracy (~48% animals) is
   the correct generalization estimate.
2. A large part of the paper's headline number is an **evaluation-protocol
   artifact** (best-epoch-on-test + single small split), which we quantified
   at ~15-18 points.
3. A residual ~20-point gap remains genuinely irreproducible.

Recommendation: evaluate DreamEEG with honest 5-fold CV, and for transparency
report BOTH the honest number and the paper-protocol number side by side.
