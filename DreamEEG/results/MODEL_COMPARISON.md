# EEGNet vs DreamEEG (animals, honest evaluation)

All numbers are leak-free, chance = 33.3%. The paper's 75.8% is not a valid
comparison point (see RECHECK_findings.md: it is an evaluation-protocol
artifact). We compare against our own honest baseline instead.

## Within-subject (6 subjects x 3 seeds, 5-fold CV; ~96 train trials/fold)
| Model | Accuracy |
|---|---|
| EEGNet (1.6K params) | **48.2% +/- 4.9%** |
| DreamEEG (40K params) | 45.3% +/- 2.4% |

With only ~96 training trials, the tiny EEGNet wins; the Transformer overfits.
DreamEEG is more consistent across subjects, but lower on average.

## Cross-subject LOSO (train on 5 subjects = 600 trials, test on held-out)
| Model | no alignment | + Euclidean Alignment |
|---|---|---|
| EEGNet | 41.4% +/- 3.7% | 41.1% +/- 3.1% |
| DreamEEG | **42.5% +/- 2.9%** | 41.7% +/- 1.2% |

With 600 pooled trials the Transformer's capacity starts to pay off: DreamEEG
edges out EEGNet and is markedly more stable (std 1.2-2.9% vs 3.1-3.7%).
Euclidean Alignment was neutral on this 6-subject set.

## Takeaways
1. Cross-subject (calibration-free) is harder than within-subject for both
   models (~41-42% vs ~48%), as expected.
2. DreamEEG's advantage appears only in the data-rich cross-subject regime,
   and mainly as improved stability so far.
3. The gains are modest because VI signal is genuinely weak (honest ceiling
   ~48% within-subject) and 600 trials is still small for a Transformer.

## Most promising next step
Scale cross-subject to ALL 22 subjects (21 x 120 = 2520 training trials),
where a Transformer should benefit most, and tune DreamEEG (augmentation,
depth, regularization). Report animals/figures/objects.
