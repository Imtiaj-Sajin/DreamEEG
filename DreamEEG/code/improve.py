"""
Can we do better than EEGNet? Scan two levers that we may have under-used:
  (1) channel selection  - visual imagery should live in occipito-parietal
      channels; frontal/temporal channels may just add noise.
  (2) a covariance / Riemannian classifier (Cov -> TangentSpace -> LogReg,
      and MDM) - the gold standard for small EEG datasets.

Runs honest 5-fold CV. Channel order (data):
 0 Fpz 1 Fp1 2 Fp2 3 Fz 4 F3 5 F4 6 F7 7 F8 8 FCz 9 FC3 10 FC4 11 FT7 12 FT8
 13 Cz 14 C3 15 C4 16 T7 17 T8 18 CP3 19 CP4 20 TP7 21 TP8 22 Pz 23 P3 24 P4
 25 P7 26 P8 27 PO3 28 PO4 29 Oz 30 O1 31 O2
"""
import sys, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from dreameeg import preprocess

CHAN_SETS = {
    "all32":     list(range(32)),
    "posterior": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],   # CP/TP/P/PO/O
    "occipital": [22, 23, 24, 27, 28, 29, 30, 31],                            # Pz P3 P4 PO3 PO4 Oz O1 O2
}

def cv(pipeline_fn, X, y, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs = []
    for tr, va in skf.split(X, y):
        clf = pipeline_fn()
        clf.fit(X[tr], y[tr]); accs.append(clf.score(X[va], y[va]))
    return np.mean(accs) * 100

def ts_pipe():
    return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"),
                         LogisticRegression(max_iter=3000, C=1.0))
def mdm_pipe():
    return make_pipeline(Covariances("oas"), MDM(metric="riemann"))

if __name__ == "__main__":
    subjects = sys.argv[1:] or ["sub-09"]
    print(f"{'subject':>8} {'chanset':>10} {'TS+LR':>7} {'MDM':>7}  (chance 33.3%)")
    agg = {}
    for s in subjects:
        X, y = preprocess(s, "AVI", use_ica=False, baseline_correct=False)
        for name, idx in CHAN_SETS.items():
            Xs = X[:, idx, :]
            a_ts = cv(ts_pipe, Xs, y); a_mdm = cv(mdm_pipe, Xs, y)
            print(f"{s:>8} {name:>10} {a_ts:6.1f}% {a_mdm:6.1f}%")
            agg.setdefault(name, {"ts": [], "mdm": []})
            agg[name]["ts"].append(a_ts); agg[name]["mdm"].append(a_mdm)
    if len(subjects) > 1:
        print("\n=== mean across subjects ===")
        for name in CHAN_SETS:
            print(f"{'MEAN':>8} {name:>10} {np.mean(agg[name]['ts']):6.1f}% {np.mean(agg[name]['mdm']):6.1f}%")
