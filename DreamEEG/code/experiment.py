"""
Stage-0 evaluation harness for DreamEEG.
=======================================
Runs a model across several subjects x several seeds, with proper aggregation,
so the numbers are trustworthy (single-subject / single-seed results are far
too noisy on ~120 trials).

For each subject:  5-fold stratified CV, repeated over N seeds, averaged.
Across subjects:   report mean +/- std of the per-subject accuracy.
Metrics:           accuracy, macro-F1, Cohen's kappa.

Usage:
    python experiment.py --model eegnet --task AVI --seeds 0,1,2
    python experiment.py --model eegnet --subjects dev --task AVI --prep lite
    python experiment.py --model dreameeg --task AVI            # once implemented

Everything runs on GPU if available. Rows are appended to results/<model>.csv.
"""
import argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score

from dreameeg import preprocess, build_eegnet, TASK_LABELS, RESULTS

# 6-subject dev set: VVIQ spread 36-80, both genders, incl. the experienced subj
DEV_SET = ["sub-01", "sub-02", "sub-05", "sub-09", "sub-10", "sub-19"]
FS = 250

# ------------------------------------------------------------------ model registry
def make_model(name, classes, time_points):
    if name == "eegnet":
        return build_eegnet(classes=classes, time_points=time_points)
    if name == "dreameeg":
        from models import build_dreameeg          # added in Stage 1
        return build_dreameeg(classes=classes, time_points=time_points)
    raise ValueError(f"unknown model {name}")

def _sliding(X, y, win_s, stride_s, fs=FS):
    w, st = int(win_s*fs), int(stride_s*fs); T = X.shape[-1]; xs, ys = [], []
    for i in range(len(X)):
        for s0 in range(0, T-w+1, st):
            xs.append(X[i, :, s0:s0+w]); ys.append(y[i])
    return np.stack(xs)[:, None], np.array(ys)

# ------------------------------------------------------------------ one CV pass
def cv_once(X, y, seed, model_name, window="full", epochs=400):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = int(y.max()) + 1
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs, f1s, kaps = [], [], []
    for tr, va in skf.split(X, y):
        Xtr, Xva = X[tr], X[va]
        m = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-8
        Xtr, Xva = (Xtr-m)/sd, (Xva-m)/sd
        if window == "slide":
            xtr, ytr = _sliding(Xtr, y[tr], 2.0, 0.5); xva, yva = _sliding(Xva, y[va], 2.0, 0.5)
        else:
            xtr, ytr = Xtr[:, None], y[tr]; xva, yva = Xva[:, None], y[va]
        tl = DataLoader(TensorDataset(torch.tensor(xtr), torch.tensor(ytr)), 64, shuffle=True)
        vl = DataLoader(TensorDataset(torch.tensor(xva), torch.tensor(yva)), 128)
        net = make_model(model_name, classes, xtr.shape[-1]).to(dev)
        opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.09)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=30)
        crit = nn.CrossEntropyLoss()
        best_a, best_f, best_k = 0., 0., 0.
        for _ in range(epochs):
            net.train()
            for xb, yb in tl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); crit(net(xb), yb).backward(); opt.step()
            net.eval(); P, Y = [], []
            with torch.no_grad():
                for xb, yb in vl:
                    P += net(xb.to(dev)).argmax(1).cpu().tolist(); Y += yb.tolist()
            a = float(np.mean(np.array(P) == np.array(Y))); sch.step(a)
            if a > best_a:
                best_a = a
                best_f = f1_score(Y, P, average="macro")
                best_k = cohen_kappa_score(Y, P)
        accs.append(best_a); f1s.append(best_f); kaps.append(best_k)
    return float(np.mean(accs)), float(np.mean(f1s)), float(np.mean(kaps))

# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eegnet")
    ap.add_argument("--subjects", default="dev", help="'dev' or comma list")
    ap.add_argument("--task", default="AVI", choices=["AVI", "FVI", "OVI"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--prep", default="lite", choices=["lite", "ica"])
    ap.add_argument("--window", default="full", choices=["full", "slide"])
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    subjects = DEV_SET if args.subjects == "dev" else args.subjects.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    chance = 100.0 / len(TASK_LABELS[args.task])
    print(f"model={args.model}  task={args.task}  prep={args.prep}  window={args.window}  "
          f"epochs={args.epochs}  seeds={seeds}")
    print(f"subjects={subjects}   chance={chance:.1f}%\n")

    out = RESULTS / f"{args.model}.csv"
    new = not out.exists()
    fout = open(out, "a")
    if new:
        fout.write("model,subject,task,prep,window,seed,acc,macroF1,kappa,chance\n")

    per_subject = []
    for sub in subjects:
        try:
            X, y = preprocess(sub, args.task, use_ica=(args.prep == "ica"), baseline_correct=False)
        except SystemExit:
            print(f"  {sub}: data not available yet, skipping"); continue
        seed_accs = []
        for s in seeds:
            a, f, k = cv_once(X, y, s, args.model, args.window, args.epochs)
            seed_accs.append(a)
            fout.write(f"{args.model},{sub},{args.task},{args.prep},{args.window},{s},"
                       f"{a:.4f},{f:.4f},{k:.4f},{chance:.1f}\n"); fout.flush()
        subj_acc = float(np.mean(seed_accs))
        per_subject.append((sub, subj_acc))
        print(f"  {sub}: {subj_acc*100:5.1f}%  (seeds {[round(x*100,1) for x in seed_accs]})")

    fout.close()
    if per_subject:
        accs = np.array([a for _, a in per_subject])
        print(f"\n===== {args.model} / {args.task} / {args.prep} =====")
        print(f"per-subject: " + ", ".join(f"{s}={a*100:.1f}%" for s, a in per_subject))
        print(f"ACROSS {len(accs)} SUBJECTS: {accs.mean()*100:.1f}% +/- {accs.std()*100:.1f}%  "
              f"(chance {chance:.1f}%, paper ~{'75.8' if args.task=='AVI' else '75.1' if args.task=='FVI' else '62.0'}%)")
        print(f"best subject {per_subject[int(accs.argmax())][0]}={accs.max()*100:.1f}%, "
              f"worst {per_subject[int(accs.argmin())][0]}={accs.min()*100:.1f}%")

if __name__ == "__main__":
    main()
