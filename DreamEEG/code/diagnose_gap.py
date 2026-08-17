"""
Why is our clean reproduction ~20-30 pts below the paper?
Isolate the two most likely causes on sub-09 animals (cached, lite preproc):

  A. CORRECT  - trial-level 5-fold split, full 4 s window, 400 epochs  (our baseline)
  B. EPOCHS   - same as A but 1500 epochs (their training budget); tests whether
                reporting best-val over many epochs on a 24-sample val set inflates.
  C. LEAKY    - generate sliding windows from ALL trials, THEN split on windows,
                so windows from the same trial land in train AND val. This is the
                classic EEG accuracy-inflation bug. If C jumps to ~75%, that is very
                likely how the paper's numbers were produced.
  D. CORRECT+AUG - trial-level split, THEN window within each fold (leak-free aug),
                to show honest augmentation does NOT inflate like C does.

All use the identical EEGNet + preprocessing, so differences are purely protocol.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from dreameeg import preprocess, build_eegnet
FS = 250

def train_eval(xtr, ytr, xva, yva, epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl = DataLoader(TensorDataset(torch.tensor(xtr), torch.tensor(ytr)), 64, shuffle=True)
    vl = DataLoader(TensorDataset(torch.tensor(xva), torch.tensor(yva)), 128)
    net = build_eegnet(classes=int(max(ytr))+1, time_points=xtr.shape[-1]).to(dev)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.09)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=30)
    crit = nn.CrossEntropyLoss(); best = 0.
    for _ in range(epochs):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); crit(net(xb), yb).backward(); opt.step()
        net.eval(); P, Y = [], []
        with torch.no_grad():
            for xb, yb in vl: P += net(xb.to(dev)).argmax(1).cpu().tolist(); Y += yb.tolist()
        a = float(np.mean(np.array(P) == np.array(Y))); sch.step(a); best = max(best, a)
    return best

def windows(X, y, win=2.0, stride=0.5):
    w, s = int(win*FS), int(stride*FS); T = X.shape[-1]; xs, ys, tid = [], [], []
    for i in range(len(X)):
        for a in range(0, T-w+1, s):
            xs.append(X[i, :, a:a+w]); ys.append(y[i]); tid.append(i)
    return np.stack(xs)[:, None], np.array(ys), np.array(tid)

def zscore(tr, va):
    m = tr.mean((0, 2, 3), keepdims=True); sd = tr.std((0, 2, 3), keepdims=True) + 1e-8
    return (tr-m)/sd, (va-m)/sd

def run_correct(X, y, epochs, aug=False):
    skf = StratifiedKFold(5, shuffle=True, random_state=0); accs = []
    for tr, va in skf.split(X, y):
        Xtr, Xva = X[tr], X[va]
        m = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True)+1e-8
        Xtr, Xva = (Xtr-m)/sd, (Xva-m)/sd
        if aug:
            xtr, ytr, _ = windows(Xtr, y[tr]); xva, yva, _ = windows(Xva, y[va])
        else:
            xtr, ytr = Xtr[:, None], y[tr]; xva, yva = Xva[:, None], y[va]
        accs.append(train_eval(xtr, ytr, xva, yva, epochs))
    return np.mean(accs)*100

def run_leaky(X, y, epochs):
    # z-score per trial-set is impossible here (we split after windowing); z-score globally
    Xw, yw, tid = windows(X, y)
    m = Xw.mean((0, 2, 3), keepdims=True); sd = Xw.std((0, 2, 3), keepdims=True)+1e-8
    Xw = (Xw-m)/sd
    skf = StratifiedKFold(5, shuffle=True, random_state=0); accs = []
    for tr, va in skf.split(Xw, yw):
        overlap = len(set(tid[tr]) & set(tid[va]))
        accs.append(train_eval(Xw[tr], yw[tr], Xw[va], yw[va], epochs))
    return np.mean(accs)*100, overlap

if __name__ == "__main__":
    X, y = preprocess("sub-09", "AVI", use_ica=False, baseline_correct=False)
    print(f"sub-09 AVI  X={X.shape}  chance=33.3%  (paper 75.8%)\n")
    a = run_correct(X, y, 400);            print(f"A. CORRECT full-window 400ep        : {a:.1f}%")
    b = run_correct(X, y, 1500);           print(f"B. CORRECT full-window 1500ep       : {b:.1f}%")
    d = run_correct(X, y, 200, aug=True);  print(f"D. CORRECT + leak-free window aug   : {d:.1f}%")
    c, ov = run_leaky(X, y, 200);          print(f"C. LEAKY window-level split 200ep   : {c:.1f}%   (train/val shared trials per fold: {ov})")
    print("\nIf C >> A/D, the paper's numbers are most likely sliding-window-leakage inflated.")
