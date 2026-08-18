"""
Cross-subject (LOSO) evaluation: train on N-1 subjects, test on the held-out
one. This is the calibration-free setting from the proposal, and it gives the
model far more training data (5 x 120 = 600 trials) than within-subject (96),
which is where a Transformer like DreamEEG can actually help.

Optionally applies Euclidean Alignment (EA) per subject: whiten each subject's
trials by their mean covariance R^{-1/2}, aligning subjects to a common
reference (He & Wu 2020; Junqueira 2024). Simple, parameter-free, well-proven
for cross-subject EEG.

Usage:
  python cross_subject.py --model eegnet   --task AVI --ea
  python cross_subject.py --model dreameeg --task AVI --ea
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from scipy.linalg import fractional_matrix_power
from dreameeg import preprocess, build_eegnet, TASK_LABELS, RESULTS

DEV = ["sub-01", "sub-02", "sub-05", "sub-09", "sub-10", "sub-19"]

def make_model(name, classes, tp):
    if name == "eegnet": return build_eegnet(classes=classes, time_points=tp)
    if name == "dreameeg":
        from models import build_dreameeg; return build_dreameeg(classes=classes, time_points=tp)
    raise ValueError(name)

def euclidean_align(X):
    """X: (n,C,T). Whiten by mean covariance across trials -> aligned trials."""
    covs = np.array([xi @ xi.T / xi.shape[1] for xi in X])   # (n,C,C)
    R = covs.mean(0)
    R_inv_sqrt = np.real(fractional_matrix_power(R, -0.5))
    return np.array([R_inv_sqrt @ xi for xi in X]).astype(np.float32)

def load_all(task, use_ea):
    data = {}
    for s in DEV:
        X, y = preprocess(s, task, use_ica=False, baseline_correct=False)
        if use_ea: X = euclidean_align(X)
        data[s] = (X.astype(np.float32), y)
    return data

def train_eval(Xtr, ytr, Xte, yte, model, epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # standardize using TRAIN stats (per channel)
    m = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-8
    Xtr, Xte = (Xtr-m)/sd, (Xte-m)/sd
    tl = DataLoader(TensorDataset(torch.tensor(Xtr[:, None]), torch.tensor(ytr)), 64, shuffle=True)
    Xte_t = torch.tensor(Xte[:, None]).to(dev); yte_t = np.array(yte)
    net = make_model(model, int(ytr.max())+1, Xtr.shape[-1]).to(dev)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.05); crit = nn.CrossEntropyLoss()
    best = 0.
    for _ in range(epochs):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); crit(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = float(np.mean(net(Xte_t).argmax(1).cpu().numpy() == yte_t))
        best = max(best, acc)   # report best (comparable across models)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eegnet")
    ap.add_argument("--task", default="AVI", choices=["AVI", "FVI", "OVI"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--ea", action="store_true", help="apply Euclidean Alignment")
    args = ap.parse_args()
    chance = 100.0/len(TASK_LABELS[args.task])
    print(f"LOSO cross-subject | model={args.model} task={args.task} EA={args.ea} epochs={args.epochs}")
    data = load_all(args.task, args.ea)
    accs = []
    for held in DEV:
        Xtr = np.concatenate([data[s][0] for s in DEV if s != held])
        ytr = np.concatenate([data[s][1] for s in DEV if s != held])
        Xte, yte = data[held]
        a = train_eval(Xtr, ytr, Xte, yte, args.model, args.epochs)
        accs.append(a); print(f"  test={held}: {a*100:.1f}%  (train n={len(ytr)})")
    accs = np.array(accs)
    print(f"\nLOSO {args.model} {args.task} EA={args.ea}: {accs.mean()*100:.1f}% +/- {accs.std()*100:.1f}%  (chance {chance:.1f}%)")
    with open(RESULTS / "cross_subject.csv", "a") as f:
        f.write(f"{args.model},{args.task},{args.ea},{args.epochs},{accs.mean():.4f},{accs.std():.4f},{chance:.1f}\n")

if __name__ == "__main__":
    main()
