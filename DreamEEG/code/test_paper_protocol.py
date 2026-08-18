"""
Did we mis-measure? Replicate the paper's EXACT Table 2 protocol and compare
to our 5-fold CV, to see whether the gap is just the evaluation method.

Paper (Table 2, EEGNet): within-subject *single* 80/20 stratified split,
Adam lr=1e-3, batch=64, 500 epochs. Their released code reports the BEST test
accuracy reached at ANY epoch (test-set used to pick the epoch).

We report three numbers per subject (animals, ses-01):
  BEST  = max test acc over 500 epochs        (their method)  <- expect ~high
  FINAL = test acc at the last epoch          (honest)        <- expect lower
  LAST50= mean test acc over final 50 epochs  (honest, stable)
averaged over several random 80/20 splits (paper uses one; we average to be
stable). Compare BEST to the paper's 75.8% and to our 5-fold 48%.
"""
import sys, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dreameeg import preprocess, build_eegnet

DEV = ["sub-01", "sub-02", "sub-05", "sub-09", "sub-10", "sub-19"]
SPLIT_SEEDS = [0, 1, 2, 3, 4]     # paper uses ONE split; we average a few
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 500   # paper Table2=500, code=1500

def one_split(X, y, seed, epochs=500):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    m = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-8
    Xtr, Xte = (Xtr-m)/sd, (Xte-m)/sd
    tl = DataLoader(TensorDataset(torch.tensor(Xtr[:, None]), torch.tensor(ytr)), 64, shuffle=True)
    Xte_t = torch.tensor(Xte[:, None]).to(dev); yte_t = np.array(yte)
    net = build_eegnet(classes=int(y.max())+1, time_points=X.shape[-1]).to(dev)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.09)
    crit = nn.CrossEntropyLoss()
    test_curve = []
    for _ in range(epochs):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); crit(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(Xte_t).argmax(1).cpu().numpy()
        test_curve.append(float(np.mean(pred == yte_t)))
    tc = np.array(test_curve)
    return tc.max(), tc[-1], tc[-50:].mean()

if __name__ == "__main__":
    print("Paper Table-2 protocol replication (animals, ses-01, single 80/20 split)\n")
    print(f"{'subject':>8}   BEST(theirs)  FINAL   LAST50")
    B, F, L = [], [], []
    for s in DEV:
        X, y = preprocess(s, "AVI", use_ica=False, baseline_correct=False)
        bs, fs, ls = [], [], []
        for seed in SPLIT_SEEDS:
            b, f, l = one_split(X, y, seed, epochs=EPOCHS)
            bs.append(b); fs.append(f); ls.append(l)
        B.append(np.mean(bs)); F.append(np.mean(fs)); L.append(np.mean(ls))
        print(f"{s:>8}   {np.mean(bs)*100:6.1f}%     {np.mean(fs)*100:5.1f}%  {np.mean(ls)*100:5.1f}%")
    print(f"\n{'MEAN':>8}   {np.mean(B)*100:6.1f}%     {np.mean(F)*100:5.1f}%  {np.mean(L)*100:5.1f}%")
    print(f"\nPaper reports 75.8% (animals).  Our 5-fold CV honest = 48.2%.")
    print("If BEST ~ 75% but FINAL/LAST50 ~ 48%, the paper's number is the")
    print("best-epoch-on-the-test-set artifact, not real generalization.")
