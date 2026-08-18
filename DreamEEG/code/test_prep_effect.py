"""
Does the authors' ICA preprocessing + their single-split best-epoch protocol
push a good subject (sub-09) up to ~76%? Compares lite vs ICA preprocessing
under the paper's protocol.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dreameeg import preprocess, build_eegnet

def one_split(X, y, seed, epochs=500):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    m = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-8
    Xtr, Xte = (Xtr-m)/sd, (Xte-m)/sd
    tl = DataLoader(TensorDataset(torch.tensor(Xtr[:, None]), torch.tensor(ytr)), 64, shuffle=True)
    Xte_t = torch.tensor(Xte[:, None]).to(dev); yte_t = np.array(yte)
    net = build_eegnet(classes=int(y.max())+1, time_points=X.shape[-1]).to(dev)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.09); crit = nn.CrossEntropyLoss()
    curve = []
    for _ in range(epochs):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); crit(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            curve.append(float(np.mean(net(Xte_t).argmax(1).cpu().numpy() == yte_t)))
    c = np.array(curve); return c.max(), c[-50:].mean()

for prep, ica in [("lite", False), ("ICA", True)]:
    X, y = preprocess("sub-09", "AVI", use_ica=ica, baseline_correct=False)
    bs, ls = [], []
    for s in [0, 1, 2, 3, 4]:
        b, l = one_split(X, y, s); bs.append(b); ls.append(l)
    print(f"sub-09 {prep:>4}:  BEST(theirs)={np.mean(bs)*100:.1f}%   honest(last50)={np.mean(ls)*100:.1f}%")
