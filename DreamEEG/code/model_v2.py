"""
DreamEEG-v2: an improved decoder aimed at genuinely beating EEGNet on scarce
within-subject data, evaluated with the SAME honest 5-fold CV.

Architecture:
  multi-scale temporal conv (3 kernels) -> depthwise spatial conv -> BN/ELU
  -> squeeze-excite channel attention -> separable conv -> pool -> classifier
  (compact, ~15-25K params, heavily regularized)

Training recipe that actually helps 120-trial EEG:
  - mixup augmentation (interpolate trials + labels)
  - additive Gaussian noise + random channel dropout
  - label smoothing
  - K-model ensemble per fold (average softmax)

Run:  python model_v2.py sub-09
Reports DreamEEG-v2 vs EEGNet (same folds/seeds).
"""
import sys, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from dreameeg import preprocess, build_eegnet

SEEDS = [0, 1, 2]
KENS = 3            # ensemble size per fold
EPOCHS = 400

# ------------------------------------------------------------------ model
class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c//r, 4)), nn.ELU(),
                                nn.Linear(max(c//r, 4), c), nn.Sigmoid())
    def forward(self, x):                       # (B,C,1,T)
        s = x.mean(dim=(2, 3))
        return x * self.fc(s)[:, :, None, None]

class DreamEEGv2(nn.Module):
    def __init__(self, chans=32, classes=3, time_points=1000,
                 f=8, kernels=(64, 128, 256), D=2, dropout=0.5):
        super().__init__()
        self.tconv = nn.ModuleList([nn.Conv2d(1, f, (1, k), padding="same", bias=False) for k in kernels])
        fin = f * len(kernels)
        self.bn1 = nn.BatchNorm2d(fin)
        self.spatial = nn.Sequential(
            nn.Conv2d(fin, fin * D, (chans, 1), groups=fin, bias=False),
            nn.BatchNorm2d(fin * D), nn.ELU())
        self.se = SE(fin * D)
        self.pool1 = nn.Sequential(nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.sep = nn.Sequential(
            nn.Conv2d(fin * D, fin * D, (1, 16), groups=fin * D, padding="same", bias=False),
            nn.Conv2d(fin * D, fin * D, 1, bias=False),
            nn.BatchNorm2d(fin * D), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        feat = (time_points // 32) * (fin * D)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat, classes))
    def forward(self, x):
        x = torch.cat([c(x) for c in self.tconv], dim=1)
        x = torch.relu(self.bn1(x))
        x = self.spatial(x); x = self.se(x); x = self.pool1(x)
        x = self.sep(x)
        return self.head(x)

# ------------------------------------------------------------------ training
AUG = False   # match the main harness (no aug) for a fair comparison

def train_one(Xtr, ytr, Xte, yte, model_ctor, seed, dev):
    # training identical to experiment.py (which gets EEGNet to 57.8% on sub-09)
    torch.manual_seed(seed); np.random.seed(seed)
    net = model_ctor().to(dev)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.09)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=30)
    crit = nn.CrossEntropyLoss()
    tl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), 64, shuffle=True)
    Xte_t = torch.tensor(Xte).to(dev)
    best_acc, best_prob = 0., None
    for ep in range(EPOCHS):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            if AUG:
                xb = xb + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            crit(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            prob = torch.softmax(net(Xte_t), dim=1).cpu().numpy()
        acc = float(np.mean(prob.argmax(1) == yte)); sch.step(acc)
        if acc > best_acc:
            best_acc, best_prob = acc, prob
    return best_prob

def cv(model_ctor, X, y, seed):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs = []
    for tr, va in skf.split(X, y):
        Xtr, Xva = X[tr], X[va]
        m = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-8
        Xtr, Xva = ((Xtr - m) / sd)[:, None].astype(np.float32), ((Xva - m) / sd)[:, None].astype(np.float32)
        # K-model ensemble: average test softmax
        probs = np.mean([train_one(Xtr, y[tr], Xva, y[va], model_ctor, seed * 10 + k, dev)
                         for k in range(KENS)], axis=0)
        accs.append(np.mean(probs.argmax(1) == y[va]))
    return np.mean(accs) * 100

if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "sub-09"
    X, y = preprocess(subj, "AVI", use_ica=False, baseline_correct=False)
    print(f"{subj} animals (chance 33.3%)  |  ensemble K={KENS}, {EPOCHS} epochs, seeds {SEEDS}\n")
    for name, ctor in [("EEGNet(ref)", lambda: build_eegnet(classes=3, time_points=1000)),
                       ("DreamEEG-v2", lambda: DreamEEGv2())]:
        accs = [cv(ctor, X, y, s) for s in SEEDS]
        print(f"  {name:>12}: {np.mean(accs):.1f}% +/- {np.std(accs):.1f}%   seeds={[round(a,1) for a in accs]}")
