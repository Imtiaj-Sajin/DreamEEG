"""
DreamEEG - Visual Imagery EEG decoding pipeline.
=================================================
Everything lives inside the repo (this file's grandparent = DreamEEG/):
    DreamEEG/data/      raw subject zips, extracted BIDS, npy caches
    DreamEEG/results/   CSV/JSON results
    DreamEEG/code/      this file + authors_reference/

Dataset: Gao et al. (2026) "EEG Dataset for Visual Imagery", figshare 30227503.
Tasks:  AVI = animals (dog/bird/fish), FVI = figures, OVI = objects.

Usage examples (run from anywhere):
    python dreameeg.py --subject sub-09 --task AVI                 # reproduce baseline
    python dreameeg.py --subject sub-09 --task AVI --window slide  # sliding-window aug
    python dreameeg.py --subject sub-09 --task AVI --no-ica        # fast, skip ICA
    python dreameeg.py --download sub-04                           # just fetch a subject

Runs on GPU automatically if CUDA is available.
"""
import os, sys, glob, json, zipfile, random, argparse, warnings, urllib.request
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- paths (all on G:, repo-relative)
REPO   = Path(__file__).resolve().parents[1]          # .../DreamEEG
DATA   = REPO / "data"
CACHE  = DATA / "cache"
RESULTS= REPO / "results"
for d in (DATA, CACHE, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

FIGSHARE_ARTICLE = "30227503"
FS_TARGET  = 250            # authors resample to 250 Hz
N_TIMES    = 1000           # 4 s @ 250 Hz
SEED       = 42
TASK_LABELS = {
    "AVI": {"dog": 1, "bird": 2, "fish": 3},                 # animals
    "FVI": {"circle": 1, "square": 2, "pentagram": 3},       # figures
    "OVI": {"cup": 1, "chair": 2, "watch": 3, "scissors": 4} # objects
}

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ----------------------------------------------------------------- data acquisition
def _figshare_files():
    fj = DATA / "fig.json"
    if not fj.exists():
        url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"
        urllib.request.urlretrieve(url, fj)
    return json.load(open(fj))["files"]

def download_subject(sub):
    """Download + extract one subject zip into DATA (skips if already present)."""
    ext = DATA / sub
    if list(ext.glob("**/*.bdf")):
        return ext
    zp = DATA / f"{sub}.zip"
    if not zp.exists():
        rec = next((f for f in _figshare_files() if f["name"] == f"{sub}.zip"), None)
        if rec is None:
            sys.exit(f"{sub}.zip not found in figshare listing")
        print(f"Downloading {sub}.zip ({rec['size']/1e6:.0f} MB) -> {zp}")
        urllib.request.urlretrieve(rec["download_url"], zp)
    print(f"Extracting {sub}.zip ...")
    with zipfile.ZipFile(zp) as z:
        z.extractall(ext)
    return ext

def find_bdf(sub, task, ses="ses-01"):
    root = download_subject(sub)
    hits = glob.glob(str(root / "**" / f"*{ses}*task-{task}*eeg.bdf"), recursive=True)
    if not hits:
        hits = glob.glob(str(root / "**" / f"*task-{task}*eeg.bdf"), recursive=True)
    if not hits:
        sys.exit(f"No {task} bdf for {sub}")
    return hits[0]

# ----------------------------------------------------------------- preprocessing (authors' pipeline)
def _events_from_status(raw):
    import numpy as np
    st = raw.get_data(picks=["Status"])[0]
    vals, cnts = np.unique(st, return_counts=True); base = vals[np.argmax(cnts)]
    codes = list(range(1, 5))
    idx = np.where(np.isin(st, codes))[0]; ev = []
    for i in idx:
        p = st[i-1] if i > 0 else base
        n = st[i+1] if i < len(st)-1 else base
        if p == base and n == base:
            ev.append([i, 0, int(st[i])])
    return np.array(ev, dtype=int)

def preprocess(sub, task, ses="ses-01", use_ica=True, baseline_correct=True):
    """Return X (n,32,1000) float32, y (n,) int. Caches to npy."""
    import mne
    mne.set_log_level("ERROR")
    tag = f"{sub}_{ses}_{task}_{'ica' if use_ica else 'lite'}_{'bl' if baseline_correct else 'nobl'}"
    cx, cy = CACHE / f"{tag}_X.npy", CACHE / f"{tag}_y.npy"
    if cx.exists() and cy.exists():
        print(f"  (cache) {tag}")
        return np.load(cx), np.load(cy)

    bdf = find_bdf(sub, task, ses)
    print(f"  preprocessing {os.path.basename(bdf)}  (ICA={use_ica})")
    raw = mne.io.read_raw_bdf(bdf, preload=True)
    ev = _events_from_status(raw)
    raw._data *= 1e-6
    raw, ev = raw.resample(FS_TARGET, events=ev, verbose=False)
    raw.set_channel_types({"Status": "stim"})
    raw.set_montage("standard_1020", on_missing="warn")

    if use_ica:
        from mne_icalabel import label_components
        from pyprep import NoisyChannels
        nc = NoisyChannels(raw.copy().pick("eeg")); nc.find_all_bads()
        raw.info["bads"] = nc.get_bads(); print(f"    bad channels: {raw.info['bads']}")
        raw.notch_filter(50, verbose=False)
        raw.filter(1., 100., method="fir", phase="zero-double", pad="edge", verbose=False)
        raw.set_eeg_reference("average", verbose=False)
        ica = mne.preprocessing.ICA(n_components=0.999999, method="picard",
              fit_params=dict(ortho=False, extended=True), random_state=SEED,
              max_iter=300, verbose=False)
        ica.fit(raw.copy().pick("eeg"))
        lab = label_components(raw.copy().pick("eeg"), ica, method="iclabel")["labels"]
        excl = [i for i, l in enumerate(lab) if l not in ("brain", "other")]
        print(f"    ICA: excluding {len(excl)}/{len(lab)} non-brain components")
        ica.apply(raw, exclude=excl)
        if raw.info["bads"]:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        raw.filter(4., 80., method="fir", phase="zero-double", pad="edge", verbose=False)
    else:
        raw.filter(1., 40., method="fir", phase="zero-double", pad="edge", verbose=False)
        raw.set_eeg_reference("average", verbose=False)

    raw.pick("eeg")
    ev_id = TASK_LABELS[task]
    bl = (-0.2, 0) if baseline_correct else None
    ep = mne.Epochs(raw, ev, event_id=ev_id, tmin=-0.2, tmax=4.0, baseline=bl,
                    preload=True, on_missing="warn", verbose=False)
    ep.crop(tmin=0)
    X = ep.get_data()[:, :32, :N_TIMES].astype(np.float32)
    y = ep.events[:, 2] - 1
    np.save(cx, X); np.save(cy, y)
    return X, y

# ----------------------------------------------------------------- model (authors' EEGNet)
def build_eegnet(chans=32, classes=3, time_points=1000):
    import torch.nn as nn
    tk, f1, f2, d, pk1, pk2, dr = 25, 8, 16, 2, 16, 8, 0.5
    lin = (time_points // (pk1 * pk2)) * f2
    class EEGNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.b1 = nn.Sequential(nn.Conv2d(1, f1, (1, tk), padding="same", bias=False), nn.BatchNorm2d(f1))
            s.b2 = nn.Sequential(nn.Conv2d(f1, d*f1, (chans, 1), groups=f1, bias=False), nn.BatchNorm2d(d*f1),
                                 nn.ELU(), nn.AvgPool2d((1, pk1)), nn.Dropout(dr))
            s.b3 = nn.Sequential(nn.Conv2d(d*f1, f2, (1, 16), groups=f2, bias=False, padding="same"),
                                 nn.Conv2d(f2, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(),
                                 nn.AvgPool2d((1, pk2)), nn.Dropout(dr))
            s.head = nn.Sequential(nn.Flatten(), nn.Linear(lin, classes))
        def forward(s, x): return s.head(s.b3(s.b2(s.b1(x))))
    return EEGNet()

def _sliding(X, y, win_s, stride_s, fs=FS_TARGET):
    w, st = int(win_s*fs), int(stride_s*fs); T = X.shape[-1]; xs, ys = [], []
    for i in range(len(X)):
        for s0 in range(0, T-w+1, st):
            xs.append(X[i, :, s0:s0+w]); ys.append(y[i])
    return np.stack(xs)[:, None], np.array(ys)

# ----------------------------------------------------------------- 5-fold CV (authors' protocol)
def run_cv(X, y, window="full", epochs=1000, n_splits=5):
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {dev} ({torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'})")
    classes = int(y.max()) + 1
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for k, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva = X[tr], X[va]
        m = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-8
        Xtr, Xva = (Xtr-m)/sd, (Xva-m)/sd
        if window == "slide":
            xtr, ytr = _sliding(Xtr, y[tr], 2.0, 0.5); xva, yva = _sliding(Xva, y[va], 2.0, 0.5)
        else:
            xtr, ytr = Xtr[:, None], y[tr]; xva, yva = Xva[:, None], y[va]
        tl = DataLoader(TensorDataset(torch.tensor(xtr), torch.tensor(ytr)), 64, shuffle=True)
        vl = DataLoader(TensorDataset(torch.tensor(xva), torch.tensor(yva)), 128)
        net = build_eegnet(classes=classes, time_points=xtr.shape[-1]).to(dev)
        opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.09)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=30)
        crit = nn.CrossEntropyLoss(); best, bf1 = 0., 0.
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
            if a > best: best, bf1 = a, f1_score(Y, P, average="macro")
        accs.append(best); f1s.append(bf1)
        print(f"    fold {k+1}: acc={best*100:5.1f}%  macroF1={bf1:.3f}")
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))

# ----------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="sub-09")
    ap.add_argument("--task", default="AVI", choices=["AVI", "FVI", "OVI"])
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--window", default="full", choices=["full", "slide"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--no-ica", action="store_true")
    ap.add_argument("--no-baseline", action="store_true", help="skip baseline correction")
    ap.add_argument("--download", metavar="SUB", help="only download+extract a subject, then exit")
    args = ap.parse_args()

    if args.download:
        download_subject(args.download); print("done."); return

    seed_all()
    chance = 100.0 / len(TASK_LABELS[args.task])
    print(f"[1] preprocess  {args.subject} {args.session} {args.task}")
    X, y = preprocess(args.subject, args.task, args.session,
                      use_ica=not args.no_ica, baseline_correct=not args.no_baseline)
    print(f"    X={X.shape}  y-counts={np.bincount(y)}  chance={chance:.1f}%")
    print(f"[2] EEGNet 5-fold CV  (window={args.window}, epochs={args.epochs})")
    acc, std, f1 = run_cv(X, y, window=args.window, epochs=args.epochs)

    print(f"\n  RESULT  {args.subject} {args.task}: {acc*100:.1f}% +/- {std*100:.1f}%  "
          f"(macroF1 {f1:.3f}, chance {chance:.1f}%)")
    out = RESULTS / "results.csv"
    new = not out.exists()
    with open(out, "a") as f:
        if new: f.write("subject,task,session,window,ica,epochs,acc,std,macroF1,chance\n")
        f.write(f"{args.subject},{args.task},{args.session},{args.window},"
                f"{not args.no_ica},{args.epochs},{acc:.4f},{std:.4f},{f1:.4f},{chance:.1f}\n")
    print(f"  saved -> {out}")

if __name__ == "__main__":
    main()
