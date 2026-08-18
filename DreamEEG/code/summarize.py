"""
Summarize results/<model>.csv into an honest baseline table:
  - per subject: mean over seeds
  - per category: mean +/- std across subjects, vs chance and the paper's claim
Writes a markdown table to results/<model>_summary.md and prints it.

Usage: python summarize.py [--model eegnet]
"""
import argparse, csv, statistics as st
from collections import defaultdict
from dreameeg import RESULTS

PAPER = {"AVI": 75.8, "FVI": 75.1, "OVI": 62.0}
CHANCE = {"AVI": 33.3, "FVI": 33.3, "OVI": 25.0}
CATNAME = {"AVI": "Animals", "FVI": "Figures", "OVI": "Objects"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="eegnet")
    args = ap.parse_args()
    path = RESULTS / f"{args.model}.csv"
    rows = list(csv.DictReader(open(path)))

    # per (subject, task): average accuracy over seeds
    bysub = defaultdict(list)
    for r in rows:
        bysub[(r["subject"], r["task"])].append(float(r["acc"]))
    subj_acc = {k: st.mean(v) for k, v in bysub.items()}
    subjects = sorted({s for s, _ in subj_acc})

    out = [f"# Honest baseline ({args.model}, clean leak-free 5-fold CV, seed-averaged)\n"]
    # per-subject table
    hdr = "| Subject | " + " | ".join(CATNAME[t] for t in ["AVI", "FVI", "OVI"]) + " |"
    out += [hdr, "|" + "---|" * 4]
    for s in subjects:
        cells = []
        for t in ["AVI", "FVI", "OVI"]:
            a = subj_acc.get((s, t))
            cells.append(f"{a*100:.1f}%" if a is not None else "-")
        out.append(f"| {s} | " + " | ".join(cells) + " |")

    # aggregate row
    out.append("")
    out.append("| Category | Our mean +/- std | Best subj | Chance | Paper | Gap |")
    out.append("|---|---|---|---|---|---|")
    for t in ["AVI", "FVI", "OVI"]:
        accs = [subj_acc[(s, t)]*100 for s in subjects if (s, t) in subj_acc]
        if not accs:
            continue
        m = st.mean(accs); sd = st.pstdev(accs)
        out.append(f"| {CATNAME[t]} | {m:.1f}% +/- {sd:.1f}% | {max(accs):.1f}% | "
                   f"{CHANCE[t]}% | {PAPER[t]}% | {m-PAPER[t]:+.1f} |")

    text = "\n".join(out) + "\n"
    (RESULTS / f"{args.model}_summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"saved -> {RESULTS / f'{args.model}_summary.md'}")

if __name__ == "__main__":
    main()
