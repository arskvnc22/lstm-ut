"""Output-gate closure tables for LSTM-UT, the linear-read and the no-forget-gate
ablations.

Reads only gate_closure.npz files written by ca_perturbation.py (gate-closure
mode) under iridis/lyapunov/results/r12_gate_closure_v1, with the definitions
of docs/gate-closure-r12.md (nothing is recomputed from a model).

Array axes: closure__<metric>[cell mode (update, frozen), start repeat s
(6, 13, 20, 30), closure length (1..15, then never reopened), repeat 0..60,
row (64)]. 'update': the cell keeps being written while the gate is closed;
'frozen': it is held at its repeat-s value.

What is averaged:
  - per-repeat metrics are means over the 64 shared input rows;
  - table 1 is also averaged over the four start repeats;
  - end states are % of branches pooled over 4 starts x 15 finite lengths x
    64 rows = 3,840 per run and mode (never pooled across seeds);
  - norms are medians over rows.
End-state classes are those of docs/perturb-v2.md (see
ca_r12_perturbation_tables.py); 'all-zero' here means all-zero output on every
repeat 51-60.

Usage:
  python -m cellular_automaton.ca_r12_gate_closure_tables
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAM = {"LSTM-UT": [113,115,117,119,121], "Linear read": [348,349,350,351,352],
       "No forget": [353,354,355,356,357], "Linear mem.": [388,389,390,391,392]}
_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--results-dir", type=Path, default=ROOT / "iridis/lyapunov/results/r12_gate_closure_v1")
_args = _parser.parse_args()
_find = lambda run: list(_args.results_dir.glob(f"run_{run}__*/best_extrapolation_unconstrained/tokens_all/gate_closure.npz"))
# A family is shown only when every seed has a result (one file per run).
_missing = [f for f, rs in FAM.items() if not all(len(_find(r)) == 1 for r in rs)]
if _missing:
    print("not shown (results missing): " + ", ".join(_missing) + "\n")
FAM = {f: rs for f, rs in FAM.items() if f not in _missing}
RUNS = [r for rs in FAM.values() for r in rs]


def _load(run):
    (path,) = _args.results_dir.glob(f"run_{run}__*/best_extrapolation_unconstrained/tokens_all/gate_closure.npz")
    return np.load(path)


Z = {r: _load(r) for r in RUNS}
STARTS = [6, 13, 20, 30]; UPD, FRZ = 0, 1; NEVER = 15
# Row label: short family tag + run number, looked up from FAM.
_TAG = {"LSTM-UT": "LSTM ", "Linear read": "Lin  ", "No forget": "NoF  ", "Linear mem.": "LMem "}
lab = lambda r: next(_TAG[f] for f, rs in FAM.items() if r in rs) + str(r)
def c(z, m): return z[f"closure__{m}"]

print("Clean baseline: centred |x| range over 6-60, min margin range, cell norm at 6/13/30/60, exact@60")
for r in RUNS:
    z = Z[r]; xn = z["clean__x_centred_norm"][6:61].mean(-1); mg = z["clean__margin_min"][6:61].mean(-1); cn = z["clean__cell_norm"].mean(-1)
    print(f" {lab(r)}: |x| {xn.min():.1f}-{xn.max():.1f}  margin {mg.min():.1f}-{mg.max():.1f}  cell " + "/".join(f"{cn[d]:.0f}" for d in (6,13,30,60))
          + f"  exact@60 {100*(z['clean__cell_accuracy'][60]==1).mean():.1f}%")

print("\n1. First closed repeat (L=1, update; mean over rows and the 4 start depths)")
print("    run   angle  clean|x| closed|x| rel.dist  cleanMargin closedMargin maxLogitΔ cellAcc exact")
for r in RUNS:
    z = Z[r]; v = {k: [] for k in "a cn xn d cm m l acc ex".split()}
    for i, s in enumerate(STARTS):
        t = s + 1; clean_n = z["clean__x_centred_norm"][t]
        v["a"].append(c(z,"x_angle_mean")[UPD,i,0,t].mean()); v["cn"].append(clean_n.mean())
        v["xn"].append(c(z,"x_centred_norm")[UPD,i,0,t].mean()); v["d"].append((c(z,"x_centred_distance")[UPD,i,0,t]/clean_n).mean())
        v["cm"].append(z["clean__margin_min"][t].mean()); v["m"].append(c(z,"margin_min")[UPD,i,0,t].mean())
        v["l"].append(c(z,"logit_distance")[UPD,i,0,t].mean()); v["acc"].append(c(z,"cell_accuracy")[UPD,i,0,t].mean())
        v["ex"].append((c(z,"cell_accuracy")[UPD,i,0,t]==1).mean())
    m = {k: np.mean(x) for k, x in v.items()}
    print(f" {lab(r)}: {m['a']:.2f}  {m['cn']:7.1f} {m['xn']:8.1f} {m['d']:8.2f}  {m['cm']:10.1f} {m['m']:11.1f} {m['l']:9.1f} {m['acc']:.3f} {100*m['ex']:4.0f}%")

print("\n2. Never reopened: cell accuracy at closed repeats +1..+4, min margin +1,+2 (mean over starts)")
for r in RUNS:
    z = Z[r]; acc = lambda k: np.mean([c(z,"cell_accuracy")[UPD,i,NEVER,s+k].mean() for i,s in enumerate(STARTS)])
    mg = lambda k: np.mean([c(z,"margin_min")[UPD,i,NEVER,s+k].mean() for i,s in enumerate(STARTS)])
    print(f" {lab(r)}: " + " ".join(f"{acc(k):.3f}" for k in (1,2,3,4)) + f" | {mg(1):6.1f} {mg(2):6.1f}")

print("\n3. After reopening. Exact rows at 60 over all modes/starts/finite lengths; rows exact at any repeat >= s+L+2; L=1 update cell acc +1..+5")
for r in RUNS:
    z = Z[r]; acc = c(z,"cell_accuracy"); ex60 = (acc[:, :, :15, 60] == 1).mean()
    late = sum(int((acc[mo,i,li,s+L+2:61] == 1).sum()) for mo in (0,1) for i,s in enumerate(STARTS) for li,L in enumerate(range(1,16)))
    tc = [np.mean([acc[UPD,i,0,s+k].mean() for i,s in enumerate(STARTS)]) for k in (1,2,3,4,5)]
    print(f" {lab(r)}: exact@60 {100*ex60:.2f}%  late-exact row-repeats {late}  L=1: " + " ".join(f"{x:.3f}" for x in tc))

def end_states(z, mode, lengths):
    acc = c(z,"cell_accuracy")[mode][:, lengths]; st = c(z,"self_transition_accuracy")[mode][:, lengths][..., 51:61, :]
    ones = c(z,"ones_fraction")[mode][:, lengths][..., 51:61, :].mean(-2); chg = c(z,"changed_fraction")[mode][:, lengths][..., 51:61, :].mean(-2)
    preds = c(z,"predictions")[mode][:, lengths][..., 51:61, :, :]
    correct = acc[..., 60, :] == 1
    faithful = ~correct & (st == 1).all(-2) & (ones >= .25) & (ones <= .75) & (chg >= .25) & (chg <= .75)
    frozen = ~correct & ~faithful & ((chg < .1) | (ones < .1) | (ones > .9))
    other = ~correct & ~faithful & ~frozen
    allzero = (preds.sum(-1) == 0).all(-2)
    return [100 * x.mean() for x in (correct, faithful, frozen, other, allzero)], faithful

print("\n4. End states, pooled over 4 starts x 15 finite lengths x 64 rows = 3840 branches: correct / faithful / frozen / other | all-zero on 51-60")
for r in RUNS:
    for mode, name in ((UPD, "update"), (FRZ, "frozen")):
        v, _ = end_states(Z[r], mode, slice(0, 15))
        print(f" {lab(r)} {name:6s}: " + " / ".join(f"{x:5.1f}" for x in v[:4]) + f" | {v[4]:5.1f}")
    v, _ = end_states(Z[r], UPD, slice(15, 16))
    print(f" {lab(r)} never : " + " / ".join(f"{x:5.1f}" for x in v[:4]) + f" | {v[4]:5.1f}")

print("\n4b. Faithful-wrong-row % by closure length 1..15 (pooled over starts and rows)")
for r in RUNS:
    for mode, name in ((UPD, "update"), (FRZ, "frozen")):
        _, f = end_states(Z[r], mode, slice(0, 15))   # f: [start, length, row]
        print(f" {lab(r)} {name:6s}: " + " ".join(f"{100*f[:, li].mean():3.0f}" for li in range(15)))

print("\n5. Self-transition after a one-repeat closure at s=13, update: open repeats +1 +2 +3 +4 +5 +6 +8 +10 +12, repeat 60, ones at +6 and 60")
for r in RUNS:
    z = Z[r]; st = c(z,"self_transition_accuracy")[UPD,1,0]; on = c(z,"ones_fraction")[UPD,1,0]
    # first open repeat is s+L+1 = 15
    print(f" {lab(r)}: " + " ".join(f"{st[14+k].mean():.2f}" for k in (1,2,3,4,5,6,8,10,12)) + f" | {st[60].mean():.2f} | {on[20].mean():.2f} {on[60].mean():.2f}")

print("\n6. Median centred |x| at 60, s=13, update: clean, L=1, L=5, L=15, never")
for r in RUNS:
    z = Z[r]; xn = c(z,"x_centred_norm")[UPD,1]
    print(f" {lab(r)}: {np.median(z['clean__x_centred_norm'][60]):7.1f} " + " ".join(f"{np.median(xn[li,60]):7.1f}" for li in (0,4,14,15)))
print("\nmax cell norm in any branch:", {r: round(float(np.nanmax(c(Z[r],'cell_norm'))),1) for r in RUNS})
print("max centred |x| in any branch:", {r: round(float(np.nanmax(c(Z[r],'x_centred_norm'))),1) for r in RUNS})
print("any +/-inf:", {r: bool(np.isinf(c(Z[r],'x_centred_norm')).any() or np.isinf(c(Z[r],'cell_norm')).any()) for r in RUNS})