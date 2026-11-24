"""Faithful recovery after a perturbation, in a window matched across models.

Question: after a `rotate_x` perturbation breaks faithful Rule-30 computation,
does the branch return to applying Rule 30 exactly to its own output, and how
often, independently of when the model would collapse on its own?

Reads only perturbation.npz files written by ca_perturbation.py (nothing is
recomputed from a model). Results are written up in docs/perturb-v2.md,
section "Addendum 2 (2026-11-14): faithful recovery in a matched window".

Array axes: [inject repeat (6, 13, 20, 30), theta (0.3, 0.6, 1.0 rad),
direction (4), sign (2), repeat 0..60, row (64)]. A branch is one
(direction, sign, row).

Definitions:
  faithful repeat  t >= inject + 1 (self-transition is undefined at the
                   injection repeat) with self-transition accuracy exactly 1
                   AND that repeat's ones and changed fractions both in
                   [0.25, 0.75]. This excludes all-zero and frozen rows. A
                   random non-degenerate 64-cell row passes by chance with
                   probability ~2^-64, so a short streak already means Rule 30
                   is being applied.
  window           repeats inject+1 .. inject+H (H = 15), the same for every
                   model.
  eligible rows    rows whose clean (unperturbed) branch is faithful at every
                   repeat of the window, so a failure to recover inside it
                   cannot be the model's own collapse.
  broke            the branch has a non-faithful repeat in the window.
  recovered(k)     after its first non-faithful repeat, the branch has k
                   consecutive faithful repeats, all inside the window.
  lag              repeats from the first non-faithful repeat to the start of
                   the first k=5 streak (median over recovered branches).
Percentages are over broken branches on eligible rows. "pooled" sums broken
and recovered branches over the five seeds.

Usage:
  python -m cellular_automaton.ca_r12_faithful_recovery
  python -m cellular_automaton.ca_r12_faithful_recovery --root <results dir>
"""
import argparse
from pathlib import Path

import numpy as np

LO, HI = 0.25, 0.75
H = 15
INJECT_DEPTHS = (6, 13)
FAMILIES = ("BUT", "gatedBUT", "BUT+inject", "LSTM-UT", "lin-read", "no-forget", "linmem", "fprm", "fprm-noinj", "decay0.5", "decay0.9")


def family(run):
    if "fprm_noinj" in run:
        return "fprm-noinj"
    if "decay_but05" in run:
        return "decay0.5"
    if "decay_but09" in run:
        return "decay0.9"
    if "fprm" in run:
        return "fprm"
    if "linmem" in run:
        return "linmem"
    if "noforget" in run:
        return "no-forget"
    if "but_inject" in run:
        return "BUT+inject"
    if "lin_lstm" in run:
        return "lin-read"
    if "_lstm_r12" in run:
        return "LSTM-UT"
    if "gated" in run:
        return "gatedBUT"
    return "BUT"


def is_faithful(self_transition, ones, changed):
    return ((self_transition == 1) & (ones >= LO) & (ones <= HI)
            & (changed >= LO) & (changed <= HI))


def first_streak(good, after, k):
    """good: [..., T] bool; after: [...] repeat index. Returns (found, start) of
    the first run of k consecutive good repeats lying entirely after `after`."""
    repeats = np.arange(good.shape[-1])
    good = good & (repeats > after[..., None])
    current = np.zeros(good.shape[:-1], dtype=int)
    found = np.zeros(good.shape[:-1], dtype=bool)
    start = np.full(good.shape[:-1], -1)
    for t in range(good.shape[-1]):
        current = np.where(good[..., t], current + 1, 0)
        new = (current >= k) & ~found
        start = np.where(new, t - k + 1, start)
        found |= current >= k
    return found, start


def summarise(path):
    data = np.load(path)
    repeats = np.arange(data["clean__cell_accuracy"].shape[0])  # 0..60
    clean_faithful = is_faithful(
        np.nan_to_num(data["clean__self_transition_accuracy"].T, nan=-1),
        data["clean__ones_fraction"].T,
        np.nan_to_num(data["clean__changed_fraction"].T, nan=-1),
    )  # [row, T]

    def load(metric, fill=None):
        values = np.moveaxis(data[f"rotate_x__{metric}"], 4, -1)  # [..., row, T]
        return values if fill is None else np.nan_to_num(values, nan=fill)

    self_transition = load("self_transition_accuracy", fill=-1)
    ones = load("ones_fraction")
    changed = load("changed_fraction", fill=-1)

    out = {}
    for depth_index, depth in enumerate(data["inject_depths"]):
        if depth not in INJECT_DEPTHS:
            continue
        window = (repeats >= depth + 1) & (repeats <= depth + H)
        eligible = clean_faithful[:, window].all(-1)  # [row]
        for theta_index, theta in enumerate(data["thetas"]):
            index = (depth_index, theta_index)
            faithful = is_faithful(self_transition[index], ones[index], changed[index]) & window
            faithful = faithful[:, :, eligible]
            not_faithful = ~faithful & window
            first_break = np.where(not_faithful.any(-1), not_faithful.argmax(-1), len(repeats))
            broke = first_break < len(repeats)
            k3, _ = first_streak(faithful, first_break, 3)
            k5, start5 = first_streak(faithful, first_break, 5)
            lag = (start5 - first_break)[k5 & broke]
            out[(int(depth), float(theta))] = {
                "rows": int(eligible.sum()), "broke": int(broke.sum()),
                "k3": int((k3 & broke).sum()), "k5": int((k5 & broke).sum()),
                "lag": float(np.median(lag)) if lag.size else float("nan"),
            }
    return out


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("iridis/lyapunov/results/r12_perturb_v2"))
    args = parser.parse_args()

    table = {}
    paths = sorted(args.root.glob("run_*/best_extrapolation_unconstrained/tokens_all/perturbation.npz"))
    for path in paths:
        run = path.parts[-4]
        for (depth, theta), r in summarise(path).items():
            table.setdefault((family(run), depth, theta), []).append((run[4:7], r))

    print(f"window = {H} repeats after injection; rows whose clean branch is faithful throughout it")
    print("per seed: run:eligible rows  k3% / k5% of broken branches (lag = median repeats to a k5 streak)")
    for key in sorted(table, key=lambda k: (k[1], k[2], FAMILIES.index(k[0]))):
        fam, depth, theta = key
        seeds = "  ".join(
            f"{run}:{r['rows']:>2}r {pct(r['k3'], r['broke']):>4.0f}/{pct(r['k5'], r['broke']):>3.0f} lag {r['lag']:>3.0f}"
            for run, r in table[key])
        broke = sum(r["broke"] for _, r in table[key])
        k3 = sum(r["k3"] for _, r in table[key])
        k5 = sum(r["k5"] for _, r in table[key])
        print(f"{fam:8} {depth:>3} {theta:>4.1f} | {seeds} | pooled {pct(k3, broke):>5.1f} {pct(k5, broke):>5.1f} (broke {broke})")


if __name__ == "__main__":
    main()
