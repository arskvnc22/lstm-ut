"""Paired-perturbation (v2) tables for LSTM-UT and the linear-read and no-forget ablations.

Reads only perturbation.npz files written by ca_perturbation.py under
iridis/lyapunov/results/r12_perturb_v2 (nothing is recomputed from a model),
with the definitions of docs/perturb-v2.md.

Array axes: [inject repeat (6, 13, 20, 30), theta (0.3, 0.6, 1.0 rad),
direction (4), sign (2), repeat 0..60, row (64)].
A branch is one (direction, sign, row): 4 * 2 * 64 = 512 branches per
(kind, inject repeat, theta). Every percentage is over those 512 branches, per
seed; seeds are never pooled.

End state of a branch from repeats 51-60:
  correct   all 64 bits right against Rule 30 at repeat 60
  faithful  not correct; self-transition accuracy exactly 1 at every repeat
            51..60; mean ones and mean changed fraction both in [0.25, 0.75]
  frozen    not correct or faithful; mean changed < 0.1, or mean ones < 0.1
            or > 0.9
  other     everything else
  all-zero  (extra column, overlaps frozen) the decoded row at repeat 60 is
            all zeros, which Rule 30 maps to itself

Usage:
  python -m cellular_automaton.ca_r12_perturbation_tables
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"LSTM-UT": [113, 115, 117, 119, 121], "Linear read": [348, 349, 350, 351, 352],
            "No forget": [353, 354, 355, 356, 357], "Linear mem.": [388, 389, 390, 391, 392]}
INJECT = {6: 0, 13: 1, 20: 2, 30: 3}
THETA = {0.3: 0, 0.6: 1, 1.0: 2}
CHECKPOINT = "best_extrapolation_unconstrained"


def end_states(z, kind, inject, theta):
    i, t = INJECT[inject], THETA[theta]
    accuracy = z[f"{kind}__cell_accuracy"][i, t]                      # [dir, sign, 61, rows]
    self_transition = z[f"{kind}__self_transition_accuracy"][i, t][..., 51:61, :]
    ones = z[f"{kind}__ones_fraction"][i, t][..., 51:61, :].mean(-2)
    changed = z[f"{kind}__changed_fraction"][i, t][..., 51:61, :].mean(-2)
    correct = accuracy[..., 60, :] == 1
    faithful = (~correct & (self_transition == 1).all(-2)
                & (ones >= .25) & (ones <= .75) & (changed >= .25) & (changed <= .75))
    frozen = ~correct & ~faithful & ((changed < .1) | (ones < .1) | (ones > .9))
    other = ~correct & ~faithful & ~frozen
    all_zero = ~correct & (z[f"{kind}__predictions"][i, t][..., 60, :, :].sum(-1) == 0)
    return [100 * m.mean() for m in (correct, faithful, frozen, other, all_zero)]


def first_wrong(accuracy, start):
    """First repeat >= start whose row is not exactly right; 61 if none. accuracy: [..., 61, rows]."""
    wrong = accuracy[..., start:61, :] < 1
    return np.where(wrong.any(-2), wrong.argmax(-2) + start, 61)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "iridis/lyapunov/results/r12_perturb_v2")
    args = parser.parse_args()

    def find(run):
        return list(args.results_dir.glob(f"run_{run}__*/{CHECKPOINT}/tokens_all/perturbation.npz"))

    # A family is shown only when every seed has a result (one file per run).
    families = {f: runs for f, runs in FAMILIES.items() if all(len(find(r)) == 1 for r in runs)}
    missing = [f for f in FAMILIES if f not in families]
    if missing:
        print("not shown (results missing): " + ", ".join(missing) + "\n")
    data = {run: np.load(find(run)[0]) for runs in families.values() for run in runs}

    print("End states, correct / faithful-wrong / frozen / other % of 512 branches")
    for kind, inject in [("rotate_x", 13), ("rotate_x", 30), ("radial_x", 13), ("cell", 13)]:
        for theta in (0.3, 0.6, 1.0):
            print(f"\n{kind} at {inject}, {theta} rad")
            for family, runs in families.items():
                for seed, run in enumerate(runs):
                    v = end_states(data[run], kind, inject, theta)
                    print(f"  {family:11s} s{seed} {run}: "
                          + " / ".join(f"{x:.0f}" for x in v[:4]) + f"  (all-zero {v[4]:.0f}%)")

    print("\nClean branch: exact rows at 60, mean self-transition over 51-60")
    for family, runs in families.items():
        for seed, run in enumerate(runs):
            z = data[run]
            print(f"  {family:11s} s{seed} {run}: exact@60={100 * (z['clean__cell_accuracy'][60] == 1).mean():.1f}%"
                  f"  selftrans={z['clean__self_transition_accuracy'][51:61].mean():.4f}")

    print("\nFirst wrong repeat, rotate_x at 13, rows exactly right on the clean branch at 13:"
          " median clean -> perturbed over branches, and % of branches earlier than clean")
    for theta in (0.3, 0.6):
        print(f" {theta} rad")
        for family, runs in families.items():
            out = []
            for run in runs:
                z = data[run]
                clean = first_wrong(z["clean__cell_accuracy"], 13)                 # [rows]
                perturbed = first_wrong(z["rotate_x__cell_accuracy"][1, THETA[theta]], 13)  # [dir, sign, rows]
                keep = z["clean__cell_accuracy"][13] == 1
                clean_b = np.broadcast_to(clean, perturbed.shape)[..., keep]
                perturbed_b = perturbed[..., keep]
                out.append(f"{int(np.median(clean_b))}->{int(np.median(perturbed_b))}"
                           f" ({100 * (perturbed_b < clean_b).mean():.0f}%)")
            print(f"  {family:11s} " + ", ".join(out))

    print("\nSelf-transition accuracy after rotate_x at 13 (mean over 512 branches),"
          " at +1 +3 +7 +10 +17 +47 repeats after injection")
    for theta in (0.6, 1.0):
        print(f" {theta} rad")
        for family, runs in families.items():
            for seed, run in enumerate(runs):
                s = data[run]["rotate_x__self_transition_accuracy"][1, THETA[theta]]
                print(f"  {family:11s} s{seed} {run}: "
                      + " ".join(f"{s[..., 13 + d, :].mean():.3f}" for d in (1, 3, 7, 10, 17, 47)))

    print("\nNorm at repeat 60 relative to clean, rotate_x at 13, median over 512 branches:"
          " centred x | cell, at 0.3 / 0.6 / 1.0 rad")
    for family, runs in families.items():
        for seed, run in enumerate(runs):
            z = data[run]
            ratio = lambda name, clean, t: np.median(
                z[f"rotate_x__{name}"][1, THETA[t]][..., 60, :] / z[clean][60])
            print(f"  {family:11s} s{seed} {run}: "
                  + " / ".join(f"{ratio('x_centred_norm', 'clean__x_centred_norm', t):.2f}" for t in THETA)
                  + "  |  "
                  + " / ".join(f"{ratio('cell_norm', 'clean__cell_norm', t):.2f}" for t in THETA))


if __name__ == "__main__":
    main()
