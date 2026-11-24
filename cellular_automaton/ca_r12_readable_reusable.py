"""Readable is not reusable: a perturbed state that still decodes the right row.

Question: right after a `rotate_x` perturbation, the perturbed state usually
still decodes to the exact correct row. Does that state then drive the next
repeats correctly, or does the branch go wrong later even though the
unperturbed model stays exact?

Reads only perturbation.npz files written by ca_perturbation.py under
iridis/lyapunov/results/r12_perturb_v2 (nothing is recomputed from a model).
Used in docs/draft-5.1.md §5.4.

Array axes: [inject repeat (6, 13, 20, 30), theta (0.3, 0.6, 1.0 rad),
direction (4), sign (2), repeat 0..60, row (64)]. A branch is one
(direction, sign, row).

Definitions:
  eligible rows  the clean (unperturbed) branch is exact (cell accuracy 1) at
                 every repeat inject..inject+H (H = 15), so a later error
                 cannot be the model's own collapse.
  readable       the branch's row decoded at the injection repeat, i.e. from
                 the perturbed state, is exactly the Rule-30 target.
  later wrong    a readable branch whose row is not exact at some repeat
                 inject+1..inject+H.
Percentages: readable is over all branches on eligible rows; later wrong is
over readable branches. "pooled" sums branches over the five seeds. A seed
with no eligible row prints "nan".

Usage:
  python -m cellular_automaton.ca_r12_readable_reusable
  python -m cellular_automaton.ca_r12_readable_reusable --theta 1.0
"""
import argparse
from pathlib import Path

import numpy as np

H = 15
INJECT_DEPTHS = (6, 13)
FAMILIES = ("LSTM-UT", "Linear read", "No forget", "Linear mem.", "FPRM BUT", "FPRM no inj.", "Decay 0.5", "Decay 0.9", "Gated BUT", "BUT + inj.", "BUT")


def family(run):
    if "fprm_noinj" in run:
        return "FPRM no inj."
    if "decay_but05" in run:
        return "Decay 0.5"
    if "decay_but09" in run:
        return "Decay 0.9"
    if "fprm" in run:
        return "FPRM BUT"
    if "linmem" in run:
        return "Linear mem."
    if "noforget" in run:
        return "No forget"
    if "but_inject" in run:
        return "BUT + inj."
    if "lin_lstm" in run:
        return "Linear read"
    if "_lstm_r12" in run:
        return "LSTM-UT"
    if "gated" in run:
        return "Gated BUT"
    return "BUT"


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else float("nan")


def summarise(path, theta):
    """{inject: (branches on eligible rows, readable, readable and later wrong)}."""
    data = np.load(path)
    theta_index = int(np.argmin(np.abs(data["thetas"] - theta)))
    assert np.isclose(data["thetas"][theta_index], theta), data["thetas"]
    clean = data["clean__cell_accuracy"]  # [T, row]
    accuracy = np.moveaxis(data["rotate_x__cell_accuracy"], 4, -1)  # [inject, theta, dir, sign, row, T]
    out = {}
    for depth_index, depth in enumerate(data["inject_depths"]):
        if depth not in INJECT_DEPTHS:
            continue
        eligible = (clean[depth:depth + H + 1] == 1).all(0)  # [row]
        branch = accuracy[depth_index, theta_index][:, :, eligible]  # [dir, sign, row, T]
        readable = branch[..., depth] == 1
        later_wrong = (branch[..., depth + 1:depth + H + 1] < 1).any(-1)
        out[int(depth)] = (int(readable.size), int(readable.sum()), int((readable & later_wrong).sum()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("iridis/lyapunov/results/r12_perturb_v2"))
    parser.add_argument("--theta", type=float, default=0.6)
    args = parser.parse_args()

    paths = sorted(args.root.glob("run_*/best_extrapolation_unconstrained/tokens_all/perturbation.npz"))
    pooled = {}
    print(f"rotate_x, {args.theta} rad; window {H} repeats; rows whose clean branch is exact throughout")
    print("per seed: readable % of branches | later wrong % of readable branches")
    for path in paths:
        run = path.parts[-4]
        cells = []
        for depth, (branches, readable, wrong) in summarise(path, args.theta).items():
            total = pooled.setdefault((family(run), depth), [0, 0, 0])
            total[0] += branches
            total[1] += readable
            total[2] += wrong
            cells.append(f"inject {depth:2d}: rows {branches // 8:2d}  readable {pct(readable, branches):5.1f}%"
                         f"  later wrong {pct(wrong, readable):5.1f}%")
        print(f"{run[4:7]} {family(run):11s} " + " | ".join(cells))

    print("\npooled over seeds")
    for depth in INJECT_DEPTHS:
        for fam in FAMILIES:
            if (fam, depth) not in pooled:
                continue
            branches, readable, wrong = pooled[(fam, depth)]
            print(f"  {fam:11s} inject {depth:2d}: readable {pct(readable, branches):5.1f}% of {branches} branches,"
                  f" later wrong {pct(wrong, readable):5.1f}% of readable")


if __name__ == "__main__":
    main()
