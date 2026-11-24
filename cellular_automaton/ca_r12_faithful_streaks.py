"""Longest faithful Rule-30 streak after a perturbation (r12_perturb_v2).

Question: once a branch has gone wrong after a `rotate_x` perturbation, does it
ever settle into a lasting, non-trivial Rule-30 computation on its own output,
and does that happen before the unperturbed (clean) branch would have failed?

Reads only perturbation.npz files written by ca_perturbation.py (nothing is
recomputed from a model). Results are written up in docs/perturb-v2.md,
section "Addendum (2026-11-14): longest faithful streak after perturbation".

Array axes: [inject repeat (6, 13, 20, 30), theta (0.3, 0.6, 1.0 rad),
direction (4), sign (2), repeat 0..60, row (64)]. A branch is one
(direction, sign, row): 512 branches per (inject repeat, theta), per seed.

Definitions:
  wrong       cell accuracy against the Rule-30 target < 1, at or after the
              injection repeat. First wrong repeat = first such repeat.
  faithful    a repeat t >= inject + 1 (self-transition is undefined at the
              injection repeat) with self-transition accuracy exactly 1 AND
              that repeat's ones fraction and changed fraction both in
              [0.25, 0.75]. The bounds exclude all-zero and frozen rows, which
              Rule 30 maps to themselves or near-themselves.
  streak      consecutive faithful repeats, counted only from the branch's
              first wrong repeat on (so an untouched, correct branch has 0).
Columns:
  >=10        share of branches whose longest streak is at least 10 repeats
  pre-clean   share with a >=10 streak that starts before the clean branch's
              own first wrong repeat on the same row (61 if it never fails)
  to60        share whose streak ending at repeat 60 is at least 10 long,
              i.e. still faithful at the end of the rollout
  p50/p90/max percentiles and maximum of the longest streak over branches

Only inject repeats 6 and 13 (before the baselines' own collapse) and
theta 0.6 and 1.0 rad (0.3 rad rarely causes an error in LSTM-UT) are shown.

Usage:
  python -m cellular_automaton.ca_r12_faithful_streaks
  python -m cellular_automaton.ca_r12_faithful_streaks --root <results dir>
"""
import argparse
from pathlib import Path

import numpy as np

LO, HI = 0.25, 0.75
STREAK = 10
INJECT_DEPTHS = (6, 13)
MIN_THETA = 0.5


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


def first_index(mask, default):
    """Index of the first True along the last axis, or `default` if none."""
    return np.where(mask.any(-1), mask.argmax(-1), default)


def streak_stats(good, clean_first_wrong):
    """good: [..., row, T] bool. Returns longest streak, streak ending at the
    last repeat, and whether a >=STREAK streak starts before the clean branch's
    first wrong repeat."""
    shape = good.shape[:-1]
    current = np.zeros(shape, dtype=int)
    longest = np.zeros(shape, dtype=int)
    pre_clean = np.zeros(shape, dtype=bool)
    for t in range(good.shape[-1]):
        current = np.where(good[..., t], current + 1, 0)
        longest = np.maximum(longest, current)
        start = t - (STREAK - 1)
        pre_clean |= (current >= STREAK) & (start < clean_first_wrong)
    return longest, current, pre_clean


def summarise(path):
    data = np.load(path)
    repeats = np.arange(data["clean__cell_accuracy"].shape[0])  # 0..60
    num_repeats = len(repeats)

    clean_wrong = (data["clean__cell_accuracy"].T < 1) & (repeats >= 1)  # [row, T]
    clean_first_wrong = first_index(clean_wrong, num_repeats)  # [row]

    def load(metric, fill=None):
        values = np.moveaxis(data[f"rotate_x__{metric}"], 4, -1)  # [..., row, T]
        return values if fill is None else np.nan_to_num(values, nan=fill)

    accuracy = load("cell_accuracy")
    self_transition = load("self_transition_accuracy", fill=-1)
    ones = load("ones_fraction")
    changed = load("changed_fraction", fill=-1)

    rows = []
    for depth_index, depth in enumerate(data["inject_depths"]):
        if depth not in INJECT_DEPTHS:
            continue
        for theta_index, theta in enumerate(data["thetas"]):
            if theta < MIN_THETA:
                continue
            index = (depth_index, theta_index)
            wrong = (accuracy[index] < 1) & (repeats >= depth)
            first_wrong = first_index(wrong, num_repeats)  # [dir, sign, row]
            faithful = (
                (self_transition[index] == 1)
                & (ones[index] >= LO) & (ones[index] <= HI)
                & (changed[index] >= LO) & (changed[index] <= HI)
                & (repeats >= depth + 1)
                & (repeats >= first_wrong[..., None])
            )
            longest, at_end, pre_clean = streak_stats(faithful, clean_first_wrong)
            rows.append({
                "depth": int(depth), "theta": float(theta),
                "ge": np.mean(longest >= STREAK), "pre": np.mean(pre_clean),
                "end": np.mean(at_end >= STREAK),
                "p50": np.percentile(longest, 50), "p90": np.percentile(longest, 90),
                "max": int(longest.max()),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("iridis/lyapunov/results/r12_perturb_v2"))
    args = parser.parse_args()

    paths = sorted(args.root.glob("run_*/best_extrapolation_unconstrained/tokens_all/perturbation.npz"))
    print(f"{'run':>4} {'fam':8} {'dep':>3} {'th':>4} | {'>=10':>6} {'>=10 pre-clean':>14} "
          f"{'to60 >=10':>9} | {'p50':>4} {'p90':>4} {'max':>4}")
    for path in paths:
        run = path.parts[-4]
        for r in summarise(path):
            print(f"{run[4:7]:>4} {family(run):8} {r['depth']:>3} {r['theta']:>4.1f} | "
                  f"{r['ge']:>6.1%} {r['pre']:>14.1%} {r['end']:>9.1%} | "
                  f"{r['p50']:>4.0f} {r['p90']:>4.0f} {r['max']:>4}")


if __name__ == "__main__":
    main()
