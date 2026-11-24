"""Contraction of a small state perturbation while every model is still correct.

Reads only perturbation.npz files written by ca_perturbation.py under
iridis/lyapunov/results/r12_perturb_v1 (the small-angle runs; nothing is
recomputed from a model). Reproduces docs/perturbation-r12-notes.md,
"Contraction while all models are correct", and gives the halving times in
docs/draft-5.1.md.

Measurement (defaults are the documented ones):
  intervention  rotate_x, theta = 0.01 rad, injected once at repeat 13. Each
                token's centred x is rotated towards a random direction; mean
                and centred norm are kept (see ca_perturbation.py).
  distance      x_centred_distance between the perturbed and the clean branch.
  rate          per branch (direction, sign, row):
                log(d_20 / d_13) / (20 - 13). Negative = shrinking.
  rows          only rows whose clean branch is exact (cell accuracy 1) at
                every repeat 1..20, so a failing model's own errors are not
                counted.
  per seed      mean rate over those branches.
  model         mean of the five seed means (seeds weighted equally), ± sample
                SD over seeds; halving time = ln 2 / -(model mean rate).

Second table (cell models only): where the same perturbation goes in the
cell. rotate_x moves only x, so the cell gap is first non-zero one repeat after
injection. Medians over branches on the same rows: cell_distance at
inject + 1, at the end of the window and at repeat 60; x_centred_distance and
cell_distance at repeat 60 divided by the clean branch's centred |x| and
|cell|. The clean cell norm is missing from the 2026-11-11 v1 files (LSTM-UT
113-121), printed as '-'. Rows are selected on repeats 1..end only, so at
repeat 60 a failing seed's own errors are included.

A run without a v1 result is reported as missing (the small-angle runs exist
for LSTM-UT, BUT and gated BUT; linear read, no forget and BUT + injection
need iridis/lyapunov/run_r12_perturbation.sh with the v1 settings).

Usage:
  python -m cellular_automaton.ca_r12_contraction_table
  python -m cellular_automaton.ca_r12_contraction_table --inject 6 --theta 0.001
"""
import argparse
import math
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "best_extrapolation_unconstrained"


def runs(numbers, name):
    return [f"run_{n}__{name}" for n in numbers]


# Seeds 0-4 in order; full folder names (run numbers repeat across folders).
FAMILIES = {
    "LSTM-UT": runs((113, 115, 117, 119, 121), "ca_lstm_r12_zero"),
    "Linear read": runs(range(348, 353), "ca_lin_lstm_r12_zero"),
    "No forget gate": runs(range(353, 358), "ca_lstm_noforget_r12_zero"),
    "Linear memory": runs(range(388, 393), "ca_linmem_r12_zero"),
    "FPRM BUT": runs(range(393, 398), "ca_fprm_but_r12_zero"),
    # FPRM with --no-fprm_input_injection: e is not re-added after h_0 = e.
    "FPRM no inj.": runs(range(398, 403), "ca_fprm_noinj_r12_zero"),
    # BUT with a fixed stream decay lambda (models/decay_but.py).
    "Decay BUT 0.5": runs(range(403, 408), "ca_decay_but05_r12_zero"),
    "Decay BUT 0.9": runs(range(408, 413), "ca_decay_but09_r12_zero"),
    "Gated BUT": runs(range(343, 348), "ca_gated_but_r12"),
    "BUT + injection": runs(range(358, 363), "ca_but_inject_r12_zero"),
    "BUT": runs((112, 114, 116, 118, 120), "ca_but_r12_zero"),
}


def seed_rate(path, inject, theta, end):
    """(mean rate over branches on clean-exact rows, number of such rows)."""
    data = np.load(path)
    inject_index = list(data["inject_depths"]).index(inject)
    theta_index = int(np.argmin(np.abs(data["thetas"] - theta)))
    assert math.isclose(data["thetas"][theta_index], theta, rel_tol=1e-6), data["thetas"]
    clean_exact = (data["clean__cell_accuracy"][1:end + 1] == 1).all(0)  # [row]
    distance = data["rotate_x__x_centred_distance"][inject_index, theta_index]  # [dir, sign, T, row]
    rate = np.log(distance[:, :, end] / distance[:, :, inject]) / (end - inject)
    return float(rate[:, :, clean_exact].mean()), int(clean_exact.sum())


def seed_cell_errors(path, inject, theta, end, late=60):
    """Where the same perturbation goes in the cell models (medians over
    branches on the same clean-exact rows). rotate_x moves only x, so the cell
    gap is first non-zero at inject + 1. Returns cell gap at inject+1, end,
    late; x gap / clean centred |x| at late; cell gap / clean |cell| at late
    (None when the file has no clean cell norm: the v1 files of 2026-11-11)."""
    data = np.load(path)
    if "rotate_x__cell_distance" not in data.files:
        return None
    inject_index = list(data["inject_depths"]).index(inject)
    theta_index = int(np.argmin(np.abs(data["thetas"] - theta)))
    clean_exact = (data["clean__cell_accuracy"][1:end + 1] == 1).all(0)
    cell = data["rotate_x__cell_distance"][inject_index, theta_index]  # [dir, sign, T, row]
    x = data["rotate_x__x_centred_distance"][inject_index, theta_index]
    at = lambda values, t: float(np.median(values[:, :, t][:, :, clean_exact]))
    relative = lambda values, norm, t: float(np.median((values[:, :, t] / norm[t])[:, :, clean_exact]))
    cell_relative = (relative(cell, data["clean__cell_norm"], late)
                     if "clean__cell_norm" in data.files else None)
    return (at(cell, inject + 1), at(cell, end), at(cell, late),
            relative(x, data["clean__x_centred_norm"], late), cell_relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "iridis/lyapunov/results/r12_perturb_v1")
    parser.add_argument("--inject", type=int, default=13)
    parser.add_argument("--theta", type=float, default=0.01)
    parser.add_argument("--end", type=int, default=20)
    args = parser.parse_args()

    print(f"rotate_x, theta {args.theta} rad at repeat {args.inject}; rate of x_centred_distance over "
          f"repeats {args.inject}-{args.end}; rows clean-exact at repeats 1-{args.end}")
    print(f"{'model':16s}{'mean ± SD':>18s}{'halving':>10s}   per seed (rows used)")
    for family, family_runs in FAMILIES.items():
        rates, cells = [], []
        for run in family_runs:
            path = args.results_dir / run / CHECKPOINT / "tokens_all" / "perturbation.npz"
            if not path.exists():
                cells.append(f"{run[4:7]}: missing")
                continue
            rate, rows = seed_rate(path, args.inject, args.theta, args.end)
            rates.append(rate)
            cells.append(f"{run[4:7]}: {rate:+.3f} ({rows})")
        if len(rates) == len(family_runs):
            mean = st.mean(rates)
            halving = f"{math.log(2) / -mean:.0f}" if mean < 0 else "none"
            summary = f"{mean:+.3f} ± {st.stdev(rates):.3f}{halving:>10s}"
        else:
            summary = f"{'incomplete':>18s}{'-':>10s}"
        print(f"{family:16s}{summary}   " + ", ".join(cells))

    print(f"\nCell models, same perturbation and rows (medians over branches): cell gap at repeats "
          f"{args.inject + 1} / {args.end} / 60 | x gap / clean |x| at 60 | cell gap / clean |cell| at 60")
    for family, family_runs in FAMILIES.items():
        for run in family_runs:
            path = args.results_dir / run / CHECKPOINT / "tokens_all" / "perturbation.npz"
            errors = seed_cell_errors(path, args.inject, args.theta, args.end) if path.exists() else None
            if errors is None:
                continue
            first, at_end, late, x_relative, cell_relative = errors
            cell_text = "-" if cell_relative is None else f"{cell_relative:.1e}"
            print(f"  {family:16s}{run[4:7]}: {first:.2e} / {at_end:.2e} / {late:.2e} | "
                  f"{x_relative:.1e} | {cell_text}")


if __name__ == "__main__":
    main()
