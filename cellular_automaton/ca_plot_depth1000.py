"""Depth plots for the seed-by-density sweeps written by ca_seed_density_sweep.py.

Reads only the saved per-condition JSON files (nothing is recomputed from a
model) and pools every condition of a sweep: each point is a count summed over
all test seeds and densities, divided by the summed total.

Three figures, each with one panel per model family and one line per seed:

  exact_rows_vs_depth.png     exact-row accuracy; top linear depth, bottom log
  cell_accuracy_vs_depth.png  cell accuracy, same layout, with the 50% chance
                              line (a wrong Rule-30 row is about half wrong)
  collapse_speed.png          top: exact rows lost per repeat, as a percentage
                              of ALL rows (10-repeat moving mean, log scale), so
                              a sudden collapse is a spike and a steady error
                              rate is a flat line; bottom: rows whose decoded
                              state is not Rule 30 of their own previous
                              decoded state

Caveat for the bottom row: an all-zeros output is a fixed point of Rule 30, so
a row collapsed to all zeros counts as following Rule 30. These sweeps save
counts only, so they cannot separate that from a faithful wrong trajectory.

Usage (defaults are the R12 depth-1000 sweeps of LSTM-UT, linear read and
no forget gate):
  python -m cellular_automaton.ca_plot_depth1000
  python -m cellular_automaton.ca_plot_depth1000 --family "LSTM-UT=363,364" \
      --output-dir some/dir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILIES = [
    "LSTM-UT=363,364,365,366,367",
    "Linear read=368,369,370,371,372",
    "No forget gate=373,374,375,376,377",
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
SMOOTH_WINDOW = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path,
                        default=ROOT / "iridis/ca-rule30/runs")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "iridis/ca-rule30/plots/depth1000_r12")
    parser.add_argument("--family", action="append", default=None,
                        help='"Name=run,run,..." of sweep run numbers, in seed '
                             "order (repeatable; at most five runs each)")
    parser.add_argument("--trained-depth", type=int, default=12)
    return parser.parse_args()


def parse_families(specs):
    families = []
    for spec in specs:
        name, runs = spec.split("=", 1)
        run_ids = [int(r) for r in runs.split(",") if r]
        if not 1 <= len(run_ids) <= len(COLORS):
            raise ValueError(f"{name}: expected 1-{len(COLORS)} runs, got {len(run_ids)}")
        families.append((name, run_ids))
    return families


def load_curves(runs_dir, run):
    """Pooled per-depth percentages for one sweep run."""
    matches = sorted(runs_dir.glob(f"run_{run}__*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one directory for run {run}, found {matches}")
    results = matches[0] / "results"
    manifest = json.loads((results / "sweep_manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError(f"Run {run} sweep status is {manifest['status']}")
    summary = json.loads((results / "condition_summary.json").read_text())
    conditions = [json.loads((results / entry["file"]).read_text())["per_depth"]
                  for entry in summary]
    depth = len(conditions[0])
    if any(len(c) != depth for c in conditions):
        raise ValueError(f"Run {run}: conditions have different depths")

    def total(key):
        return np.array([sum(c[t][key] for c in conditions) for t in range(depth)], float)

    rows, cells = total("row_total"), total("cell_total")
    return {
        "source": manifest["source_run_id"],
        "exact": 100 * total("exact_rows") / rows,
        "cell": 100 * total("cell_correct") / cells,
        "rule_break": 100 * (1 - total("decoded_recurrence_exact_rows") / rows),
    }


def style_axis(ax, max_depth, trained_depth, logx=False):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(trained_depth, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(1, max_depth)


def plot_lines(ax, families_data, run_ids, key, transform=lambda y: y):
    for i, run in enumerate(run_ids):
        curve = families_data[run][key]
        depth = np.arange(1, len(curve) + 1)
        ax.plot(depth, transform(curve), color=COLORS[i], ls=STYLES[i], lw=1.5,
                label=f"seed {i} (run {run})")


def new_figure(n_cols, title):
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 6.6), sharey="row",
                             constrained_layout=True, squeeze=False)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    return fig, axes


def accuracy_figure(families, data, key, ylabel, title, path, max_depth,
                    trained_depth, chance=None):
    fig, axes = new_figure(len(families), title)
    for col, (name, run_ids) in enumerate(families):
        for row, logx in enumerate([False, True]):
            ax = axes[row, col]
            style_axis(ax, max_depth, trained_depth, logx)
            if chance is not None:
                ax.axhline(chance, color=MUTED, lw=0.8, ls="--")
            plot_lines(ax, data, run_ids, key)
            ax.set_ylim(-2, 102)
            if row == 0:
                ax.set_title(name, color=INK, loc="left", fontsize=10)
            else:
                ax.set_xlabel("repeat (depth)")
                ax.legend(fontsize=7, frameon=False, loc="lower left")
            if col == 0:
                ax.set_ylabel(ylabel + (" (log depth)" if logx else ""))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def collapse_figure(families, data, path, max_depth, trained_depth):
    fig, axes = new_figure(
        len(families),
        "How fast rows are lost. Sudden collapse = spike on top; steady error = "
        f"flat line. Dotted line: trained depth {trained_depth}.",
    )
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    for col, (name, run_ids) in enumerate(families):
        ax = axes[0, col]
        style_axis(ax, max_depth, trained_depth)
        for i, run in enumerate(run_ids):
            exact = data[run]["exact"]
            # Rows lost at each repeat; repeat 0 is the input, counted as 100%.
            lost = np.maximum(-np.diff(np.concatenate([[100.0], exact])), 0)
            smoothed = np.convolve(lost, kernel, mode="same")
            ax.plot(np.arange(1, len(exact) + 1), np.maximum(smoothed, 1e-4),
                    color=COLORS[i], ls=STYLES[i], lw=1.4, label=f"seed {i} (run {run})")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 20)
        ax.set_title(name, color=INK, loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        if col == 0:
            ax.set_ylabel(f"exact rows lost per repeat, %\n({SMOOTH_WINDOW}-repeat mean)")

        ax = axes[1, col]
        style_axis(ax, max_depth, trained_depth)
        plot_lines(ax, data, run_ids, "rule_break")
        ax.set_ylim(-2, 102)
        ax.set_xlabel("repeat (depth)")
        if col == 0:
            ax.set_ylabel("rows not equal to Rule 30 of\nown previous decoded row, %")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    families = parse_families(args.family or DEFAULT_FAMILIES)
    data = {run: load_curves(args.runs_dir, run)
            for _, run_ids in families for run in run_ids}
    max_depth = max(len(d["exact"]) for d in data.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })
    accuracy_figure(
        families, data, "exact", "exact rows %",
        f"Exact-row accuracy vs depth (pooled over all sweep conditions). "
        f"Top: linear depth; bottom: log depth. Dotted line: trained depth {args.trained_depth}.",
        args.output_dir / "exact_rows_vs_depth.png", max_depth, args.trained_depth,
    )
    accuracy_figure(
        families, data, "cell", "cell accuracy %",
        "Cell accuracy vs depth (same runs). Dashed: 50% chance. "
        "Top: linear depth; bottom: log depth.",
        args.output_dir / "cell_accuracy_vs_depth.png", max_depth,
        args.trained_depth, chance=50,
    )
    collapse_figure(families, data, args.output_dir / "collapse_speed.png",
                    max_depth, args.trained_depth)
    for name, run_ids in families:
        for run in run_ids:
            d = data[run]
            print(f"{name:15s} run {run} ({d['source']}): exact@{len(d['exact'])}="
                  f"{d['exact'][-1]:.2f}% cell={d['cell'][-1]:.2f}%")
    print(f"Wrote 3 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
