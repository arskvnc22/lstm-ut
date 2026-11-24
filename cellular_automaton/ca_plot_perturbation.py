"""Per-seed trajectory plots for the paired perturbation results.

Reads the files written by ca_perturbation.py (v2 or later: they must contain
self-transition accuracy) and draws one figure per trained run. Nothing is
recomputed from a model and no result file is modified.

Each figure is a grid: one column per injection depth, one row per metric.
Every panel shows, against the repeat number,

  - the unperturbed (clean) trajectory of that run, over all repeats;
  - one line per perturbation size, starting at the injection repeat;
  - a vertical line at the repeat where the perturbation was applied;
  - a shaded band over the repeats seen in training.

Lines are means over the input rows, and for perturbed branches also over
direction draws and signs. The rows are the same in every line of a figure, so
a perturbed line can be compared directly with the clean line under it.

Metrics (all are fractions of the 64 bits of a row):

  cell_accuracy              bits that match the Rule-30 target for that repeat
  self_transition_accuracy   bits that match Rule 30 applied once to the
                             branch's OWN decoded row at the previous repeat
  changed_fraction           bits that differ from the branch's own previous
                             repeat; a real Rule-30 row stays near 0.5

Self-transition accuracy is drawn next to the changed fraction on purpose: a
frozen all-zeros output is a fixed point of Rule 30 and scores 1, so a high
value only means "still computing Rule 30" where the output keeps changing.

Usage:
    python -m cellular_automaton.ca_plot_perturbation \
        --results-dir iridis/lyapunov/results/r12_perturb_v2 \
        --output-dir iridis/lyapunov/plots/r12_perturb_v2
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "lstm_ut_bidir": "LSTM-UT",
    "lin_lstm_ut_bidir": "Linear-read LSTM-UT",
    "but_full_depth": "BUT",
    "gated_but_bidir": "Gated BUT",
    "but_input_inject": "BUT + input injection",
    "linmem_but": "Fixed-decay linear memory BUT",
    "fprm_but": "BUT + FPRM residual scaling",
    "decay_but": "BUT + fixed stream decay",
}
METRIC_LABELS = {
    "cell_accuracy": "Bits right\nagainst Rule 30",
    "self_transition_accuracy": "Bits following Rule 30\nfrom own previous row",
    "changed_fraction": "Bits changed since\nprevious repeat",
    "ones_fraction": "Predicted ones",
    "bit_disagreement": "Bits differing\nfrom clean branch",
}
# Metrics that only exist as a comparison with the clean branch.
METRICS_WITHOUT_CLEAN_LINE = ("bit_disagreement",)
DEFAULT_METRICS = ("cell_accuracy", "self_transition_accuracy", "changed_fraction")

# One fixed colour per perturbation size, assigned in order of the saved
# angles, so an angle keeps its colour in every panel and figure. These three
# pass the colour-blind separation check against each other; the legend
# carries identity for the lightest one.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
CLEAN_COLOUR = "#0b0b0b"
SURFACE = "#fcfcfb"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BAND = "#efeeea"


def load_run(result_dir):
    """Arrays and provenance of one run's perturbation result."""
    arrays = np.load(result_dir / "perturbation.npz")
    provenance = json.loads((result_dir / "summary.json").read_text())["provenance"]
    return arrays, provenance


def training_seed(repo_root, run_id):
    """Training seed from the source run's manifest, or None if unavailable."""
    manifest_path = repo_root / "iridis" / "ca-rule30" / "runs" / run_id / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())

    def find_seed(node):
        if isinstance(node, dict):
            if isinstance(node.get("seed"), int):
                return node["seed"]
            for value in node.values():
                found = find_seed(value)
                if found is not None:
                    return found
        return None

    return find_seed(manifest)


def clean_curve(arrays, metric):
    """Mean over rows of the clean branch, indexed by repeat; NaN where undefined."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN repeats stay NaN
        return np.nanmean(arrays[f"clean__{metric}"], axis=1)


def perturbed_curve(arrays, kind, metric, depth_index, theta_index):
    """Mean over directions, signs and rows of one perturbed condition, by repeat.

    Repeats before the injection are NaN in the saved arrays and stay NaN, so
    the line starts where the perturbation was applied.
    """
    values = arrays[f"{kind}__{metric}"][depth_index, theta_index]  # [direction, sign, depth, row]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN repeats stay NaN
        return np.nanmean(values, axis=(0, 1, 3))


def plot_run(arrays, *, kind, metrics, title, trained_depth, output_path):
    inject_depths = [int(depth) for depth in arrays["inject_depths"]]
    thetas = [float(theta) for theta in arrays["thetas"]]
    if len(thetas) > len(SERIES_COLOURS):
        raise ValueError(
            f"{len(thetas)} perturbation sizes but only {len(SERIES_COLOURS)} validated colours; "
            "plot a subset or extend SERIES_COLOURS."
        )
    max_depth = arrays["clean__cell_accuracy"].shape[0] - 1
    repeats = np.arange(max_depth + 1)

    figure, axes = plt.subplots(
        len(metrics), len(inject_depths),
        figsize=(4.3 * len(inject_depths), 2.5 * len(metrics) + 1.3),
        sharex=True, sharey=True, squeeze=False, facecolor=SURFACE,
    )
    for row, metric in enumerate(metrics):
        for column, inject_depth in enumerate(inject_depths):
            axis = axes[row, column]
            axis.set_facecolor(SURFACE)
            axis.axvspan(1, trained_depth, color=BAND, zorder=0, linewidth=0)
            axis.axvline(inject_depth, color=TEXT_SECONDARY, linewidth=1.0, zorder=1)
            if metric not in METRICS_WITHOUT_CLEAN_LINE:
                # Drawn underneath and wider than the perturbed lines, so it
                # stays visible as a dark edge where a perturbed line lies
                # exactly on top of it.
                axis.plot(repeats, clean_curve(arrays, metric), color=CLEAN_COLOUR, linewidth=4.0,
                          solid_capstyle="round", zorder=2, label="Unperturbed")
            for theta_index, theta in enumerate(thetas):
                axis.plot(
                    repeats, perturbed_curve(arrays, kind, metric, column, theta_index),
                    color=SERIES_COLOURS[theta_index], linewidth=2.0, solid_capstyle="round",
                    zorder=3 + theta_index, label=f"Perturbed, size {theta:g}",
                )
            axis.set_ylim(-0.03, 1.03)
            axis.set_xlim(0, max_depth)
            axis.set_yticks([0.0, 0.5, 1.0])
            axis.grid(axis="y", color=GRID, linewidth=1.0)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color(GRID)
            axis.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
            if row == 0:
                axis.set_title(f"Perturbed at repeat {inject_depth}", fontsize=10.5,
                               color=CLEAN_COLOUR, loc="left")
            if column == 0:
                axis.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9.5, color=TEXT_SECONDARY)
            if row == len(metrics) - 1:
                axis.set_xlabel("Repeat", fontsize=9.5, color=TEXT_SECONDARY)

    # The two reference marks are explained once, in the first panel.
    first = axes[0, 0]
    first.text(trained_depth / 2 + 0.5, 0.04, "trained\nrepeats", fontsize=8, color=TEXT_SECONDARY,
               ha="center", va="bottom")
    first.annotate("perturbation", xy=(inject_depths[0], 0.30), xytext=(inject_depths[0] + 6, 0.30),
                   fontsize=8, color=TEXT_SECONDARY, va="center",
                   arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "linewidth": 0.8})

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=len(handles), frameon=False,
                  fontsize=9.5, labelcolor=TEXT_SECONDARY, bbox_to_anchor=(0.995, 0.955))
    figure.suptitle(title, x=0.01, ha="left", fontsize=13, color=CLEAN_COLOUR)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def kind_description(kind):
    return {
        "rotate_x": "rotation of every token's centred hidden state; size in radians",
        "radial_x": "centred hidden state scaled by exp(±size)",
        "cell": "cell moved by size × its norm",
        "shift_x": "per-token constant offset (control)",
    }.get(kind, kind)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory holding one sub-directory per run, e.g. .../r12_perturb_v2.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-type", default="best_extrapolation_unconstrained")
    parser.add_argument("--tokens", choices=("all", "single"), default="all")
    parser.add_argument("--kind", default="rotate_x",
                        help="Perturbation kind to plot; runs that did not run it are skipped.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--trained-depth", type=int, default=12,
                        help="Largest repeat count seen in training (shaded band).")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Run ids to plot (default: every run found).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    result_dirs = sorted(
        args.results_dir.glob(f"*/{args.checkpoint_type}/tokens_{args.tokens}")
    )
    if not result_dirs:
        raise ValueError(f"no perturbation results found under {args.results_dir}")
    written = []
    for result_dir in result_dirs:
        run_id = result_dir.parents[1].name
        if args.runs is not None and run_id not in args.runs:
            continue
        arrays, provenance = load_run(result_dir)
        missing = [metric for metric in args.metrics if f"{args.kind}__{metric}" not in arrays.files]
        if missing:
            print(f"skip {run_id}: no {args.kind} data for {', '.join(missing)}")
            continue
        model = MODEL_LABELS.get(provenance["model"], provenance["model"])
        # Recorded since Nov 2026; older results have no such key.
        if provenance.get("lstm_forget_gate") == "none":
            model += ", no forget gate"
        seed = training_seed(repo_root, run_id)
        seed_text = f"seed {seed}, " if seed is not None else ""
        title = (
            f"{model}, {seed_text}{run_id.split('__')[0].replace('_', ' ')} — "
            f"{kind_description(args.kind)}"
        )
        output_path = args.output_dir / f"{run_id}__{args.kind}.png"
        plot_run(arrays, kind=args.kind, metrics=args.metrics, title=title,
                 trained_depth=args.trained_depth, output_path=output_path)
        written.append(output_path)
        print(f"wrote {output_path}")
    if not written:
        raise ValueError("no figure was written; check --kind, --metrics and --runs")


if __name__ == "__main__":
    main()
