"""Tables comparing the R12 Lyapunov results across model families.

Reads only summary.json files written by ca_lyapunov.py under
iridis/lyapunov/results/r12_v1 (nothing is recomputed from a model).

What is averaged:
  - Per seed, every number in summary.json is already a mean over the 64
    shared input rows; window rates are also time-averaged over the repeats in
    the window (log growth of the first QR direction, 32 tangent vectors).
  - Model rows weight the five training seeds equally; "±" is the sample SD
    across seeds (not a standard error).
  - The same 64 input rows are used by every model and seed.

"SQ" = shift-quotiented rate (each token's hidden-feature mean removed from
the tangents before QR); "raw" = without that projection. See
docs/lyapunov-r12-results.md for the definitions.

Usage:
  python -m cellular_automaton.ca_r12_lyapunov_tables
  python -m cellular_automaton.ca_r12_lyapunov_tables --checkpoint-type best_id
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "LSTM-UT": {113: "ca_lstm_r12_zero", 115: "ca_lstm_r12_zero", 117: "ca_lstm_r12_zero",
                119: "ca_lstm_r12_zero", 121: "ca_lstm_r12_zero"},
    "Linear read": {r: "ca_lin_lstm_r12_zero" for r in range(348, 353)},
    "BUT": {112: "ca_but_r12_zero", 114: "ca_but_r12_zero", 116: "ca_but_r12_zero",
            118: "ca_but_r12_zero", 120: "ca_but_r12_zero"},
    "Gated BUT": {r: "ca_gated_but_r12" for r in range(343, 348)},
    "No forget": {r: "ca_lstm_noforget_r12_zero" for r in range(353, 358)},
    "BUT + inj.": {r: "ca_but_inject_r12_zero" for r in range(358, 363)},
    # Fixed-decay linear memory (models/linmem_but.py): its "forget" statistics
    # are the learned per-channel decay lambda (ca_lyapunov.step_with_forget_gate).
    "Linear mem.": {r: "ca_linmem_r12_zero" for r in range(388, 393)},
    # BUT with FPRM residual scaling (models/fprm_but.py): no cell; its
    # summaries carry the learned scales (fprm_residual_scales).
    "FPRM BUT": {r: "ca_fprm_but_r12_zero" for r in range(393, 398)},
    # FPRM with --no-fprm_input_injection: e is not re-added after h_0 = e.
    "FPRM no inj.": {r: "ca_fprm_noinj_r12_zero" for r in range(398, 403)},
    # BUT with a fixed stream decay lambda (models/decay_but.py). A pure
    # per-token offset is invisible to the block, so its rate is exactly
    # ln(lambda): -0.693 for 0.5, -0.105 for 0.9.
    "Decay BUT 0.5": {r: "ca_decay_but05_r12_zero" for r in range(403, 408)},
    "Decay BUT 0.9": {r: "ca_decay_but09_r12_zero" for r in range(408, 413)},
}
# Models with a cell. The no-forget model has a cell but no forget gate: its
# cell-to-cell Jacobian block is the identity (c_t = c_{t-1} + ...), so its
# 'cell lead' rate is 0 by construction, and its forget statistics are NaN in
# summary.json and printed as '-'.
CELL_FAMILIES = ("LSTM-UT", "Linear read", "No forget", "Linear mem.")
WINDOWS = ["1-12", "13-20", "21-30", "31-60"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "iridis/lyapunov/results/r12_v1")
    parser.add_argument("--checkpoint-type", default="best_extrapolation_unconstrained")
    args = parser.parse_args()

    def path(run, name):
        return args.results_dir / f"run_{run}__{name}" / args.checkpoint_type / "summary.json"

    # A family is shown only when all five seeds have results for this
    # checkpoint (e.g. best_id is missing for the no-forget and injection runs).
    data, missing = {}, []
    for family, runs in FAMILIES.items():
        if all(path(r, n).exists() for r, n in runs.items()):
            data[family] = [json.loads(path(r, n).read_text()) for r, n in runs.items()]
        else:
            missing.append(family)
    mean_sd = lambda v: f"{st.mean(v):+.4f} ± {st.stdev(v):.4f}"
    seed_mean = lambda runs, get: st.mean(get(s) for s in runs)

    print(f"checkpoint: {args.checkpoint_type}")
    if missing:
        print("not shown (results missing for this checkpoint): " + ", ".join(missing))
    print()
    print("Shift-quotiented leading QR rate per window (mean ± SD over 5 seeds)")
    print(f"{'model':12s}" + "".join(f"{w:>20s}" for w in WINDOWS))
    for family, runs in data.items():
        print(f"{family:12s}" + "".join(
            f"{mean_sd([s['windows'][w]['shift_quotient']['top_exponent_mean'] for s in runs]):>20s}"
            for w in WINDOWS))

    print("\nRepeats 31-60 rates; accuracy and shift energy at repeat 60 (seed means)")
    print(f"{'model':12s}{'exact@60':>10s}{'bit@60':>9s}{'raw lead':>10s}{'SQ lead':>10s}"
          f"{'shift E@60':>11s}{'cell lead':>10s}{'x->x lead':>10s}")
    for family, runs in data.items():
        late = lambda s, key: s["windows"]["31-60"][key]["top_exponent_mean"]
        line = (f"{family:12s}"
                f"{100 * seed_mean(runs, lambda s: s['per_depth']['exact_accuracy'][59]):10.3f}"
                f"{100 * seed_mean(runs, lambda s: s['per_depth']['cell_accuracy'][59]):9.3f}"
                f"{seed_mean(runs, lambda s: late(s, 'full')):+10.5f}"
                f"{seed_mean(runs, lambda s: late(s, 'shift_quotient')):+10.5f}"
                f"{100 * seed_mean(runs, lambda s: s['per_depth']['top_vector_shift_fraction'][59]):10.2f}%")
        if family in CELL_FAMILIES:
            line += (f"{seed_mean(runs, lambda s: late(s, 'cell_to_cell')):+10.5f}"
                     f"{seed_mean(runs, lambda s: late(s, 'x_to_x')):+10.5f}")
        else:
            line += f"{'-':>10s}{'-':>10s}"
        print(line)

    print("\nPer seed (all models): SQ rate per window " + " / ".join(WINDOWS) + ", exact@60")
    for family, runs in data.items():
        for seed, (run, s) in enumerate(zip(FAMILIES[family], runs)):
            rates = " / ".join(f"{s['windows'][w]['shift_quotient']['top_exponent_mean']:+.4f}" for w in WINDOWS)
            print(f"{family:12s} s{seed} run {run}: {rates}  exact@60={100 * s['per_depth']['exact_accuracy'][59]:6.2f}%")

    print("\nPer seed (cell models): SQ rate 31-60 and 13-20, exact@60, cell norm @12/@60,"
          " forget mean @60, share of forget gates > 0.95 @60, |x| @60, checkpoint step")
    for family in (f for f in CELL_FAMILIES if f in data):
        for seed, ((run, _), s) in enumerate(zip(FAMILIES[family].items(), data[family])):
            p = s["per_depth"]
            if p["forget_mean"][59] == p["forget_mean"][59]:  # not NaN
                forget = (f"forget@60={p['forget_mean'][59]:.3f} "
                          f"retain@60={100 * p['forget_retaining_fraction'][59]:5.2f}% ")
            else:
                forget = "forget@60=- (no forget gate) "
            print(f"{family:12s} s{seed} run {run}: "
                  f"SQ31-60={s['windows']['31-60']['shift_quotient']['top_exponent_mean']:+.5f} "
                  f"SQ13-20={s['windows']['13-20']['shift_quotient']['top_exponent_mean']:+.5f} "
                  f"exact@60={100 * p['exact_accuracy'][59]:6.2f}% "
                  f"cell@12={p['cell_norm'][11]:7.1f} @60={p['cell_norm'][59]:7.1f} "
                  + forget +
                  f"|x|@60={p['x_norm'][59]:6.1f} step={s['provenance']['checkpoint_step']}")

    if "FPRM BUT" in data:
        print("\nFPRM BUT learned scales per seed (mean [min, max] over channels); 'kept' is"
              " alpha_2 * alpha_1^2, the share of the old stream one repeat keeps"
              " (0.14 at initialisation)")
        for seed, (run, s) in enumerate(zip(FAMILIES["FPRM BUT"], data["FPRM BUT"])):
            scales = s["fprm_residual_scales"]
            text = "  ".join(f"{name}={v['mean']:.3f} [{v['min']:.3f}, {v['max']:.3f}]"
                             for name, v in ((n, scales[n]) for n in
                                             ("alpha_1", "alpha_2", "beta_1", "beta_2", "stream_kept_per_repeat")))
            print(f"  s{seed} run {run}: " + text.replace("stream_kept_per_repeat", "kept"))


if __name__ == "__main__":
    main()
