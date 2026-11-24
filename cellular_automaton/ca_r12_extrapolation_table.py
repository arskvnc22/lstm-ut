"""R12 depth-extrapolation table: exact-row accuracy per depth and per-seed reach.

Reads only each run's eval_metrics.jsonl (nothing is recomputed from a model).
Used for docs/draft-5.1.md Table 3 and the headline accuracy numbers.

Extraction rule (the same as docs/rebuttal-ablations-results.md and the
paper's appendix table): keep the rows with
  data_split == "final_test"
  evaluation_role == "repeat_horizon_diagnostic"
  checkpoint_type == the requested checkpoint
  num_repeats == ca_steps   (the model runs exactly as many repeats as CA steps)
and read `value` (a fraction) for `exact_sequence_accuracy` and `cell_accuracy`.
The final test set has 4,096 rows.

Aggregation: model rows weight the five training seeds equally; "±" is the
sample SD across seeds (not a standard error).

Reach (per seed): the largest depth d >= 12 such that exact-row accuracy is at
least 95% at every depth 12..d. Evaluation stops at depth 60, so 60 means "at
least 60". "-" means the seed is below 95% already at depth 12, the deepest
trained depth.

Run folders are named in full: run numbers repeat across folders (run_113, for
example, exists for more than one model).

Usage:
  python -m cellular_automaton.ca_r12_extrapolation_table
  python -m cellular_automaton.ca_r12_extrapolation_table --checkpoint-type best_id
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "iridis/ca-rule30/runs"


def runs(numbers, name):
    return [f"run_{n}__{name}" for n in numbers]


# Seeds 0-4 in order.
FAMILIES = {
    "LSTM-UT": runs((113, 115, 117, 119, 121), "ca_lstm_r12_zero"),
    "Linear read": runs(range(348, 353), "ca_lin_lstm_r12_zero"),
    "No forget gate": runs(range(353, 358), "ca_lstm_noforget_r12_zero"),
    "Gated BUT": runs(range(343, 348), "ca_gated_but_r12"),
    "BUT + injection": runs(range(358, 363), "ca_but_inject_r12_zero"),
    "BUT": runs((112, 114, 116, 118, 120), "ca_but_r12_zero"),
    "BUT width 80": ["run_302__ca_but_pm80_r12_seed0", "run_303__ca_but_pm80_r12_seed1",
                     "run_304__ca_but_pm80_r12_seed2", "run_305__ca_but_pm80_r12_seed3",
                     "run_306__ca_but_pm80_r12"],
    "LSTM-UT width 48": runs(range(307, 312), "ca_lstm_pm48_r12_zero"),
    "Linear memory": runs(range(388, 393), "ca_linmem_r12_zero"),
    "FPRM BUT": runs(range(393, 398), "ca_fprm_but_r12_zero"),
    # FPRM with --no-fprm_input_injection: e is not re-added after h_0 = e.
    "FPRM no inj.": runs(range(398, 403), "ca_fprm_noinj_r12_zero"),
    # BUT with a fixed stream decay lambda (models/decay_but.py).
    "Decay BUT 0.5": runs(range(403, 408), "ca_decay_but05_r12_zero"),
    "Decay BUT 0.9": runs(range(408, 413), "ca_decay_but09_r12_zero"),
    "BUT d96 4 heads": runs(range(383, 388), "ca_but_d96_r12_zero"),
    "CoTFormer": runs(range(378, 383), "ca_cotf_r12_zero"),
}
DEPTHS = (12, 15, 20, 25, 30, 40, 60)
METRICS = ("exact_sequence_accuracy", "cell_accuracy")
THRESHOLD = 0.95


def load(run, checkpoint_type):
    """{(metric, depth): value} for one run, following the extraction rule."""
    values = {}
    # The files hold ~1M lines; a substring check skips most of them before
    # parsing. Every kept line is still checked field by field below.
    needles = ("repeat_horizon_diagnostic", "final_test", f'"{checkpoint_type}"')
    with open(RUNS_DIR / run / "eval_metrics.jsonl") as f:
        for line in f:
            if not all(n in line for n in needles):
                continue
            row = json.loads(line)
            if (row.get("data_split") != "final_test"
                    or row.get("evaluation_role") != "repeat_horizon_diagnostic"
                    or row.get("checkpoint_type") != checkpoint_type
                    or row.get("metric") not in METRICS
                    or row.get("num_repeats") != row.get("ca_steps")):
                continue
            key = (row["metric"], row["ca_steps"])
            # A duplicate row must agree, otherwise the rule is ambiguous for this run.
            assert key not in values or values[key] == row["value"], (run, key)
            values[key] = row["value"]
    return values


def reach(values):
    exact = lambda d: values.get(("exact_sequence_accuracy", d))
    if exact(12) is None or exact(12) < THRESHOLD:
        return "-"
    depth = 12
    while exact(depth + 1) is not None and exact(depth + 1) >= THRESHOLD:
        depth += 1
    return str(depth)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-type", default="best_extrapolation_unconstrained")
    args = parser.parse_args()

    print(f"checkpoint: {args.checkpoint_type}; exact-row accuracy (%), mean ± SD over seeds")
    print(f"{'model':18s}" + "".join(f"{f'd{d}':>15s}" for d in DEPTHS) + f"{'cell d60':>10s}  reach per seed")
    for family, family_runs in FAMILIES.items():
        loaded = [(run, load(run, args.checkpoint_type)) if (RUNS_DIR / run / "eval_metrics.jsonl").exists()
                  else (run, {}) for run in family_runs]
        # A run without final-test rows at every reported depth (still
        # training, or not yet pulled) is left out and named.
        incomplete = [run[4:7] for run, v in loaded
                      if any(("exact_sequence_accuracy", d) not in v for d in DEPTHS + (60,))]
        family_runs = [run for run, v in loaded if run[4:7] not in incomplete]
        data = [v for run, v in loaded if run[4:7] not in incomplete]
        if incomplete:
            print(f"{family:18s}left out (no final-test rows yet): {', '.join(incomplete)}")
        if len(data) < 2:
            continue
        cells = []
        for depth in DEPTHS:
            xs = [100 * v[("exact_sequence_accuracy", depth)] for v in data]
            cells.append(f"{st.mean(xs):6.2f} ± {st.stdev(xs):5.2f}")
        cell60 = st.mean(100 * v[("cell_accuracy", 60)] for v in data)
        print(f"{family:18s}" + "".join(f"{c:>15s}" for c in cells)
              + f"{cell60:>10.2f}  " + ", ".join(reach(v) for v in data))
        print(f"{'':18s}per seed exact d20/30/40/60: " + "; ".join(
            f"{run[4:7]}:" + "/".join(f"{100 * v[('exact_sequence_accuracy', d)]:.0f}" for d in (20, 30, 40, 60))
            for run, v in zip(family_runs, data)))


if __name__ == "__main__":
    main()
