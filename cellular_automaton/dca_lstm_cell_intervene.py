"""Delayed-recall cell interventions for LSTM-UT (reviewer zXsf Q2, Nov 2026).

For one trained ``dca_lstm_ut`` run, evaluate delayed recall under four
conditions that differ only in the recall repeats (evolution is untouched):

  baseline           unmodified model
  zero_cell          cell set to 0 before the first recall repeat
  freeze_cell        cell held at its end-of-evolution value and read
  close_output_gate  cell written but not read into x
  swap_cell          before the first recall repeat every row takes the cell
                     of the previous row in its batch (its "donor"); the
                     stream is not swapped

Each condition uses the paper's delayed-recall evaluation unchanged
(``evaluate_delayed_recall_pairs`` and ``summarize_delayed_recall_pairs``):
every horizon H in 1..Hmax, every query repeat q in 1..H, the no-op query
q = H excluded from the nontrivial macro, queries averaged within a horizon
and horizons averaged equally. All conditions see the same rows in the same
order. The intervention reaches the model only through recall calls; the
model reports the condition it applied and the evaluator checks it.

Two references are reported. ``cell_accuracy`` and ``exact_sequence_accuracy``
score the recalled row against the true Rule 30 row. ``internal_cell_accuracy``
and ``internal_exact_sequence_accuracy`` score it against the row the model
itself decoded at the requested repeat. Evolution is untouched, so that
internal reference is the same in every condition.

``max_horizon_donor_scores`` (every condition, longest horizon, one entry per
age) compares the recalled row with two true rows: the row's own target and the
target of its donor, i.e. the previous row of the batch at the same age. Cells
are binary, so on the cells where the two targets differ the output matches
exactly one of them: ``own_match_on_differing_cells`` is 1 when the output
ignores the donor, 0.5 when it carries no information, and below 0.5 when it
follows the donor. Hamming distance (mean wrong cells per row), exact-row match
and Matthews correlation are given against both targets. The baseline entry is
the "before" value for the swap.

Example:
    python -m cellular_automaton.dca_lstm_cell_intervene \
        --run-dir iridis/dca-30/runs/run_115__dca_lstm_ut_h12_recall1 \
        --device cuda --output-dir iridis/dca-30/results/cell_intervention_v1
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import torch

import config
import models

try:
    from .ca_forward import CAForwardContext, CAForwardPolicy
    from .ca_gen import rule30
    from .ca_reporting import read_json, utc_now, validate_manifest, write_json
    from .dca_eval import evaluate_delayed_recall_pairs, summarize_delayed_recall_pairs
    from .dca_intervene_eval import autocast_context, build_fixed_diagnostic_dataset
except ImportError:
    from ca_forward import CAForwardContext, CAForwardPolicy
    from ca_gen import rule30
    from ca_reporting import read_json, utc_now, validate_manifest, write_json
    from dca_eval import evaluate_delayed_recall_pairs, summarize_delayed_recall_pairs
    from dca_intervene_eval import autocast_context, build_fixed_diagnostic_dataset


SOURCE_MODEL = "dca_lstm_ut"
CONDITIONS = ("baseline", "zero_cell", "freeze_cell", "close_output_gate", "swap_cell")
REPORTED_METRICS = ("cell_accuracy", "exact_sequence_accuracy", "loss")


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run directory containing run_manifest.json.")
    parser.add_argument("--checkpoint-name", default="best_delayed_recall.pt")
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        help="Override provenance.checkpoint_dir from the manifest.",
    )
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--conditions", choices=CONDITIONS, nargs="+",
                        default=list(CONDITIONS))
    parser.add_argument(
        "--horizons", type=positive_int, nargs="+",
        help="Evolution horizons H. Defaults to the run's training horizons 1..Hmax.",
    )
    parser.add_argument(
        "--max-horizon-only", action="store_true",
        help="Evaluate only the longest horizon (about 7x faster at H = 20). The "
             "all-horizon averages then cover that one horizon.",
    )
    parser.add_argument(
        "--num-recall-repeats", type=positive_int,
        help="Defaults to ca_recall_repeats from the manifest (as in training).",
    )
    parser.add_argument("--eval-batch-size", type=positive_int)
    parser.add_argument("--max-batches", type=positive_int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _require_mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_source_manifest(run_dir):
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run manifest does not exist: {manifest_path}")
    manifest = _require_mapping(read_json(manifest_path), "run manifest")
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid run manifest: " + "; ".join(errors))
    if manifest["status"] != "completed":
        raise ValueError(f"run status is {manifest['status']!r}, not 'completed'")
    model_name = _require_mapping(manifest["model"], "model").get("model")
    resolved = _require_mapping(manifest["resolved_args"], "resolved_args")
    if model_name != SOURCE_MODEL or resolved.get("model") != SOURCE_MODEL:
        raise ValueError(f"expected model {SOURCE_MODEL!r}, found {model_name!r}")
    return manifest


def resolve_model_args(manifest):
    values = dict(_require_mapping(manifest["resolved_args"], "resolved_args"))
    config_format = values.get("config_format")
    if config_format not in config.registered_formats():
        raise ValueError(f"unsupported config format: {config_format!r}")
    parsed = config.parse_args_with_format(
        format=config_format,
        base_parser=argparse.ArgumentParser(add_help=False, allow_abbrev=False),
        args=[],
        namespace=argparse.Namespace(**values),
    )
    if not isinstance(parsed.dtype, torch.dtype):
        raise ValueError(f"unsupported dtype: {parsed.dtype!r}")
    model_args = argparse.Namespace(**values)
    model_args.dtype = parsed.dtype
    return model_args


def load_model(model_args, checkpoint_path, device):
    checkpoint = _require_mapping(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        "checkpoint",
    )
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint.model must be a non-empty state dictionary")
    step = checkpoint.get("itr")
    if hasattr(step, "item"):
        step = step.item()
    if isinstance(step, bool) or not isinstance(step, Integral):
        raise ValueError("checkpoint.itr must be an integer")
    model = models.make_model_from_args(model_args)
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device)).eval()
    return model, int(step)


def training_horizons(model_args):
    pairs = [tuple(int(value) for value in pair) for pair in model_args.ca_train_pairs]
    if any(steps != repeats for steps, repeats in pairs):
        raise ValueError("delayed recall requires diagonal training pairs")
    return sorted({repeats for _, repeats in pairs})


class CellInterventionContext:
    """Add one recall cell intervention to delayed-recall model calls only."""

    def __init__(self, base_context, condition):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition!r}")
        self.base_context = base_context
        self.model = base_context.model
        self.policy = base_context.policy
        self.condition = condition
        self.recall_calls = 0

    def call(self, inputs, **call_kwargs):
        is_recall = bool(call_kwargs.get("delayed_recall", False))
        if is_recall and self.condition != "baseline":
            call_kwargs["recall_cell_intervention"] = self.condition
        outputs = self.base_context.call(inputs, **call_kwargs)
        if is_recall:
            expected = None if self.condition == "baseline" else self.condition
            if outputs.get("recall_cell_intervention", "missing") != expected:
                raise RuntimeError(
                    "model did not report the requested recall cell intervention"
                )
            self.recall_calls += 1
        return outputs


def _select(metrics, names=REPORTED_METRICS):
    if metrics is None:
        return None
    return {name: float(metrics[name]) for name in names if name in metrics}


def _select_with_internal(metrics, internal_macro):
    """Ground-truth metrics plus the evaluator's internal-consistency macro."""
    selected = _select(metrics)
    if selected is None or internal_macro is None:
        return selected
    selected["internal_cell_accuracy"] = float(
        internal_macro["decoded_requested_repeat_cell_accuracy"])
    selected["internal_exact_sequence_accuracy"] = float(
        internal_macro["decoded_requested_repeat_exact_sequence_accuracy"])
    return selected


def _mcc(prediction, target):
    """Matthews correlation between two 0/1 tensors, pooled over all cells."""
    prediction, target = prediction.bool(), target.bool()
    tp = float((prediction & target).sum())
    tn = float((~prediction & ~target).sum())
    fp = float((prediction & ~target).sum())
    fn = float((~prediction & target).sum())
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0


def score_against_own_and_donor(prediction, own_target, donor_target):
    """Scores for one batch of recalled rows, all tensors [rows, cells] of 0/1."""
    differ = own_target != donor_target
    own_correct = prediction == own_target
    donor_correct = prediction == donor_target
    return {
        "rows": int(prediction.shape[0]),
        "differing_cells": int(differ.sum()),
        "agreeing_cells": int((~differ).sum()),
        "own_matches_on_differing_cells": int((own_correct & differ).sum()),
        "correct_on_agreeing_cells": int((own_correct & ~differ).sum()),
        "wrong_cells_vs_own": int((~own_correct).sum()),
        "wrong_cells_vs_donor": int((~donor_correct).sum()),
        "exact_rows_own": int(own_correct.all(dim=1).sum()),
        "exact_rows_donor": int(donor_correct.all(dim=1).sum()),
        "prediction": prediction.cpu(),
        "own_target": own_target.cpu(),
        "donor_target": donor_target.cpu(),
    }


def donor_scores(model, loader, device, *, condition, horizon, num_recall_repeats,
                 max_batches=None, ctx=None):
    """Recall at one horizon scored against the own and the donor target, by age.

    The donor of a row is the previous row of its batch (the same pairing that
    swap_cell uses), so the scores are comparable across conditions.
    """
    base_context = CAForwardContext(model, CAForwardPolicy(repeat_cache_window=None))
    context = CellInterventionContext(base_context, condition)
    per_age = {age: [] for age in range(horizon)}
    ctx = ctx or nullcontext()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["input_id"].to(device, dtype=torch.long)
        if inputs.shape[0] < 2:
            raise ValueError("donor scoring needs at least two rows per batch")
        state, true_rows = inputs, {}
        for repeat in range(1, horizon + 1):
            state = rule30(state)
            true_rows[repeat] = state
        for age in range(horizon):
            with ctx:
                outputs = context.call(
                    inputs, get_logits=True, return_all_logits=True, num_repeats=horizon,
                    delayed_recall=True, recall_age=age, num_recall_repeats=num_recall_repeats,
                )
            own_target = true_rows[horizon - age]
            per_age[age].append(score_against_own_and_donor(
                outputs["logits"].argmax(dim=-1), own_target,
                torch.roll(own_target, shifts=1, dims=0),
            ))

    rows = []
    for age, batches in per_age.items():
        total = {key: sum(b[key] for b in batches) for key in batches[0]
                 if key not in ("prediction", "own_target", "donor_target")}
        prediction = torch.cat([b["prediction"] for b in batches])
        cells = prediction.shape[1]
        rows.append({
            "recall_age": age,
            "query_repeat": horizon - age,
            "differing_cells_fraction": total["differing_cells"] / (total["rows"] * cells),
            "own_match_on_differing_cells":
                total["own_matches_on_differing_cells"] / max(total["differing_cells"], 1),
            "accuracy_on_agreeing_cells":
                total["correct_on_agreeing_cells"] / max(total["agreeing_cells"], 1),
            "hamming_to_own": total["wrong_cells_vs_own"] / total["rows"],
            "hamming_to_donor": total["wrong_cells_vs_donor"] / total["rows"],
            "exact_match_own": total["exact_rows_own"] / total["rows"],
            "exact_match_donor": total["exact_rows_donor"] / total["rows"],
            "mcc_to_own": _mcc(prediction, torch.cat([b["own_target"] for b in batches])),
            "mcc_to_donor": _mcc(prediction, torch.cat([b["donor_target"] for b in batches])),
        })
    return rows


def evaluate_condition(model, loader, device, *, condition, horizons,
                       num_recall_repeats, max_batches=None, ctx=None):
    base_context = CAForwardContext(model, CAForwardPolicy(repeat_cache_window=None))
    context = CellInterventionContext(base_context, condition)
    pair_results = evaluate_delayed_recall_pairs(
        context, loader, device,
        pairs=[(horizon, horizon) for horizon in horizons],
        max_batches=max_batches,
        num_recall_repeats=num_recall_repeats,
        num_examples=0,
        ctx=ctx,
    )
    summary = summarize_delayed_recall_pairs(pair_results)
    if context.recall_calls == 0:
        raise RuntimeError("no delayed-recall calls were made")

    max_horizon = max(horizons)
    final_pair = next(
        result for result in pair_results.values() if result["num_repeats"] == max_horizon
    )
    by_query = [
        {
            "query_repeat": int(query["query_repeat"]),
            "recall_age": int(query["recall_age"]),
            **_select(query["metrics"]),
            "internal_cell_accuracy": float(
                query["internal_consistency"]["decoded_requested_repeat"]["cell_accuracy"]),
            "internal_exact_sequence_accuracy": float(
                query["internal_consistency"]["decoded_requested_repeat"]["exact_sequence_accuracy"]),
        }
        for query in final_pair["queries"].values()
    ]
    return {
        "condition": condition,
        "recall_calls": context.recall_calls,
        "nontrivial_pair_macro": _select_with_internal(
            summary["nontrivial_queries_pair_macro"],
            summary["nontrivial_internal_consistency_pair_macro"]),
        "all_queries_pair_macro": _select_with_internal(
            summary["all_queries_pair_macro"],
            summary["all_internal_consistency_pair_macro"]),
        "max_horizon_nontrivial_macro": _select_with_internal(
            final_pair["nontrivial_queries_macro"],
            final_pair["nontrivial_internal_consistency_macro"]),
        "max_horizon_by_query": sorted(by_query, key=lambda row: row["query_repeat"]),
    }


def deltas_from_baseline(results):
    baseline = results.get("baseline")
    if baseline is None:
        return {}
    deltas = {}
    for condition, result in results.items():
        if condition == "baseline":
            continue
        deltas[condition] = {
            field: {
                name: result[field][name] - baseline[field][name]
                for name in baseline[field]
            }
            for field in ("nontrivial_pair_macro", "max_horizon_nontrivial_macro")
            if result[field] is not None and baseline[field] is not None
        }
    return deltas


def run(args):
    manifest = load_source_manifest(args.run_dir)
    model_args = resolve_model_args(manifest)
    checkpoint_dir = args.checkpoint_dir or Path(manifest["provenance"]["checkpoint_dir"])
    checkpoint_path = Path(checkpoint_dir) / args.checkpoint_name
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
    model, step = load_model(model_args, checkpoint_path, args.device)

    horizons = sorted(set(args.horizons)) if args.horizons else training_horizons(model_args)
    if args.max_horizon_only:
        horizons = [max(horizons)]
    if max(horizons) > int(model_args.ca_max_relative_age):
        raise ValueError("horizon exceeds ca_max_relative_age")
    num_recall_repeats = int(args.num_recall_repeats or model_args.ca_recall_repeats)
    loader, data_metadata = build_fixed_diagnostic_dataset(
        model_args, args.split, args.device, "manifest", args.eval_batch_size,
    )

    results = {}
    for condition in args.conditions:
        with torch.no_grad():
            results[condition] = evaluate_condition(
                model, loader, args.device,
                condition=condition,
                horizons=horizons,
                num_recall_repeats=num_recall_repeats,
                max_batches=args.max_batches,
                ctx=autocast_context(model_args, args.device),
            )
            results[condition]["max_horizon_donor_scores"] = donor_scores(
                model, loader, args.device,
                condition=condition,
                horizon=max(horizons),
                num_recall_repeats=num_recall_repeats,
                max_batches=args.max_batches,
                ctx=autocast_context(model_args, args.device),
            )

    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_id": manifest["run_id"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": step,
        "model_seed": manifest["resolved_args"].get("seed"),
        "max_relative_age": int(model_args.ca_max_relative_age),
        "num_recall_repeats": num_recall_repeats,
        "horizons": horizons,
        "data": data_metadata,
        "max_batches": args.max_batches,
        "device": str(args.device),
        "results": results,
        "deltas_from_baseline": deltas_from_baseline(results),
    }
    output_path = Path(args.output_dir) / manifest["run_id"] / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, summary)
    return summary, output_path


def print_summary(summary, output_path):
    print(f"{summary['run_id']}  step {summary['checkpoint_step']}  "
          f"K={summary['num_recall_repeats']}  H=1..{max(summary['horizons'])}")
    print(f"{'condition':<20}{'cell %':>9}{'exact %':>9}{'int cell %':>12}{'int exact %':>13}"
          f"{'d cell':>9}{'d exact':>9}{'d int cell':>12}{'d int exact':>13}")
    for condition, result in summary["results"].items():
        macro = result["nontrivial_pair_macro"]
        delta = summary["deltas_from_baseline"].get(condition, {}).get(
            "nontrivial_pair_macro", {}
        )
        print(
            f"{condition:<20}{100 * macro['cell_accuracy']:>9.2f}"
            f"{100 * macro['exact_sequence_accuracy']:>9.2f}"
            f"{100 * macro['internal_cell_accuracy']:>12.2f}"
            f"{100 * macro['internal_exact_sequence_accuracy']:>13.2f}"
            f"{100 * delta.get('cell_accuracy', 0.0):>9.2f}"
            f"{100 * delta.get('exact_sequence_accuracy', 0.0):>9.2f}"
            f"{100 * delta.get('internal_cell_accuracy', 0.0):>12.2f}"
            f"{100 * delta.get('internal_exact_sequence_accuracy', 0.0):>13.2f}"
        )
    print("own-target match on cells where own and donor targets differ, % by age "
          "(100 = donor ignored, 50 = no information, below 50 = follows donor):")
    for condition, result in summary["results"].items():
        values = " ".join(f"{100 * row['own_match_on_differing_cells']:.0f}"
                          for row in result["max_horizon_donor_scores"])
        print(f"{condition:<20}{values}")
    print(f"wrote {output_path}")


def main(argv=None):
    args = parse_args(argv)
    summary, output_path = run(args)
    print_summary(summary, output_path)


if __name__ == "__main__":
    main()
