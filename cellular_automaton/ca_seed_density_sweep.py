"""Compact Rule 30 seed-by-density sweep for looped CA checkpoints.

Supports every model with the fast per-repeat readout (LSTM-UT, BUT, GUT,
gated BUT and their ablations); originally written for LSTM-UT.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch

import models
from .ca_checkpoint_eval import (
    AVAILABLE_CHECKPOINT_TYPES,
    _autocast_context,
    _read_checkpoint_metadata,
    load_checkpoint_weights,
    resolve_source_args,
    validate_source_run,
)
from .ca_eval import _forward_all_cells
from .ca_forward import CAForwardContext
from .ca_gen import rule30
from .ca_main import make_ca_fixed_loaders
from .ca_reporting import forward_policy_metadata, utc_now, write_json


# Models whose returned repeat states decode exactly like independent forwards
# (the same set as the fast readout in ca_train/ca_eval).
SUPPORTED_MODELS = (
    "lstm_ut_bidir",
    "lin_lstm_ut_bidir",
    "but_full_depth",
    "gut_bidir",
    "gated_but_bidir",
    "but_input_inject",
    "linmem_but",
    "fprm_but",
    "decay_but",
    "decay_gated_but",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-type", choices=AVAILABLE_CHECKPOINT_TYPES, required=True
    )
    parser.add_argument("--test-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bernoulli-ps", type=float, nargs="+", required=True)
    parser.add_argument("--max-repeats", type=int, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--num-cells", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

@torch.inference_mode()
def evaluate_condition(context, loader, device, max_repeats, autocast):
    # Rows: target cells, target rows, recurrence cells, recurrence rows,
    # rows containing a non-finite logit. All counters stay on the GPU.
    counts = torch.zeros((5, max_repeats), dtype=torch.int64, device=device)
    rows_seen = 0
    num_cells = loader.dataset.inputs.shape[1]

    for batch_index, batch in enumerate(loader):
        inputs = batch["input_id"].to(device, dtype=torch.long, non_blocking=True)
        rows_seen += inputs.shape[0]
        target = inputs
        previous_prediction = inputs
        previous_finite = torch.ones(inputs.shape[0], dtype=torch.bool, device=device)

        with autocast:
            final_logits, _, outputs = _forward_all_cells(
                context, inputs, num_repeats=max_repeats,
                return_repeat_states=True, return_outputs=True,
            )
            states = outputs.get("repeat_states")
            if states is None or len(states) != max_repeats + 1:
                raise ValueError("The model did not return every repeat state.")

            for repeat in range(1, max_repeats + 1):
                logits = (
                    final_logits if repeat == max_repeats
                    else context.model.lm_head(
                        context.model.transformer.ln_f(states[repeat])
                    )
                )
                if batch_index == 0 and repeat in {1, max(1, max_repeats // 2)}:
                    reference, _ = _forward_all_cells(
                        context, inputs, num_repeats=repeat
                    )
                    if (
                        not torch.allclose(
                            logits, reference, rtol=1e-3, atol=1e-3,
                            equal_nan=True,
                        )
                        or not torch.equal(
                            logits.argmax(-1), reference.argmax(-1)
                        )
                    ):
                        raise AssertionError(
                            f"Intermediate readout differs at repeat {repeat}."
                        )

                finite = torch.isfinite(logits).all(dim=(1, 2))
                prediction = logits.argmax(-1)
                target = rule30(target)
                target_match = prediction.eq(target) & finite[:, None]
                recurrence_target = rule30(previous_prediction)
                recurrence_match = (
                    prediction.eq(recurrence_target)
                    & finite[:, None]
                    & previous_finite[:, None]
                )

                index = repeat - 1
                counts[0, index] += target_match.sum()
                counts[1, index] += target_match.all(dim=1).sum()
                counts[2, index] += recurrence_match.sum()
                counts[3, index] += recurrence_match.all(dim=1).sum()
                counts[4, index] += (~finite).sum()
                previous_prediction = prediction
                previous_finite = finite

    if rows_seen == 0:
        raise ValueError("Evaluation loader contained no rows.")
    correct_cells, correct_rows, recurrence_cells, recurrence_rows, bad_rows = (
        counts.cpu().tolist()
    )
    total_cells = rows_seen * num_cells
    return [
        {
            "repeat": index + 1,
            "cell_correct": correct_cells[index],
            "cell_total": total_cells,
            "cell_accuracy": correct_cells[index] / total_cells,
            "exact_rows": correct_rows[index],
            "row_total": rows_seen,
            "exact_sequence_accuracy": correct_rows[index] / rows_seen,
            "decoded_recurrence_cell_correct": recurrence_cells[index],
            "decoded_recurrence_cell_accuracy":
                recurrence_cells[index] / total_cells,
            "decoded_recurrence_exact_rows": recurrence_rows[index],
            "decoded_recurrence_exact_sequence_accuracy":
                recurrence_rows[index] / rows_seen,
            "nonfinite_logit_rows": bad_rows[index],
        }
        for index in range(max_repeats)
    ]

def main():
    args = parse_args()
    if args.max_repeats < 1 or args.samples < 1 or args.num_cells < 1:
        raise ValueError("Repeats, samples, and row length must be positive.")
    if len(set(args.test_seeds)) != len(args.test_seeds):
        raise ValueError("Test seeds must be distinct.")
    if len(set(args.bernoulli_ps)) != len(args.bernoulli_ps):
        raise ValueError("Bernoulli probabilities must be distinct.")
    if not all(0.0 <= p <= 1.0 for p in args.bernoulli_ps):
        raise ValueError("Bernoulli probabilities must lie in [0, 1].")

    source, source_path, checkpoint_dir = validate_source_run(args.source_run_dir)
    model_args, _ = resolve_source_args(source, args.device)
    if model_args.model not in SUPPORTED_MODELS or model_args.n_layer_end != 0:
        raise ValueError(
            f"This sweep requires one of {SUPPORTED_MODELS} with zero end blocks."
        )
    if model_args.ca_data_mode != "materialized":
        raise ValueError("This paired sweep requires materialized sampling.")
    if forward_policy_metadata(source)["repeat_cache_window"] is not None:
        raise ValueError("This sweep requires the full forward policy.")
    if args.num_cells not in model_args.ca_eval_num_cells:
        raise ValueError("Row length must be an original evaluation length.")
    if any(steps != repeats for steps, repeats in model_args.ca_train_pairs):
        raise ValueError("Training pairs must align CA steps with repeats.")

    metadata, checkpoint_path = _read_checkpoint_metadata(
        checkpoint_dir, args.checkpoint_type
    )
    model = models.make_model_from_args(model_args)
    load_checkpoint_weights(model, checkpoint_path, metadata, model_args.device)
    context = CAForwardContext(model)
    training_max_repeats = max(
        (repeats for _, repeats in model_args.ca_train_pairs),
        default=model_args.n_repeat,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "started_at": utc_now(),
        "source_run_id": source["run_id"],
        "source_manifest": str(source_path),
        "checkpoint_type": args.checkpoint_type,
        "checkpoint_step": metadata["step"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_bernoulli_p": float(model_args.ca_bernoulli_p),
        "test_seeds": args.test_seeds,
        "test_bernoulli_ps": args.bernoulli_ps,
        "samples_per_condition": args.samples,
        "num_cells": args.num_cells,
        "max_repeats": args.max_repeats,
        "dataset_mode": "materialized",
        "dataset_seed_rule": "test_seed + num_cells",
        "target_rule": "Rule30 applied repeat times to the initial row",
        "recurrence_rule": "prediction at repeat r versus Rule30(prediction at r-1); prediction at repeat 0 is the input",
        "nonfinite_rule": "A row with any non-finite logit is counted incorrect for all cells and for recurrence.",
        "completed_conditions": 0,
    }
    write_json(output_dir / "sweep_manifest.json", report)
    summaries = []
    try:
        for seed in args.test_seeds:
            for probability in args.bernoulli_ps:
                condition_args = copy.copy(model_args)
                condition_args.ca_eval_num_cells = [args.num_cells]
                condition_args.ca_bernoulli_p = probability
                loader = make_ca_fixed_loaders(
                    condition_args, base_seed=seed, num_samples=args.samples
                )[args.num_cells]
                rows = evaluate_condition(
                    context, loader, model_args.device, args.max_repeats,
                    _autocast_context(model_args),
                )
                inputs = loader.dataset.inputs
                input_ones = int(inputs.sum().item())
                row_ones = inputs.sum(dim=1)
                extrapolation = [
                    row["exact_sequence_accuracy"] for row in rows
                    if row["repeat"] > training_max_repeats
                ]
                summary = {
                    "test_seed": seed,
                    "actual_dataset_seed": seed + args.num_cells,
                    "test_bernoulli_p": probability,
                    "observed_input_one_rate":
                        input_ones / (args.samples * args.num_cells),
                    "all_zero_input_rows": int(row_ones.eq(0).sum().item()),
                    "all_one_input_rows":
                        int(row_ones.eq(args.num_cells).sum().item()),
                    "cell_accuracy_at_max": rows[-1]["cell_accuracy"],
                    "exact_sequence_accuracy_at_max":
                        rows[-1]["exact_sequence_accuracy"],
                    "decoded_recurrence_cell_accuracy_at_max":
                        rows[-1]["decoded_recurrence_cell_accuracy"],
                    "decoded_recurrence_exact_sequence_accuracy_at_max":
                        rows[-1]["decoded_recurrence_exact_sequence_accuracy"],
                    "worst_extrapolation_exact_sequence_accuracy":
                        min(extrapolation) if extrapolation else None,
                    "nonfinite_logit_rows_at_max":
                        rows[-1]["nonfinite_logit_rows"],
                }
                condition_id = len(summaries) + 1
                filename = f"condition_{condition_id:03d}.json"
                write_json(output_dir / filename, {
                    "summary": summary, "per_depth": rows,
                })
                summaries.append({"file": filename, **summary})
                write_json(output_dir / "condition_summary.json", summaries)
                report["completed_conditions"] = len(summaries)
                write_json(output_dir / "sweep_manifest.json", report)
                print(
                    f"seed={seed} p={probability:g} "
                    f"exact@{args.max_repeats}="
                    f"{summary['exact_sequence_accuracy_at_max']:.6f} "
                    f"recurrence@{args.max_repeats}="
                    f"{summary['decoded_recurrence_cell_accuracy_at_max']:.6f}",
                    flush=True,
                )
        report["status"] = "completed"
        report["completed_at"] = utc_now()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        write_json(output_dir / "sweep_manifest.json", report)


if __name__ == "__main__":
    main()