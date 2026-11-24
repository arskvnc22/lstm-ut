"""Paired finite perturbations of the looped block's recurrent state on Rule 30.

The Lyapunov tool (ca_lyapunov.py) only pushes infinitesimal tangents through
the Jacobian; the rollout itself is never changed. This tool changes it. The
unmodified model is run from the embedding to repeat k, the state s_k is
copied and modified ONCE, and the clean and the perturbed copy are then both
run freely through the same block up to the maximum depth. The perturbation
is never re-applied, and there is no cache (lm_cache=none), so nothing is
masked or reset. Both branches are measured at every repeat from k onwards.

The blocks are pre-LayerNorm with an un-normalised residual x, so a change to
one token's x has three parts that LayerNorm treats differently:

  shift       a constant on all hidden features. Every LayerNorm removes it
              and the residual carries it forward unchanged: an exact
              symmetry that never reaches a bit.
  radial      along the token's own centred x. LayerNorm ignores it to first
              order on this repeat, but it changes the ratio of x to the next
              update, so it is NOT a symmetry.
  tangential  everything else. LayerNorm passes it on divided by the centred
              norm of x, so a fixed absolute size would be a different
              effective perturbation for every model, depth and token.

The perturbation kinds therefore are:

  rotate_x  the main intervention. Each token's centred x is rotated by an
            angle theta towards a random direction orthogonal to it and to
            the all-ones vector. The token's mean and centred norm are kept
            exactly, at any theta, so the perturbation has no shift and no
            radial part, and theta is the angle moved in the pre-gain output
            of every LayerNorm.
  shift_x   negative control. theta * r / sqrt(D) is added to every feature
            of a token (r = centred norm, D = width), with a random sign per
            token; the added vector has norm theta * r. It must change no
            logit, and its offset must be carried unchanged.
  radial_x  the centred x is multiplied by exp(+-theta). Reported separately.
  cell      LSTM-UT only. The token's cell is moved by theta * ||cell|| in a
            random direction. No LayerNorm acts on the cell, so nothing is
            projected out of it.

With --tokens single only one randomly chosen token per row is perturbed (a
new token for every direction draw); the other tokens start identical to the
clean branch.

Besides distances to the clean branch, every branch's decoded bits are saved
(except for shift_x, whose bits equal the clean branch's), together with its
self-transition accuracy: the fraction of bits at repeat t that equal Rule 30
applied once to the branch's OWN decoded row at repeat t - 1. That separates a
model that faithfully evolves a wrong row (near 1) from one whose output no
longer follows from its previous output (near 0.5 for unrelated bits). It must
be read next to `ones_fraction` and `changed_fraction`: Rule 30 maps the
all-zeros row to itself, so a constant all-zeros output also scores 1.

These are random directions. They do not probe the worst case, and a
recovery here says nothing about what causes it.

Gate closure (LSTM-UT only, --closure-lengths). A second, separate
intervention that changes the block instead of the state: the model's output
gate is closed for a window of repeats and then released. With it closed
(models/lstm_ut_bidir.py, Block.forward(close_hidden_gate=True)) a repeat is
x <- x + attention + MLP and the cell update, without x <- x + hidden_gate *
tanh(cell). See rollout_gate_closure. When --closure-lengths is given the
perturbations above are not run, and the output is gate_closure.npz.

Usage:
    python -m cellular_automaton.ca_perturbation \
        --source-run-dir iridis/ca-rule30/runs/run_113__ca_lstm_r12_zero \
        --output-dir <new directory> --device cuda:0
    # gate closed for 1..15 repeats after each of repeats 6, 13, 20, 30:
    python -m cellular_automaton.ca_perturbation \
        --source-run-dir iridis/ca-rule30/runs/run_115__ca_lstm_r12_zero \
        --output-dir <new directory> --device cuda:0 \
        --inject-depths 6 13 20 30 --closure-lengths 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .ca_checkpoint_eval import AVAILABLE_CHECKPOINT_TYPES
from .ca_gen import apply_rule30
from .ca_lyapunov import BlockStep, _math_attention, jvp_columns, load_model, test_rows
from .ca_reporting import utc_now


X_KINDS = ("rotate_x", "shift_x", "radial_x")
CELL_KIND = "cell"
ALL_KINDS = X_KINDS + (CELL_KIND,)
# Decoded bits are not stored for this kind: they equal the clean branch's.
KINDS_WITHOUT_SAVED_PREDICTIONS = ("shift_x",)
# Value of the uint8 prediction arrays at repeats before the injection depth.
NOT_RUN = 255
DEFAULT_THETAS = (1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0)
DEFAULT_INJECT_DEPTHS = (6, 13, 20, 30)
SIGNS = (1.0, -1.0)
# Relative size below which a distance is treated as rounding error in the
# summary. The smallest default theta (1e-4) is well above it.
ROUNDING_FLOOR = 1e-6
# Gate closure: what happens to the cell while the output gate is closed.
CELL_MODES = ("update", "frozen")
# Entry of `closure_lengths` for the branch whose gate is never reopened.
NEVER_REOPENED = -1
# The closure summary reports the branch's own-output measurements over this
# many final repeats (51-60 at the default depth, as in docs/perturb-v2.md).
FINAL_WINDOW = 10


def centre(x):
    """Remove each token's mean over its hidden features (the shift part)."""
    return x - x.mean(dim=-1, keepdim=True)


def rotation_direction(x, gaussian):
    """Unit direction per token, orthogonal to its centred x and to all-ones."""
    centred = centre(x)
    unit = centred / centred.norm(dim=-1, keepdim=True)
    direction = centre(gaussian)
    direction = direction - (direction * unit).sum(dim=-1, keepdim=True) * unit
    return direction / direction.norm(dim=-1, keepdim=True)


def rotate_tokens(x, gaussian, theta):
    """Rotate each token's centred x by `theta`; mean and centred norm are kept."""
    centred = centre(x)
    radius = centred.norm(dim=-1, keepdim=True)
    direction = rotation_direction(x, gaussian)
    # cos(theta) - 1 written as -2 sin^2(theta / 2): exact at theta = 0 and
    # accurate for very small angles.
    cosine_minus_one = -2.0 * math.sin(theta / 2) ** 2
    return x + cosine_minus_one * centred + math.sin(theta) * radius * direction


def shift_tokens(x, token_sign, theta):
    """Add a per-token constant whose vector norm is theta * centred norm."""
    radius = centre(x).norm(dim=-1, keepdim=True)
    return x + token_sign * theta * radius / math.sqrt(x.shape[-1])


def scale_tokens(x, theta):
    """Multiply each token's centred x by exp(theta); the mean is kept."""
    return x + math.expm1(theta) * centre(x)


def perturb_cell(cell, gaussian, theta):
    """Move each token's cell by theta * ||cell|| along a random unit direction."""
    direction = gaussian / gaussian.norm(dim=-1, keepdim=True)
    return cell + theta * cell.norm(dim=-1, keepdim=True) * direction


def new_draw(batch, cells, width, generator):
    """The random choices shared by every theta and sign of one direction draw."""
    return {
        "x_gaussian": torch.randn(batch, cells, width, generator=generator, dtype=torch.float64),
        "cell_gaussian": torch.randn(batch, cells, width, generator=generator, dtype=torch.float64),
        "token_sign": torch.randint(0, 2, (batch, cells, 1), generator=generator).double() * 2 - 1,
        "token_index": torch.randint(0, cells, (batch,), generator=generator),
    }


def perturb_state(state, kind, theta, sign, draw, token_mask):
    """A perturbed copy of `state` ([B, S, T, D]); only masked tokens change.

    `sign` is +1 or -1 and flips the perturbation along the same direction.
    """
    perturbed = state.clone()
    signed_theta = sign * theta
    if kind == "rotate_x":
        changed, component = rotate_tokens(state[:, 0], draw["x_gaussian"], signed_theta), 0
    elif kind == "shift_x":
        changed, component = shift_tokens(state[:, 0], draw["token_sign"], signed_theta), 0
    elif kind == "radial_x":
        changed, component = scale_tokens(state[:, 0], signed_theta), 0
    elif kind == CELL_KIND:
        changed, component = perturb_cell(state[:, 1], draw["cell_gaussian"], signed_theta), 1
    else:
        raise ValueError(f"unknown perturbation kind: {kind!r}")
    perturbed[:, component] = torch.where(token_mask, changed, state[:, component])
    return perturbed


def logits_of(step, state):
    """Readout of the recurrent x, as GPTBase.forward does after the repeats."""
    return step.model.lm_head(step.model.transformer.ln_f(state[:, 0]))


def target_margin_min(logits, targets):
    """Smallest over cells of (target logit - best other logit); < 0 means a wrong bit."""
    target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    is_target = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, targets.unsqueeze(-1), True)
    best_other = logits.masked_fill(is_target, float("-inf")).max(dim=-1).values
    return (target_logit - best_other).min(dim=1).values


def own_output_metrics(predictions, previous_predictions):
    """Measurements of a branch's decoded bits against its own previous repeat.

    `previous_predictions` is None at the first measured repeat, where the
    two transition measurements are undefined and left out.
    """
    metrics = {"ones_fraction": predictions.double().mean(dim=1)}
    if previous_predictions is not None:
        expected = apply_rule30(previous_predictions, steps=1)
        metrics["self_transition_accuracy"] = (predictions == expected).double().mean(dim=1)
        metrics["changed_fraction"] = (predictions != previous_predictions).double().mean(dim=1)
    return metrics


def branch_metrics(step, clean_state, state, targets, token_mask, previous_predictions=None):
    """Per-row measurements of one perturbed branch, and its decoded bits.

    Distances are over the whole row and are against the clean branch.
    `x_centred_distance` ignores per-token shifts, matching the shift-quotient
    tangents of the Lyapunov tool; `x_raw_distance` does not. Returns
    (metrics, predictions); pass the predictions back in at the next repeat.
    """
    clean_x, x = clean_state[:, 0], state[:, 0]
    difference = x - clean_x
    centred_difference = centre(difference)
    centred_energy = centred_difference.pow(2).sum(dim=-1)  # [B, T]
    cosine = torch.nn.functional.cosine_similarity(centre(x), centre(clean_x), dim=-1)
    clean_logits = logits_of(step, clean_state)
    logits = logits_of(step, state)
    predictions = logits.argmax(dim=-1)
    metrics = {
        "x_centred_distance": centred_energy.sum(dim=1).sqrt(),
        "x_raw_distance": difference.pow(2).sum(dim=(1, 2)).sqrt(),
        "x_angle_mean": torch.acos(cosine.clamp(-1.0, 1.0)).mean(dim=1),
        "logit_distance": (logits - clean_logits).abs().amax(dim=(1, 2)),
        "bit_disagreement": (predictions != clean_logits.argmax(dim=-1)).double().mean(dim=1),
        "cell_accuracy": (predictions == targets).double().mean(dim=1),
        "margin_min": target_margin_min(logits, targets),
        # The branch's own centred norm: tells a norm blow-up from a change
        # of direction.
        "x_centred_norm": centre(x).pow(2).sum(dim=(1, 2)).sqrt(),
    }
    metrics.update(own_output_metrics(predictions, previous_predictions))
    if not bool(token_mask.all()):
        # Share of the centred difference that sits outside the perturbed token.
        off_token = (centred_energy * (~token_mask.squeeze(-1))).sum(dim=1)
        metrics["off_token_fraction"] = off_token / centred_energy.sum(dim=1)
    if step.has_cell:
        clean_cell, cell = clean_state[:, 1], state[:, 1]
        metrics["cell_distance"] = (cell - clean_cell).pow(2).sum(dim=(1, 2)).sqrt()
        # tanh(cell) is what the block exposes to x.
        metrics["cell_tanh_distance"] = (
            (torch.tanh(cell) - torch.tanh(clean_cell)).pow(2).sum(dim=(1, 2)).sqrt()
        )
        metrics["cell_norm"] = cell.pow(2).sum(dim=(1, 2)).sqrt()
    return metrics, predictions


def clean_rollout(step, rows, max_depth):
    """Unmodified states s_0..s_max and Rule-30 targets (index 0 is the input row).

    For but_input_inject, s_0 has x = 0 (see ca_lyapunov.INJECT_MODELS), so
    clean measurements at repeat 0, and the clean self-transition and changed
    fraction at repeat 1, describe a zero state rather than the input row.
    fprm_but starts from x = e, the input row, so this does not apply to it.
    """
    with torch.no_grad():
        state = step.initial_state(rows)
        states = [state]
        for _ in range(max_depth):
            state = step(state)
            states.append(state)
    targets = [rows] + [apply_rule30(rows, steps=depth) for depth in range(1, max_depth + 1)]
    return states, targets


def clean_branch_arrays(step, clean_states, targets):
    """`clean__<metric>` arrays [depth, row] and `clean__predictions` [depth, row, cell]."""
    max_depth = len(clean_states) - 1
    batch, device = clean_states[0].shape[0], clean_states[0].device
    arrays = {}
    nan_row = torch.full((batch,), float("nan"), dtype=torch.float64, device=device)
    clean = {"cell_accuracy": [], "margin_min": [], "x_centred_norm": [], "ones_fraction": [],
             "self_transition_accuracy": [], "changed_fraction": []}
    if step.has_cell:
        clean["cell_norm"] = []
    clean_predictions = []
    for depth in range(max_depth + 1):
        with torch.no_grad():
            logits = logits_of(step, clean_states[depth])
        predictions = logits.argmax(dim=-1)
        clean["cell_accuracy"].append((predictions == targets[depth]).double().mean(dim=1))
        clean["margin_min"].append(target_margin_min(logits, targets[depth]))
        clean["x_centred_norm"].append(centre(clean_states[depth][:, 0]).pow(2).sum(dim=(1, 2)).sqrt())
        if step.has_cell:
            clean["cell_norm"].append(clean_states[depth][:, 1].pow(2).sum(dim=(1, 2)).sqrt())
        own = own_output_metrics(predictions, clean_predictions[-1] if clean_predictions else None)
        for name in ("ones_fraction", "self_transition_accuracy", "changed_fraction"):
            clean[name].append(own.get(name, nan_row))  # undefined at depth 0
        clean_predictions.append(predictions)
    for name, values in clean.items():
        arrays[f"clean__{name}"] = torch.stack(values).cpu().numpy()
    arrays["clean__predictions"] = torch.stack(clean_predictions).cpu().numpy().astype(np.uint8)
    return arrays


def linear_centred_norms(step, clean_states, tangent, inject_depth, max_depth):
    """Centred x norm of a tangent pushed through the clean Jacobians.

    `tangent` is [B, S, T, D], placed at s_k. Entry t of the result is the
    first-order prediction of x_centred_distance / theta at repeat t.
    """
    batch = tangent.shape[0]
    norms = {inject_depth: centre(tangent[:, 0]).pow(2).sum(dim=(1, 2)).sqrt()}
    for depth in range(inject_depth + 1, max_depth + 1):
        columns = tangent.reshape(batch, -1, 1)
        tangent = jvp_columns(step, clean_states[depth - 1], columns)[:, :, 0].reshape(tangent.shape)
        norms[depth] = centre(tangent[:, 0]).pow(2).sum(dim=(1, 2)).sqrt()
    return norms


def rollout_perturbations(
    step,
    rows,
    *,
    max_depth,
    inject_depths,
    thetas,
    num_directions,
    direction_seed,
    tokens,
    kinds=None,
):
    """Clean and perturbed rollouts; returns numpy arrays.

    Branch arrays are named `<kind>__<metric>` and indexed
    [inject depth, theta, direction, sign, depth, row]; the depth axis runs
    0..max_depth and is NaN before the injection depth. `<kind>__predictions`
    has one more axis (cell), is uint8 and holds NOT_RUN before injection.

    `kinds` restricts the perturbation kinds (default: all). The cell kind is
    skipped for models without a cell. The random draws do not depend on which
    kinds run.
    """
    requested = ALL_KINDS if kinds is None else tuple(kinds)
    unknown = [kind for kind in requested if kind not in ALL_KINDS]
    if unknown or not requested:
        raise ValueError(f"kinds must be a non-empty subset of {ALL_KINDS}, got {requested}")
    if tokens not in ("all", "single"):
        raise ValueError(f"tokens must be 'all' or 'single', got {tokens!r}")
    if any(not 1 <= depth < max_depth for depth in inject_depths):
        raise ValueError("every injection depth must satisfy 1 <= k < max_depth")
    generator = torch.Generator().manual_seed(int(direction_seed))
    rows = rows.to(next(step.model.parameters()).device)
    clean_states, targets = clean_rollout(step, rows, max_depth)
    batch, _, cells, width = clean_states[0].shape
    dtype, device = clean_states[0].dtype, clean_states[0].device
    kinds = tuple(kind for kind in requested if kind != CELL_KIND or step.has_cell)

    arrays = clean_branch_arrays(step, clean_states, targets)

    branch_shape = (len(inject_depths), len(thetas), num_directions, len(SIGNS), max_depth + 1, batch)
    linear = np.full((len(inject_depths), num_directions, max_depth + 1, batch), np.nan, dtype=np.float32)
    token_index = np.full((len(inject_depths), num_directions, batch), -1, dtype=np.int64)

    for depth_index, inject_depth in enumerate(inject_depths):
        base = clean_states[inject_depth]
        for direction_index in range(num_directions):
            draw = new_draw(batch, cells, width, generator)
            draw = {name: value.to(device) for name, value in draw.items()}
            for name in ("x_gaussian", "cell_gaussian", "token_sign"):
                draw[name] = draw[name].to(dtype)
            if tokens == "single":
                token_mask = torch.zeros(batch, cells, 1, dtype=torch.bool, device=device)
                token_mask[torch.arange(batch, device=device), draw["token_index"]] = True
                token_index[depth_index, direction_index] = draw["token_index"].cpu().numpy()
            else:
                token_mask = torch.ones(batch, cells, 1, dtype=torch.bool, device=device)

            # First-order prediction for rotate_x: d(rotated x)/d(theta) at
            # theta = 0 is (centred norm) * direction, on the masked tokens.
            if "rotate_x" in kinds:
                # Dynamic components only (jvp_columns leaves out the
                # constant injected input of but_input_inject and fprm_but).
                tangent = torch.zeros_like(base[:, :step.num_components])
                radius = centre(base[:, 0]).norm(dim=-1, keepdim=True)
                tangent[:, 0] = radius * rotation_direction(base[:, 0], draw["x_gaussian"]) * token_mask
                for depth, norms in linear_centred_norms(
                    step, clean_states, tangent, inject_depth, max_depth
                ).items():
                    linear[depth_index, direction_index, depth] = norms.cpu().numpy()

            # Every theta and sign of one kind runs as a single batch of
            # `copies` * B examples, ordered theta-major then sign. Examples
            # do not interact inside the block, so this equals separate runs.
            copies = len(thetas) * len(SIGNS)
            copied_mask = token_mask.repeat(copies, 1, 1)
            for kind in kinds:
                with torch.no_grad():
                    state = torch.cat([
                        perturb_state(base, kind, theta, sign, draw, token_mask)
                        for theta in thetas
                        for sign in SIGNS
                    ])
                    # The transition measurements start at inject_depth + 1:
                    # they compare a branch with its own previous repeat.
                    predictions = None
                    for depth in range(inject_depth, max_depth + 1):
                        if depth > inject_depth:
                            state = step(state)
                        metrics, predictions = branch_metrics(
                            step,
                            clean_states[depth].repeat(copies, 1, 1, 1),
                            state,
                            targets[depth].repeat(copies, 1),
                            copied_mask,
                            previous_predictions=predictions,
                        )
                        if kind not in KINDS_WITHOUT_SAVED_PREDICTIONS:
                            key = f"{kind}__predictions"
                            if key not in arrays:
                                arrays[key] = np.full(branch_shape + (cells,), NOT_RUN, dtype=np.uint8)
                            arrays[key][depth_index, :, direction_index, :, depth] = (
                                predictions.reshape(len(thetas), len(SIGNS), batch, cells).cpu().numpy()
                            )
                        for name, values in metrics.items():
                            key = f"{kind}__{name}"
                            if key not in arrays:
                                arrays[key] = np.full(branch_shape, np.nan, dtype=np.float32)
                            arrays[key][depth_index, :, direction_index, :, depth] = (
                                values.reshape(len(thetas), len(SIGNS), batch).cpu().numpy()
                            )

    if "rotate_x" in kinds:
        arrays["linear__rotate_x_centred_norm"] = linear
    arrays["token_index"] = token_index
    arrays["inject_depths"] = np.asarray(inject_depths, dtype=np.int64)
    arrays["thetas"] = np.asarray(thetas, dtype=np.float64)
    return arrays


def rollout_gate_closure(step, rows, *, max_depth, start_depths, closure_lengths, cell_modes=CELL_MODES):
    """LSTM-UT rollouts with the output gate closed for a window of repeats.

    For a start depth s and a length L:
      repeats 1 .. s        unmodified, identical to the clean branch;
      repeats s+1 .. s+L    produced with the gate closed: x gets attention
                            and the MLP only, the cell is still written;
      repeats s+L+1 .. max  unmodified again. Nothing stops early.
    A branch whose gate stays closed from s+1 to max_depth is always added as
    the last length entry (NEVER_REOPENED). Length 0 is the clean branch.

    While the gate is closed x does not depend on the cell at all, so the two
    cell modes share the same x during the closure and differ only after it:
      update  the cell is whatever the block wrote during the closure;
      frozen  the cell is put back to its repeat-s value after every closed
              step, so the first open step reads the pre-closure cell.

    Returns numpy arrays. `closure__<metric>` is indexed [cell mode, start,
    length, depth, row] and is NaN before the start depth;
    `closure__predictions` has one more axis (cell), is uint8 and holds
    NOT_RUN there. `closure__gate_closed` [start, length, depth] marks the
    repeats produced with the gate closed. Distances are against the clean
    branch. The transition measurements start at s + 1.
    """
    if not step.has_cell:
        raise ValueError("gate closure needs a model with a cell (an LSTM-UT variant).")
    unknown = [mode for mode in cell_modes if mode not in CELL_MODES]
    if unknown or not cell_modes:
        raise ValueError(f"cell_modes must be a non-empty subset of {CELL_MODES}, got {tuple(cell_modes)}")
    if not closure_lengths or any(length < 0 for length in closure_lengths):
        raise ValueError("closure_lengths must be non-empty and non-negative")
    if any(start < 1 or start + max(closure_lengths) >= max_depth for start in start_depths):
        raise ValueError("every start depth s must satisfy 1 <= s and s + longest closure < max_depth")
    rows = rows.to(next(step.model.parameters()).device)
    clean_states, targets = clean_rollout(step, rows, max_depth)
    batch, _, cells, _ = clean_states[0].shape
    device = clean_states[0].device
    arrays = clean_branch_arrays(step, clean_states, targets)

    lengths = list(closure_lengths) + [NEVER_REOPENED]
    branch_shape = (len(cell_modes), len(start_depths), len(lengths), max_depth + 1, batch)
    gate_closed = np.zeros((len(start_depths), len(lengths), max_depth + 1), dtype=bool)
    all_tokens = torch.ones(batch, cells, 1, dtype=torch.bool, device=device)
    for mode_index, cell_mode in enumerate(cell_modes):
        for start_index, start in enumerate(start_depths):
            for length_index, length in enumerate(lengths):
                last_closed = max_depth if length == NEVER_REOPENED else start + length
                gate_closed[start_index, length_index, start + 1:last_closed + 1] = True
                with torch.no_grad():
                    state = clean_states[start]
                    predictions = None
                    for depth in range(start, max_depth + 1):
                        if depth > start:
                            closed = depth <= last_closed
                            state = step(state, close_hidden_gate=closed)
                            if closed and cell_mode == "frozen":
                                state = torch.stack((state[:, 0], clean_states[start][:, 1]), dim=1)
                        metrics, predictions = branch_metrics(
                            step, clean_states[depth], state, targets[depth], all_tokens,
                            previous_predictions=predictions,
                        )
                        index = (mode_index, start_index, length_index, depth)
                        if "closure__predictions" not in arrays:
                            arrays["closure__predictions"] = np.full(
                                branch_shape + (cells,), NOT_RUN, dtype=np.uint8
                            )
                        arrays["closure__predictions"][index] = predictions.cpu().numpy()
                        for name, values in metrics.items():
                            key = f"closure__{name}"
                            if key not in arrays:
                                arrays[key] = np.full(branch_shape, np.nan, dtype=np.float32)
                            arrays[key][index] = values.cpu().numpy()

    arrays["closure__gate_closed"] = gate_closed
    arrays["closure_start_depths"] = np.asarray(start_depths, dtype=np.int64)
    arrays["closure_lengths"] = np.asarray(lengths, dtype=np.int64)
    arrays["closure_cell_modes"] = np.asarray(cell_modes)
    return arrays


def _log_growth_rate(distance, clean_norm, inject_depth, max_depth):
    """Mean per-repeat log growth of a distance from injection to the end.

    Branches whose distance at either end is zero up to rounding (below
    ROUNDING_FLOOR of the clean centred norm; e.g. the centred distance of a
    pure shift) are left out rather than reported as growth of noise.
    """
    start, end = distance[..., inject_depth, :], distance[..., max_depth, :]
    valid = (start > ROUNDING_FLOOR * clean_norm[inject_depth]) & (
        end > ROUNDING_FLOOR * clean_norm[max_depth]
    )
    if not valid.any():
        return None
    return float(np.mean(np.log(end[valid] / start[valid])) / (max_depth - inject_depth))


def first_wrong_depth(cell_accuracy, inject_depth, max_depth):
    """First repeat at or after injection where a row is not exactly right.

    `cell_accuracy` is [..., depth, row]. Rows that are never wrong get
    max_depth + 1, so means of this are censored at the end of the rollout.
    """
    wrong = cell_accuracy[..., inject_depth:max_depth + 1, :] < 1
    first = inject_depth + wrong.argmax(axis=-2)
    return np.where(wrong.any(axis=-2), first, max_depth + 1)


def _clean_summary(arrays):
    """Per-depth means over rows of the clean branch."""
    return {
        "exact_accuracy_per_depth": (arrays["clean__cell_accuracy"] == 1).mean(axis=1).tolist(),
        "cell_accuracy_per_depth": arrays["clean__cell_accuracy"].mean(axis=1).tolist(),
        # Index 0 (the embedded input row) has no previous repeat.
        "ones_fraction_per_depth": arrays["clean__ones_fraction"].mean(axis=1).tolist(),
        "self_transition_accuracy_per_depth": [None] + arrays[
            "clean__self_transition_accuracy"][1:].mean(axis=1).tolist(),
        "changed_fraction_per_depth": [None] + arrays[
            "clean__changed_fraction"][1:].mean(axis=1).tolist(),
    }


def summarise(arrays, max_depth):
    """Means over directions, signs and rows for every kind, injection depth and theta."""
    inject_depths = [int(depth) for depth in arrays["inject_depths"]]
    thetas = [float(theta) for theta in arrays["thetas"]]
    kinds = sorted({name.split("__")[0] for name in arrays if name.endswith("__bit_disagreement")})
    clean_norm = arrays["clean__x_centred_norm"]  # [depth, row]
    summary = {
        "clean": _clean_summary(arrays),
        "kinds": {},
        "linear_check": {},
    }
    for kind in kinds:
        summary["kinds"][kind] = {}
        for depth_index, inject_depth in enumerate(inject_depths):
            per_theta = {}
            for theta_index, theta in enumerate(thetas):
                # [direction, sign, depth, row]
                def branch(metric):
                    return arrays[f"{kind}__{metric}"][depth_index, theta_index]

                disagreement = branch("bit_disagreement")
                after = slice(inject_depth, max_depth + 1)
                later = slice(inject_depth + 2, max_depth + 1)
                first_wrong = first_wrong_depth(branch("cell_accuracy"), inject_depth, max_depth)
                clean_first_wrong = first_wrong_depth(
                    arrays["clean__cell_accuracy"], inject_depth, max_depth
                )  # [row], broadcast over direction and sign
                per_theta[f"{theta:g}"] = {
                    "bit_disagreement_at_injection": float(disagreement[..., inject_depth, :].mean()),
                    "bit_disagreement_at_end": float(disagreement[..., max_depth, :].mean()),
                    # Decoded bits equal the clean branch's at the last repeat.
                    "agrees_with_clean_at_end_fraction": float(
                        (disagreement[..., max_depth, :] == 0).mean()
                    ),
                    "exact_accuracy_at_end": float((branch("cell_accuracy")[..., max_depth, :] == 1).mean()),
                    "centred_log_growth_rate": _log_growth_rate(
                        branch("x_centred_distance"), clean_norm, inject_depth, max_depth
                    ),
                    "raw_log_growth_rate": _log_growth_rate(
                        branch("x_raw_distance"), clean_norm, inject_depth, max_depth
                    ),
                    "max_logit_distance": float(branch("logit_distance")[..., after, :].max()),
                    # Is the branch still applying Rule 30 to its own decoded
                    # row? Read with the two fractions below: a constant
                    # all-zeros output is also "self-consistent".
                    "self_transition_accuracy_first_step": float(
                        branch("self_transition_accuracy")[..., inject_depth + 1, :].mean()
                    ),
                    "self_transition_accuracy_later": float(
                        branch("self_transition_accuracy")[..., later, :].mean()
                    ) if inject_depth + 2 <= max_depth else None,
                    "ones_fraction_after_injection": float(
                        branch("ones_fraction")[..., inject_depth + 1:, :].mean()
                    ),
                    "changed_fraction_after_injection": float(
                        branch("changed_fraction")[..., inject_depth + 1:, :].mean()
                    ),
                    # Does the perturbation bring failure against Rule 30
                    # forward? Both means are censored at max_depth + 1.
                    "first_wrong_depth_mean": float(first_wrong.mean()),
                    "clean_first_wrong_depth_mean": float(clean_first_wrong.mean()),
                    "wrong_earlier_than_clean_fraction": float((first_wrong < clean_first_wrong).mean()),
                    "wrong_later_than_clean_fraction": float((first_wrong > clean_first_wrong).mean()),
                }
            summary["kinds"][kind][f"inject_{inject_depth}"] = per_theta

    # Smallest theta against the Jacobian prediction for the same direction:
    # |log(finite difference / theta) - log(linear)|, over directions, signs,
    # depths after injection and rows.
    smallest = int(np.argmin(arrays["thetas"]))
    for depth_index, inject_depth in enumerate(inject_depths):
        if "rotate_x" not in kinds:
            break
        finite = arrays["rotate_x__x_centred_distance"][depth_index, smallest] / thetas[smallest]
        predicted = arrays["linear__rotate_x_centred_norm"][depth_index][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            error = np.abs(np.log(finite) - np.log(predicted))[..., inject_depth:, :]
        error = error[np.isfinite(error)]
        summary["linear_check"][f"inject_{inject_depth}"] = {
            "theta": thetas[smallest],
            "median_abs_log_error": float(np.median(error)),
            "max_abs_log_error": float(error.max()),
        }
    return summary


def summarise_gate_closure(arrays, max_depth):
    """Means over rows for every cell mode, start depth and closure length."""
    starts = [int(depth) for depth in arrays["closure_start_depths"]]
    lengths = [int(length) for length in arrays["closure_lengths"]]
    clean_norm = arrays["clean__x_centred_norm"]  # [depth, row]
    summary = {"clean": _clean_summary(arrays), "closure": {}}
    for mode_index, cell_mode in enumerate(str(mode) for mode in arrays["closure_cell_modes"]):
        summary["closure"][cell_mode] = {}
        for start_index, start in enumerate(starts):
            per_length = {}
            clean_first_wrong = first_wrong_depth(arrays["clean__cell_accuracy"], start, max_depth)
            for length_index, length in enumerate(lengths):
                # [depth, row]
                def branch(metric):
                    return arrays[f"closure__{metric}"][mode_index, start_index, length_index]

                never_reopened = length == NEVER_REOPENED
                last_closed = max_depth if never_reopened else start + length
                closed = slice(start + 1, last_closed + 1)
                final = slice(max(start + 1, max_depth - FINAL_WINDOW + 1), max_depth + 1)
                accuracy = branch("cell_accuracy")
                first_wrong = first_wrong_depth(accuracy, start, max_depth)
                relative_distance = branch("x_centred_distance") / clean_norm
                entry = {
                    "last_closed_depth": last_closed,
                    # How well x carries the computation on its own.
                    "exact_accuracy_at_last_closed_depth": float((accuracy[last_closed] == 1).mean()),
                    "cell_accuracy_at_last_closed_depth": float(accuracy[last_closed].mean()),
                    "relative_x_distance_at_last_closed_depth": float(relative_distance[last_closed].mean()),
                    "exact_accuracy_at_end": float((accuracy[max_depth] == 1).mean()),
                    "cell_accuracy_at_end": float(accuracy[max_depth].mean()),
                    "relative_x_distance_at_end": float(relative_distance[max_depth].mean()),
                    # Is the branch applying Rule 30 to its own decoded row at
                    # the end? Read all three together: a constant all-zeros
                    # output also has self-transition accuracy 1.
                    "final_window": [final.start, max_depth],
                    "self_transition_accuracy_final": float(branch("self_transition_accuracy")[final].mean()),
                    "ones_fraction_final": float(branch("ones_fraction")[final].mean()),
                    "changed_fraction_final": float(branch("changed_fraction")[final].mean()),
                    # Both means are censored at max_depth + 1.
                    "first_wrong_depth_mean": float(first_wrong.mean()),
                    "clean_first_wrong_depth_mean": float(clean_first_wrong.mean()),
                }
                if length > 0 or never_reopened:
                    entry["self_transition_accuracy_during_closure"] = float(
                        branch("self_transition_accuracy")[closed].mean()
                    )
                per_length["never_reopened" if never_reopened else f"length_{length}"] = entry
            summary["closure"][cell_mode][f"start_{start}"] = per_length
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-type", default="best_extrapolation_unconstrained",
                        choices=AVAILABLE_CHECKPOINT_TYPES)
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Override manifest.provenance.checkpoint_dir (e.g. a local copy).")
    parser.add_argument("--device", default="cpu")
    # float64 by default: a 1e-4 rad rotation of a state with a large norm is
    # close to float32 rounding.
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--num-rows", type=int, default=64)
    parser.add_argument("--num-cells", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=60)
    parser.add_argument("--inject-depths", type=int, nargs="+", default=list(DEFAULT_INJECT_DEPTHS))
    parser.add_argument("--thetas", type=float, nargs="+", default=list(DEFAULT_THETAS))
    parser.add_argument("--num-directions", type=int, default=4)
    parser.add_argument("--direction-seed", type=int, default=0)
    parser.add_argument("--tokens", choices=("all", "single"), default="all",
                        help="Perturb every token, or one random token per row and direction.")
    parser.add_argument("--kinds", nargs="+", choices=ALL_KINDS, default=list(ALL_KINDS),
                        help="Perturbation kinds to run; 'cell' is skipped for models without a cell.")
    parser.add_argument("--closure-lengths", type=int, nargs="+", default=None,
                        help="LSTM-UT only. Run the gate-closure rollouts instead of the perturbations: "
                             "close the output gate for this many repeats after each --inject-depths "
                             "entry, then reopen it. A never-reopened branch is always added.")
    parser.add_argument("--closure-cell-modes", nargs="+", choices=CELL_MODES, default=list(CELL_MODES),
                        help="Cell during the closure: 'update' as the block writes it, or 'frozen' at "
                             "its value from the start depth.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    dtype = getattr(torch, args.dtype)
    model, model_args, metadata, checkpoint_path = load_model(
        args.source_run_dir, args.checkpoint_type, args.checkpoint_dir, args.device, dtype
    )
    step = BlockStep(model)
    rows = test_rows(model_args, args.num_rows, args.num_cells)
    # Same attention arithmetic for the rollouts and the Jacobian products.
    with _math_attention():
        if args.closure_lengths:
            arrays = rollout_gate_closure(
                step,
                rows,
                max_depth=args.max_depth,
                start_depths=args.inject_depths,
                closure_lengths=args.closure_lengths,
                cell_modes=args.closure_cell_modes,
            )
        else:
            arrays = rollout_perturbations(
                step,
                rows,
                max_depth=args.max_depth,
                inject_depths=args.inject_depths,
                thetas=args.thetas,
                num_directions=args.num_directions,
                direction_seed=args.direction_seed,
                tokens=args.tokens,
                kinds=args.kinds,
            )
    if args.closure_lengths:
        summary, array_file = summarise_gate_closure(arrays, args.max_depth), "gate_closure.npz"
    else:
        summary, array_file = summarise(arrays, args.max_depth), "perturbation.npz"
    summary["provenance"] = {
        "created_at": utc_now(),
        "source_run_dir": str(args.source_run_dir.resolve()),
        "model": model_args.model,
        # Separates the no-forget ablation from LSTM-UT (same model name).
        # Only meaningful for cell models; other manifests carry the default.
        "lstm_forget_gate": getattr(model_args, "lstm_forget_gate", None) if step.has_cell else None,
        "checkpoint_type": args.checkpoint_type,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(metadata["step"]),
        "test_seed": int(model_args.ca_test_seed) + int(args.num_cells),
        "arguments": {key: (str(value) if isinstance(value, Path) else value)
                      for key, value in vars(args).items()},
        "torch_version": torch.__version__,
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / array_file, rows=rows.numpy(), **arrays)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.closure_lengths:
        print(f"gate closure: wrote {array_file} and summary.json to {args.output_dir}")
    else:
        print(json.dumps(summary["linear_check"], indent=2))


if __name__ == "__main__":
    main()
