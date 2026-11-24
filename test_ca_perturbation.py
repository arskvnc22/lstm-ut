"""Tests for the paired perturbation tool (cellular_automaton/ca_perturbation.py).

These check software correctness only, on tiny randomly initialised models:
that each perturbation is the one its docstring describes, that the clean
branch is the model's own repeat, that a per-token shift is invisible, and
that small rotations agree with the Jacobian. They say nothing about how a
trained model responds to perturbations.
"""

import math
import sys

import numpy as np
import pytest
import torch

from cellular_automaton import ca_lyapunov, ca_perturbation
from test_ca_lyapunov import MODEL_NAMES, _rows, _state_after, _tiny_model


def _draw(state, seed=0):
    batch, _, cells, width = state.shape
    return ca_perturbation.new_draw(batch, cells, width, torch.Generator().manual_seed(seed))


def _all_tokens(state):
    return torch.ones(state.shape[0], state.shape[2], 1, dtype=torch.bool)


def _rollout(model_name, **overrides):
    step = ca_lyapunov.BlockStep(_tiny_model(model_name))
    options = dict(max_depth=6, inject_depths=(2,), thetas=(1e-6, 0.1), num_directions=2,
                   direction_seed=0, tokens="all")
    options.update(overrides)
    return step, ca_perturbation.rollout_perturbations(step, _rows(3, 8), **options), options


@pytest.mark.parametrize("theta", (1e-3, 0.3, -1.0))
def test_rotation_keeps_mean_and_norm_and_moves_by_theta(theta):
    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 3 + 2
    gaussian = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(2), dtype=torch.float64)

    rotated = ca_perturbation.rotate_tokens(x, gaussian, theta)

    torch.testing.assert_close(rotated.mean(dim=-1), x.mean(dim=-1), rtol=0, atol=1e-12)
    centred, centred_rotated = ca_perturbation.centre(x), ca_perturbation.centre(rotated)
    torch.testing.assert_close(centred_rotated.norm(dim=-1), centred.norm(dim=-1), rtol=1e-12, atol=0)
    cosine = torch.nn.functional.cosine_similarity(centred, centred_rotated, dim=-1)
    torch.testing.assert_close(cosine, torch.full_like(cosine, math.cos(theta)), rtol=0, atol=1e-12)


def test_rotation_direction_is_orthogonal_to_centred_x_and_to_ones():
    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    gaussian = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(2), dtype=torch.float64)

    direction = ca_perturbation.rotation_direction(x, gaussian)

    zeros = torch.zeros(2, 5, dtype=torch.float64)
    torch.testing.assert_close(direction.sum(dim=-1), zeros, rtol=0, atol=1e-12)
    torch.testing.assert_close((direction * ca_perturbation.centre(x)).sum(dim=-1), zeros, rtol=0, atol=1e-12)
    torch.testing.assert_close(direction.norm(dim=-1), torch.ones(2, 5, dtype=torch.float64))


def test_shift_and_radial_change_only_their_own_part():
    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(1), dtype=torch.float64) + 1
    token_sign = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0], dtype=torch.float64).reshape(1, 5, 1)
    centred = ca_perturbation.centre(x)

    shifted = ca_perturbation.shift_tokens(x, token_sign, 0.2)
    scaled = ca_perturbation.scale_tokens(x, 0.2)

    # Shift: centred part untouched, added vector has norm theta * centred norm.
    torch.testing.assert_close(ca_perturbation.centre(shifted), centred, rtol=0, atol=1e-12)
    torch.testing.assert_close((shifted - x).norm(dim=-1), 0.2 * centred.norm(dim=-1))
    # Radial: mean untouched, centred part multiplied by exp(theta).
    torch.testing.assert_close(scaled.mean(dim=-1), x.mean(dim=-1), rtol=0, atol=1e-12)
    torch.testing.assert_close(ca_perturbation.centre(scaled), math.exp(0.2) * centred)


def test_cell_perturbation_leaves_x_and_moves_cell_by_theta_times_its_norm():
    step = ca_lyapunov.BlockStep(_tiny_model("lstm_ut_bidir"))
    state = _state_after(step, _rows(2, 8), depth=2)

    perturbed = ca_perturbation.perturb_state(state, "cell", 0.1, 1.0, _draw(state), _all_tokens(state))

    assert torch.equal(perturbed[:, 0], state[:, 0])
    torch.testing.assert_close(
        (perturbed[:, 1] - state[:, 1]).norm(dim=-1), 0.1 * state[:, 1].norm(dim=-1)
    )


def test_opposite_signs_are_mirror_perturbations_and_zero_theta_is_the_identity():
    step = ca_lyapunov.BlockStep(_tiny_model("lstm_ut_bidir"))
    state = _state_after(step, _rows(2, 8), depth=2)
    draw, mask = _draw(state), _all_tokens(state)

    for kind in ("rotate_x", "shift_x", "cell"):
        plus = ca_perturbation.perturb_state(state, kind, 1e-3, 1.0, draw, mask)
        minus = ca_perturbation.perturb_state(state, kind, 1e-3, -1.0, draw, mask)
        # Antisymmetric to first order (exactly, for the two additive kinds).
        torch.testing.assert_close(plus - state, -(minus - state), rtol=0, atol=1e-5)
    for kind in ("rotate_x", "shift_x", "radial_x", "cell"):
        unchanged = ca_perturbation.perturb_state(state, kind, 0.0, 1.0, draw, mask)
        assert torch.equal(unchanged, state)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_clean_rollout_is_the_models_own_repeats(model_name):
    model = _tiny_model(model_name)
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)

    states, targets = ca_perturbation.clean_rollout(step, rows, max_depth=4)

    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=4, return_repeat_states=True)
        for depth in range(5):
            torch.testing.assert_close(
                states[depth][:, 0], outputs["repeat_states"][depth], rtol=0, atol=1e-12
            )
        torch.testing.assert_close(
            ca_perturbation.logits_of(step, states[4]), outputs["logits"], rtol=0, atol=1e-12
        )
    assert torch.equal(targets[0], rows)
    assert torch.equal(targets[3], ca_perturbation.apply_rule30(rows, steps=3))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_zero_theta_branch_equals_the_clean_branch(model_name):
    _, arrays, options = _rollout(model_name, thetas=(0.0,))
    after = slice(options["inject_depths"][0], None)

    for name in ("x_centred_distance", "x_raw_distance", "logit_distance", "bit_disagreement"):
        assert np.all(arrays[f"rotate_x__{name}"][..., after, :] == 0)
    np.testing.assert_array_equal(
        arrays["rotate_x__cell_accuracy"][0, 0, 0, 0, after],
        arrays["clean__cell_accuracy"][after].astype(np.float32),
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_shift_control_changes_no_logit_and_is_carried_unchanged(model_name):
    _, arrays, options = _rollout(model_name, thetas=(0.5,))
    inject = options["inject_depths"][0]
    after = slice(inject, None)

    assert arrays["shift_x__logit_distance"][..., after, :].max() < 1e-10
    assert arrays["shift_x__x_centred_distance"][..., after, :].max() < 1e-10
    assert np.all(arrays["shift_x__bit_disagreement"][..., after, :] == 0)
    raw = arrays["shift_x__x_raw_distance"][..., after, :]
    assert raw.min() > 0
    np.testing.assert_allclose(raw, raw[..., :1, :].repeat(raw.shape[-2], axis=-2), rtol=1e-6)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_small_rotation_matches_the_jacobian_prediction(model_name):
    _, arrays, options = _rollout(model_name)
    summary = ca_perturbation.summarise(arrays, options["max_depth"])

    check = summary["linear_check"]["inject_2"]
    assert check["theta"] == 1e-6
    assert check["max_abs_log_error"] < 1e-3


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_single_token_mode_perturbs_exactly_the_recorded_token(model_name):
    step = ca_lyapunov.BlockStep(_tiny_model(model_name))
    state = _state_after(step, _rows(3, 8), depth=2)
    draw = _draw(state)
    mask = torch.zeros(3, 8, 1, dtype=torch.bool)
    mask[torch.arange(3), draw["token_index"]] = True

    perturbed = ca_perturbation.perturb_state(state, "rotate_x", 0.1, 1.0, draw, mask)

    changed = (perturbed != state).any(dim=-1)[:, 0]  # [B, T] on the x component
    assert torch.equal(changed, mask.squeeze(-1))

    _, arrays, options = _rollout(model_name, tokens="single")
    inject = options["inject_depths"][0]
    token_index = arrays["token_index"]
    assert token_index.shape == (1, 2, 3)
    assert token_index.min() >= 0 and token_index.max() < 8
    # At injection the whole difference sits on the perturbed token; attention
    # then spreads it to the others.
    off_token = arrays["rotate_x__off_token_fraction"]
    assert np.all(off_token[..., inject, :] == 0)
    assert np.all(off_token[..., inject + 1, :] > 0)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_rollout_and_summary_smoke(model_name):
    _, arrays, options = _rollout(model_name)
    summary = ca_perturbation.summarise(arrays, options["max_depth"])

    has_cell = model_name in ca_lyapunov.CELL_MODELS
    kinds = {"rotate_x", "shift_x", "radial_x"} | ({"cell"} if has_cell else set())
    assert set(summary["kinds"]) == kinds
    assert ("cell__cell_distance" in arrays) == has_cell
    assert "rotate_x__off_token_fraction" not in arrays  # only in single-token mode
    assert np.all(arrays["token_index"] == -1)

    distance = arrays["rotate_x__x_centred_distance"]
    # [inject depth, theta, direction, sign, depth, row]
    assert distance.shape == (1, 2, 2, 2, 7, 3)
    assert np.all(np.isnan(distance[..., :2, :]))
    assert np.all(np.isfinite(distance[..., 2:, :]))
    # At injection the centred distance of a rotation is 2 r sin(theta / 2)
    # per token, so it is the same for both signs and close to theta * r.
    np.testing.assert_allclose(distance[0, 1, :, 0, 2], distance[0, 1, :, 1, 2], rtol=1e-6)
    np.testing.assert_allclose(
        distance[0, 1, 0, 0, 2], 2 * math.sin(0.05) * arrays["clean__x_centred_norm"][2], rtol=1e-6
    )
    # The radial kind tells the two signs apart: exp(+theta) and exp(-theta)
    # move the centred x by different amounts. This pins the sign axis.
    radial = arrays["radial_x__x_centred_distance"][0, 1, 0, :, 2]
    clean_norm = arrays["clean__x_centred_norm"][2]
    np.testing.assert_allclose(radial[0], math.expm1(0.1) * clean_norm, rtol=1e-6)
    np.testing.assert_allclose(radial[1], -math.expm1(-0.1) * clean_norm, rtol=1e-6)

    entry = summary["kinds"]["rotate_x"]["inject_2"]["0.1"]
    assert 0.0 <= entry["bit_disagreement_at_end"] <= 1.0
    assert entry["centred_log_growth_rate"] is not None
    assert summary["kinds"]["shift_x"]["inject_2"]["0.1"]["centred_log_growth_rate"] is None
    assert len(summary["clean"]["exact_accuracy_per_depth"]) == 7


def test_rollout_rejects_bad_options():
    step = ca_lyapunov.BlockStep(_tiny_model("but_full_depth"))
    common = dict(max_depth=4, thetas=(0.1,), num_directions=1, direction_seed=0)
    with pytest.raises(ValueError, match="injection depth"):
        ca_perturbation.rollout_perturbations(step, _rows(2, 8), inject_depths=(4,), tokens="all", **common)
    with pytest.raises(ValueError, match="tokens"):
        ca_perturbation.rollout_perturbations(step, _rows(2, 8), inject_depths=(2,), tokens="some", **common)


# --- decoded bits, self-transition accuracy, kind selection ------------------


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_saved_predictions_reproduce_the_saved_bit_metrics(model_name):
    _, arrays, options = _rollout(model_name)
    inject, last = options["inject_depths"][0], options["max_depth"]
    rows = _rows(3, 8)
    clean = arrays["clean__predictions"]  # [depth, row, cell]
    assert clean.shape == (last + 1, 3, 8) and clean.dtype == np.uint8

    for kind in ("rotate_x", "radial_x"):
        predictions = arrays[f"{kind}__predictions"]  # [k, theta, direction, sign, depth, row, cell]
        assert predictions.shape == (1, 2, 2, 2, last + 1, 3, 8) and predictions.dtype == np.uint8
        # The sentinel fills exactly the repeats before injection.
        assert np.all(predictions[..., :inject, :, :] == ca_perturbation.NOT_RUN)
        assert set(np.unique(predictions[..., inject:, :, :])) <= {0, 1}
        for depth in range(inject, last + 1):
            target = ca_perturbation.apply_rule30(rows, steps=depth).numpy()
            bits = predictions[..., depth, :, :]
            np.testing.assert_array_equal(
                (bits == target).mean(axis=-1).astype(np.float32),
                arrays[f"{kind}__cell_accuracy"][..., depth, :],
            )
            np.testing.assert_array_equal(
                (bits != clean[depth]).mean(axis=-1).astype(np.float32),
                arrays[f"{kind}__bit_disagreement"][..., depth, :],
            )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_self_transition_metrics_match_a_recomputation_from_the_saved_bits(model_name):
    _, arrays, options = _rollout(model_name)
    inject, last = options["inject_depths"][0], options["max_depth"]

    def recompute(bits, previous):
        expected = ca_perturbation.apply_rule30(torch.from_numpy(previous.astype(np.int64)), steps=1).numpy()
        return ((bits == expected).mean(axis=-1).astype(np.float32),
                (bits != previous).mean(axis=-1).astype(np.float32),
                bits.mean(axis=-1).astype(np.float32))

    predictions = arrays["rotate_x__predictions"].reshape(-1, last + 1, 3, 8)  # branches flattened
    saved = {name: arrays[f"rotate_x__{name}"].reshape(-1, last + 1, 3)
             for name in ("self_transition_accuracy", "changed_fraction", "ones_fraction")}
    # Undefined at the injection repeat: there is no previous repeat of the branch.
    assert np.all(np.isnan(saved["self_transition_accuracy"][:, inject]))
    assert np.all(np.isnan(saved["changed_fraction"][:, inject]))
    assert np.all(np.isfinite(saved["ones_fraction"][:, inject]))
    for depth in range(inject + 1, last + 1):
        transition, changed, ones = recompute(predictions[:, depth], predictions[:, depth - 1])
        np.testing.assert_array_equal(saved["self_transition_accuracy"][:, depth], transition)
        np.testing.assert_array_equal(saved["changed_fraction"][:, depth], changed)
        np.testing.assert_array_equal(saved["ones_fraction"][:, depth], ones)

    clean = arrays["clean__predictions"]
    assert np.all(np.isnan(arrays["clean__self_transition_accuracy"][0]))
    for depth in range(1, last + 1):
        transition, changed, ones = recompute(clean[depth], clean[depth - 1])
        np.testing.assert_array_equal(arrays["clean__self_transition_accuracy"][depth].astype(np.float32), transition)
        np.testing.assert_array_equal(arrays["clean__changed_fraction"][depth].astype(np.float32), changed)
        np.testing.assert_array_equal(arrays["clean__ones_fraction"][depth].astype(np.float32), ones)


def test_self_transition_accuracy_on_known_rows():
    row = _rows(2, 16, seed=3)
    faithful = ca_perturbation.apply_rule30(row, steps=1)
    zeros = torch.zeros_like(row)

    assert torch.all(ca_perturbation.own_output_metrics(faithful, row)["self_transition_accuracy"] == 1)
    flipped = faithful.clone()
    flipped[:, :4] = 1 - flipped[:, :4]
    torch.testing.assert_close(
        ca_perturbation.own_output_metrics(flipped, row)["self_transition_accuracy"],
        torch.full((2,), 0.75, dtype=torch.float64),
    )
    # The trap documented in the module: a constant all-zeros output is a
    # fixed point of Rule 30, so it is "self-consistent" while carrying no row.
    constant = ca_perturbation.own_output_metrics(zeros, zeros)
    assert torch.all(constant["self_transition_accuracy"] == 1)
    assert torch.all(constant["ones_fraction"] == 0) and torch.all(constant["changed_fraction"] == 0)
    assert set(ca_perturbation.own_output_metrics(row, None)) == {"ones_fraction"}


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_zero_theta_branch_has_the_clean_predictions_and_norm(model_name):
    _, arrays, options = _rollout(model_name, thetas=(0.0,))
    inject = options["inject_depths"][0]

    np.testing.assert_array_equal(
        arrays["rotate_x__predictions"][0, 0, 0, 0, inject:], arrays["clean__predictions"][inject:]
    )
    np.testing.assert_array_equal(
        arrays["rotate_x__x_centred_norm"][0, 0, 0, 0, inject:],
        arrays["clean__x_centred_norm"][inject:].astype(np.float32),
    )
    if model_name in ca_lyapunov.CELL_MODELS:
        np.testing.assert_array_equal(
            arrays["cell__cell_norm"][0, 0, 0, 0, inject:],
            arrays["clean__cell_norm"][inject:].astype(np.float32),
        )


def test_rotation_keeps_and_radial_scales_the_branch_norm_at_injection():
    _, arrays, options = _rollout("but_full_depth")
    inject = options["inject_depths"][0]
    clean_norm = arrays["clean__x_centred_norm"][inject]

    np.testing.assert_allclose(arrays["rotate_x__x_centred_norm"][0, 1, 0, 0, inject], clean_norm, rtol=1e-6)
    np.testing.assert_allclose(
        arrays["radial_x__x_centred_norm"][0, 1, 0, 0, inject], math.exp(0.1) * clean_norm, rtol=1e-6
    )


def test_kinds_option_selects_kinds_without_changing_their_results():
    _, everything, _ = _rollout("lstm_ut_bidir")
    _, selected, options = _rollout("lstm_ut_bidir", kinds=("rotate_x", "cell"))

    kinds = {name.split("__")[0] for name in selected if "__" in name} - {"clean", "linear"}
    assert kinds == {"rotate_x", "cell"}
    # Same random draws whichever kinds run.
    for name in selected:
        np.testing.assert_array_equal(selected[name], everything[name])
    # The shift control runs by default but its bits are not stored.
    assert "shift_x__bit_disagreement" in everything
    assert "shift_x__predictions" not in everything
    assert "cell__predictions" in everything

    summary = ca_perturbation.summarise(selected, options["max_depth"])
    assert set(summary["kinds"]) == {"rotate_x", "cell"}
    entry = summary["kinds"]["rotate_x"]["inject_2"]["0.1"]
    for field in ("self_transition_accuracy_first_step", "self_transition_accuracy_later",
                  "ones_fraction_after_injection", "changed_fraction_after_injection",
                  "first_wrong_depth_mean", "clean_first_wrong_depth_mean",
                  "wrong_earlier_than_clean_fraction", "wrong_later_than_clean_fraction"):
        assert entry[field] is not None and np.isfinite(entry[field])
    assert summary["clean"]["self_transition_accuracy_per_depth"][0] is None


def test_kinds_option_skips_the_cell_for_models_without_one_and_the_linear_check_without_rotation():
    _, arrays, options = _rollout("but_full_depth", kinds=("radial_x", "cell"))

    assert not any(name.startswith(("cell__", "rotate_x__", "linear__")) for name in arrays)
    summary = ca_perturbation.summarise(arrays, options["max_depth"])
    assert set(summary["kinds"]) == {"radial_x"}
    assert summary["linear_check"] == {}

    step = ca_lyapunov.BlockStep(_tiny_model("but_full_depth"))
    with pytest.raises(ValueError, match="kinds"):
        ca_perturbation.rollout_perturbations(
            step, _rows(2, 8), max_depth=4, inject_depths=(2,), thetas=(0.1,), num_directions=1,
            direction_seed=0, tokens="all", kinds=("rotate",),
        )


def test_first_wrong_depth_is_censored_at_the_end():
    # [depth 0..5, row]: row 0 first wrong at 3, row 1 wrong before injection
    # only, row 2 wrong at injection, row 3 never wrong.
    accuracy = np.ones((6, 4), dtype=np.float32)
    accuracy[3:, 0] = 0.5
    accuracy[1, 1] = 0.0
    accuracy[2, 2] = 0.9

    first = ca_perturbation.first_wrong_depth(accuracy, inject_depth=2, max_depth=5)

    np.testing.assert_array_equal(first, [3, 6, 2, 6])


# --- LSTM-UT output-gate closure ---------------------------------------------
#
# Software correctness only: that the closed gate removes exactly the cell
# term, that nothing changes with the flag off, and that the closure window,
# the reopening and the two cell modes are what the docstring says. Tiny
# random models; nothing here says how a trained model responds.

# Block parameters before close_hidden_gate was added. The flag must not add
# or rename any, or old checkpoints would stop loading.
LSTM_BLOCK_STATE_KEYS = [
    "attn.c_attn.weight", "attn.c_proj.weight", "attn.q_attn.weight", "ln_1.weight", "ln_2.weight",
    "memory_gate_norm.bias", "memory_gate_norm.weight", "mlp.c_fc.weight", "mlp.c_proj.weight",
    "previous_to_forget.weight", "previous_to_hidden.weight", "previous_to_memory_proposal.weight",
    "previous_to_write.weight", "proposed_to_forget.weight", "proposed_to_hidden.weight",
    "proposed_to_memory_proposal.weight", "proposed_to_write.weight",
]


def _closure(**overrides):
    step = ca_lyapunov.BlockStep(_tiny_model("lstm_ut_bidir"))
    options = dict(max_depth=8, start_depths=(2, 3), closure_lengths=(0, 1, 3))
    options.update(overrides)
    return step, ca_perturbation.rollout_gate_closure(step, _rows(3, 8), **options), options


def _manual_block_parts(step, state):
    """x after attention and MLP, and the gated cell term, written out by hand."""
    block, x = step.block, state[:, 0]
    after = x + block.attn(block.ln_1(x), step.pos_emb_closure, None, 0)
    after = after + block.mlp(block.ln_2(after))
    gate = torch.sigmoid(block._control_projection(
        block.memory_gate_norm(x), block.memory_gate_norm(after),
        block.previous_to_hidden, block.proposed_to_hidden,
    ))
    return after, gate


def test_gate_flag_off_is_the_original_block_and_on_drops_only_the_cell_term():
    step = ca_lyapunov.BlockStep(_tiny_model("lstm_ut_bidir"))
    state = _state_after(step, _rows(3, 8), depth=2)

    with torch.no_grad():
        default = step(state)
        explicit_open = step(state, close_hidden_gate=False)
        closed = step(state, close_hidden_gate=True)
        after_attention_and_mlp, gate = _manual_block_parts(step, state)

    assert torch.equal(default, explicit_open)
    # Open: the block as it was before the flag existed.
    assert torch.equal(default[:, 0], after_attention_and_mlp + gate * torch.tanh(default[:, 1]))
    # Closed: x is attention + MLP only; the cell is written exactly as usual.
    assert torch.equal(closed[:, 0], after_attention_and_mlp)
    assert torch.equal(closed[:, 1], default[:, 1])
    assert not torch.equal(closed[:, 0], default[:, 0])


def test_linear_read_gate_flag_reads_the_cell_without_tanh_and_closing_drops_it():
    # Same check as above for the linear-read ablation: open, x gains o * c
    # (no tanh); closed, x is attention + MLP only and the cell is unchanged.
    step = ca_lyapunov.BlockStep(_tiny_model("lin_lstm_ut_bidir"))
    assert step.has_cell
    state = _state_after(step, _rows(3, 8), depth=2)

    with torch.no_grad():
        default = step(state)
        closed = step(state, close_hidden_gate=True)
        after_attention_and_mlp, gate = _manual_block_parts(step, state)

    assert torch.equal(default[:, 0], after_attention_and_mlp + gate * default[:, 1])
    assert torch.equal(closed[:, 0], after_attention_and_mlp)
    assert torch.equal(closed[:, 1], default[:, 1])


def test_gate_flag_adds_no_parameters():
    model = _tiny_model("lstm_ut_bidir")
    assert sorted(model.transformer.h_mid[0].state_dict()) == LSTM_BLOCK_STATE_KEYS


@pytest.mark.parametrize("model_name", ("but_full_depth", "gated_but_bidir"))
def test_closed_gate_is_rejected_for_models_without_a_cell(model_name):
    step = ca_lyapunov.BlockStep(_tiny_model(model_name))
    state = _state_after(step, _rows(2, 8), depth=1)
    with pytest.raises(ValueError, match="cell"):
        step(state, close_hidden_gate=True)
    with pytest.raises(ValueError, match="cell"):
        ca_perturbation.rollout_gate_closure(
            step, _rows(2, 8), max_depth=6, start_depths=(2,), closure_lengths=(1,)
        )


def test_gate_closure_rejects_bad_options():
    step = ca_lyapunov.BlockStep(_tiny_model("lstm_ut_bidir"))
    common = dict(max_depth=6, start_depths=(2,), closure_lengths=(1,))
    for bad, match in (
        (dict(closure_lengths=(-1,)), "non-negative"),
        (dict(closure_lengths=()), "non-empty"),
        (dict(closure_lengths=(4,)), "start depth"),  # 2 + 4 is not < 6
        (dict(start_depths=(0,)), "start depth"),
        (dict(cell_modes=("melted",)), "cell_modes"),
    ):
        with pytest.raises(ValueError, match=match):
            ca_perturbation.rollout_gate_closure(step, _rows(2, 8), **{**common, **bad})


def test_zero_length_closure_is_the_clean_branch():
    _, arrays, _ = _closure()

    # [cell mode, start, length, depth, row]; length index 0 is L = 0.
    assert arrays["closure__x_raw_distance"].shape == (2, 2, 4, 9, 3)
    assert arrays["closure_lengths"].tolist() == [0, 1, 3, ca_perturbation.NEVER_REOPENED]
    assert arrays["closure_cell_modes"].tolist() == ["update", "frozen"]
    for start_index, start in enumerate((2, 3)):
        for mode_index in range(2):
            branch = lambda name: arrays[f"closure__{name}"][mode_index, start_index, 0]  # noqa: E731
            assert np.all(np.isnan(branch("x_raw_distance")[:start]))
            assert np.all(branch("predictions")[:start] == ca_perturbation.NOT_RUN)
            for name in ("x_raw_distance", "cell_distance", "logit_distance", "bit_disagreement"):
                assert np.all(branch(name)[start:] == 0), name
            np.testing.assert_array_equal(
                branch("predictions")[start:], arrays["clean__predictions"][start:]
            )
            np.testing.assert_array_equal(
                branch("cell_accuracy")[start:], arrays["clean__cell_accuracy"][start:].astype(np.float32)
            )


def test_closure_window_is_closed_exactly_for_the_requested_repeats():
    _, arrays, _ = _closure()

    closed = arrays["closure__gate_closed"]  # [start, length, depth]
    for start_index, start in enumerate((2, 3)):
        for length_index, length in enumerate((0, 1, 3)):
            expected = np.zeros(9, dtype=bool)
            expected[start + 1:start + length + 1] = True
            np.testing.assert_array_equal(closed[start_index, length_index], expected)
        never = np.zeros(9, dtype=bool)
        never[start + 1:] = True
        np.testing.assert_array_equal(closed[start_index, 3], never)


@pytest.mark.parametrize("cell_mode", ca_perturbation.CELL_MODES)
def test_closure_branch_matches_a_step_by_step_recomputation(cell_mode):
    step, arrays, _ = _closure()
    rows, start, length = _rows(3, 8), 3, 3
    clean_states, _ = ca_perturbation.clean_rollout(step, rows, max_depth=8)
    mode_index = ca_perturbation.CELL_MODES.index(cell_mode)

    # Written out for s = 3, L = 3: repeats 4, 5, 6 closed, repeats 7, 8 open.
    with torch.no_grad():
        state = clean_states[start]
        for depth in range(start + 1, 9):
            closed = depth in (4, 5, 6)
            state = step(state, close_hidden_gate=closed)
            if closed and cell_mode == "frozen":
                state = torch.stack((state[:, 0], clean_states[start][:, 1]), dim=1)
            x_distance = (state[:, 0] - clean_states[depth][:, 0]).pow(2).sum(dim=(1, 2)).sqrt()
            cell_norm = state[:, 1].pow(2).sum(dim=(1, 2)).sqrt()
            index = (mode_index, 1, 2, depth)  # start index 1, length index 2
            np.testing.assert_allclose(
                arrays["closure__x_raw_distance"][index], x_distance.numpy(), rtol=1e-6
            )
            np.testing.assert_allclose(arrays["closure__cell_norm"][index], cell_norm.numpy(), rtol=1e-6)
            np.testing.assert_array_equal(
                arrays["closure__predictions"][index], step.decode(state).numpy()
            )


def test_cell_modes_share_x_during_the_closure_and_the_frozen_cell_is_the_start_cell():
    _, arrays, _ = _closure()
    update, frozen = 0, 1

    for start_index, start in enumerate((2, 3)):
        for length_index, length in enumerate((1, 3), start=1):
            closed = slice(start, start + length + 1)  # start repeat and the closed repeats
            for name in ("x_raw_distance", "predictions", "cell_accuracy"):
                np.testing.assert_array_equal(
                    arrays[f"closure__{name}"][update, start_index, length_index, closed],
                    arrays[f"closure__{name}"][frozen, start_index, length_index, closed],
                )
            np.testing.assert_allclose(
                arrays["closure__cell_norm"][frozen, start_index, length_index, closed],
                np.broadcast_to(arrays["clean__cell_norm"][start], (length + 1, 3)),
                rtol=1e-6,
            )
    # After reopening the two modes read different cells, so x differs.
    assert not np.array_equal(
        arrays["closure__x_raw_distance"][update, 0, 2, 6:],
        arrays["closure__x_raw_distance"][frozen, 0, 2, 6:],
    )


def test_never_reopened_equals_a_window_that_runs_to_the_end():
    # s = 2, max depth 8: L = 5 closes repeats 3..7; never-reopened closes 3..8.
    _, arrays, _ = _closure(start_depths=(2,), closure_lengths=(5,))

    for name in ("x_raw_distance", "cell_norm", "predictions", "self_transition_accuracy"):
        values = arrays[f"closure__{name}"]
        np.testing.assert_array_equal(values[:, 0, 0, 2:8], values[:, 0, 1, 2:8], err_msg=name)
    # While the gate never reopens, the cell mode cannot reach x.
    np.testing.assert_array_equal(
        arrays["closure__predictions"][0, 0, 1], arrays["closure__predictions"][1, 0, 1]
    )


def test_gate_closure_summary_and_cli(tmp_path):
    import json
    import types
    from unittest import mock

    step, arrays, options = _closure()
    summary = ca_perturbation.summarise_gate_closure(arrays, options["max_depth"])

    assert set(summary["closure"]) == {"update", "frozen"}
    per_length = summary["closure"]["update"]["start_3"]
    assert list(per_length) == ["length_0", "length_1", "length_3", "never_reopened"]
    assert per_length["length_3"]["last_closed_depth"] == 6
    assert per_length["never_reopened"]["last_closed_depth"] == 8
    assert per_length["length_3"]["final_window"] == [4, 8]
    # L = 0 is the clean branch: no distance, no closure, same first error.
    assert per_length["length_0"]["relative_x_distance_at_end"] == 0.0
    assert "self_transition_accuracy_during_closure" not in per_length["length_0"]
    assert per_length["length_0"]["first_wrong_depth_mean"] == per_length["length_0"]["clean_first_wrong_depth_mean"]
    assert 0.0 <= per_length["length_3"]["self_transition_accuracy_during_closure"] <= 1.0
    json.dumps(summary)

    model_args = types.SimpleNamespace(model="lstm_ut_bidir", ca_test_seed=7)
    loaded = (step.model, model_args, {"step": 1}, tmp_path / "ckpt.pt")
    output_dir = tmp_path / "out"
    with mock.patch.object(ca_perturbation, "load_model", return_value=loaded), \
            mock.patch.object(ca_perturbation, "test_rows", return_value=_rows(3, 8)):
        ca_perturbation.main([
            "--source-run-dir", str(tmp_path), "--output-dir", str(output_dir), "--num-cells", "8",
            "--max-depth", "8", "--inject-depths", "2", "3", "--closure-lengths", "0", "1", "3",
        ])

    saved = np.load(output_dir / "gate_closure.npz")
    assert not (output_dir / "perturbation.npz").exists()
    np.testing.assert_array_equal(saved["closure__predictions"], arrays["closure__predictions"])
    np.testing.assert_array_equal(saved["rows"], _rows(3, 8).numpy())
    written = json.loads((output_dir / "summary.json").read_text())
    # main() runs under _math_attention, so floats can differ from the direct
    # call above in the last digit; the structure and the bits may not.
    assert list(written["closure"]["frozen"]["start_2"]) == list(per_length)
    assert written["closure"]["update"]["start_3"]["length_3"]["last_closed_depth"] == 6
    assert written["closure"]["update"]["start_3"]["length_3"]["cell_accuracy_at_end"] == pytest.approx(
        per_length["length_3"]["cell_accuracy_at_end"]
    )
    assert written["provenance"]["arguments"]["closure_lengths"] == [0, 1, 3]


# --- BUT with input injection: state (x, e), e constant, x_0 = 0 -------------

INJECT_MODEL = "but_input_inject"


@pytest.mark.parametrize("check", (
    test_rollout_and_summary_smoke,
    test_zero_theta_branch_equals_the_clean_branch,
    test_shift_control_changes_no_logit_and_is_carried_unchanged,
    test_small_rotation_matches_the_jacobian_prediction,
    test_saved_predictions_reproduce_the_saved_bit_metrics,
    test_self_transition_metrics_match_a_recomputation_from_the_saved_bits,
    test_zero_theta_branch_has_the_clean_predictions_and_norm,
), ids=lambda check: check.__name__)
def test_input_injection_passes_the_model_independent_checks(check):
    check(INJECT_MODEL)


def test_input_injection_clean_rollout_is_the_models_own_repeats():
    model = _tiny_model(INJECT_MODEL)
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)

    states, _ = ca_perturbation.clean_rollout(step, rows, max_depth=4)

    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=4, return_repeat_states=True)
    assert torch.equal(states[0][:, 0], torch.zeros_like(states[0][:, 0]))
    for depth in range(5):
        assert torch.equal(states[depth][:, 1], outputs["repeat_states"][0])
    for depth in range(1, 5):
        assert torch.equal(states[depth][:, 0], outputs["repeat_states"][depth])
    assert torch.equal(ca_perturbation.logits_of(step, states[4]), outputs["logits"])


def test_input_injection_perturbed_branch_keeps_each_rows_own_input():
    step = ca_lyapunov.BlockStep(_tiny_model(INJECT_MODEL))
    rows = _rows(3, 8)
    states, _ = ca_perturbation.clean_rollout(step, rows, max_depth=5)
    base = states[2]
    draw = _draw(base)
    perturbed = ca_perturbation.perturb_state(base, "rotate_x", 0.3, 1.0, draw, _all_tokens(base))
    assert torch.equal(perturbed[:, 1], base[:, 1])

    _, arrays, _ = _rollout(INJECT_MODEL, inject_depths=(2,), thetas=(0.3,), num_directions=1,
                            max_depth=5)
    # Recompute direction 0, sign +1 by hand: one row at a time, so a wrong
    # pairing of rows with injected inputs inside the batched branch would show.
    with torch.no_grad():
        for row in range(3):
            state = perturbed[row:row + 1]
            for _ in range(3):
                state = step(state)
            expected = ca_perturbation.logits_of(step, state).argmax(dim=-1)[0].numpy()
            np.testing.assert_array_equal(arrays["rotate_x__predictions"][0, 0, 0, 0, 5, row], expected)


def test_gate_closure_is_rejected_for_input_injection():
    step = ca_lyapunov.BlockStep(_tiny_model(INJECT_MODEL))
    with pytest.raises(ValueError, match="cell"):
        ca_perturbation.rollout_gate_closure(
            step, _rows(2, 8), max_depth=6, start_depths=(2,), closure_lengths=(1,)
        )


# --- BUT with FPRM residual scaling: state (h, e), e constant, h_0 = e -------

FPRM_MODEL = "fprm_but"


# The shift control is left out here: it assumes a per-token offset is carried
# unchanged, which holds for a plain residual stream but not for fprm_but (see
# the next test).
@pytest.mark.parametrize("check", (
    test_rollout_and_summary_smoke,
    test_zero_theta_branch_equals_the_clean_branch,
    test_small_rotation_matches_the_jacobian_prediction,
    test_saved_predictions_reproduce_the_saved_bit_metrics,
    test_self_transition_metrics_match_a_recomputation_from_the_saved_bits,
    test_zero_theta_branch_has_the_clean_predictions_and_norm,
), ids=lambda check: check.__name__)
def test_fprm_passes_the_model_independent_checks(check):
    check(FPRM_MODEL)


def test_fprm_shift_decays_by_the_kept_share_each_repeat():
    # A per-token offset k*1 passes LayerNorm unseen, so after one repeat it is
    # alpha_2 * alpha_1^2 * k*1. With equal scales in every channel (the
    # initialisation) it stays an offset: still invisible, but it shrinks by
    # the kept share each repeat instead of being carried unchanged. (With
    # unequal per-channel scales, as after training, it is no longer a pure
    # offset and becomes visible to LayerNorm.)
    _, arrays, options = _rollout(FPRM_MODEL, thetas=(0.5,))
    inject = options["inject_depths"][0]
    after = slice(inject, None)
    assert arrays["shift_x__logit_distance"][..., after, :].max() < 1e-6
    assert arrays["shift_x__x_centred_distance"][..., after, :].max() < 1e-6
    assert np.all(arrays["shift_x__bit_disagreement"][..., after, :] == 0)
    raw = arrays["shift_x__x_raw_distance"][..., after, :]
    kept = 0.25 * 0.75 ** 2  # default fprm_alpha_2_init * fprm_alpha_1_init^2
    np.testing.assert_allclose(raw[..., 1:, :] / raw[..., :-1, :], kept, rtol=1e-4)


def test_fprm_clean_rollout_is_the_models_own_repeats():
    model = _tiny_model(FPRM_MODEL)
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)

    states, _ = ca_perturbation.clean_rollout(step, rows, max_depth=4)

    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=4, return_repeat_states=True)
    for depth in range(5):
        assert torch.equal(states[depth][:, 0], outputs["repeat_states"][depth])
        assert torch.equal(states[depth][:, 1], outputs["repeat_states"][0])
    assert torch.equal(ca_perturbation.logits_of(step, states[4]), outputs["logits"])


def test_gate_closure_is_rejected_for_fprm():
    step = ca_lyapunov.BlockStep(_tiny_model(FPRM_MODEL))
    with pytest.raises(ValueError, match="cell"):
        ca_perturbation.rollout_gate_closure(
            step, _rows(2, 8), max_depth=6, start_depths=(2,), closure_lengths=(1,)
        )


# --- BUT with a fixed stream decay (lambda = 0.5) ----------------------------


@pytest.mark.parametrize("check", (
    test_rollout_and_summary_smoke,
    test_zero_theta_branch_equals_the_clean_branch,
    test_small_rotation_matches_the_jacobian_prediction,
    test_saved_predictions_reproduce_the_saved_bit_metrics,
    test_self_transition_metrics_match_a_recomputation_from_the_saved_bits,
    test_zero_theta_branch_has_the_clean_predictions_and_norm,
), ids=lambda check: check.__name__)
def test_decay_but_passes_the_model_independent_checks(check, monkeypatch):
    # The shared checks build models by name only (through this module's
    # _tiny_model); give decay_but lambda = 0.5 instead of the default 1.0.
    build = _tiny_model

    def with_decay(model_name, **kwargs):
        if model_name == "decay_but":
            kwargs["extra_args"] = tuple(kwargs.get("extra_args", ())) + ("--but_stream_decay", "0.5")
        return build(model_name, **kwargs)

    monkeypatch.setattr(sys.modules[__name__], "_tiny_model", with_decay)
    check("decay_but")


def test_decay_but_shift_decays_by_lambda_each_repeat():
    # A per-token offset passes LayerNorm unseen and is multiplied by lambda
    # each repeat: still invisible, but not carried unchanged as in BUT.
    step = ca_lyapunov.BlockStep(_tiny_model("decay_but", extra_args=("--but_stream_decay", "0.5")))
    arrays = ca_perturbation.rollout_perturbations(
        step, _rows(3, 8), max_depth=6, inject_depths=(2,), thetas=(0.5,), num_directions=2,
        direction_seed=0, tokens="all",
    )
    after = slice(2, None)
    assert arrays["shift_x__logit_distance"][..., after, :].max() < 1e-6
    assert np.all(arrays["shift_x__bit_disagreement"][..., after, :] == 0)
    raw = arrays["shift_x__x_raw_distance"][..., after, :]
    np.testing.assert_allclose(raw[..., 1:, :] / raw[..., :-1, :], 0.5, rtol=1e-4)


# --- LSTM-UT without a forget gate -------------------------------------------


def test_no_forget_gate_runs_perturbations_and_gate_closure():
    model = _tiny_model("lstm_ut_bidir", extra_args=("--lstm_forget_gate", "none"))
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)

    arrays = ca_perturbation.rollout_perturbations(
        step, rows, max_depth=6, inject_depths=(2,), thetas=(1e-6, 0.1), num_directions=2,
        direction_seed=0, tokens="all",
    )
    summary = ca_perturbation.summarise(arrays, 6)
    assert set(summary["kinds"]) == {"rotate_x", "shift_x", "radial_x", "cell"}
    assert summary["linear_check"]["inject_2"]["max_abs_log_error"] < 1e-3

    closure = ca_perturbation.rollout_gate_closure(
        step, rows, max_depth=6, start_depths=(2,), closure_lengths=(0, 1),
    )
    np.testing.assert_array_equal(
        closure["closure__predictions"][0, 0, 0, 2:], closure["clean__predictions"][2:]
    )
