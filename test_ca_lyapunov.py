"""Tests for the finite-time Lyapunov exponent tool (cellular_automaton/ca_lyapunov.py).

These check software correctness only: that the one-step map is the model's
own repeat, that the JVPs are the Jacobian of that map, and that the
Benettin/QR procedure recovers known exponents. They say nothing about what
the exponents of a trained model mean.
"""

import sys
from unittest import mock

import numpy as np
import pytest
import torch

import models
from cellular_automaton import ca_lyapunov, ca_main

MODEL_NAMES = ("lstm_ut_bidir", "lin_lstm_ut_bidir", "but_full_depth", "gated_but_bidir")


def _tiny_model(model_name, *, n_embd=16, n_head=2, cells=8, seed=0, extra_args=()):
    argv = [
        str(ca_main.__file__),
        "--model", model_name,
        "--n_embd", str(n_embd),
        "--n_head", str(n_head),
        "--n_layer", "1",
        "--sequence_length", str(cells),
        "--vocab_size", "2",
        "--attention_mode", "bidirectional",
        "--dropout", "0",
        *extra_args,
    ]
    with mock.patch.object(sys, "argv", argv):
        args = ca_main.get_args()
    torch.manual_seed(seed)
    model = models.make_model_from_args(args).double().eval()
    # The 0.02 init makes the block nearly the identity; larger random weights
    # give a Jacobian with real structure to test against.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 2 and "wte" not in name:
                parameter.normal_(0.0, 0.3, generator=generator)
    model.requires_grad_(False)
    return model


def _rows(batch, cells, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, cells), generator=generator)


def _state_after(step, rows, depth):
    with torch.no_grad():
        state = step.initial_state(rows)
        for _ in range(depth):
            state = step(state)
    return state


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_rollout_matches_model_repeat_states_and_logits(model_name):
    model = _tiny_model(model_name)
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)
    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=4, return_repeat_states=True)
        state = step.initial_state(rows)
        torch.testing.assert_close(state[:, 0], outputs["repeat_states"][0], rtol=0, atol=1e-12)
        for depth in range(1, 5):
            state = step(state)
            torch.testing.assert_close(
                state[:, 0], outputs["repeat_states"][depth], rtol=0, atol=1e-12
            )
        assert torch.equal(step.decode(state), outputs["logits"].argmax(dim=-1))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_jvp_matches_central_finite_differences(model_name):
    model = _tiny_model(model_name)
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(1, 8), depth=2)
    dim = state[0].numel()
    tangents = torch.randn(1, dim, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64)

    images = ca_lyapunov.jvp_columns(step, state, tangents)

    h = 1e-6
    for column in range(3):
        direction = tangents[:, :, column].reshape(state.shape)
        with torch.no_grad():
            difference = (step(state + h * direction) - step(state - h * direction)) / (2 * h)
        torch.testing.assert_close(images[:, :, column], difference.reshape(1, -1), rtol=1e-6, atol=1e-7)


def test_restricted_jvp_is_a_block_of_the_full_jacobian():
    model = _tiny_model("lstm_ut_bidir")
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(2, 8), depth=2)
    block_dim = state[0, 0].numel()
    tangents = torch.randn(2, block_dim, 2, generator=torch.Generator().manual_seed(4), dtype=torch.float64)

    embedded = torch.zeros(2, 2 * block_dim, 2, dtype=torch.float64)
    embedded[:, block_dim:] = tangents  # tangent in the cell only
    full = ca_lyapunov.jvp_columns(step, state, embedded)
    cell_to_cell = ca_lyapunov.jvp_columns(step, state, tangents, source=1, target=1)
    cell_to_x = ca_lyapunov.jvp_columns(step, state, tangents, source=1, target=0)

    torch.testing.assert_close(cell_to_cell, full[:, block_dim:], rtol=0, atol=1e-12)
    torch.testing.assert_close(cell_to_x, full[:, :block_dim], rtol=0, atol=1e-12)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_examples_do_not_interact(model_name):
    model = _tiny_model(model_name)
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(3, 8), depth=1)
    dim = state[0].numel()
    tangents = torch.randn(3, dim, 2, generator=torch.Generator().manual_seed(5), dtype=torch.float64)

    batched = ca_lyapunov.jvp_columns(step, state, tangents)
    alone = ca_lyapunov.jvp_columns(step, state[1:2], tangents[1:2])

    torch.testing.assert_close(batched[1:2], alone, rtol=1e-12, atol=1e-12)


def test_benettin_recovers_exponents_of_a_diagonal_linear_map():
    scales = torch.exp(torch.linspace(-1.0, 0.5, 8, dtype=torch.float64)).reshape(1, 1, 4, 2)

    def linear_step(state):
        return state * scales

    state = torch.ones(1, 1, 4, 2, dtype=torch.float64)
    tangents = ca_lyapunov.orthonormal_tangents(
        1, 8, 3, generator=torch.Generator().manual_seed(6), dtype=torch.float64, device="cpu"
    )
    total = torch.zeros(1, 3, dtype=torch.float64)
    steps = 400
    for _ in range(steps):
        images = ca_lyapunov.jvp_columns(linear_step, state, tangents)
        tangents, log_growth = ca_lyapunov.benettin_update(images)
        total += log_growth

    expected = torch.log(scales.flatten()).sort(descending=True).values[:3]
    torch.testing.assert_close(total[0] / steps, expected, rtol=0, atol=0.02)


def test_full_rank_benettin_step_equals_log_abs_determinant():
    model = _tiny_model("lstm_ut_bidir", n_embd=8, cells=4)
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(1, 4), depth=2)
    dim = state[0].numel()

    jacobian = ca_lyapunov.full_jacobian(step, state, chunk_size=16)
    reference = torch.autograd.functional.jacobian(
        lambda flat: step(flat.reshape(state.shape)).reshape(-1), state.reshape(-1)
    )
    torch.testing.assert_close(jacobian, reference, rtol=1e-10, atol=1e-12)

    tangents = ca_lyapunov.orthonormal_tangents(
        1, dim, dim, generator=torch.Generator().manual_seed(7), dtype=torch.float64, device="cpu"
    )
    _, log_growth = ca_lyapunov.benettin_update(ca_lyapunov.jvp_columns(step, state, tangents))
    _, log_abs_det = torch.linalg.slogdet(jacobian)
    torch.testing.assert_close(log_growth.sum(), log_abs_det, rtol=1e-8, atol=1e-8)


def test_forget_gate_hook_reads_the_gate_used_by_the_block():
    model = _tiny_model("lstm_ut_bidir")
    step = ca_lyapunov.BlockStep(model)
    block = step.block
    state = _state_after(step, _rows(2, 8), depth=1)

    with torch.no_grad():
        _, forget_gate = step.step_with_forget_gate(state)
        x = state[:, 0]
        x_block = x + block.attn(block.ln_1(x), step.pos_emb_closure, None, 0)
        x_block = x_block + block.mlp(block.ln_2(x_block))
        expected = torch.sigmoid(
            block.previous_to_forget(block.memory_gate_norm(x))
            + block.proposed_to_forget(block.memory_gate_norm(x_block))
        )

    torch.testing.assert_close(forget_gate, expected, rtol=0, atol=1e-12)


def test_token_uniformity_limits():
    uniform = torch.ones(1, 4 * 3, dtype=torch.float64)
    alternating = torch.tensor([[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]],
                               dtype=torch.float64)
    assert ca_lyapunov.token_uniformity(uniform, 4, 3).item() == pytest.approx(1.0)
    assert ca_lyapunov.token_uniformity(alternating, 4, 3).item() == pytest.approx(0.0)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_rollout_and_summary_smoke(model_name):
    model = _tiny_model(model_name)
    step = ca_lyapunov.BlockStep(model)
    arrays, checks = ca_lyapunov.rollout_lyapunov(
        step, _rows(3, 8), max_depth=5, num_vectors=4, tangent_seed=0,
        jacobian_check_depths=(2,), jacobian_check_rows=1,
    )

    assert arrays["full_log_growth"].shape == (5, 3, 4)
    assert arrays["cell_accuracy"].shape == (5, 3)
    for values in arrays.values():
        assert not (values != values).all()  # not all NaN
    assert len(checks) == 1
    # |R_11| = |J q_1| can never exceed the largest singular value of J.
    assert checks[0]["full_local_log_growth_top"] <= checks[0]["log_top_singular_values"][0] + 1e-9

    if model_name in ca_lyapunov.CELL_MODELS:
        assert arrays["cell_to_cell_log_growth"].shape == (5, 3, 4)
        assert ((arrays["forget_mean"] > 0) & (arrays["forget_mean"] < 1)).all()
    else:
        assert "cell_to_cell_log_growth" not in arrays

    summary = ca_lyapunov.summarise(arrays, ((1, 2), (3, 10)), max_depth=5)
    assert set(summary["windows"]) == {"1-2", "3-5"}
    assert len(summary["per_depth"]["exact_accuracy"]) == 5


def test_block_step_rejects_unsupported_layouts():
    model = _tiny_model("but_full_depth")
    model.config.lm_cache = "kv"
    with pytest.raises(ValueError, match="lm_cache"):
        ca_lyapunov.BlockStep(model)


def test_remove_token_shift_removes_x_offsets_and_keeps_the_cell():
    vectors = torch.randn(2, 2 * 4 * 3, 5, generator=torch.Generator().manual_seed(8), dtype=torch.float64)

    projected = ca_lyapunov.remove_token_shift(vectors, 2, 4, 3)

    per_feature = projected.reshape(2, 2, 4, 3, 5)
    torch.testing.assert_close(per_feature[:, 0].mean(dim=2), torch.zeros(2, 4, 5, dtype=torch.float64),
                               rtol=0, atol=1e-15)
    torch.testing.assert_close(per_feature[:, 1], vectors.reshape(2, 2, 4, 3, 5)[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(ca_lyapunov.remove_token_shift(projected, 2, 4, 3), projected,
                               rtol=0, atol=1e-15)


def test_token_shift_energy_fraction_limits():
    shift = torch.zeros(1, 2, 4, 3, dtype=torch.float64)
    shift[0, 0, 2] = 1.0  # one constant across token 2's hidden features
    mean_free = torch.zeros(1, 2, 4, 3, dtype=torch.float64)
    mean_free[0, 0, 1] = torch.tensor([1.0, -1.0, 0.0])
    half = shift.clone()
    half[0, 1, 0] = torch.tensor([3.0 ** 0.5, 0.0, 0.0])  # equal energy in the cell

    def fraction(vector):
        return ca_lyapunov.token_shift_energy_fraction(vector.reshape(1, -1), 2, 4, 3).item()

    assert fraction(shift) == pytest.approx(1.0)
    assert fraction(mean_free) == pytest.approx(0.0)
    assert fraction(half) == pytest.approx(0.5)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_token_shift_is_an_exact_neutral_direction(model_name):
    model = _tiny_model(model_name)
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(1, 8), depth=2)
    # A different constant on every token's hidden features; no cell change.
    shift = torch.zeros_like(state)
    offsets = torch.randn(state.shape[2], 1, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    shift[0, 0] = offsets.expand(-1, state.shape[3])

    image = ca_lyapunov.jvp_columns(step, state, shift.reshape(1, -1, 1))[0, :, 0]

    torch.testing.assert_close(image, shift.reshape(-1), rtol=0, atol=1e-12)


class _LinearStep:
    """Known linear map: x offsets are kept, the rest of x is halved, c is scaled by 0.8."""

    def __init__(self, has_cell):
        self.has_cell = has_cell
        self.model = torch.nn.Linear(1, 1)

    def initial_state(self, rows):
        generator = torch.Generator().manual_seed(10)
        components = 2 if self.has_cell else 1
        return torch.randn(rows.shape[0], components, 4, 4, generator=generator, dtype=torch.float64)

    def __call__(self, state):
        x = state[:, 0]
        offset = x.mean(dim=2, keepdim=True).expand_as(x)
        parts = [offset + 0.5 * (x - offset)]
        if self.has_cell:
            parts.append(0.8 * state[:, 1])
        return torch.stack(parts, dim=1)

    def step_with_forget_gate(self, state):
        return self(state), None

    def decode(self, state):
        return torch.zeros(state.shape[0], state.shape[2], dtype=torch.long)


@pytest.mark.parametrize("has_cell", (False, True), ids=("x_only", "with_cell"))
def test_shift_quotient_recovers_the_contraction_hidden_by_neutral_offsets(has_cell):
    arrays, _ = ca_lyapunov.rollout_lyapunov(
        _LinearStep(has_cell), torch.zeros(1, 4, dtype=torch.long),
        max_depth=150, num_vectors=3, tangent_seed=0,
    )

    # Raw set: the 4 neutral offset directions take over the top of the spectrum.
    np = pytest.importorskip("numpy")
    assert np.allclose(arrays["full_log_growth"][-1, 0], 0.0, atol=1e-6)
    assert arrays["top_vector_shift_fraction"][-1, 0] == pytest.approx(1.0, abs=1e-6)
    # Shift-quotient set. Without a cell every offset-free direction is halved,
    # so every step shows exactly log 0.5. With a cell the slowest remaining
    # direction is the (never projected) cell at x0.8, which the leading
    # direction reaches once it has aligned.
    shift_quotient = arrays["shift_quotient_log_growth"]
    if has_cell:
        assert np.allclose(shift_quotient[-1, 0], np.log(0.8), atol=1e-10)
    else:
        assert np.allclose(shift_quotient[:, 0], np.log(0.5), atol=1e-10)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_shift_quotient_outputs_smoke(model_name):
    model = _tiny_model(model_name)
    step = ca_lyapunov.BlockStep(model)
    arrays, _ = ca_lyapunov.rollout_lyapunov(
        step, _rows(3, 8), max_depth=5, num_vectors=4, tangent_seed=0,
    )

    assert arrays["shift_quotient_log_growth"].shape == (5, 3, 4)
    assert ((arrays["top_vector_shift_fraction"] >= 0) & (arrays["top_vector_shift_fraction"] <= 1)).all()
    summary = ca_lyapunov.summarise(arrays, ((1, 5),), max_depth=5)
    assert "shift_quotient" in summary["windows"]["1-5"]
    assert len(summary["per_depth"]["shift_quotient_top_local_log_growth"]) == 5
    assert len(summary["per_depth"]["top_vector_shift_fraction"]) == 5


# BUT with input injection: the state is (x, e) with e constant and x_0 = 0.


def test_input_injection_rollout_matches_model_repeat_states_and_logits():
    model = _tiny_model("but_input_inject")
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)
    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=5, return_repeat_states=True)
        state = step.initial_state(rows)
        assert state.shape[1] == 2
        assert torch.equal(state[:, 0], torch.zeros_like(state[:, 0]))
        # e is the state entering the model's first repeat.
        assert torch.equal(state[:, 1], outputs["repeat_states"][0])
        for depth in range(1, 6):
            state = step(state)
            assert torch.equal(state[:, 0], outputs["repeat_states"][depth])
            assert torch.equal(state[:, 1], outputs["repeat_states"][0])
        assert torch.equal(step.decode(state), outputs["logits"].argmax(dim=-1))


def test_input_injection_jvp_is_the_jacobian_in_x_with_e_held_fixed():
    model = _tiny_model("but_input_inject")
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(1, 8), depth=2)
    dim = state[0, 0].numel()
    tangents = torch.randn(1, dim, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64)

    images = ca_lyapunov.jvp_columns(step, state, tangents)

    assert images.shape == (1, dim, 3)
    h = 1e-6
    for column in range(3):
        direction = torch.zeros_like(state)
        direction[:, 0] = tangents[:, :, column].reshape(state[:, 0].shape)
        with torch.no_grad():
            difference = (step(state + h * direction) - step(state - h * direction)) / (2 * h)
        torch.testing.assert_close(images[:, :, column], difference[:, 0].reshape(1, -1),
                                   rtol=1e-6, atol=1e-7)


def test_input_injection_full_jacobian_matches_autograd_in_x():
    model = _tiny_model("but_input_inject", n_embd=8, cells=4)
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(1, 4), depth=2)
    injected = state[:, 1]

    jacobian = ca_lyapunov.full_jacobian(step, state, chunk_size=16)
    reference = torch.autograd.functional.jacobian(
        lambda flat: step(torch.stack((flat.reshape(injected.shape), injected), dim=1))[:, 0].reshape(-1),
        state[:, 0].reshape(-1),
    )
    torch.testing.assert_close(jacobian, reference, rtol=1e-10, atol=1e-12)


def test_input_injection_examples_do_not_interact():
    model = _tiny_model("but_input_inject")
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(3, 8), depth=1)
    dim = state[0, 0].numel()
    tangents = torch.randn(3, dim, 2, generator=torch.Generator().manual_seed(5), dtype=torch.float64)

    batched = ca_lyapunov.jvp_columns(step, state, tangents)
    alone = ca_lyapunov.jvp_columns(step, state[1:2], tangents[1:2])

    torch.testing.assert_close(batched[1:2], alone, rtol=1e-12, atol=1e-12)


def test_input_injection_rollout_and_summary_smoke():
    model = _tiny_model("but_input_inject")
    step = ca_lyapunov.BlockStep(model)
    arrays, checks = ca_lyapunov.rollout_lyapunov(
        step, _rows(3, 8), max_depth=5, num_vectors=4, tangent_seed=0,
        jacobian_check_depths=(2,), jacobian_check_rows=1,
    )

    assert arrays["full_log_growth"].shape == (5, 3, 4)
    assert np.isfinite(arrays["full_log_growth"]).all()
    assert np.isfinite(arrays["shift_quotient_log_growth"]).all()
    assert "cell_to_cell_log_growth" not in arrays
    # The radial direction x / |x| is undefined at x_0 = 0 only.
    assert np.isnan(arrays["radial_log_growth"][0]).all()
    assert np.isfinite(arrays["radial_log_growth"][1:]).all()
    assert checks[0]["full_local_log_growth_top"] <= checks[0]["log_top_singular_values"][0] + 1e-9
    summary = ca_lyapunov.summarise(arrays, ((1, 2), (3, 5)), max_depth=5)
    assert np.isfinite(summary["windows"]["1-2"]["radial_log_growth_mean"])


# BUT with FPRM residual scaling: state (h, e), e constant, h_0 = e.


def _fprm_model(**kwargs):
    model = _tiny_model("fprm_but", **kwargs)
    # Distinct per-channel scales, so a wrong broadcast or a swapped alpha shows.
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        model.fprm_alpha_1_logit.copy_(torch.randn(model.fprm_alpha_1_logit.shape, generator=generator))
        model.fprm_alpha_2_logit.copy_(torch.randn(model.fprm_alpha_2_logit.shape, generator=generator))
    return model


def test_fprm_rollout_matches_model_repeat_states_and_logits():
    model = _fprm_model()
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)
    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=5, return_repeat_states=True)
        state = step.initial_state(rows)
        assert state.shape[1] == 2 and step.num_static_components == 1
        # h_0 = e = the state entering the model's first repeat.
        assert torch.equal(state[:, 0], outputs["repeat_states"][0])
        assert torch.equal(state[:, 1], outputs["repeat_states"][0])
        for depth in range(1, 6):
            state = step(state)
            assert torch.equal(state[:, 0], outputs["repeat_states"][depth])
            assert torch.equal(state[:, 1], outputs["repeat_states"][0])
        assert torch.equal(step.decode(state), outputs["logits"].argmax(dim=-1))


def test_fprm_full_jacobian_matches_autograd_in_h_with_e_held_fixed():
    model = _fprm_model(n_embd=8, cells=4)
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(1, 4), depth=2)
    injected = state[:, 1]

    jacobian = ca_lyapunov.full_jacobian(step, state, chunk_size=16)
    reference = torch.autograd.functional.jacobian(
        lambda flat: step(torch.stack((flat.reshape(injected.shape), injected), dim=1))[:, 0].reshape(-1),
        state[:, 0].reshape(-1),
    )
    torch.testing.assert_close(jacobian, reference, rtol=1e-10, atol=1e-12)


def test_fprm_rollout_summary_and_scale_summary():
    model = _fprm_model()
    step = ca_lyapunov.BlockStep(model)
    arrays, checks = ca_lyapunov.rollout_lyapunov(
        step, _rows(3, 8), max_depth=5, num_vectors=4, tangent_seed=0,
        jacobian_check_depths=(2,), jacobian_check_rows=1,
    )
    # h_0 = e is not zero, so every quantity is defined from step 1 on.
    for name in ("full_log_growth", "shift_quotient_log_growth", "radial_log_growth"):
        assert np.isfinite(arrays[name]).all(), name
    assert "cell_to_cell_log_growth" not in arrays
    assert checks[0]["full_local_log_growth_top"] <= checks[0]["log_top_singular_values"][0] + 1e-9

    scales = ca_lyapunov.fprm_residual_scale_summary(model)
    alpha_1, _, alpha_2, _ = (v.double() for v in model.residual_scales())
    np.testing.assert_allclose(scales["stream_kept_per_repeat"]["per_channel"],
                               (alpha_2 * alpha_1 ** 2).detach().numpy(), rtol=1e-6)
    assert scales["alpha_1"]["min"] <= scales["alpha_1"]["mean"] <= scales["alpha_1"]["max"]


def test_fprm_without_injection_rollout_matches_model():
    model = _tiny_model("fprm_but", extra_args=("--no-fprm_input_injection",))
    assert model.input_injection is False
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)
    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=5, return_repeat_states=True)
        state = step.initial_state(rows)
        for depth in range(1, 6):
            state = step(state)
            assert torch.equal(state[:, 0], outputs["repeat_states"][depth])
        assert torch.equal(step.decode(state), outputs["logits"].argmax(dim=-1))


# BUT with a fixed stream decay: state x only, x_t = block(x) - (1 - lambda) x.


@pytest.mark.parametrize("decay", ("0.5", "1.0"))
def test_decay_but_rollout_matches_model_repeat_states_and_logits(decay):
    model = _tiny_model("decay_but", extra_args=("--but_stream_decay", decay))
    assert model.stream_decay == float(decay)
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)
    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=5, return_repeat_states=True)
        state = step.initial_state(rows)
        assert state.shape[1] == 1
        for depth in range(1, 6):
            state = step(state)
            assert torch.equal(state[:, 0], outputs["repeat_states"][depth])
        assert torch.equal(step.decode(state), outputs["logits"].argmax(dim=-1))


def test_decay_but_full_jacobian_matches_autograd():
    model = _tiny_model("decay_but", n_embd=8, cells=4, extra_args=("--but_stream_decay", "0.5"))
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(1, 4), depth=2)

    jacobian = ca_lyapunov.full_jacobian(step, state, chunk_size=16)
    reference = torch.autograd.functional.jacobian(
        lambda flat: step(flat.reshape(state.shape)).reshape(-1), state.reshape(-1))
    torch.testing.assert_close(jacobian, reference, rtol=1e-10, atol=1e-12)


# Gated BUT with a fixed stream decay: state x only, driven as decay_but.


@pytest.mark.parametrize("decay", ("0.9", "1.0"))
def test_decay_gated_but_rollout_matches_model_repeat_states_and_logits(decay):
    model = _tiny_model("decay_gated_but", extra_args=("--but_stream_decay", decay))
    assert model.stream_decay == float(decay)
    step = ca_lyapunov.BlockStep(model)
    rows = _rows(3, 8)
    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=5, return_repeat_states=True)
        state = step.initial_state(rows)
        assert state.shape[1] == 1
        for depth in range(1, 6):
            state = step(state)
            assert torch.equal(state[:, 0], outputs["repeat_states"][depth])
        assert torch.equal(step.decode(state), outputs["logits"].argmax(dim=-1))


def test_decay_gated_but_full_jacobian_matches_autograd():
    model = _tiny_model("decay_gated_but", n_embd=8, cells=4, extra_args=("--but_stream_decay", "0.9"))
    step = ca_lyapunov.BlockStep(model)
    state = _state_after(step, _rows(1, 4), depth=2)

    jacobian = ca_lyapunov.full_jacobian(step, state, chunk_size=16)
    reference = torch.autograd.functional.jacobian(
        lambda flat: step(flat.reshape(state.shape)).reshape(-1), state.reshape(-1))
    torch.testing.assert_close(jacobian, reference, rtol=1e-10, atol=1e-12)


# LSTM-UT without a forget gate (--lstm_forget_gate none).


def test_no_forget_gate_rollout_matches_model_and_reports_no_forget_gate():
    model = _tiny_model("lstm_ut_bidir", extra_args=("--lstm_forget_gate", "none"))
    step = ca_lyapunov.BlockStep(model)
    assert step.block.forget_gate_mode == "none"
    rows = _rows(3, 8)
    with torch.no_grad():
        outputs = model(rows, get_logits=True, return_all_logits=True,
                        num_repeats=4, return_repeat_states=True)
        state = step.initial_state(rows)
        for depth in range(1, 5):
            state, forget_gate = step.step_with_forget_gate(state)
            assert forget_gate is None
            torch.testing.assert_close(state[:, 0], outputs["repeat_states"][depth], rtol=0, atol=1e-12)
        assert torch.equal(step.decode(state), outputs["logits"].argmax(dim=-1))

    arrays, _ = ca_lyapunov.rollout_lyapunov(step, rows, max_depth=4, num_vectors=4, tangent_seed=0)
    assert np.isnan(arrays["forget_mean"]).all()
    assert np.isfinite(arrays["cell_to_cell_log_growth"]).all()
    summary = ca_lyapunov.summarise(arrays, ((1, 4),), max_depth=4)
    assert "cell_to_cell" in summary["windows"]["1-4"]
