"""Focused wiring tests for the LSTM Universal Transformer ablations.

These tests distinguish parser/configuration wiring from the numerical block
semantics.  The latter use deterministic projections so a test cannot pass
merely because two randomly initialized model variants happen to differ.
"""

import math
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from cellular_automaton import ca_main, dca_main
from models import dca_lstm_ut, lstm_ut_bidir


MODEL_CASES = (
    pytest.param(ca_main, lstm_ut_bidir, "lstm_ut_bidir", id="ca"),
    pytest.param(dca_main, dca_lstm_ut, "dca_lstm_ut", id="dca"),
)


class _NoCache:
    """Minimal cache interface needed to construct an isolated block."""

    def get_storage_for_layer(self, _layer):
        return None


class _ConstantAttention(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(
        self,
        x,
        _pos_emb_closure,
        _cache_context,
        _start_index,
    ):
        return torch.full_like(x, self.value)


class _ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _RecordingProjection(nn.Module):
    """Return zero while retaining the tensor supplied to one gate branch."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return torch.zeros_like(x)


class _ConstantProjection(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


def _block_config(*, forget_gate, control_input):
    return SimpleNamespace(
        n_embd=4,
        n_head=1,
        bias=False,
        dropout=0.0,
        attention_window_length=None,
        attention_mode="bidirectional",
        sequence_length=3,
        allow_cache_during_training=False,
        lstm_forget_gate=forget_gate,
        lstm_control_input=control_input,
    )


def _make_block(model_module, *, forget_gate, control_input):
    return model_module.Block(
        _block_config(
            forget_gate=forget_gate,
            control_input=control_input,
        ),
        _NoCache(),
    )


def _replace_hidden_path(block, *, attention_value):
    """Make previous and proposed hidden states analytically observable."""
    block.ln_1 = nn.Identity()
    block.ln_2 = nn.Identity()
    block.memory_gate_norm = nn.Identity()
    block.attn = _ConstantAttention(attention_value)
    block.mlp = _ZeroMLP()


@pytest.mark.parametrize(
    "main_module,model_module,model_name",
    MODEL_CASES,
)
@pytest.mark.parametrize(
    "forget_gate,control_input",
    (
        ("learned", "previous_and_proposed"),
        ("learned", "proposed"),
        ("none", "previous_and_proposed"),
        ("none", "proposed"),
    ),
)
def test_runner_arguments_reach_the_selected_lstm_block(
    main_module,
    model_module,
    model_name,
    forget_gate,
    control_input,
):
    """Both CA parsers must carry their CLI values into model construction."""
    argv = [
        str(main_module.__file__),
        "--model",
        model_name,
        "--n_embd",
        "4",
        "--n_head",
        "1",
        "--lstm_forget_gate",
        forget_gate,
        "--lstm_control_input",
        control_input,
    ]

    with mock.patch.object(sys, "argv", argv):
        args = main_module.get_args()

    block = model_module.Block(args, _NoCache())

    assert args.lstm_forget_gate == forget_gate
    assert args.lstm_control_input == control_input
    assert block.forget_gate_mode == forget_gate
    assert block.control_input == control_input

    expect_previous = control_input == "previous_and_proposed"
    for name in ("write", "memory_proposal", "hidden"):
        assert (getattr(block, f"previous_to_{name}") is not None) is expect_previous

    assert (block.previous_to_forget is not None) is (
        expect_previous and forget_gate == "learned"
    )
    assert (block.proposed_to_forget is not None) is (
        forget_gate == "learned"
    )


@pytest.mark.parametrize("model_module", (lstm_ut_bidir, dca_lstm_ut))
@pytest.mark.parametrize(
    "control_input",
    ("previous_and_proposed", "proposed"),
)
def test_every_gate_receives_the_configured_control_inputs(
    model_module,
    control_input,
):
    """Exercise the actual Block.forward control-projection calls."""
    block = _make_block(
        model_module,
        forget_gate="learned",
        control_input=control_input,
    )
    _replace_hidden_path(block, attention_value=0.75)

    projection_names = ("forget", "write", "memory_proposal", "hidden")
    proposed_recorders = {}
    previous_recorders = {}

    for name in projection_names:
        proposed = _RecordingProjection()
        setattr(block, f"proposed_to_{name}", proposed)
        proposed_recorders[name] = proposed

        if control_input == "previous_and_proposed":
            previous = _RecordingProjection()
            setattr(block, f"previous_to_{name}", previous)
            previous_recorders[name] = previous
        else:
            assert getattr(block, f"previous_to_{name}") is None

    x = torch.tensor(
        [[[-1.0, -0.5, 0.25, 1.0], [0.5, -0.25, 1.5, -1.0]]]
    )
    cell = torch.zeros_like(x)

    block(
        x,
        cell,
        pos_emb_closure=None,
        cache_context=None,
        start_index=0,
    )

    previous_hidden = x
    proposed_hidden = x + 0.75

    for name, recorder in proposed_recorders.items():
        assert len(recorder.inputs) == 1, name
        torch.testing.assert_close(recorder.inputs[0], proposed_hidden)

    if control_input == "previous_and_proposed":
        for name, recorder in previous_recorders.items():
            assert len(recorder.inputs) == 1, name
            torch.testing.assert_close(recorder.inputs[0], previous_hidden)


@pytest.mark.parametrize("model_module", (lstm_ut_bidir, dca_lstm_ut))
@pytest.mark.parametrize(
    "forget_gate,expected_retention",
    (("learned", 0.5), ("none", 1.0)),
)
def test_forget_gate_ablation_changes_only_the_retention_term(
    model_module,
    forget_gate,
    expected_retention,
):
    """No-forget mode must implement cell + write, not a fixed gate value."""
    block = _make_block(
        model_module,
        forget_gate=forget_gate,
        control_input="proposed",
    )
    _replace_hidden_path(block, attention_value=0.0)

    if forget_gate == "learned":
        # sigmoid(0) gives a known 0.5 retention gate.
        block.proposed_to_forget = _ConstantProjection(0.0)

    # sigmoid(0) = 0.5 and tanh(atanh(0.4)) = 0.4, giving a 0.2 write.
    block.proposed_to_write = _ConstantProjection(0.0)
    block.proposed_to_memory_proposal = _ConstantProjection(math.atanh(0.4))
    block.proposed_to_hidden = _ConstantProjection(0.0)

    x = torch.zeros(1, 2, 4)
    initial_cell = torch.tensor(
        [[[1.0, -1.0, 0.5, -0.5], [0.25, 0.75, -0.25, -0.75]]]
    )

    _, next_cell = block(
        x,
        initial_cell,
        pos_emb_closure=None,
        cache_context=None,
        start_index=0,
    )

    expected_cell = expected_retention * initial_cell + 0.2
    torch.testing.assert_close(next_cell, expected_cell)


def _dca_model_config():
    return SimpleNamespace(
        model="dca_lstm_ut",
        vocab_size=2,
        sequence_length=4,
        n_embd=4,
        n_head=1,
        n_layer=1,
        n_layer_begin=0,
        n_layer_end=0,
        n_repeat=3,
        min_repeat=1,
        depth_random_method="uniform",
        bias=False,
        dropout=0.0,
        attention_window_length=None,
        attention_mode="bidirectional",
        positional_encoder="rotary",
        lm_cache="none",
        allow_cache_during_training=False,
        lstm_forget_gate="none",
        lstm_control_input="proposed",
        ca_max_relative_age=3,
        ca_controller_application="subtract",
    )


def test_dca_forward_returns_only_requested_memory_snapshots():
    """Smoke-test evolution and recall snapshot selection on the full model."""
    with mock.patch.object(
        dca_lstm_ut.tiktoken,
        "get_encoding",
        return_value=None,
    ):
        model = dca_lstm_ut.GPTBase(_dca_model_config()).eval()

    inputs = torch.tensor(
        [[0, 1, 0, 1], [1, 1, 0, 0]],
        dtype=torch.long,
    )

    outputs = model(
        inputs,
        get_logits=True,
        return_all_logits=True,
        num_repeats=3,
        delayed_recall=True,
        recall_age=1,
        num_recall_repeats=2,
        return_memory_states=True,
        memory_evolution_repeats=(1, 3),
        memory_recall_steps=(2,),
    )

    memory_states = outputs["memory_states"]
    assert set(memory_states) == {"evolution", "recall"}
    assert set(memory_states["evolution"]) == {1, 3}
    assert set(memory_states["recall"]) == {2}

    for snapshot in (
        *memory_states["evolution"].values(),
        *memory_states["recall"].values(),
    ):
        assert snapshot.shape == (2, 4, 4)
        assert not snapshot.requires_grad
        assert torch.isfinite(snapshot).all()

    assert outputs["logits"].shape == (2, 4, 2)


def test_dca_forward_does_not_return_memory_by_default():
    """The diagnostic must not alter the default training/evaluation output."""
    with mock.patch.object(
        dca_lstm_ut.tiktoken,
        "get_encoding",
        return_value=None,
    ):
        model = dca_lstm_ut.GPTBase(_dca_model_config()).eval()

    outputs = model(
        torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
        get_logits=True,
        return_all_logits=True,
        num_repeats=2,
    )

    assert "memory_states" not in outputs
