"""Tests for the rebuttal ablations (Nov 2026).

- lin_lstm_ut_bidir: LSTM-UT with a linear cell read, x + o * c, instead of
  x + o * tanh(c). Everything else must match lstm_ut_bidir.
- but_input_inject: BUT with additive input injection,
  x_1 = block(e), x_t = block(x_{t-1} + e) for t >= 2.

The block-level tests replace attention and the MLP with constants so the
read is checked exactly. The fast per-repeat readout must give the same
numbers as independent forwards, as it does for the existing models.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from cellular_automaton.ca_eval import evaluate_ca_repeat_horizon_diagnostics
from cellular_automaton.ca_forward import CAForwardContext, CAForwardPolicy
from cellular_automaton.ca_gen import Rule30Dataset
from models import (
    MODELS,
    but_full_depth,
    but_input_inject,
    lin_lstm_ut_bidir,
    lstm_ut_bidir,
)


class _NoCache:
    def get_storage_for_layer(self, _layer):
        return None


class _ConstantAttention(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, _pos_emb_closure, _cache_context, _start_index):
        return torch.full_like(x, self.value)


class _ConstantMLP(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


def _args(model_name, *, forget_gate="learned", n_embd=32):
    return SimpleNamespace(
        model=model_name,
        vocab_size=2,
        sequence_length=16,
        n_repeat=4,
        n_embd=n_embd,
        n_head=4,
        n_layer=1,
        n_layer_begin=0,
        n_layer_end=0,
        bias=False,
        dropout=0.0,
        attention_mode="bidirectional",
        attention_window_length=None,
        lstm_forget_gate=forget_gate,
        lstm_control_input="previous_and_proposed",
        lstm_initial_cell="zero",
        lm_cache="none",
        positional_encoder="rotary",
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    for module in ("lstm_ut_bidir", "lin_lstm_ut_bidir", "but_full_depth", "but_input_inject"):
        monkeypatch.setattr(f"models.{module}.tiktoken.get_encoding", lambda _: None)


def _model(module, name, seed=7, **kwargs):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return module.GPTBase(_args(name, **kwargs)).eval()


def _parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def test_ablations_are_registered():
    assert MODELS["lin_lstm_ut_bidir"] is lin_lstm_ut_bidir.GPTBase
    assert MODELS["but_input_inject"] is but_input_inject.GPTBase


def test_ablations_add_no_parameters():
    assert _parameter_count(_model(lin_lstm_ut_bidir, "lin_lstm_ut_bidir")) == (
        _parameter_count(_model(lstm_ut_bidir, "lstm_ut_bidir"))
    )
    assert _parameter_count(_model(but_input_inject, "but_input_inject")) == (
        _parameter_count(_model(but_full_depth, "but_full_depth"))
    )


@pytest.mark.parametrize("forget_gate", ("learned", "none"))
def test_linear_read_differs_from_lstm_ut_only_in_the_read(forget_gate):
    torch.manual_seed(0)
    lstm_block = lstm_ut_bidir.Block(_args("lstm_ut_bidir", forget_gate=forget_gate), _NoCache())
    lin_block = lin_lstm_ut_bidir.Block(_args("lin_lstm_ut_bidir", forget_gate=forget_gate), _NoCache())
    lin_block.load_state_dict(lstm_block.state_dict())
    for block in (lstm_block, lin_block):
        block.attn = _ConstantAttention(0.3)
        block.mlp = _ConstantMLP(-0.1)

    x = torch.randn(2, 5, 32)
    # A large cell makes tanh(c) and c clearly different.
    cell = 3.0 * torch.randn(2, 5, 32)

    x_lstm, cell_lstm = lstm_block(x, cell, None, None, 0)
    x_lin, cell_lin = lin_block(x, cell, None, None, 0)
    x_closed, cell_closed = lin_block(x, cell, None, None, 0, close_hidden_gate=True)

    # Cell update is identical; closing the gate skips only the read.
    assert torch.equal(cell_lin, cell_lstm)
    assert torch.equal(cell_closed, cell_lstm)
    assert torch.allclose(x_closed, x + 0.3 - 0.1)

    gate = torch.sigmoid(
        lin_block._control_projection(
            lin_block.memory_gate_norm(x),
            lin_block.memory_gate_norm(x_closed),
            lin_block.previous_to_hidden,
            lin_block.proposed_to_hidden,
        )
    )
    assert torch.allclose(x_lin, x_closed + gate * cell_lin, atol=1e-6)
    assert torch.allclose(x_lstm, x_closed + gate * torch.tanh(cell_lstm), atol=1e-6)
    assert not torch.allclose(x_lin, x_lstm)


def test_input_injection_matches_but_on_repeat_one_and_adds_input_after():
    but = _model(but_full_depth, "but_full_depth")
    inject = _model(but_input_inject, "but_input_inject", seed=99)
    inject.load_state_dict(but.state_dict())

    idx = torch.randint(0, 2, (3, 12))
    with torch.no_grad():
        but_states = but(idx, num_repeats=3, return_repeat_states=True)["repeat_states"]
        states = inject(idx, num_repeats=3, return_repeat_states=True)["repeat_states"]

        # Recompute repeats 2 and 3 by hand with the shared block.
        _, pos_emb_closure = inject.transformer.wpe(idx)
        block = inject.transformer.h_mid[0]
        expected_2 = block(states[1] + states[0], pos_emb_closure, None, start_index=0)
        expected_3 = block(states[2] + states[0], pos_emb_closure, None, start_index=0)

    assert len(states) == 4
    assert torch.equal(states[0], but_states[0])  # e, the embedded input row
    assert torch.equal(states[1], but_states[1])  # repeat 1 identical to BUT
    assert torch.allclose(states[2], expected_2, atol=1e-6)
    assert torch.allclose(states[3], expected_3, atol=1e-6)
    assert not torch.allclose(states[2], but_states[2])


@pytest.mark.parametrize(
    "module, name",
    ((lin_lstm_ut_bidir, "lin_lstm_ut_bidir"), (but_input_inject, "but_input_inject")),
)
def test_fast_readout_matches_independent_forwards(module, name):
    model = _model(module, name)
    dataloader = DataLoader(
        Rule30Dataset(num_samples=4, num_cells=12, seed=33),
        batch_size=2,
        shuffle=False,
    )
    inputs = next(iter(dataloader))["input_id"]
    max_repeats = 4
    with torch.no_grad():
        states = model(
            inputs, get_logits=True, return_all_logits=True,
            num_repeats=max_repeats, return_repeat_states=True,
        )["repeat_states"]
        for repeats in range(1, max_repeats + 1):
            direct = model(
                inputs, get_logits=True, return_all_logits=True, num_repeats=repeats,
            )["logits"]
            decoded = model.lm_head(model.transformer.ln_f(states[repeats]))
            assert torch.equal(decoded, direct)

    context = CAForwardContext(model, CAForwardPolicy(repeat_cache_window=None))
    options = dict(
        max_repeats=max_repeats,
        target_horizons=range(max_repeats + 1),
        num_examples=1,
    )
    slow = evaluate_ca_repeat_horizon_diagnostics(context, dataloader, "cpu", **options)
    fast = evaluate_ca_repeat_horizon_diagnostics(
        context, dataloader, "cpu", fast_lstm_readout=True, **options
    )
    assert repr(fast) == repr(slow)
