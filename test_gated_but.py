"""Tests for the highway-gated BUT control (gated_but_bidir and dca_gated_but).

The control keeps LSTM-UT's gate inputs, LN(x_in) and LN(x_block), and its
bias-free projections, but gates the residual stream directly and has no cell:
x_out = G * x_block + (1 - G) * x_in. The analytic tests replace sub-modules
with constants so the update rule is checked exactly.
"""

import argparse
import sys
from unittest import mock

import pytest
import torch
from torch import nn

from cellular_automaton import ca_analyze, ca_main
from models import MODELS, but_full_depth, dca_gated_but, gated_but_bidir, lstm_ut_bidir

MODEL_MODULES = (gated_but_bidir, dca_gated_but)


class _NoCache:
    def get_storage_for_layer(self, _layer):
        return None


class _ConstantAttention(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, _pos_emb_closure, _cache_context, _start_index):
        return torch.full_like(x, self.value)


class _ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _RecordingConstantProjection(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return torch.full_like(x, self.value)


def _ca_args(model_name, n_embd=64, n_head=4):
    argv = [
        str(ca_main.__file__),
        "--model", model_name,
        "--n_embd", str(n_embd),
        "--n_head", str(n_head),
        "--n_layer", "1",
        "--sequence_length", "16",
        "--vocab_size", "2",
        "--attention_mode", "bidirectional",
    ]
    with mock.patch.object(sys, "argv", argv):
        return ca_main.get_args()


def _dca_args(n_embd=16):
    # dca_main.get_args cannot be used here: it currently raises an argparse
    # conflict on --lstm_forget_gate (also defined in config/base.py).
    args = argparse.Namespace(**vars(_ca_args("dca_gated_but", n_embd=n_embd)))
    args.ca_max_relative_age = 4
    args.ca_controller_application = "subtract"
    return args


def _parameter_count(module):
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize(
    "name,module",
    (("gated_but_bidir", gated_but_bidir), ("dca_gated_but", dca_gated_but)),
)
def test_gated_but_models_are_registered(name, module):
    assert MODELS[name] is module.GPTBase


@pytest.mark.parametrize("module", MODEL_MODULES)
def test_block_gates_residual_stream_from_previous_and_proposed_states(module):
    block = module.Block(_ca_args("gated_but_bidir", n_embd=4, n_head=1), _NoCache())
    block.ln_1 = nn.Identity()
    block.ln_2 = nn.Identity()
    block.gate_norm = nn.Identity()
    block.attn = _ConstantAttention(0.5)
    block.mlp = _ZeroMLP()
    block.previous_to_gate = _RecordingConstantProjection(0.3)
    block.proposed_to_gate = _RecordingConstantProjection(-0.1)

    x = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    out = block(x, None, None, 0)

    gate = torch.sigmoid(torch.tensor(0.2))
    # x_block = x + 0.5, so G * x_block + (1 - G) * x = x + 0.5 * G.
    torch.testing.assert_close(out, x + 0.5 * gate)
    torch.testing.assert_close(block.previous_to_gate.inputs[0], x)
    torch.testing.assert_close(block.proposed_to_gate.inputs[0], x + 0.5)


@pytest.mark.parametrize("module", MODEL_MODULES)
@pytest.mark.parametrize("gate_logit,expect_but", ((50.0, True), (-50.0, False)))
def test_saturated_gate_reduces_to_but_step_or_copy(module, gate_logit, expect_but):
    torch.manual_seed(0)
    args = _ca_args("gated_but_bidir", n_embd=8, n_head=2)
    block = module.Block(args, _NoCache()).eval()
    but_block = but_full_depth.Block(args, _NoCache()).eval()
    but_block.load_state_dict(block.state_dict(), strict=False)
    block.previous_to_gate = _RecordingConstantProjection(gate_logit)
    block.proposed_to_gate = _RecordingConstantProjection(0.0)

    model = but_full_depth.GPTBase(args)
    idx = torch.randint(0, 2, (2, 16))
    _, pos_emb_closure = model.transformer.wpe(idx)
    x = torch.randn(2, 16, 8)

    out = block(x, pos_emb_closure, None, 0)
    expected = but_block(x, pos_emb_closure, None, 0) if expect_but else x
    torch.testing.assert_close(out, expected)


def test_gate_overhead_is_two_bias_free_projections_and_one_norm():
    args = _ca_args("gated_but_bidir")
    base = _parameter_count(but_full_depth.Block(args, _NoCache()))
    for module in MODEL_MODULES:
        assert _parameter_count(module.Block(args, _NoCache())) - base == 8_320
    # LSTM-UT spends 32,896 on four gates and the same normalisation.
    assert _parameter_count(lstm_ut_bidir.Block(args, _NoCache())) - base == 32_896


def test_gate_starts_near_one_half():
    torch.manual_seed(0)
    model = gated_but_bidir.GPTBase(_ca_args("gated_but_bidir"))
    block = model.transformer.h_mid[0]
    assert block.previous_to_gate.bias is None
    assert block.proposed_to_gate.bias is None

    x = torch.randn(8, 16, 64)
    logits = block.previous_to_gate(block.gate_norm(x)) + block.proposed_to_gate(block.gate_norm(x))
    assert torch.sigmoid(logits).mean().item() == pytest.approx(0.5, abs=0.02)


def test_gated_but_bidir_forward_returns_every_repeat_state():
    torch.manual_seed(0)
    model = gated_but_bidir.GPTBase(_ca_args("gated_but_bidir", n_embd=16)).eval()
    idx = torch.randint(0, 2, (2, 16))
    out = model(idx, get_logits=True, return_all_logits=True,
                num_repeats=3, return_repeat_states=True)
    assert out["logits"].shape == (2, 16, 2)
    assert len(out["repeat_states"]) == 4


def test_dca_gated_but_delayed_recall_forward_is_finite():
    torch.manual_seed(0)
    model = dca_gated_but.GPTBase(_dca_args()).eval()
    idx = torch.randint(0, 2, (2, 16))
    out = model(idx, targets=idx, get_logits=True, num_repeats=3,
                delayed_recall=True, recall_age=1, num_recall_repeats=2)
    assert torch.isfinite(out["loss"])


@pytest.mark.parametrize("name", ("gated_but_bidir", "dca_gated_but"))
def test_analysis_labels_gated_but_runs(name):
    manifest = {"model": {"model": name, "n_embd": 64}}
    assert ca_analyze._plot_configuration_label(manifest) == "Gated BUT"
