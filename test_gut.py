"""Tests for the GUT baseline (gut_bidir and dca_gut).

GUT is the BUT block with the Eq. 9-11 transition gate of Ray Chowdhury &
Caragea (arXiv 2402.00976), without halting. The analytic test replaces the
attention, MLP and gate with constants so the update rule is checked exactly
rather than inferred from two random models differing.
"""

import argparse
import sys
from unittest import mock

import pytest
import torch
from torch import nn

from cellular_automaton import ca_analyze, ca_main
from models import MODELS, but_full_depth, dca_gut, gut_bidir, lstm_ut_bidir


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


class _RecordingConstantGate(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.inputs = []

    def forward(self, attended):
        self.inputs.append(attended.detach().clone())
        return torch.full_like(attended, self.value)


def _ca_args(model_name, n_embd=64, extra=()):
    argv = [
        str(ca_main.__file__),
        "--model", model_name,
        "--n_embd", str(n_embd),
        "--n_head", "4",
        "--n_layer", "1",
        "--sequence_length", "16",
        "--vocab_size", "2",
        "--attention_mode", "bidirectional",
        *extra,
    ]
    with mock.patch.object(sys, "argv", argv):
        return ca_main.get_args()


def _dca_args(n_embd=16):
    # dca_main.get_args cannot be used here: it currently raises an argparse
    # conflict on --lstm_forget_gate (also defined in config/base.py).
    args = argparse.Namespace(**vars(_ca_args("dca_gut", n_embd=n_embd)))
    args.ca_max_relative_age = 4
    args.ca_controller_application = "subtract"
    return args


def _block_overhead(block_module, base_block, config):
    block = block_module.Block(config, _NoCache())
    base = base_block.Block(config, _NoCache())
    count = lambda module: sum(p.numel() for p in module.parameters())
    return count(block) - count(base)


@pytest.mark.parametrize("name,module", (("gut_bidir", gut_bidir), ("dca_gut", dca_gut)))
def test_gut_models_are_registered(name, module):
    assert MODELS[name] is module.GPTBase


@pytest.mark.parametrize("module", (gut_bidir, dca_gut))
def test_block_applies_eq_11_and_gate_reads_post_attention_state(module):
    args = _ca_args("gut_bidir", n_embd=4, extra=("--n_head", "1"))
    block = module.Block(args, _NoCache())
    block.ln_1 = nn.Identity()
    block.ln_2 = nn.Identity()
    block.attn = _ConstantAttention(0.5)
    block.mlp = _ZeroMLP()
    block.gate = _RecordingConstantGate(0.25)

    x = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    out = block(x, None, None, 0)

    # A = x + 0.5, H = A, so G*H + (1-G)*x = x + 0.25 * 0.5.
    torch.testing.assert_close(out, x + 0.125)
    torch.testing.assert_close(block.gate.inputs[0], x + 0.5)


def test_gate_overhead_is_near_matched_to_lstm_ut_at_d64():
    args = _ca_args("gut_bidir")
    assert _block_overhead(gut_bidir, but_full_depth, args) == 33_216
    assert _block_overhead(dca_gut, but_full_depth, args) == 33_216
    assert _block_overhead(lstm_ut_bidir, but_full_depth, args) == 32_896


def test_gate_starts_near_one_half_with_zero_biases():
    torch.manual_seed(0)
    model = gut_bidir.GPTBase(_ca_args("gut_bidir"))
    gate = model.transformer.h_mid[0].gate
    assert torch.count_nonzero(gate.fc_in.bias) == 0
    assert torch.count_nonzero(gate.fc_out.bias) == 0
    # The gate is outside the c_proj scaled-init rule.
    assert gate.fc_out.weight.std().item() == pytest.approx(0.02, rel=0.1)

    attended = torch.randn(8, 16, 64)
    assert gate(attended).mean().item() == pytest.approx(0.5, abs=0.02)


def test_gut_bidir_forward_returns_every_repeat_state():
    torch.manual_seed(0)
    model = gut_bidir.GPTBase(_ca_args("gut_bidir", n_embd=16)).eval()
    idx = torch.randint(0, 2, (2, 16))
    out = model(idx, get_logits=True, return_all_logits=True,
                num_repeats=3, return_repeat_states=True)
    assert out["logits"].shape == (2, 16, 2)
    assert len(out["repeat_states"]) == 4


def test_dca_gut_delayed_recall_forward_is_finite():
    torch.manual_seed(0)
    model = dca_gut.GPTBase(_dca_args()).eval()
    idx = torch.randint(0, 2, (2, 16))
    out = model(idx, targets=idx, get_logits=True, num_repeats=3,
                delayed_recall=True, recall_age=1, num_recall_repeats=2)
    assert torch.isfinite(out["loss"])


@pytest.mark.parametrize("name", ("gut_bidir", "dca_gut"))
def test_analysis_labels_gut_runs(name):
    manifest = {"model": {"model": name, "n_embd": 64}}
    assert ca_analyze._plot_configuration_label(manifest) == "GUT"
