"""Tests for the fixed-decay linear-memory BUT ablation (models/linmem_but.py).

Intended behaviour: BUT plus a separate memory,
    c = lambda * c + W_prev LN_m(x) + W_prop LN_m(x_block),   x = x_block + c,
with lambda = sigmoid(logit) a learned constant per channel. Checked here:
  - with the memory's write zeroed, the model is exactly BUT (no unintended
    change to the copied BUT code);
  - one block step follows the formula above, and close_hidden_gate skips
    only the read;
  - parameter count and optimiser groups (the decay logit is not decayed);
  - the fast per-repeat readout and the Lyapunov BlockStep drive it exactly
    as its own forward does, and the reported "forget gate" is lambda.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from cellular_automaton import ca_lyapunov
from cellular_automaton.ca_eval import evaluate_ca_repeat_horizon_diagnostics
from cellular_automaton.ca_forward import CAForwardContext, CAForwardPolicy
from cellular_automaton.ca_gen import Rule30Dataset
from models import MODELS, but_full_depth, linmem_but

D = 32


class _NoCache:
    def get_storage_for_layer(self, _layer):
        return None


class _Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, *_args):
        return torch.full_like(x, self.value)


def _args(model_name, control_input="previous_and_proposed"):
    return SimpleNamespace(
        model=model_name, vocab_size=2, sequence_length=16, n_repeat=4, n_embd=D,
        n_head=4, n_layer=1, n_layer_begin=0, n_layer_end=0, bias=False, dropout=0.0,
        attention_mode="bidirectional", attention_window_length=None,
        lstm_forget_gate="learned", lstm_control_input=control_input,
        lstm_initial_cell="zero", lm_cache="none", positional_encoder="rotary",
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    for module in ("but_full_depth", "linmem_but"):
        monkeypatch.setattr(f"models.{module}.tiktoken.get_encoding", lambda _: None)


def _model(module, name, seed=7, **kwargs):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return module.GPTBase(_args(name, **kwargs)).eval()


def _count(model):
    return sum(p.numel() for p in model.parameters())


def test_registered():
    assert MODELS["linmem_but"] is linmem_but.GPTBase


def test_parameter_count_is_but_plus_memory():
    but = _count(_model(but_full_depth, "but_full_depth"))
    # W_prev, W_prop (no bias), LN_m weight and bias, decay logit.
    assert _count(_model(linmem_but, "linmem_but")) == but + 2 * D * D + 2 * D + D
    assert _count(_model(linmem_but, "linmem_but", control_input="proposed")) == but + D * D + 2 * D + D


def test_decay_starts_at_one_half_and_is_not_weight_decayed():
    model = _model(linmem_but, "linmem_but")
    assert torch.equal(model.transformer.h_mid[0].memory_decay(), torch.full((D,), 0.5))
    decay_group, no_decay_group = model.get_parameter_group_specs()
    assert "transformer.h_mid.0.memory_decay_logit" in no_decay_group["params"]
    assert "transformer.h_mid.0.memory_decay_logit" not in decay_group["params"]
    assert "transformer.h_mid.0.proposed_to_memory_proposal.weight" in decay_group["params"]


def test_zero_write_is_exactly_but():
    but = _model(but_full_depth, "but_full_depth")
    linmem = _model(linmem_but, "linmem_but", seed=99)
    missing, unexpected = linmem.load_state_dict(but.state_dict(), strict=False)
    assert not unexpected
    assert all("memory" in name for name in missing)
    block = linmem.transformer.h_mid[0]
    with torch.no_grad():
        block.previous_to_memory_proposal.weight.zero_()
        block.proposed_to_memory_proposal.weight.zero_()
        block.memory_decay_logit.fill_(3.0)  # any decay: the cell stays 0

    idx = torch.randint(0, 2, (3, 12))
    with torch.no_grad():
        for repeats in (1, 3, 6):
            expected = but(idx, get_logits=True, return_all_logits=True, num_repeats=repeats)["logits"]
            got = linmem(idx, get_logits=True, return_all_logits=True, num_repeats=repeats)["logits"]
            assert torch.equal(got, expected)


@pytest.mark.parametrize("control_input", ("previous_and_proposed", "proposed"))
def test_block_step_follows_the_linear_memory_formula(control_input):
    torch.manual_seed(0)
    block = linmem_but.Block(_args("linmem_but", control_input), _NoCache())
    block.attn = _Constant(0.3)
    block.mlp = _Constant(-0.1)
    with torch.no_grad():
        block.memory_decay_logit.copy_(torch.randn(D))

    x = torch.randn(2, 5, D)
    cell = 3.0 * torch.randn(2, 5, D)
    x_new, cell_new = block(x, cell, None, None, 0)
    x_closed, cell_closed = block(x, cell, None, None, 0, close_hidden_gate=True)

    x_block = x + 0.3 - 0.1
    write = block.proposed_to_memory_proposal(block.memory_gate_norm(x_block))
    if control_input == "previous_and_proposed":
        write = write + block.previous_to_memory_proposal(block.memory_gate_norm(x))
    else:
        assert block.previous_to_memory_proposal is None
    expected_cell = torch.sigmoid(block.memory_decay_logit) * cell + write

    assert torch.allclose(cell_new, expected_cell, atol=1e-6)
    assert torch.allclose(x_new, x_block + expected_cell, atol=1e-6)
    # Closing the read updates the cell identically and leaves x = x_block.
    assert torch.equal(cell_closed, cell_new)
    assert torch.allclose(x_closed, x_block)


def test_fast_readout_matches_independent_forwards():
    model = _model(linmem_but, "linmem_but")
    dataloader = DataLoader(Rule30Dataset(num_samples=4, num_cells=12, seed=33), batch_size=2, shuffle=False)
    inputs = next(iter(dataloader))["input_id"]
    max_repeats = 4
    with torch.no_grad():
        states = model(inputs, get_logits=True, return_all_logits=True,
                       num_repeats=max_repeats, return_repeat_states=True)["repeat_states"]
        for repeats in range(1, max_repeats + 1):
            direct = model(inputs, get_logits=True, return_all_logits=True, num_repeats=repeats)["logits"]
            assert torch.equal(model.lm_head(model.transformer.ln_f(states[repeats])), direct)

    context = CAForwardContext(model, CAForwardPolicy(repeat_cache_window=None))
    options = dict(max_repeats=max_repeats, target_horizons=range(max_repeats + 1), num_examples=1)
    slow = evaluate_ca_repeat_horizon_diagnostics(context, dataloader, "cpu", **options)
    fast = evaluate_ca_repeat_horizon_diagnostics(context, dataloader, "cpu", fast_lstm_readout=True, **options)
    assert repr(fast) == repr(slow)


def test_lyapunov_block_step_matches_forward_and_reports_lambda():
    model = _model(linmem_but, "linmem_but")
    block = model.transformer.h_mid[0]
    with torch.no_grad():
        block.memory_decay_logit.copy_(torch.linspace(-2, 3, D))
    step = ca_lyapunov.BlockStep(model)
    assert step.has_cell and step.num_components == 2

    rows = torch.randint(0, 2, (3, 12))
    with torch.no_grad():
        states = model(rows, num_repeats=3, return_repeat_states=True)["repeat_states"]
        state = step.initial_state(rows)
        for repeat in range(1, 4):
            state, decay = step.step_with_forget_gate(state)
            assert torch.allclose(state[:, 0], states[repeat], atol=1e-6)
            assert decay.shape == state[:, 1].shape
            assert torch.equal(decay[0, 0], block.memory_decay())
        # The closed step is the block's own intervention.
        closed = step(state, close_hidden_gate=True)
        x_block = closed[:, 0]
        opened = step(state)
        assert torch.allclose(opened[:, 0], x_block + opened[:, 1], atol=1e-6)
