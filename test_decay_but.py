"""Tests for BUT with a fixed stream decay (models/decay_but.py) and for the
no-re-injection switch of models/fprm_but.py.

Intended behaviour:
  decay_but:  x_t = lambda * x_{t-1} + Delta_t, Delta_t = block(x_{t-1}) - x_{t-1},
              lambda fixed; lambda = 1 is exactly BUT.
  fprm_but --no-fprm_input_injection: h = alpha_2 * h before each repeat
              (no "+ beta_2 * e"), betas unchanged.
These check software correctness only.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from cellular_automaton.ca_eval import evaluate_ca_repeat_horizon_diagnostics
from cellular_automaton.ca_forward import CAForwardContext, CAForwardPolicy
from cellular_automaton.ca_gen import Rule30Dataset
from models import MODELS, but_full_depth, decay_but, fprm_but

D = 32


class _Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, *_args):
        return torch.full_like(x, self.value)


def _args(model_name, **extra):
    return SimpleNamespace(
        model=model_name, vocab_size=2, sequence_length=16, n_repeat=4, n_embd=D,
        n_head=4, n_layer=1, n_layer_begin=0, n_layer_end=0, bias=False, dropout=0.0,
        attention_mode="bidirectional", attention_window_length=None,
        lm_cache="none", positional_encoder="rotary", **extra,
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    for module in ("but_full_depth", "decay_but", "fprm_but"):
        monkeypatch.setattr(f"models.{module}.tiktoken.get_encoding", lambda _: None)


def _model(module, name, seed=7, **extra):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return module.GPTBase(_args(name, **extra)).eval()


def test_registered_and_same_parameters_as_but():
    assert MODELS["decay_but"] is decay_but.GPTBase
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(_model(decay_but, "decay_but", but_stream_decay=0.5)) == count(_model(but_full_depth, "but_full_depth"))


@pytest.mark.parametrize("bad", (0.0, -0.1, 1.5))
def test_decay_outside_unit_interval_is_rejected(bad):
    with pytest.raises(ValueError):
        _model(decay_but, "decay_but", but_stream_decay=bad)


def test_unit_decay_is_exactly_but():
    but = _model(but_full_depth, "but_full_depth")
    decay = _model(decay_but, "decay_but", seed=99, but_stream_decay=1.0)
    decay.load_state_dict(but.state_dict())
    idx = torch.randint(0, 2, (3, 12))
    with torch.no_grad():
        for repeats in (1, 3, 6):
            expected = but(idx, get_logits=True, return_all_logits=True, num_repeats=repeats)["logits"]
            got = decay(idx, get_logits=True, return_all_logits=True, num_repeats=repeats)["logits"]
            assert torch.equal(got, expected)


@pytest.mark.parametrize("lam", (0.5, 0.9))
def test_each_repeat_is_lambda_times_carried_plus_the_block_write(lam):
    model = _model(decay_but, "decay_but", but_stream_decay=lam)
    block = model.transformer.h_mid[0]
    block.attn = _Constant(0.3)
    block.mlp = _Constant(-0.1)  # Delta = 0.2 at every repeat
    idx = torch.randint(0, 2, (2, 10))
    with torch.no_grad():
        states = model(idx, num_repeats=4, return_repeat_states=True)["repeat_states"]
    for t in range(1, 5):
        assert torch.allclose(states[t], lam * states[t - 1] + 0.2, atol=1e-6)


def test_decay_but_fast_readout_matches_independent_forwards():
    model = _model(decay_but, "decay_but", but_stream_decay=0.5)
    dataloader = DataLoader(Rule30Dataset(num_samples=4, num_cells=12, seed=33), batch_size=2, shuffle=False)
    inputs = next(iter(dataloader))["input_id"]
    with torch.no_grad():
        states = model(inputs, get_logits=True, return_all_logits=True,
                       num_repeats=4, return_repeat_states=True)["repeat_states"]
        for repeats in range(1, 5):
            direct = model(inputs, get_logits=True, return_all_logits=True, num_repeats=repeats)["logits"]
            assert torch.equal(model.lm_head(model.transformer.ln_f(states[repeats])), direct)
    context = CAForwardContext(model, CAForwardPolicy(repeat_cache_window=None))
    options = dict(max_repeats=4, target_horizons=range(5), num_examples=1)
    slow = evaluate_ca_repeat_horizon_diagnostics(context, dataloader, "cpu", **options)
    fast = evaluate_ca_repeat_horizon_diagnostics(context, dataloader, "cpu", fast_lstm_readout=True, **options)
    assert repr(fast) == repr(slow)


def test_fprm_without_injection_follows_its_formula_and_keeps_the_betas():
    model = _model(fprm_but, "fprm_but", fprm_input_injection=False)
    reference = _model(fprm_but, "fprm_but")
    assert model.input_injection is False and reference.input_injection is True
    for a, b in zip(model.residual_scales(), reference.residual_scales()):
        assert torch.equal(a, b)  # betas still tied to the alphas as in FPRM
    with torch.no_grad():
        model.fprm_alpha_1_logit.copy_(torch.randn(D))
        model.fprm_alpha_2_logit.copy_(torch.randn(D))
    block = model.transformer.h_mid[0]
    block.attn = _Constant(0.3)
    block.mlp = _Constant(-0.1)
    idx = torch.randint(0, 2, (2, 10))
    with torch.no_grad():
        states = model(idx, num_repeats=3, return_repeat_states=True)["repeat_states"]
        alpha_1, beta_1, alpha_2, _ = model.residual_scales()
        h = states[0]
        for t in (1, 2, 3):
            h = alpha_2 * h  # no "+ beta_2 * e"
            h = alpha_1 * h + beta_1 * 0.3
            h = alpha_1 * h + beta_1 * -0.1
            assert torch.allclose(states[t], h, atol=1e-6)
