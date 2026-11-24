"""Tests for BUT with FPRM residual scaling (models/fprm_but.py).

Intended behaviour (Movahedi et al., 2026; their models/transformer.py):
per repeat
    h = alpha_2 * h + beta_2 * e
    h = alpha_1 * h + beta_1 * attn(ln_1(h));  h = alpha_1 * h + beta_1 * mlp(ln_2(h))
with h_0 = e, learned per-channel alphas, and the betas tied to the alphas.
Checked here:
  - the tied betas match FPRM's formula and the alphas start at their inits;
  - with alpha_1 = beta_1 = alpha_2 = 1 and beta_2 = 0 the model is exactly
    BUT (no unintended change to the copied BUT code);
  - one repeat follows the formula above;
  - parameter count and optimiser groups (the alpha logits are not decayed);
  - the fast per-repeat readout matches independent forwards.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from cellular_automaton.ca_eval import evaluate_ca_repeat_horizon_diagnostics
from cellular_automaton.ca_forward import CAForwardContext, CAForwardPolicy
from cellular_automaton.ca_gen import Rule30Dataset
from models import MODELS, but_full_depth, fprm_but

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
    for module in ("but_full_depth", "fprm_but"):
        monkeypatch.setattr(f"models.{module}.tiktoken.get_encoding", lambda _: None)


def _model(module, name, seed=7, **extra):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return module.GPTBase(_args(name, **extra)).eval()


def _count(model):
    return sum(p.numel() for p in model.parameters())


def test_registered():
    assert MODELS["fprm_but"] is fprm_but.GPTBase


def test_parameter_count_is_but_plus_two_alpha_vectors():
    assert _count(_model(fprm_but, "fprm_but")) == _count(_model(but_full_depth, "but_full_depth")) + 2 * D


@pytest.mark.parametrize("inits", ((0.75, 0.25), (0.5, 0.5)))
def test_alphas_start_at_init_and_betas_follow_fprm(inits):
    model = _model(fprm_but, "fprm_but", fprm_alpha_1_init=inits[0], fprm_alpha_2_init=inits[1])
    alpha_1, beta_1, alpha_2, beta_2 = model.residual_scales()
    assert torch.allclose(alpha_1, torch.full((D,), inits[0]))
    assert torch.allclose(alpha_2, torch.full((D,), inits[1]))
    a1, a2 = inits
    # One looped block: 2L = 2 sub-layers (FPRM transformer.py).
    b2 = 1 - a2 * a1 ** 2
    b1 = b2 * (1 - a1) / (1 - a1 ** 2 + 1e-5)
    assert torch.allclose(beta_2, torch.full((D,), b2))
    assert torch.allclose(beta_1, torch.full((D,), b1))


def test_default_inits_and_alphas_not_weight_decayed():
    model = _model(fprm_but, "fprm_but")
    alpha_1, _, alpha_2, _ = model.residual_scales()
    assert torch.allclose(alpha_1, torch.full((D,), 0.75)) and torch.allclose(alpha_2, torch.full((D,), 0.25))
    decay_group, no_decay_group = model.get_parameter_group_specs()
    for name in ("fprm_alpha_1_logit", "fprm_alpha_2_logit"):
        assert name in no_decay_group["params"] and name not in decay_group["params"]
    assert "transformer.h_mid.0.mlp.c_fc.weight" in decay_group["params"]


def test_unit_scales_are_exactly_but():
    but = _model(but_full_depth, "but_full_depth")
    fprm = _model(fprm_but, "fprm_but", seed=99)
    missing, unexpected = fprm.load_state_dict(but.state_dict(), strict=False)
    assert not unexpected and set(missing) == {"fprm_alpha_1_logit", "fprm_alpha_2_logit"}
    ones = torch.ones(D)
    fprm.residual_scales = lambda: (ones, ones, ones, torch.zeros(D))

    idx = torch.randint(0, 2, (3, 12))
    with torch.no_grad():
        for repeats in (1, 3, 6):
            expected = but(idx, get_logits=True, return_all_logits=True, num_repeats=repeats)["logits"]
            got = fprm(idx, get_logits=True, return_all_logits=True, num_repeats=repeats)["logits"]
            assert torch.equal(got, expected)


def test_one_repeat_follows_the_fprm_formula():
    model = _model(fprm_but, "fprm_but")
    with torch.no_grad():
        model.fprm_alpha_1_logit.copy_(torch.randn(D))
        model.fprm_alpha_2_logit.copy_(torch.randn(D))
    block = model.transformer.h_mid[0]
    block.attn = _Constant(0.3)
    block.mlp = _Constant(-0.1)
    idx = torch.randint(0, 2, (2, 10))
    with torch.no_grad():
        states = model(idx, num_repeats=2, return_repeat_states=True)["repeat_states"]
        alpha_1, beta_1, alpha_2, beta_2 = model.residual_scales()
        e = states[0]
        h = e
        for repeat in (1, 2):
            h = alpha_2 * h + beta_2 * e
            h = alpha_1 * h + beta_1 * 0.3
            h = alpha_1 * h + beta_1 * -0.1
            assert torch.allclose(states[repeat], h, atol=1e-6)


def test_fast_readout_matches_independent_forwards():
    model = _model(fprm_but, "fprm_but")
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
