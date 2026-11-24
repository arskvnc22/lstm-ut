"""Tests for the delayed-recall BUT with FPRM residual scaling (models/dca_fprm_but.py).

Intended behaviour, every evolution and recall repeat (c = controller):
    h = alpha_2 * h [+ beta_2 * e] + c
    h = alpha_1 * h + beta_1 * attn(ln_1(h));  h = alpha_1 * h + beta_1 * mlp(ln_2(h))
    h = h - alpha_1^2 * c                      (controller "subtract")
with h_0 = e and the betas tied to the alphas as in fprm_but.
Checked here (software correctness only):
  - with alpha_1 = beta_1 = alpha_2 = 1 and beta_2 = 0 the model is exactly
    dca_but with the same weights (both controller modes, with and without
    recall);
  - it matches an independent reference loop written from the equations
    above, with and without re-injection, for 1 and 2 recall repeats;
  - with the sub-layer outputs silenced, "subtract" leaves no trace of the
    controller in the stream, while "persistent" does;
  - re-injection changes the output, and --no-fprm_input_injection reaches
    the model;
  - with zero controller embeddings and no recall it is exactly the Rule 30
    fprm_but with the same weights;
  - parameter groups (alpha logits not decayed) and gradients to the alphas.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from models import MODELS, dca_but, dca_fprm_but, fprm_but

D = 16
H = 5


def _args(model_name, controller="subtract", injection=True):
    return SimpleNamespace(
        model=model_name, vocab_size=2, sequence_length=16, n_repeat=H, n_embd=D,
        n_head=2, n_layer=1, n_layer_begin=0, n_layer_end=0, bias=False, dropout=0.0,
        attention_mode="bidirectional", attention_window_length=None,
        lm_cache="none", positional_encoder="rotary",
        ca_max_relative_age=H, ca_controller_application=controller,
        fprm_alpha_1_init=0.75, fprm_alpha_2_init=0.25, fprm_input_injection=injection,
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    for module in ("dca_but", "dca_fprm_but", "fprm_but"):
        monkeypatch.setattr(f"models.{module}.tiktoken.get_encoding", lambda _: None)


def _model(module, name, seed=3, controller="subtract", injection=True):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = module.GPTBase(_args(name, controller, injection)).eval()
    with torch.no_grad():
        if hasattr(model, "direction_embedding"):
            model.direction_embedding.embd.weight.normal_(0, 1, generator=torch.Generator().manual_seed(11))
            model.age_embedding.embd.weight.normal_(0, 1, generator=torch.Generator().manual_seed(12))
        if hasattr(model, "fprm_alpha_1_logit"):
            # Unequal per-channel alphas, so a wrong broadcast would show.
            model.fprm_alpha_1_logit.add_(torch.randn(D, generator=torch.Generator().manual_seed(13)))
            model.fprm_alpha_2_logit.add_(torch.randn(D, generator=torch.Generator().manual_seed(14)))
    return model


def _idx(batch=3, seed=0):
    return torch.randint(0, 2, (batch, 16), generator=torch.Generator().manual_seed(seed))


def _logits(model, idx, recall, recall_age=2, recall_repeats=1):
    out = model(idx, get_logits=True, return_all_logits=True, num_repeats=H,
                delayed_recall=recall, recall_age=recall_age if recall else None,
                num_recall_repeats=recall_repeats)
    return out["logits"]


def test_registered():
    assert MODELS["dca_fprm_but"] is dca_fprm_but.GPTBase


def test_scales_start_at_init_and_betas_follow_fprm():
    model = dca_fprm_but.GPTBase(_args("dca_fprm_but"))
    a1, b1, a2, b2 = model.residual_scales()
    torch.testing.assert_close(a1, torch.full((D,), 0.75))
    torch.testing.assert_close(a2, torch.full((D,), 0.25))
    torch.testing.assert_close(b2, 1 - a2 * a1 ** 2)
    torch.testing.assert_close(b1, b2 * (1 - a1) / (1 - a1 ** 2 + 1e-5))


@pytest.mark.parametrize("controller", ("subtract", "persistent"))
@pytest.mark.parametrize("recall,recall_repeats", ((False, 1), (True, 1), (True, 2)))
def test_unit_scales_is_exactly_dca_but(controller, recall, recall_repeats):
    fprm = _model(dca_fprm_but, "dca_fprm_but", controller=controller)
    ones = torch.ones(D)
    fprm.residual_scales = lambda: (ones, ones, ones, torch.zeros(D))
    but = dca_but.GPTBase(_args("dca_but", controller)).eval()
    loaded = but.load_state_dict(fprm.state_dict(), strict=False)
    assert loaded.missing_keys == []
    assert sorted(loaded.unexpected_keys) == ["fprm_alpha_1_logit", "fprm_alpha_2_logit"]
    idx = _idx()
    with torch.no_grad():
        torch.testing.assert_close(
            _logits(fprm, idx, recall, recall_repeats=recall_repeats),
            _logits(but, idx, recall, recall_repeats=recall_repeats),
            rtol=0, atol=0,
        )


def _reference(model, idx, recall_age, recall_repeats):
    """The model's equations written out independently of its forward."""
    block = model.transformer.h_mid[0]
    idx, pos = model.transformer.wpe(idx)
    e = pos.adapt_model_input(model.transformer.wte(idx), start_index=0)
    a1 = torch.sigmoid(model.fprm_alpha_1_logit)
    a2 = torch.sigmoid(model.fprm_alpha_2_logit)
    b2 = 1 - a2 * a1 * a1
    b1 = b2 * (1 - a1) / (1 - a1 * a1 + 1e-5)
    B = e.shape[0]

    def controller(direction, age):
        d = model.direction_embedding.embd.weight[direction]
        a = model.age_embedding.embd.weight[age]
        return (d + a).expand(B, 1, D)

    def step(h, c):
        h = a2 * h + (b2 * e if model.input_injection else 0) + c
        h = a1 * h + b1 * block.attn(block.ln_1(h), pos, None, 0)
        h = a1 * h + b1 * block.mlp(block.ln_2(h))
        if model.controller_application == "subtract":
            h = h - a1 * a1 * c
        return h

    h = e
    for rep in range(1, H + 1):
        h = step(h, controller(0, H - rep + 1))
    for _ in range(recall_repeats):
        h = step(h, controller(1, recall_age))
    return model.lm_head(model.transformer.ln_f(h))


@pytest.mark.parametrize("controller", ("subtract", "persistent"))
@pytest.mark.parametrize("injection", (True, False))
@pytest.mark.parametrize("recall_repeats", (1, 2))
def test_matches_reference(controller, injection, recall_repeats):
    model = _model(dca_fprm_but, "dca_fprm_but", controller=controller, injection=injection)
    idx = _idx()
    with torch.no_grad():
        torch.testing.assert_close(
            _logits(model, idx, True, recall_age=3, recall_repeats=recall_repeats),
            _reference(model, idx, 3, recall_repeats),
        )


@pytest.mark.parametrize("injection", (True, False))
def test_subtract_leaves_no_controller_in_the_stream(injection):
    """With attention and MLP outputs zeroed the stream is linear in c, so the
    final state must not depend on the controller under "subtract". Two
    different random controllers are compared (not a rescaled one: the final
    LayerNorm would hide a pure rescaling)."""
    outputs = {}
    for controller in ("subtract", "persistent"):
        for draw in (21, 22):
            model = _model(dca_fprm_but, "dca_fprm_but", controller=controller, injection=injection)
            with torch.no_grad():
                for block in model.transformer.h_mid:
                    block.attn.c_proj.weight.zero_()
                    block.mlp.c_proj.weight.zero_()
                gen = torch.Generator().manual_seed(draw)
                model.direction_embedding.embd.weight.normal_(0, 1, generator=gen)
                model.age_embedding.embd.weight.normal_(0, 1, generator=gen)
                outputs[controller, draw] = _logits(model, _idx(), True, recall_age=2, recall_repeats=2)
    torch.testing.assert_close(outputs["subtract", 21], outputs["subtract", 22])
    assert (outputs["persistent", 21] - outputs["persistent", 22]).abs().max() > 1e-3


def test_injection_switch_changes_output_and_reaches_model_from_cli():
    idx = _idx()
    with torch.no_grad():
        on = _logits(_model(dca_fprm_but, "dca_fprm_but", injection=True), idx, True)
        off = _logits(_model(dca_fprm_but, "dca_fprm_but", injection=False), idx, True)
    assert (on - off).abs().max() > 1e-3

    from cellular_automaton import dca_main
    for flag, expected in (([], True), (["--no-fprm_input_injection"], False)):
        argv = ["dca_main.py", "--model", "dca_fprm_but", *flag]
        with mock.patch.object(sys, "argv", argv):
            assert dca_main.get_args().fprm_input_injection is expected


@pytest.mark.parametrize("injection", (True, False))
def test_evolution_without_controller_is_rule30_fprm_but(injection):
    dca = _model(dca_fprm_but, "dca_fprm_but", injection=injection)
    with torch.no_grad():
        dca.direction_embedding.embd.weight.zero_()
        dca.age_embedding.embd.weight.zero_()
    ca = fprm_but.GPTBase(_args("fprm_but", injection=injection)).eval()
    loaded = ca.load_state_dict(dca.state_dict(), strict=False)
    assert loaded.missing_keys == []
    assert all(k.startswith(("direction_embedding", "age_embedding")) for k in loaded.unexpected_keys)
    idx = _idx()
    with torch.no_grad():
        torch.testing.assert_close(
            _logits(dca, idx, False),
            ca(idx, get_logits=True, return_all_logits=True, num_repeats=H)["logits"],
            rtol=0, atol=0,
        )


def test_alphas_not_weight_decayed_and_get_gradient():
    model = _model(dca_fprm_but, "dca_fprm_but").train()
    decay, no_decay = model.get_parameter_group_specs()
    for name in ("fprm_alpha_1_logit", "fprm_alpha_2_logit"):
        assert name in no_decay["params"] and name not in decay["params"]
    out = model(_idx(), targets=_idx(seed=1), num_repeats=H, delayed_recall=True, recall_age=1)
    out["loss"].backward()
    for p in (model.fprm_alpha_1_logit, model.fprm_alpha_2_logit):
        assert p.grad is not None and p.grad.abs().sum() > 0
