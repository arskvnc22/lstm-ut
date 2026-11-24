"""Tests for the delayed-recall BUT with a fixed stream decay (models/dca_decay_but.py).

Intended behaviour, every evolution and recall repeat (c = controller,
Delta(.) = what the looped block adds to its input):
    x = lambda * x + Delta(x + c)            (controller "subtract")
    x = lambda * x + c + Delta(x + c)        (controller "persistent")
with x_0 the embedded input row and lambda = --but_stream_decay.
Checked here (software correctness only):
  - lambda = 1 is exactly dca_but with the same weights (both controller
    modes, with and without recall, 1 and 2 recall repeats);
  - it matches an independent reference loop written from the equations
    above for lambda = 0.9 and 0.5;
  - with zero controller embeddings and no recall it is exactly the Rule 30
    decay_but with the same weights;
  - the decay changes the output, lambda outside (0, 1] is rejected, and
    --but_stream_decay reaches the model;
  - the parameters are those of dca_but, and a training step gives gradients.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from models import MODELS, dca_but, dca_decay_but, decay_but

D = 16
H = 5


def _args(model_name, controller="subtract", decay=1.0):
    return SimpleNamespace(
        model=model_name, vocab_size=2, sequence_length=16, n_repeat=H, n_embd=D,
        n_head=2, n_layer=1, n_layer_begin=0, n_layer_end=0, bias=False, dropout=0.0,
        attention_mode="bidirectional", attention_window_length=None,
        lm_cache="none", positional_encoder="rotary",
        ca_max_relative_age=H, ca_controller_application=controller,
        but_stream_decay=decay,
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    for module in ("dca_but", "dca_decay_but", "decay_but"):
        monkeypatch.setattr(f"models.{module}.tiktoken.get_encoding", lambda _: None)


def _model(module, name, seed=3, controller="subtract", decay=1.0):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = module.GPTBase(_args(name, controller, decay)).eval()
    with torch.no_grad():
        if hasattr(model, "direction_embedding"):
            model.direction_embedding.embd.weight.normal_(0, 1, generator=torch.Generator().manual_seed(11))
            model.age_embedding.embd.weight.normal_(0, 1, generator=torch.Generator().manual_seed(12))
    return model


def _idx(batch=3, seed=0):
    return torch.randint(0, 2, (batch, 16), generator=torch.Generator().manual_seed(seed))


def _logits(model, idx, recall, recall_age=2, recall_repeats=1):
    out = model(idx, get_logits=True, return_all_logits=True, num_repeats=H,
                delayed_recall=recall, recall_age=recall_age if recall else None,
                num_recall_repeats=recall_repeats)
    return out["logits"]


def test_registered_and_same_parameters_as_dca_but():
    assert MODELS["dca_decay_but"] is dca_decay_but.GPTBase
    decayed = dca_decay_but.GPTBase(_args("dca_decay_but", decay=0.9))
    but = dca_but.GPTBase(_args("dca_but"))
    assert {n: p.shape for n, p in decayed.named_parameters()} == {n: p.shape for n, p in but.named_parameters()}


@pytest.mark.parametrize("bad", (0.0, -0.5, 1.5))
def test_decay_outside_unit_interval_is_rejected(bad):
    with pytest.raises(ValueError):
        dca_decay_but.GPTBase(_args("dca_decay_but", decay=bad))


@pytest.mark.parametrize("controller", ("subtract", "persistent"))
@pytest.mark.parametrize("recall,recall_repeats", ((False, 1), (True, 1), (True, 2)))
def test_unit_decay_is_exactly_dca_but(controller, recall, recall_repeats):
    decayed = _model(dca_decay_but, "dca_decay_but", controller=controller, decay=1.0)
    but = dca_but.GPTBase(_args("dca_but", controller)).eval()
    but.load_state_dict(decayed.state_dict())
    idx = _idx()
    with torch.no_grad():
        torch.testing.assert_close(
            _logits(decayed, idx, recall, recall_repeats=recall_repeats),
            _logits(but, idx, recall, recall_repeats=recall_repeats),
            rtol=0, atol=0,
        )


def _reference(model, idx, recall_age, recall_repeats, lam):
    """The model's equations written out independently of its forward."""
    block = model.transformer.h_mid[0]
    idx, pos = model.transformer.wpe(idx)
    x = pos.adapt_model_input(model.transformer.wte(idx), start_index=0)
    B = x.shape[0]

    def controller(direction, age):
        d = model.direction_embedding.embd.weight[direction]
        a = model.age_embedding.embd.weight[age]
        return (d + a).expand(B, 1, D)

    def delta(h):
        """What the pre-norm block adds to its input h."""
        after_attention = h + block.attn(block.ln_1(h), pos, None, 0)
        return after_attention + block.mlp(block.ln_2(after_attention)) - h

    def step(x, c):
        write = delta(x + c)
        if model.controller_application == "subtract":
            return lam * x + write
        return lam * x + c + write

    for rep in range(1, H + 1):
        x = step(x, controller(0, H - rep + 1))
    for _ in range(recall_repeats):
        x = step(x, controller(1, recall_age))
    return model.lm_head(model.transformer.ln_f(x))


@pytest.mark.parametrize("controller", ("subtract", "persistent"))
@pytest.mark.parametrize("lam", (0.9, 0.5))
@pytest.mark.parametrize("recall_repeats", (1, 2))
def test_matches_reference(controller, lam, recall_repeats):
    model = _model(dca_decay_but, "dca_decay_but", controller=controller, decay=lam)
    idx = _idx()
    with torch.no_grad():
        torch.testing.assert_close(
            _logits(model, idx, True, recall_age=3, recall_repeats=recall_repeats),
            _reference(model, idx, 3, recall_repeats, lam),
            rtol=1e-5, atol=1e-5,
        )


@pytest.mark.parametrize("lam", (0.9, 0.5))
def test_evolution_without_controller_is_rule30_decay_but(lam):
    dca = _model(dca_decay_but, "dca_decay_but", decay=lam)
    with torch.no_grad():
        dca.direction_embedding.embd.weight.zero_()
        dca.age_embedding.embd.weight.zero_()
    ca = decay_but.GPTBase(_args("decay_but", decay=lam)).eval()
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


def test_decay_changes_the_output_and_flag_reaches_the_model():
    idx = _idx()
    with torch.no_grad():
        plain = _logits(_model(dca_decay_but, "dca_decay_but", decay=1.0), idx, True)
        decayed = _logits(_model(dca_decay_but, "dca_decay_but", decay=0.9), idx, True)
    assert (plain - decayed).abs().max() > 1e-3

    from cellular_automaton import dca_main
    for flag, expected in (([], 1.0), (["--but_stream_decay", "0.9"], 0.9)):
        argv = ["dca_main.py", "--model", "dca_decay_but", *flag]
        with mock.patch.object(sys, "argv", argv):
            assert dca_main.get_args().but_stream_decay == expected


def test_training_step_gives_gradients():
    model = _model(dca_decay_but, "dca_decay_but", decay=0.9).train()
    out = model(_idx(), targets=_idx(seed=1), num_repeats=H, delayed_recall=True, recall_age=1)
    out["loss"].backward()
    block = model.transformer.h_mid[0]
    for p in (block.mlp.c_proj.weight, model.age_embedding.embd.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
