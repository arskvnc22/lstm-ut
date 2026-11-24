"""Tests for gated BUT with a fixed stream decay (models/decay_gated_but.py).

Intended behaviour, every repeat (G = gate, x_block = block output from x):
    x = lambda * x + G * (x_block - x)
with x_0 the embedded input row and lambda = --but_stream_decay.
Checked here (software correctness only):
  - lambda = 1 is exactly gated_but_bidir with the same weights;
  - each repeat state equals lambda * previous + the gated write, where the
    gated write is taken from the unmodified gated block;
  - the decay changes the output, lambda outside (0, 1] is rejected, and
    --but_stream_decay reaches the model;
  - the parameters are those of gated_but_bidir, and a training step gives
    gradients.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from models import MODELS, decay_gated_but, gated_but_bidir

D = 16
REPEATS = 5


def _args(model_name, decay=1.0):
    return SimpleNamespace(
        model=model_name, vocab_size=2, sequence_length=16, n_repeat=REPEATS, n_embd=D,
        n_head=2, n_layer=1, n_layer_begin=0, n_layer_end=0, bias=False, dropout=0.0,
        attention_mode="bidirectional", attention_window_length=None,
        lm_cache="none", positional_encoder="rotary", min_repeat=1,
        but_stream_decay=decay,
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    for module in ("gated_but_bidir", "decay_gated_but"):
        monkeypatch.setattr(f"models.{module}.tiktoken.get_encoding", lambda _: None)


def _model(module, name, seed=3, decay=1.0):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return module.GPTBase(_args(name, decay)).eval()


def _idx(batch=3, seed=0):
    return torch.randint(0, 2, (batch, 16), generator=torch.Generator().manual_seed(seed))


def test_registered_and_same_parameters_as_gated_but():
    assert MODELS["decay_gated_but"] is decay_gated_but.GPTBase
    decayed = decay_gated_but.GPTBase(_args("decay_gated_but", decay=0.9))
    gated = gated_but_bidir.GPTBase(_args("gated_but_bidir"))
    assert {n: p.shape for n, p in decayed.named_parameters()} == {n: p.shape for n, p in gated.named_parameters()}


@pytest.mark.parametrize("bad", (0.0, -0.5, 1.5))
def test_decay_outside_unit_interval_is_rejected(bad):
    with pytest.raises(ValueError):
        decay_gated_but.GPTBase(_args("decay_gated_but", decay=bad))


def test_unit_decay_is_exactly_gated_but():
    decayed = _model(decay_gated_but, "decay_gated_but", decay=1.0)
    gated = gated_but_bidir.GPTBase(_args("gated_but_bidir")).eval()
    gated.load_state_dict(decayed.state_dict())
    idx = _idx()
    with torch.no_grad():
        torch.testing.assert_close(
            decayed(idx, get_logits=True, return_all_logits=True, num_repeats=REPEATS)["logits"],
            gated(idx, get_logits=True, return_all_logits=True, num_repeats=REPEATS)["logits"],
            rtol=0, atol=0,
        )


@pytest.mark.parametrize("lam", (0.9, 0.5))
def test_each_repeat_is_lambda_times_carried_plus_the_gated_write(lam):
    model = _model(decay_gated_but, "decay_gated_but", decay=lam)
    idx = _idx()
    block = model.transformer.h_mid[0]
    with torch.no_grad():
        out = model(idx, get_logits=True, return_all_logits=True,
                    num_repeats=REPEATS, return_repeat_states=True)
        states = out["repeat_states"]
        _, pos = model.transformer.wpe(idx)
        for previous, current in zip(states[:-1], states[1:]):
            # The gated block returns previous + G * (x_block - previous).
            gated_write = block(previous, pos, None, 0) - previous
            torch.testing.assert_close(current, lam * previous + gated_write, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            out["logits"], model.lm_head(model.transformer.ln_f(states[-1])), rtol=1e-5, atol=1e-5)


def test_decay_changes_the_output_and_flag_reaches_the_model():
    idx = _idx()
    with torch.no_grad():
        plain = _model(decay_gated_but, "decay_gated_but", decay=1.0)(
            idx, get_logits=True, return_all_logits=True, num_repeats=REPEATS)["logits"]
        decayed = _model(decay_gated_but, "decay_gated_but", decay=0.9)(
            idx, get_logits=True, return_all_logits=True, num_repeats=REPEATS)["logits"]
    assert (plain - decayed).abs().max() > 1e-3

    from cellular_automaton import ca_main
    for flag, expected in (([], 1.0), (["--but_stream_decay", "0.9"], 0.9)):
        argv = ["ca_main.py", "--model", "decay_gated_but", *flag]
        with mock.patch.object(sys, "argv", argv):
            assert ca_main.get_args().but_stream_decay == expected


def test_training_step_gives_gradients():
    model = _model(decay_gated_but, "decay_gated_but", decay=0.9).train()
    out = model(_idx(), targets=_idx(seed=1), num_repeats=REPEATS)
    out["loss"].backward()
    block = model.transformer.h_mid[0]
    for p in (block.mlp.c_proj.weight, block.previous_to_gate.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
