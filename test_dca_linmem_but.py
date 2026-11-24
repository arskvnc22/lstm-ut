"""Tests for the delayed-recall linear memory BUT (models/dca_linmem_but.py).

Intended behaviour: dca_but with the linear-memory block of linmem_but,
    c = lambda * c + W_prev LN_m(x) + W_prop LN_m(x_block);  x = x_block + c,
with the cell starting at zero and carried, without a reset, from the
evolution repeats into the recall repeats (as in dca_lstm_ut), and the
controller added to / subtracted from x around every repeat as in dca_but.
Checked here (software correctness only):
  - with zero memory writes the model is exactly dca_but with the same
    weights, in both controller modes, with and without recall;
  - it matches an independent reference loop written from the equations
    above, including the cell carried into recall (and differs from a
    reference that resets the cell at recall);
  - with zero controller embeddings and no recall it is exactly the Rule 30
    linmem_but with the same weights;
  - parameter groups (lambda not decayed) and gradients to lambda.
"""

from types import SimpleNamespace

import pytest
import torch

from models import MODELS, dca_but, dca_linmem_but, linmem_but

D = 16
H = 5


def _args(model_name, controller="subtract"):
    return SimpleNamespace(
        model=model_name, vocab_size=2, sequence_length=16, n_repeat=H, n_embd=D,
        n_head=2, n_layer=1, n_layer_begin=0, n_layer_end=0, bias=False, dropout=0.0,
        attention_mode="bidirectional", attention_window_length=None,
        lm_cache="none", positional_encoder="rotary",
        ca_max_relative_age=H, ca_controller_application=controller,
        lstm_control_input="previous_and_proposed", lstm_initial_cell="zero",
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    for module in ("dca_but", "dca_linmem_but", "linmem_but"):
        monkeypatch.setattr(f"models.{module}.tiktoken.get_encoding", lambda _: None)


def _model(module, name, seed=3, controller="subtract"):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = module.GPTBase(_args(name, controller)).eval()
    with torch.no_grad():
        # Make the controller and the memory large enough to matter.
        if hasattr(model, "direction_embedding"):
            model.direction_embedding.embd.weight.normal_(0, 1, generator=torch.Generator().manual_seed(11))
            model.age_embedding.embd.weight.normal_(0, 1, generator=torch.Generator().manual_seed(12))
        for block in model.transformer.h_mid:
            if hasattr(block, "memory_decay_logit"):
                block.memory_decay_logit.normal_(0, 1, generator=torch.Generator().manual_seed(13))
                block.proposed_to_memory_proposal.weight.normal_(0, 0.3, generator=torch.Generator().manual_seed(14))
                block.previous_to_memory_proposal.weight.normal_(0, 0.3, generator=torch.Generator().manual_seed(15))
    return model


def _idx(batch=3, seed=0):
    return torch.randint(0, 2, (batch, 16), generator=torch.Generator().manual_seed(seed))


def _logits(model, idx, recall, recall_age=2, recall_repeats=1):
    out = model(idx, get_logits=True, return_all_logits=True, num_repeats=H,
                delayed_recall=recall, recall_age=recall_age if recall else None,
                num_recall_repeats=recall_repeats)
    return out["logits"]


def test_registered():
    assert MODELS["dca_linmem_but"] is dca_linmem_but.GPTBase


@pytest.mark.parametrize("controller", ("subtract", "persistent"))
@pytest.mark.parametrize("recall,recall_repeats", ((False, 1), (True, 1), (True, 2)))
def test_zero_memory_writes_is_exactly_dca_but(controller, recall, recall_repeats):
    linmem = _model(dca_linmem_but, "dca_linmem_but", controller=controller)
    but = dca_but.GPTBase(_args("dca_but", controller)).eval()
    missing = but.load_state_dict(linmem.state_dict(), strict=False)
    assert missing.missing_keys == []
    assert all("memory" in key for key in missing.unexpected_keys)
    with torch.no_grad():
        for block in linmem.transformer.h_mid:
            block.proposed_to_memory_proposal.weight.zero_()
            block.previous_to_memory_proposal.weight.zero_()
    idx = _idx()
    with torch.no_grad():
        torch.testing.assert_close(
            _logits(linmem, idx, recall, recall_repeats=recall_repeats),
            _logits(but, idx, recall, recall_repeats=recall_repeats),
            rtol=0, atol=0,
        )


def _reference(model, idx, recall_age, recall_repeats, reset_cell_at_recall=False):
    """The model's equations written out independently of its forward."""
    block = model.transformer.h_mid[0]
    idx, pos = model.transformer.wpe(idx)
    x = pos.adapt_model_input(model.transformer.wte(idx), start_index=0)
    cell = torch.zeros_like(x)
    B = x.shape[0]
    subtract = model.controller_application == "subtract"

    def controller(direction, age):
        d = model.direction_embedding.embd.weight[direction]
        a = model.age_embedding.embd.weight[age]
        return (d + a).expand(B, 1, D)

    def step(x, cell, c):
        x = x + c
        previous = block.memory_gate_norm(x)
        x_block = x + block.attn(block.ln_1(x), pos, None, 0)
        x_block = x_block + block.mlp(block.ln_2(x_block))
        proposed = block.memory_gate_norm(x_block)
        write = block.proposed_to_memory_proposal(proposed) + block.previous_to_memory_proposal(previous)
        cell = torch.sigmoid(block.memory_decay_logit) * cell + write
        x = x_block + cell
        if subtract:
            x = x - c
        return x, cell

    for rep in range(1, H + 1):
        x, cell = step(x, cell, controller(0, H - rep + 1))
    if reset_cell_at_recall:
        cell = torch.zeros_like(cell)
    for _ in range(recall_repeats):
        x, cell = step(x, cell, controller(1, recall_age))
    return model.lm_head(model.transformer.ln_f(x))


@pytest.mark.parametrize("controller", ("subtract", "persistent"))
@pytest.mark.parametrize("recall_repeats", (1, 2))
def test_matches_reference_and_carries_cell_into_recall(controller, recall_repeats):
    model = _model(dca_linmem_but, "dca_linmem_but", controller=controller)
    idx = _idx()
    with torch.no_grad():
        got = _logits(model, idx, True, recall_age=2, recall_repeats=recall_repeats)
        torch.testing.assert_close(got, _reference(model, idx, 2, recall_repeats))
        reset = _reference(model, idx, 2, recall_repeats, reset_cell_at_recall=True)
    assert (got - reset).abs().max() > 1e-3


def test_evolution_without_controller_is_rule30_linmem_but():
    dca = _model(dca_linmem_but, "dca_linmem_but")
    with torch.no_grad():
        dca.direction_embedding.embd.weight.zero_()
        dca.age_embedding.embd.weight.zero_()
    ca = linmem_but.GPTBase(_args("linmem_but")).eval()
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


def test_decay_not_weight_decayed_and_gets_gradient():
    model = _model(dca_linmem_but, "dca_linmem_but").train()
    decay, no_decay = model.get_parameter_group_specs()
    assert "transformer.h_mid.0.memory_decay_logit" in no_decay["params"]
    assert "transformer.h_mid.0.memory_decay_logit" not in decay["params"]
    idx = _idx()
    targets = _idx(seed=1)
    out = model(idx, targets=targets, num_repeats=H, delayed_recall=True, recall_age=1)
    out["loss"].backward()
    grad = model.transformer.h_mid[0].memory_decay_logit.grad
    assert grad is not None and grad.abs().sum() > 0
