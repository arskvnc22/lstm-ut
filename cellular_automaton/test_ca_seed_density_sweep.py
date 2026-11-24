"""Tests for the seed-by-density depth sweep (ca_seed_density_sweep).

The sweep decodes every repeat from returned repeat states. For each
supported model this must give the same per-depth counts as independent
forwards with num_repeats = r.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from cellular_automaton.ca_eval import _forward_all_cells
from cellular_automaton.ca_forward import CAForwardContext
from cellular_automaton.ca_gen import MaterializedRule30Dataset, apply_rule30
from cellular_automaton.ca_seed_density_sweep import (
    SUPPORTED_MODELS,
    evaluate_condition,
)
from models import MODELS

MAX_REPEATS = 5


def _args(model):
    return SimpleNamespace(
        model=model, vocab_size=2, sequence_length=16, n_repeat=4, n_embd=16,
        n_head=2, n_layer=1, n_layer_begin=0, n_layer_end=0, bias=False,
        dropout=0.0, attention_mode="bidirectional",
        attention_window_length=None, lstm_forget_gate="learned",
        lstm_control_input="previous_and_proposed", lstm_initial_cell="zero",
        lm_cache="none", positional_encoder="rotary",
    )


def test_supported_models_are_registered():
    assert set(SUPPORTED_MODELS) <= set(MODELS)


@pytest.mark.parametrize("model_name", ("lstm_ut_bidir", "lin_lstm_ut_bidir", "but_input_inject"))
def test_sweep_counts_match_independent_forwards(monkeypatch, model_name):
    monkeypatch.setattr(f"models.{model_name}.tiktoken.get_encoding", lambda _: None)
    with torch.random.fork_rng():
        torch.manual_seed(1)
        model = MODELS[model_name](_args(model_name)).eval()
    dataset = MaterializedRule30Dataset(
        num_samples=6, num_cells=12, steps=1, bernoulli_p=0.5, seed=11
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    context = CAForwardContext(model)

    rows = evaluate_condition(context, loader, "cpu", MAX_REPEATS, nullcontext())

    inputs = dataset.inputs.long()
    previous = inputs
    for repeat in range(1, MAX_REPEATS + 1):
        with torch.no_grad():
            logits, _ = _forward_all_cells(context, inputs, num_repeats=repeat)
        prediction = logits.argmax(-1)
        target_match = prediction.eq(apply_rule30(inputs, steps=repeat))
        recurrence_match = prediction.eq(apply_rule30(previous, steps=1))
        row = rows[repeat - 1]
        assert row["repeat"] == repeat
        assert row["cell_correct"] == int(target_match.sum())
        assert row["exact_rows"] == int(target_match.all(dim=1).sum())
        assert row["decoded_recurrence_cell_correct"] == int(recurrence_match.sum())
        assert row["decoded_recurrence_exact_rows"] == int(recurrence_match.all(dim=1).sum())
        assert row["nonfinite_logit_rows"] == 0
        previous = prediction
