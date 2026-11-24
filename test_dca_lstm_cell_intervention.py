"""Tests for the delayed-recall cell interventions on dca_lstm_ut (zXsf Q2).

The model switches are evaluation-only and act on recall repeats only:
zero_cell (cell set to 0 before recall), freeze_cell (cell held and read),
close_output_gate (cell written but not read). Block-level tests use constant
attention/MLP so the read is checked exactly; model-level tests rebuild the
recall phase by hand from the shared block.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from cellular_automaton.ca_forward import CAForwardContext, CAForwardPolicy
from cellular_automaton.ca_gen import Rule30Dataset
from cellular_automaton.dca_eval import evaluate_delayed_recall_pairs, summarize_delayed_recall_pairs
from cellular_automaton.dca_lstm_cell_intervene import (
    CONDITIONS,
    CellInterventionContext,
    donor_scores,
    score_against_own_and_donor,
    deltas_from_baseline,
    evaluate_condition,
)
from models import dca_lstm_ut

H = 4
INTERVENTIONS = ("zero_cell", "freeze_cell", "close_output_gate", "swap_cell")


class _NoCache:
    def get_storage_for_layer(self, _layer):
        return None


class _Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, *_args):
        return torch.full_like(x, self.value)


def _args():
    return SimpleNamespace(
        model="dca_lstm_ut", vocab_size=2, sequence_length=16, n_repeat=H,
        n_embd=16, n_head=2, n_layer=1, n_layer_begin=0, n_layer_end=0,
        bias=False, dropout=0.0, attention_mode="bidirectional",
        attention_window_length=None, lstm_forget_gate="learned",
        lstm_control_input="previous_and_proposed", lm_cache="none",
        positional_encoder="rotary", ca_max_relative_age=H,
        ca_controller_application="subtract",
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    monkeypatch.setattr("models.dca_lstm_ut.tiktoken.get_encoding", lambda _: None)


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(3)
        return dca_lstm_ut.GPTBase(_args()).eval()


def _recall(model, idx, **kwargs):
    return model(idx, get_logits=True, return_all_logits=True, num_repeats=H,
                 delayed_recall=True, recall_age=2, num_recall_repeats=2,
                 return_repeat_logits=True, return_memory_states=True, **kwargs)


def test_block_switches():
    torch.manual_seed(0)
    block = dca_lstm_ut.Block(_args(), _NoCache())
    block.attn, block.mlp = _Constant(0.2), _Constant(-0.1)
    x = torch.randn(2, 5, 16)
    cell = 2.0 * torch.randn(2, 5, 16)

    x_normal, cell_normal = block(x, cell, None, None, 0)
    x_closed, cell_closed = block(x, cell, None, None, 0, close_hidden_gate=True)
    x_frozen, cell_frozen = block(x, cell, None, None, 0, freeze_cell=True)

    proposed = x + 0.2 - 0.1
    gate = torch.sigmoid(block._control_projection(
        block.memory_gate_norm(x), block.memory_gate_norm(proposed),
        block.previous_to_hidden, block.proposed_to_hidden,
    ))
    assert torch.allclose(x_closed, proposed)
    assert torch.equal(cell_closed, cell_normal)
    assert torch.allclose(x_normal, proposed + gate * torch.tanh(cell_normal), atol=1e-6)
    assert torch.equal(cell_frozen, cell)
    assert torch.allclose(x_frozen, proposed + gate * torch.tanh(cell), atol=1e-6)


def test_baseline_reports_none_and_evolution_is_untouched(model):
    idx = torch.randint(0, 2, (3, 12))
    with torch.no_grad():
        baseline = _recall(model, idx)
        assert baseline["recall_cell_intervention"] is None
        for condition in INTERVENTIONS:
            out = _recall(model, idx, recall_cell_intervention=condition)
            assert out["recall_cell_intervention"] == condition
            assert torch.equal(out["repeat_logits"], baseline["repeat_logits"])
            for repeat, cell in baseline["memory_states"]["evolution"].items():
                assert torch.equal(out["memory_states"]["evolution"][repeat], cell)
            assert not torch.allclose(out["logits"], baseline["logits"])


@pytest.mark.parametrize("condition", (None,) + INTERVENTIONS)
def test_recall_matches_hand_built_recall(model, condition):
    idx = torch.randint(0, 2, (3, 12))
    with torch.no_grad():
        out = _recall(model, idx, recall_cell_intervention=condition)

        # Rebuild evolution and recall from the shared block.
        _, pos = model.transformer.wpe(idx)
        x = model.transformer.wte(idx)
        cell = torch.zeros_like(x)
        block = model.transformer.h_mid[0]

        def controller(direction, age):
            ids = torch.full((idx.shape[0],), age, dtype=torch.long)
            dirs = torch.full((idx.shape[0],), direction, dtype=torch.long)
            return model.direction_embedding(dirs) + model.age_embedding(ids)

        for rep in range(1, H + 1):
            c = controller(0, H - rep + 1)
            x, cell = block(x + c, cell, pos, None, start_index=0)
            x = x - c
        evolution_cell = cell
        kwargs = {}
        if condition == "zero_cell":
            cell = torch.zeros_like(cell)
        elif condition == "freeze_cell":
            kwargs = {"freeze_cell": True}
        elif condition == "close_output_gate":
            kwargs = {"close_hidden_gate": True}
        elif condition == "swap_cell":
            # Row i takes the end-of-evolution cell of row i - 1; streams stay put.
            cell = torch.stack([evolution_cell[i - 1] for i in range(idx.shape[0])])
        c = controller(1, 2)
        for _ in range(2):
            x, cell = block(x + c, cell, pos, None, start_index=0, **kwargs)
            x = x - c
        expected = model.lm_head(model.transformer.ln_f(x))

    assert torch.allclose(out["logits"], expected, atol=1e-5)
    if condition == "freeze_cell":
        for recall_cell in out["memory_states"]["recall"].values():
            assert torch.equal(recall_cell, evolution_cell)
    if condition == "zero_cell":
        assert not torch.allclose(out["memory_states"]["recall"][1], evolution_cell)


def test_invalid_requests_raise(model):
    idx = torch.randint(0, 2, (2, 12))
    with pytest.raises(ValueError):
        _recall(model, idx, recall_cell_intervention="erase")
    with pytest.raises(ValueError):
        model(idx, num_repeats=H, recall_cell_intervention="zero_cell")


def _loader():
    return DataLoader(Rule30Dataset(num_samples=6, num_cells=12, seed=5),
                      batch_size=3, shuffle=False)


def test_baseline_evaluation_equals_unwrapped_paper_evaluation(model):
    loader = _loader()
    base = CAForwardContext(model, CAForwardPolicy(repeat_cache_window=None))
    reference = evaluate_delayed_recall_pairs(
        base, loader, "cpu", pairs=[(h, h) for h in range(1, H + 1)],
        num_recall_repeats=2, num_examples=0,
    )
    wrapped = CellInterventionContext(base, "baseline")
    result = evaluate_delayed_recall_pairs(
        wrapped, loader, "cpu", pairs=[(h, h) for h in range(1, H + 1)],
        num_recall_repeats=2, num_examples=0,
    )
    assert repr(result) == repr(reference)
    assert wrapped.recall_calls > 0


def test_every_condition_evaluates_and_reports_deltas(model):
    results = {
        condition: evaluate_condition(
            model, _loader(), "cpu", condition=condition,
            horizons=list(range(1, H + 1)), num_recall_repeats=2,
        )
        for condition in CONDITIONS
    }
    for condition, result in results.items():
        assert result["recall_calls"] > 0
        assert 0.0 <= result["nontrivial_pair_macro"]["cell_accuracy"] <= 1.0
        assert [row["query_repeat"] for row in result["max_horizon_by_query"]] == list(range(1, H + 1))
    deltas = deltas_from_baseline(results)
    assert set(deltas) == set(INTERVENTIONS)
    assert deltas["zero_cell"]["nontrivial_pair_macro"]["cell_accuracy"] == pytest.approx(
        results["zero_cell"]["nontrivial_pair_macro"]["cell_accuracy"]
        - results["baseline"]["nontrivial_pair_macro"]["cell_accuracy"]
    )


def test_internal_accuracy_is_reported_and_matches_the_evaluator(model):
    """Internal accuracy (recall scored against the model's own decoded row) is
    passed through unchanged from the paper's evaluator, for every condition."""
    horizons = list(range(1, H + 1))
    internal_names = ("internal_cell_accuracy", "internal_exact_sequence_accuracy")
    results = {
        condition: evaluate_condition(
            model, _loader(), "cpu", condition=condition,
            horizons=horizons, num_recall_repeats=2,
        )
        for condition in CONDITIONS
    }
    for result in results.values():
        for field in ("nontrivial_pair_macro", "all_queries_pair_macro",
                      "max_horizon_nontrivial_macro"):
            for name in internal_names:
                assert 0.0 <= result[field][name] <= 1.0
        for row in result["max_horizon_by_query"]:
            for name in internal_names:
                assert 0.0 <= row[name] <= 1.0

    base = CAForwardContext(model, CAForwardPolicy(repeat_cache_window=None))
    reference = summarize_delayed_recall_pairs(evaluate_delayed_recall_pairs(
        base, _loader(), "cpu", pairs=[(h, h) for h in horizons],
        num_recall_repeats=2, num_examples=0,
    ))["nontrivial_internal_consistency_pair_macro"]
    baseline = results["baseline"]["nontrivial_pair_macro"]
    assert baseline["internal_cell_accuracy"] == pytest.approx(
        reference["decoded_requested_repeat_cell_accuracy"])
    assert baseline["internal_exact_sequence_accuracy"] == pytest.approx(
        reference["decoded_requested_repeat_exact_sequence_accuracy"])

    deltas = deltas_from_baseline(results)
    for condition in INTERVENTIONS:
        for name in internal_names:
            assert deltas[condition]["nontrivial_pair_macro"][name] == pytest.approx(
                results[condition]["nontrivial_pair_macro"][name] - baseline[name])


def test_context_rejects_a_model_that_ignores_the_intervention(model):
    class _Silent:
        def __init__(self, inner):
            self.model, self.policy, self.inner = inner.model, inner.policy, inner

        def call(self, inputs, **kwargs):
            kwargs.pop("recall_cell_intervention", None)
            return self.inner.call(inputs, **kwargs)

    base = CAForwardContext(model, CAForwardPolicy(repeat_cache_window=None))
    context = CellInterventionContext(_Silent(base), "zero_cell")
    idx = torch.randint(0, 2, (2, 12))
    with pytest.raises(RuntimeError):
        context.call(idx, get_logits=True, num_repeats=H, delayed_recall=True, recall_age=1)


def test_swap_cell_is_a_no_op_for_identical_rows_and_needs_two_rows(model):
    """With the same input in every row all cells are equal, so swapping them
    cannot change anything. A single row has no donor."""
    row = torch.randint(0, 2, (1, 12))
    idx = row.repeat(3, 1)
    with torch.no_grad():
        baseline = _recall(model, idx)
        swapped = _recall(model, idx, recall_cell_intervention="swap_cell")
        assert torch.allclose(swapped["logits"], baseline["logits"], atol=1e-6)
        with pytest.raises(ValueError):
            _recall(model, row, recall_cell_intervention="swap_cell")


def test_own_and_donor_scoring_on_a_hand_made_case():
    # Two rows of four cells. Row 0: own 1100, donor 1010 -> they differ in cells 1, 2.
    # Row 1: own 0000, donor 0011 -> they differ in cells 2, 3.
    own = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
    donor = torch.tensor([[1, 0, 1, 0], [0, 0, 1, 1]])
    prediction = torch.tensor([[1, 1, 1, 0],   # cell 1 own, cell 2 donor; agreeing cells right
                               [1, 0, 1, 1]])  # cells 2, 3 donor; agreeing cell 0 wrong
    scores = score_against_own_and_donor(prediction, own, donor)
    assert scores["rows"] == 2
    assert scores["differing_cells"] == 4 and scores["agreeing_cells"] == 4
    assert scores["own_matches_on_differing_cells"] == 1
    assert scores["correct_on_agreeing_cells"] == 3
    assert scores["wrong_cells_vs_own"] == 1 + 3
    assert scores["wrong_cells_vs_donor"] == 1 + 1
    assert scores["exact_rows_own"] == 0 and scores["exact_rows_donor"] == 0
    assert score_against_own_and_donor(donor, own, donor)["exact_rows_donor"] == 2
    assert score_against_own_and_donor(own, own, donor)["own_matches_on_differing_cells"] == 4


def test_donor_scores_cover_every_age_and_agree_with_plain_accuracy(model):
    """Hamming distance to the own target, divided by the row length, is one
    minus the evaluator's cell accuracy for the same query."""
    for condition in ("baseline", "swap_cell"):
        rows = donor_scores(model, _loader(), "cpu", condition=condition,
                            horizon=H, num_recall_repeats=2)
        assert [row["recall_age"] for row in rows] == list(range(H))
        reference = evaluate_condition(
            model, _loader(), "cpu", condition=condition, horizons=[H], num_recall_repeats=2,
        )["max_horizon_by_query"]
        accuracy_by_age = {row["recall_age"]: row["cell_accuracy"] for row in reference}
        for row in rows:
            assert 0.0 <= row["own_match_on_differing_cells"] <= 1.0
            assert -1.0 <= row["mcc_to_own"] <= 1.0 and -1.0 <= row["mcc_to_donor"] <= 1.0
            assert 1 - row["hamming_to_own"] / 12 == pytest.approx(accuracy_by_age[row["recall_age"]])

