"""Focused backend-selection tests for instrumented CA CoTFormer attention."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from models import ca_cotf_cache_attn


def make_attention(implementation="sdpa"):
    config = SimpleNamespace(
        n_embd=8,
        n_head=2,
        bias=False,
        dropout=0.0,
        attention_window_length=None,
        attention_mode="bidirectional",
        attention_implementation=implementation,
        allow_cache_during_training=False,
        sequence_length=4,
        lm_cache="none",
        positional_encoder="rotary",
    )
    lm_cache = ca_cotf_cache_attn.caches.get_cache("none")(config)
    attention = ca_cotf_cache_attn.CausalSelfAttention(config, lm_cache)
    encoder = ca_cotf_cache_attn.positional_encoders.get_encoder("rotary")(config)
    _, position_closure = encoder(torch.zeros((2, 4), dtype=torch.long))
    return attention, position_closure


def make_model(
    attention_mode="bidirectional",
    implementation="manual",
    n_layer_begin=0,
):
    config = SimpleNamespace(
        vocab_size=2,
        sequence_length=4,
        n_embd=8,
        n_head=2,
        n_layer=1 + n_layer_begin,
        n_layer_begin=n_layer_begin,
        n_layer_end=0,
        n_repeat=3,
        bias=False,
        dropout=0.0,
        attention_window_length=None,
        attention_mode=attention_mode,
        attention_implementation=implementation,
        allow_cache_during_training=False,
        positional_encoder="rotary",
        lm_cache="none",
    )
    with mock.patch.object(
        ca_cotf_cache_attn.tiktoken, "get_encoding", return_value=None
    ):
        return ca_cotf_cache_attn.GPTBase(config)


def make_uniform_value_attention():
    """Build manual attention whose output is the mean of visible values."""
    attention, position_closure = make_attention("manual")
    attention.eval()
    with torch.no_grad():
        attention.c_attn.weight.zero_()
        attention.c_attn.weight[16:24].copy_(torch.eye(8))
        attention.c_proj.weight.copy_(torch.eye(8))
    return attention, position_closure


def run_repeat_sequence(repeat_cache_window):
    attention, position_closure = make_uniform_value_attention()
    repeats = [
        torch.full((2, 4, 8), float(value)) for value in (1, 2, 3)
    ]
    outputs = []
    diagnostics = []
    cache_lengths = []
    attention.init_cache(3 * 4)
    try:
        for values in repeats:
            output, record = attention(
                values,
                position_closure,
                cache_context=None,
                start_index=0,
                indices=None,
                collect_attention_diagnostics=True,
                repeat_cache_window=repeat_cache_window,
            )
            outputs.append(output)
            diagnostics.append(record)
            cache_lengths.append(attention.all_keys[1].shape[2])
    finally:
        attention.drop_cache()
    return outputs, diagnostics, cache_lengths


def run_attention(attention, position_closure, x=None):
    if x is None:
        x = torch.randn(2, 4, 8, requires_grad=True)
    attention.init_cache(4)
    try:
        return attention(
            x,
            position_closure,
            cache_context=None,
            start_index=0,
            indices=None,
        )[0]
    finally:
        attention.drop_cache()


def test_manual_training_skips_sdpa_and_diagnostics():
    attention, position_closure = make_attention("manual")
    attention.train()
    with mock.patch.object(
        torch.nn.functional, "scaled_dot_product_attention"
    ) as sdpa, mock.patch.object(
        attention, "_summarize_attention"
    ) as summarize:
        output = run_attention(attention, position_closure)
    sdpa.assert_not_called()
    summarize.assert_not_called()
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert attention.c_attn.weight.grad is not None


def test_sdpa_remains_the_default_backend():
    attention, position_closure = make_attention()
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    with mock.patch.object(
        torch.nn.functional,
        "scaled_dot_product_attention",
        wraps=original_sdpa,
    ) as sdpa:
        run_attention(attention, position_closure)
    sdpa.assert_called_once()


def test_manual_and_sdpa_match_with_the_same_parameters():
    torch.manual_seed(0)
    manual, manual_position_closure = make_attention("manual")
    sdpa, sdpa_position_closure = make_attention("sdpa")
    sdpa.load_state_dict(manual.state_dict(), strict=True)
    x = torch.randn(2, 4, 8)

    manual_output = run_attention(manual, manual_position_closure, x)
    sdpa_output = run_attention(sdpa, sdpa_position_closure, x)

    assert manual.state_dict().keys() == sdpa.state_dict().keys()
    torch.testing.assert_close(manual_output, sdpa_output)


def test_requesting_diagnostics_uses_manual_attention():
    attention, position_closure = make_attention("sdpa")
    attention.eval()
    expected_diagnostics = {"checked": True}
    attention.init_cache(4)
    try:
        with mock.patch.object(
            torch.nn.functional, "scaled_dot_product_attention"
        ) as sdpa, mock.patch.object(
            attention,
            "_summarize_attention",
            return_value=expected_diagnostics,
        ) as summarize:
            _, diagnostics = attention(
                torch.randn(2, 4, 8),
                position_closure,
                cache_context=None,
                start_index=0,
                indices=None,
                collect_attention_diagnostics=True,
            )
    finally:
        attention.drop_cache()

    sdpa.assert_not_called()
    summarize.assert_called_once()
    assert diagnostics is expected_diagnostics


def test_invalid_attention_implementation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported attention implementation"):
        make_attention("unknown")


def test_repeat_cache_window_none_preserves_full_history():
    outputs, diagnostics, cache_lengths = run_repeat_sequence(None)
    final = diagnostics[-1]

    torch.testing.assert_close(outputs[-1], torch.full_like(outputs[-1], 2.0))
    torch.testing.assert_close(
        final["repeat_mass"],
        torch.full_like(final["repeat_mass"], 1.0 / 3.0),
    )
    assert torch.equal(
        final["visible_token_count"],
        torch.full_like(final["visible_token_count"], 12),
    )
    assert torch.equal(
        final["visible_repeat_count"],
        torch.full_like(final["visible_repeat_count"], 3),
    )
    torch.testing.assert_close(
        final["normalised_attention_entropy"],
        torch.ones_like(final["normalised_attention_entropy"]),
    )
    torch.testing.assert_close(
        final["normalised_repeat_entropy"],
        torch.ones_like(final["normalised_repeat_entropy"]),
    )
    torch.testing.assert_close(
        final["effective_support_fraction"],
        torch.ones_like(final["effective_support_fraction"]),
    )
    assert cache_lengths == [4, 8, 12]


def test_repeat_cache_window_one_exposes_only_current_repeat():
    outputs, diagnostics, cache_lengths = run_repeat_sequence(1)
    final_repeat_mass = diagnostics[-1]["repeat_mass"]

    torch.testing.assert_close(outputs[-1], torch.full_like(outputs[-1], 3.0))
    torch.testing.assert_close(
        final_repeat_mass[..., :2], torch.zeros_like(final_repeat_mass[..., :2])
    )
    torch.testing.assert_close(
        final_repeat_mass[..., 2], torch.ones_like(final_repeat_mass[..., 2])
    )
    assert torch.equal(
        diagnostics[-1]["repeat_logsumexp_valid"][..., :2],
        torch.zeros_like(
            diagnostics[-1]["repeat_logsumexp_valid"][..., :2]
        ),
    )
    assert torch.equal(
        diagnostics[-1]["visible_token_count"],
        torch.full_like(diagnostics[-1]["visible_token_count"], 4),
    )
    assert torch.equal(
        diagnostics[-1]["visible_repeat_count"],
        torch.ones_like(diagnostics[-1]["visible_repeat_count"]),
    )
    torch.testing.assert_close(
        diagnostics[-1]["normalised_attention_entropy"],
        torch.ones_like(diagnostics[-1]["normalised_attention_entropy"]),
    )
    torch.testing.assert_close(
        diagnostics[-1]["normalised_repeat_entropy"],
        torch.zeros_like(diagnostics[-1]["normalised_repeat_entropy"]),
    )
    torch.testing.assert_close(
        diagnostics[-1]["effective_support"],
        torch.full_like(diagnostics[-1]["effective_support"], 4.0),
    )
    torch.testing.assert_close(
        diagnostics[-1]["effective_support_fraction"],
        torch.ones_like(diagnostics[-1]["effective_support_fraction"]),
    )
    assert cache_lengths == [4, 8, 12]


def test_repeat_cache_window_two_exposes_two_most_recent_repeats():
    outputs, diagnostics, cache_lengths = run_repeat_sequence(2)
    final_repeat_mass = diagnostics[-1]["repeat_mass"]

    torch.testing.assert_close(outputs[-1], torch.full_like(outputs[-1], 2.5))
    torch.testing.assert_close(
        final_repeat_mass[..., 0], torch.zeros_like(final_repeat_mass[..., 0])
    )
    torch.testing.assert_close(
        final_repeat_mass[..., 1:],
        torch.full_like(final_repeat_mass[..., 1:], 0.5),
    )
    torch.testing.assert_close(
        final_repeat_mass.sum(dim=-1),
        torch.ones_like(final_repeat_mass.sum(dim=-1)),
    )
    assert torch.equal(
        diagnostics[-1]["visible_token_count"],
        torch.full_like(diagnostics[-1]["visible_token_count"], 8),
    )
    assert torch.equal(
        diagnostics[-1]["visible_repeat_count"],
        torch.full_like(diagnostics[-1]["visible_repeat_count"], 2),
    )
    torch.testing.assert_close(
        diagnostics[-1]["normalised_attention_entropy"],
        torch.ones_like(diagnostics[-1]["normalised_attention_entropy"]),
    )
    torch.testing.assert_close(
        diagnostics[-1]["normalised_repeat_entropy"],
        torch.ones_like(diagnostics[-1]["normalised_repeat_entropy"]),
    )
    torch.testing.assert_close(
        diagnostics[-1]["effective_support"],
        torch.full_like(diagnostics[-1]["effective_support"], 8.0),
    )
    torch.testing.assert_close(
        diagnostics[-1]["effective_support_fraction"],
        torch.ones_like(diagnostics[-1]["effective_support_fraction"]),
    )
    assert cache_lengths == [4, 8, 12]


def test_window_at_least_as_large_as_history_is_a_no_op():
    full_outputs, full_diagnostics, _ = run_repeat_sequence(None)
    window_outputs, window_diagnostics, _ = run_repeat_sequence(3)

    for full, windowed in zip(full_outputs, window_outputs):
        torch.testing.assert_close(windowed, full)
    torch.testing.assert_close(
        window_diagnostics[-1]["repeat_mass"],
        full_diagnostics[-1]["repeat_mass"],
    )


@pytest.mark.parametrize(
    ("invalid_window", "expected_error", "message"),
    (
        (0, ValueError, "must be positive"),
        (-1, ValueError, "must be positive"),
        (1.5, TypeError, "positive integer or None"),
        (True, TypeError, "positive integer or None"),
    ),
)
def test_invalid_repeat_cache_windows_are_rejected(
    invalid_window, expected_error, message
):
    model = make_model()
    model.eval()
    with pytest.raises(expected_error, match=message):
        model(
            torch.zeros((2, 4), dtype=torch.long),
            num_repeats=3,
            repeat_cache_window=invalid_window,
        )


def test_repeat_cache_window_supports_manual_training_and_backward():
    model = make_model()
    model.train()
    inputs = torch.randint(0, 2, (2, 4))
    targets = torch.randint(0, 2, (2, 4))

    outputs = model(
        inputs,
        targets=targets,
        num_repeats=3,
        repeat_cache_window=2,
    )
    loss = outputs["loss"]
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert torch.isfinite(loss)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_repeat_cache_window_rejects_sdpa_flash_attention():
    model = make_model(implementation="sdpa")
    model.train()

    with pytest.raises(ValueError, match="requires manual attention"):
        model(
            torch.zeros((2, 4), dtype=torch.long),
            num_repeats=3,
            repeat_cache_window=2,
        )


def test_repeat_cache_window_rejects_causal_attention():
    model = make_model(attention_mode="causal")
    model.eval()
    with pytest.raises(ValueError, match="repeat_cache_window"):
        model(
            torch.zeros((2, 4), dtype=torch.long),
            num_repeats=3,
            repeat_cache_window=2,
        )


def test_model_propagates_window_and_cleans_cache():
    model = make_model()
    model.eval()
    attention = model.transformer.h_mid[0].attn

    with mock.patch.object(
        attention, "forward", wraps=attention.forward
    ) as attention_forward:
        model(
            torch.zeros((2, 4), dtype=torch.long),
            num_repeats=3,
            repeat_cache_window=2,
        )

    assert attention_forward.call_count == 3
    assert all(
        call.kwargs["repeat_cache_window"] == 2
        for call in attention_forward.call_args_list
    )
    assert attention.all_keys is None
    assert attention.all_values is None
    assert attention._lazy_init_cache_length is None


def test_repeat_cache_is_cleaned_after_exception():
    model = make_model()
    model.eval()
    attention = model.transformer.h_mid[0].attn

    with mock.patch.object(
        attention, "forward", side_effect=RuntimeError("synthetic failure")
    ), pytest.raises(RuntimeError, match="synthetic failure"):
        model(
            torch.zeros((2, 4), dtype=torch.long),
            num_repeats=3,
            repeat_cache_window=2,
        )

    assert attention.all_keys is None
    assert attention.all_values is None
    assert attention._lazy_init_cache_length is None


def test_no_intervention_uses_the_existing_input_and_repeat_path():
    model = make_model()
    model.eval()
    inputs = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])

    with mock.patch.object(
        model.transformer.wte, "forward", wraps=model.transformer.wte.forward
    ) as embedding_forward:
        outputs = model(
            inputs,
            get_logits=True,
            return_all_logits=True,
            num_repeats=3,
        )

    assert embedding_forward.call_count == 1
    assert torch.equal(embedding_forward.call_args.args[0], inputs)
    assert outputs["logits"].shape == (2, 4, 2)


def test_intervention_replaces_current_x_and_preserves_historical_kv():
    model = make_model()
    model.eval()
    inputs = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 0]])
    clean_state = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]])
    block = model.transformer.h_mid[0]
    attention = block.attn
    block_inputs = []
    cache_lengths = []
    historical_keys = None
    appended_keys = None
    original_block_forward = block.forward

    def record_block_input(x, *args, **kwargs):
        nonlocal historical_keys, appended_keys
        block_inputs.append(x.detach().clone())
        current_keys = attention.all_keys
        cache_lengths.append(
            0 if current_keys is None else current_keys[1].shape[2]
        )
        if len(block_inputs) == 3:
            historical_keys = current_keys[1].detach().clone()
        result = original_block_forward(x, *args, **kwargs)
        if len(block_inputs) == 3:
            appended_keys = attention.all_keys[1].detach().clone()
        return result

    with mock.patch.object(block, "forward", side_effect=record_block_input):
        model(
            inputs,
            num_repeats=3,
            intervention_source_depth=2,
            intervention_input_ids=clean_state,
        )

    canonical_clean_x = model.transformer.wte(clean_state)
    _, position_closure = model.transformer.wpe(inputs)
    canonical_clean_x = position_closure.adapt_model_input(
        canonical_clean_x, start_index=0
    )
    torch.testing.assert_close(block_inputs[2], canonical_clean_x)
    assert cache_lengths == [0, 4, 8]
    torch.testing.assert_close(appended_keys[..., :8, :], historical_keys)

    normalized_clean_x = block.ln_1(canonical_clean_x)
    _, expected_keys, _ = attention.c_attn(normalized_clean_x).split(
        model.config.n_embd, dim=2
    )
    expected_keys = expected_keys.view(
        inputs.shape[0],
        inputs.shape[1],
        model.config.n_head,
        model.config.n_embd // model.config.n_head,
    ).transpose(1, 2)
    expected_keys = position_closure.adapt_keys(
        expected_keys, start_index=0, indices=None
    )
    torch.testing.assert_close(appended_keys[..., 8:, :], expected_keys)
    assert attention.all_keys is None
    assert attention.all_values is None
    assert attention._lazy_init_cache_length is None


@pytest.mark.parametrize(
    "kwargs",
    (
        {"intervention_source_depth": 1},
        {"intervention_input_ids": torch.zeros((2, 4), dtype=torch.long)},
    ),
)
def test_partial_intervention_arguments_are_rejected(kwargs):
    model = make_model()
    model.eval()
    with pytest.raises(ValueError, match="must be supplied together"):
        model(torch.zeros((2, 4), dtype=torch.long), num_repeats=3, **kwargs)


def test_training_mode_intervention_is_rejected():
    model = make_model()
    model.train()
    inputs = torch.zeros((2, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="only supported in evaluation mode"):
        model(
            inputs,
            num_repeats=2,
            intervention_source_depth=1,
            intervention_input_ids=inputs.clone(),
        )


@pytest.mark.parametrize("invalid_depth", (-1, 3, True, 1.5))
def test_invalid_intervention_depth_is_rejected(invalid_depth):
    model = make_model()
    model.eval()
    inputs = torch.zeros((2, 4), dtype=torch.long)
    expected_error = TypeError if isinstance(invalid_depth, (bool, float)) else ValueError
    with pytest.raises(expected_error, match="intervention_source_depth"):
        model(
            inputs,
            num_repeats=3,
            intervention_source_depth=invalid_depth,
            intervention_input_ids=inputs.clone(),
        )


def test_intervention_rejects_unsupported_architecture_and_lm_cache_mode():
    inputs = torch.zeros((2, 4), dtype=torch.long)
    begin_model = make_model(n_layer_begin=1)
    begin_model.eval()
    with pytest.raises(ValueError, match="n_layer_begin == 0"):
        begin_model(
            inputs,
            num_repeats=2,
            intervention_source_depth=1,
            intervention_input_ids=inputs.clone(),
        )

    model = make_model()
    model.eval()
    with pytest.raises(ValueError, match="external LM cache mode"):
        model(
            inputs,
            use_cache=True,
            num_repeats=2,
            intervention_source_depth=1,
            intervention_input_ids=inputs.clone(),
        )


def test_intervention_rejects_replacement_with_wrong_shape():
    model = make_model()
    model.eval()
    inputs = torch.zeros((2, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="same shape"):
        model(
            inputs,
            num_repeats=2,
            intervention_source_depth=1,
            intervention_input_ids=torch.zeros((2, 3), dtype=torch.long),
        )


def test_intervention_cache_is_cleaned_after_exception():
    model = make_model()
    model.eval()
    attention = model.transformer.h_mid[0].attn
    inputs = torch.zeros((2, 4), dtype=torch.long)

    with mock.patch.object(
        attention, "forward", side_effect=RuntimeError("synthetic intervention failure")
    ), pytest.raises(RuntimeError, match="synthetic intervention failure"):
        model(
            inputs,
            num_repeats=2,
            intervention_source_depth=0,
            intervention_input_ids=torch.ones_like(inputs),
        )

    assert attention.all_keys is None
    assert attention.all_values is None
    assert attention._lazy_init_cache_length is None


@pytest.mark.parametrize(
    ("baseline_window", "expected_windows"),
    ((None, [None, None, 1, 2, 3]), (8, [8, 8, 1, 2, 3])),
)
def test_cache_reset_permanently_excludes_the_pre_reset_segment(
    baseline_window, expected_windows
):
    model = make_model()
    model.eval()
    inputs = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    baseline_states = model(
        inputs,
        num_repeats=5,
        repeat_cache_window=baseline_window,
        return_repeat_states=True,
    )["repeat_states"]
    block = model.transformer.h_mid[0]
    attention = block.attn
    active_windows = []
    cache_lengths = []
    original_block_forward = block.forward

    def record_cache_policy(x, *args, **kwargs):
        active_windows.append(kwargs["repeat_cache_window"])
        cache_lengths.append(
            0 if attention.all_keys is None else attention.all_keys[1].shape[2]
        )
        return original_block_forward(x, *args, **kwargs)

    with mock.patch.object(block, "forward", side_effect=record_cache_policy):
        intervened_states = model(
            inputs,
            num_repeats=5,
            repeat_cache_window=baseline_window,
            return_repeat_states=True,
            cache_reset_source_depth=2,
        )["repeat_states"]

    assert active_windows == expected_windows
    assert cache_lengths == [0, 4, 8, 12, 16]
    for baseline, intervened in zip(baseline_states[:3], intervened_states[:3]):
        torch.testing.assert_close(intervened, baseline)
    assert attention.all_keys is None
    assert attention.all_values is None
    assert attention._lazy_init_cache_length is None


def test_clean_state_and_cache_reset_compose_at_the_same_source_depth():
    model = make_model()
    model.eval()
    inputs = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 0]])
    clean_state = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]])
    block = model.transformer.h_mid[0]
    block_inputs = []
    active_windows = []
    original_block_forward = block.forward

    def record_target_inputs(x, *args, **kwargs):
        block_inputs.append(x.detach().clone())
        active_windows.append(kwargs["repeat_cache_window"])
        return original_block_forward(x, *args, **kwargs)

    with mock.patch.object(block, "forward", side_effect=record_target_inputs):
        model(
            inputs,
            num_repeats=3,
            repeat_cache_window=2,
            intervention_source_depth=2,
            intervention_input_ids=clean_state,
            cache_reset_source_depth=2,
        )

    canonical_clean_x = model.transformer.wte(clean_state)
    _, position_closure = model.transformer.wpe(inputs)
    canonical_clean_x = position_closure.adapt_model_input(
        canonical_clean_x, start_index=0
    )
    assert active_windows == [2, 2, 1]
    torch.testing.assert_close(block_inputs[2], canonical_clean_x)


@pytest.mark.parametrize("invalid_depth", (-1, 3, True, 1.5))
def test_invalid_cache_reset_depth_is_rejected(invalid_depth):
    model = make_model()
    model.eval()
    expected_error = TypeError if isinstance(invalid_depth, (bool, float)) else ValueError
    with pytest.raises(expected_error, match="cache_reset_source_depth"):
        model(
            torch.zeros((2, 4), dtype=torch.long),
            num_repeats=3,
            cache_reset_source_depth=invalid_depth,
        )


def test_cache_reset_rejects_training_and_mismatched_state_depth():
    inputs = torch.zeros((2, 4), dtype=torch.long)
    model = make_model()
    model.train()
    with pytest.raises(ValueError, match="only supported in evaluation mode"):
        model(
            inputs,
            num_repeats=3,
            cache_reset_source_depth=1,
        )

    model.eval()
    with pytest.raises(ValueError, match="same source depth"):
        model(
            inputs,
            num_repeats=3,
            intervention_source_depth=1,
            intervention_input_ids=inputs.clone(),
            cache_reset_source_depth=2,
        )


def test_cache_reset_rejects_the_sdpa_backend_and_legacy_arguments():
    model = make_model(implementation="sdpa")
    model.eval()
    with pytest.raises(ValueError, match="requires manual attention"):
        model(
            torch.zeros((2, 4), dtype=torch.long),
            num_repeats=3,
            cache_reset_source_depth=1,
        )

    manual_model = make_model()
    manual_model.eval()
    with pytest.raises(TypeError, match="cache_intervention_source_depth"):
        manual_model(
            torch.zeros((2, 4), dtype=torch.long),
            num_repeats=3,
            cache_intervention_source_depth=1,
            cache_intervention_window=1,
        )
