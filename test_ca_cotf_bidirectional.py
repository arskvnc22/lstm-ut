#!/usr/bin/env python3
"""Focused tests for causal and bidirectional attention in ``models.ca_cotf``.

These tests exercise the recurrent CoT K/V cache directly and also check the
model interface used by the cellular-automaton trainer. They deliberately use
small CPU tensors and replace tokenizer construction so that no network access
or tokenizer cache is required.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from models import ca_cotf


BATCH_SIZE = 2
SEQUENCE_LENGTH = 4
N_EMBD = 8
N_HEAD = 2
VOCAB_SIZE = 2


def make_config(attention_mode):
    """Return the minimal configuration needed by the CA CoTFormer."""
    return SimpleNamespace(
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=1,
        n_layer_begin=0,
        n_layer_end=0,
        n_repeat=3,
        bias=False,
        dropout=0.0,
        attention_window_length=None,
        attention_mode=attention_mode,
        positional_encoder="rotary",
        lm_cache="none",
        allow_cache_during_training=False,
    )


def make_attention(attention_mode, *, use_sdpa):
    """Build attention whose output is the exact mean of its visible values."""
    config = make_config(attention_mode)
    lm_cache = ca_cotf.caches.get_cache(config.lm_cache)(config)
    attention = ca_cotf.CausalSelfAttention(config, lm_cache).cpu()
    attention.flash = use_sdpa

    # Q=K=0 makes every permitted attention score equal. V and the output
    # projection are identities, so the result is the mean of exactly those
    # input positions admitted by the mask.
    with torch.no_grad():
        attention.c_attn.weight.zero_()
        attention.c_attn.weight[2 * N_EMBD : 3 * N_EMBD].copy_(
            torch.eye(N_EMBD)
        )
        attention.c_proj.weight.copy_(torch.eye(N_EMBD))

    encoder = ca_cotf.positional_encoders.get_encoder(
        config.positional_encoder
    )(config)
    _, position_closure = encoder(
        torch.zeros((BATCH_SIZE, SEQUENCE_LENGTH), dtype=torch.long)
    )
    return attention, position_closure


def make_model(attention_mode):
    """Build a complete tensor-only model without loading GPT-2 tokenizer data."""
    with mock.patch.object(ca_cotf.tiktoken, "get_encoding", return_value=None):
        return ca_cotf.GPTBase(make_config(attention_mode)).cpu()


class TestCACoTFormerAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_invalid_attention_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported attention mode"):
            make_model("not-an-attention-mode")

    def test_accumulated_cache_semantics_and_gradients(self):
        """Check exact visible values for both modes and both attention paths."""
        for attention_mode in ("causal", "bidirectional"):
            for use_sdpa in (False, True):
                backend = "sdpa" if use_sdpa else "manual"
                with self.subTest(mode=attention_mode, backend=backend):
                    attention, position_closure = make_attention(
                        attention_mode,
                        use_sdpa=use_sdpa,
                    )
                    first = (
                        torch.arange(
                            BATCH_SIZE * SEQUENCE_LENGTH * N_EMBD,
                            dtype=torch.float32,
                        )
                        .view(BATCH_SIZE, SEQUENCE_LENGTH, N_EMBD)
                        .div(10.0)
                        .requires_grad_()
                    )
                    second = (
                        first.detach().add(5.0).clone().requires_grad_()
                    )

                    attention.init_cache(2 * SEQUENCE_LENGTH)
                    try:
                        attention(
                            first,
                            position_closure,
                            cache_context=None,
                            start_index=0,
                            indices=None,
                        )
                        output = attention(
                            second,
                            position_closure,
                            cache_context=None,
                            start_index=0,
                            indices=None,
                        )

                        self.assertEqual(
                            attention.all_keys[1].shape[2],
                            2 * SEQUENCE_LENGTH,
                        )
                        self.assertEqual(
                            attention.all_values[1].shape[2],
                            2 * SEQUENCE_LENGTH,
                        )

                        if attention_mode == "bidirectional":
                            expected = (
                                torch.cat((first, second), dim=1)
                                .mean(dim=1, keepdim=True)
                                .expand_as(output)
                            )
                        else:
                            # Query i may see positions 0..i from both repeats.
                            expected = torch.stack(
                                [
                                    torch.cat(
                                        (
                                            first[:, : position + 1],
                                            second[:, : position + 1],
                                        ),
                                        dim=1,
                                    ).mean(dim=1)
                                    for position in range(SEQUENCE_LENGTH)
                                ],
                                dim=1,
                            )

                        torch.testing.assert_close(output, expected)
                    finally:
                        attention.drop_cache()

                    output.sum().backward()
                    for label, tensor in (
                        ("first repeat", first),
                        ("second repeat", second),
                    ):
                        self.assertIsNotNone(tensor.grad, label)
                        self.assertTrue(
                            torch.isfinite(tensor.grad).all().item(),
                            label,
                        )
                        self.assertGreater(tensor.grad.abs().sum().item(), 0.0)

    def test_sdpa_receives_the_intended_mask_and_causal_flag(self):
        """Distinguish configured causality from SDPA's built-in mask switch."""
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        for attention_mode in ("causal", "bidirectional"):
            with self.subTest(mode=attention_mode):
                attention, position_closure = make_attention(
                    attention_mode,
                    use_sdpa=True,
                )
                calls = []

                def record_sdpa(q, k, v, *, attn_mask, dropout_p, is_causal):
                    calls.append(
                        {
                            "query_length": q.shape[-2],
                            "key_length": k.shape[-2],
                            "mask": (
                                None
                                if attn_mask is None
                                else attn_mask.detach().clone()
                            ),
                            "is_causal": is_causal,
                        }
                    )
                    return original_sdpa(
                        q,
                        k,
                        v,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                    )

                inputs = torch.randn(
                    BATCH_SIZE,
                    SEQUENCE_LENGTH,
                    N_EMBD,
                )
                attention.init_cache(2 * SEQUENCE_LENGTH)
                try:
                    with mock.patch.object(
                        torch.nn.functional,
                        "scaled_dot_product_attention",
                        side_effect=record_sdpa,
                    ):
                        for _ in range(2):
                            attention(
                                inputs,
                                position_closure,
                                cache_context=None,
                                start_index=0,
                                indices=None,
                            )
                finally:
                    attention.drop_cache()

                self.assertEqual(
                    [call["key_length"] for call in calls],
                    [SEQUENCE_LENGTH, 2 * SEQUENCE_LENGTH],
                )
                self.assertTrue(
                    all(not call["is_causal"] for call in calls)
                )

                if attention_mode == "bidirectional":
                    self.assertTrue(
                        all(call["mask"] is None for call in calls)
                    )
                else:
                    triangle = torch.tril(
                        torch.ones(
                            SEQUENCE_LENGTH,
                            SEQUENCE_LENGTH,
                            dtype=torch.bool,
                        )
                    )
                    for repeat, call in enumerate(calls, start=1):
                        expected_mask = torch.cat(
                            [triangle] * repeat,
                            dim=-1,
                        ).view(
                            1,
                            1,
                            SEQUENCE_LENGTH,
                            repeat * SEQUENCE_LENGTH,
                        )
                        torch.testing.assert_close(
                            call["mask"],
                            expected_mask,
                        )

    def test_full_bidirectional_model_matches_ca_forward_contract(self):
        """Exercise runtime repeats, all-cell logits, loss, and cache cleanup."""
        model = make_model("bidirectional")
        model.train()
        inputs = torch.randint(
            0,
            VOCAB_SIZE,
            (BATCH_SIZE, SEQUENCE_LENGTH),
        )
        targets = torch.randint(
            0,
            VOCAB_SIZE,
            (BATCH_SIZE, SEQUENCE_LENGTH),
        )

        for repeats in (1, 2, 3):
            with self.subTest(repeats=repeats):
                model.zero_grad(set_to_none=True)
                outputs = model(
                    inputs,
                    targets=targets,
                    get_logits=True,
                    num_repeats=repeats,
                    return_repeat_states=True,
                )

                self.assertEqual(
                    tuple(outputs["logits"].shape),
                    (BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE),
                )
                self.assertEqual(len(outputs["repeat_states"]), repeats + 1)
                self.assertTrue(torch.isfinite(outputs["loss"]).item())
                self.assertEqual(outputs["average_depth"].item(), repeats)

                for block in model.transformer.h_mid:
                    self.assertIsNone(block.attn.all_keys)
                    self.assertIsNone(block.attn.all_values)
                    self.assertIsNone(block.attn._lazy_init_cache_length)

                outputs["loss"].backward()
                gradient = model.transformer.h_mid[0].attn.c_attn.weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertGreater(gradient.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
