from types import SimpleNamespace

import pytest
import torch

from models.but_full_depth import CausalSelfAttention
from models.caches.cache import LMCache
from models.positional_encoders.encoder import PositionalEncoderClosure


@pytest.fixture
def attention_config():
    return SimpleNamespace(
        n_embd=8,
        n_head=2,
        bias=False,
        dropout=0.25,
        sequence_length=4,
        attention_window_length=None,
        attention_mode="bidirectional",
    )


def test_sdpa_dropout_is_disabled_during_evaluation(monkeypatch, attention_config):
    observed_dropout = []

    def record_sdpa(q, k, v, *, attn_mask, dropout_p, is_causal):
        observed_dropout.append(dropout_p)
        return torch.zeros_like(q)

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        record_sdpa,
    )

    attention = CausalSelfAttention(
        attention_config,
        LMCache(attention_config),
    )
    x = torch.randn(2, 4, attention_config.n_embd)
    positional_adapter = PositionalEncoderClosure(encoder=None)

    attention.train()
    attention(
        x,
        positional_adapter,
        cache_context=None,
        start_index=0,
    )

    attention.eval()
    attention(
        x,
        positional_adapter,
        cache_context=None,
        start_index=0,
    )

    assert observed_dropout == [attention_config.dropout, 0.0]
