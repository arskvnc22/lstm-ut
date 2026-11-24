"""Tests for the iterative Jacobian check in cellular_automaton/ca_eff_lyapunov.py.

ca_eff_lyapunov.py differs from ca_lyapunov.py only in how the optional
sanity check is computed. These tests check that the iterative values match
the dense decomposition and that every other output is unchanged. Software
correctness only.
"""

import numpy as np
import pytest
import torch

from cellular_automaton import ca_eff_lyapunov, ca_lyapunov
from test_ca_lyapunov import MODEL_NAMES, _rows, _state_after, _tiny_model


def _small_state(model_name):
    # n = S * 4 * 8: 64 for LSTM-UT, 32 otherwise, so dense references are cheap.
    step = ca_eff_lyapunov.BlockStep(_tiny_model(model_name, n_embd=8, cells=4))
    return step, _state_after(step, _rows(1, 4), depth=2)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_vjp_is_the_transpose_of_the_dense_jacobian(model_name):
    step, state = _small_state(model_name)
    dim = state[0].numel()
    cotangents = torch.randn(1, dim, 3, generator=torch.Generator().manual_seed(11), dtype=torch.float64)

    jacobian = ca_eff_lyapunov.full_jacobian(step, state, chunk_size=16)
    pulled_back = ca_eff_lyapunov.vjp_columns(step, state, cotangents)

    torch.testing.assert_close(pulled_back[0], jacobian.T @ cotangents[0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_iterative_check_matches_dense_decomposition(model_name):
    step, state = _small_state(model_name)

    check = ca_eff_lyapunov.iterative_jacobian_check(step, state, tol=1e-12)

    jacobian = ca_eff_lyapunov.full_jacobian(step, state, chunk_size=16)
    dense_singular_values = torch.linalg.svdvals(jacobian)[:8].numpy()
    dense_radius = torch.linalg.eigvals(jacobian).abs().max().item()
    assert check["converged"]
    assert check["method"] == "iterative"
    assert check["jacobian_products"] > 0
    np.testing.assert_allclose(np.exp(check["log_top_singular_values"]), dense_singular_values, rtol=1e-8)
    assert np.exp(check["log_spectral_radius"]) == pytest.approx(dense_radius, rel=1e-8)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_outputs_other_than_the_check_match_ca_lyapunov(model_name):
    kwargs = dict(max_depth=4, num_vectors=4, tangent_seed=0, jacobian_check_depths=(2,), jacobian_check_rows=1)
    reference, dense_checks = ca_lyapunov.rollout_lyapunov(
        ca_lyapunov.BlockStep(_tiny_model(model_name)), _rows(3, 8), **kwargs
    )
    efficient, iterative_checks = ca_eff_lyapunov.rollout_lyapunov(
        ca_eff_lyapunov.BlockStep(_tiny_model(model_name)), _rows(3, 8), **kwargs
    )

    assert set(reference) == set(efficient)
    for name in reference:
        np.testing.assert_array_equal(efficient[name], reference[name], err_msg=name)
    assert len(iterative_checks) == len(dense_checks) == 1
    iterative, dense = iterative_checks[0], dense_checks[0]
    assert (iterative["depth"], iterative["row"]) == (dense["depth"], dense["row"])
    assert iterative["full_local_log_growth_top"] == dense["full_local_log_growth_top"]
    # ARPACK's default tolerance here is 1e-6.
    np.testing.assert_allclose(iterative["log_top_singular_values"], dense["log_top_singular_values"], atol=1e-5)
    assert iterative["log_spectral_radius"] == pytest.approx(dense["log_spectral_radius"], abs=1e-5)
    # |R_11| = |J q_1| can never exceed the largest singular value of J.
    assert iterative["full_local_log_growth_top"] <= iterative["log_top_singular_values"][0] + 1e-9
