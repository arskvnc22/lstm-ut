"""Finite-time Lyapunov exponents (FTLE) of the looped block on Rule 30.

For a trained 0/1/0 model the recurrent state after loop t is s_t = x_t for
BUT and gated BUT, and s_t = (x_t, c_t) for LSTM-UT, with s_t = Phi(s_{t-1})
where Phi is the single shared middle block. Attention only sees the current
repeat (lm_cache=none), so s_t is the whole state.

Nothing in the rollout is modified. Alongside the free rollout, p orthonormal
tangent vectors are pushed through the Jacobian of Phi with forward-mode JVPs
and re-orthonormalised by a QR decomposition after every step (Benettin's
method). log|R_ii| at step t is the local growth rate of the i-th direction;
its mean over a window of steps is the finite-time Lyapunov exponent. A
negative top exponent means infinitesimal state errors shrink along the
model's own trajectory; a positive one means they grow.

Adding one constant to every hidden feature of a token is an exact symmetry
of these blocks: every LayerNorm (block, gate and readout) removes it, so it
is carried forward unchanged and never affects a decoded bit. Those 64
directions per row have exponent exactly 0 and can occupy the top of the raw
spectrum. The `shift_quotient` tangent set therefore removes the per-token
hidden-feature mean from its tangents at initialisation and after every
Jacobian step, before the QR; the cell part is never projected. Its exponents
are "growth after removing invisible hidden-state offsets". They are not a
direct measure of bit sensitivity: other directions may still have no
immediate effect on the readout. Per-token scaling is deliberately not
removed, because a scale change alters the direction of x + f(LN(x)) and can
become visible on later loops.

This module is ca_lyapunov.py with one change: the optional Jacobian sanity
check (--jacobian-check-depths) never forms the dense n x n Jacobian. It
treats J as an operator (J v by JVP, J^T u by VJP) and asks SciPy's ARPACK
solvers for the top singular values (Lanczos, svds) and the largest-magnitude
eigenvalues (Arnoldi, eigs). Every other output is identical.

Usage:
    python -m cellular_automaton.ca_eff_lyapunov \
        --source-run-dir iridis/ca-rule30/runs/run_113__ca_lstm_r12_zero \
        --output-dir <new directory> --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # PyTorch < 2.3, e.g. the iridis environment (2.2.0)
    SDPBackend = sdpa_kernel = None

import models

from .ca_checkpoint_eval import (
    AVAILABLE_CHECKPOINT_TYPES,
    _read_checkpoint_metadata,
    load_checkpoint_weights,
    read_validated_manifest,
    resolve_source_args,
)
from .ca_gen import Rule30Dataset, apply_rule30
from .ca_reporting import MANIFEST_FILENAME, utc_now


SUPPORTED_MODELS = ("lstm_ut_bidir", "but_full_depth", "gated_but_bidir")
DEFAULT_WINDOWS = ((1, 12), (13, 20), (21, 30), (31, 60))
# A forget gate above this value retains at least 95% of the cell per step.
FORGET_RETENTION_THRESHOLD = 0.95


def _math_attention():
    """Context restricting scaled_dot_product_attention to its math kernel.

    Fused SDPA kernels do not all support forward-mode AD; the math kernel
    does. PyTorch < 2.3 has no torch.nn.attention, so the older global switch
    (available since 2.0) is used there instead.
    """
    if sdpa_kernel is not None:
        return sdpa_kernel(SDPBackend.MATH)
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    )


class BlockStep:
    """The one-loop map Phi on a state tensor of shape [B, S, T, D].

    S = 1 holds x; S = 2 holds (x, c) for LSTM-UT. Calling the object runs
    the model's own middle block once, exactly as GPTBase.forward does inside
    its repeat loop.
    """

    def __init__(self, model):
        config = model.config
        if config.model not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported model for FTLE: {config.model!r}")
        if (
            len(model.transformer.h_begin) != 0
            or len(model.transformer.h_mid) != 1
            or len(model.transformer.h_end) != 0
        ):
            raise ValueError("FTLE requires the 0/1/0 layout (one shared middle block).")
        if getattr(config, "lm_cache", "none") != "none":
            raise ValueError("FTLE requires lm_cache=none so the state is (x[, c]) only.")
        self.model = model
        self.block = model.transformer.h_mid[0]
        self.has_cell = config.model == "lstm_ut_bidir"
        self.num_components = 2 if self.has_cell else 1
        # The rotary closure depends only on positions, not on the tokens or
        # the batch (models/positional_encoders), so one closure serves the
        # rollout and every replicated tangent batch.
        _, self.pos_emb_closure = model.transformer.wpe(None)
        self._capture_gates = False
        self._forget_logits = []
        if self.has_cell and self.block.forget_gate_mode == "learned":
            # Read-only hooks: the forget gate is sigmoid of the summed outputs
            # of these projections (models/lstm_ut_bidir.py, Block.forward).
            for projection in (self.block.previous_to_forget, self.block.proposed_to_forget):
                if projection is not None:
                    projection.register_forward_hook(self._record_forget_logit)

    def _record_forget_logit(self, _module, _inputs, output):
        if self._capture_gates:
            self._forget_logits.append(output.detach())

    def initial_state(self, rows):
        """Embed binary rows as GPTBase.forward does before the first repeat."""
        x = self.model.transformer.wte(rows)
        x = self.model.transformer.drop(x)
        x = self.pos_emb_closure.adapt_model_input(x, start_index=0)
        if not self.has_cell:
            return x.unsqueeze(1)
        if self.model.initial_cell == "initial_hidden":
            cell = x.clone()
        else:
            cell = torch.zeros_like(x)
        return torch.stack((x, cell), dim=1)

    def __call__(self, state):
        x = state[:, 0]
        if self.has_cell:
            x, cell = self.block(x, state[:, 1], self.pos_emb_closure, None, start_index=0)
            return torch.stack((x, cell), dim=1)
        x = self.block(x, self.pos_emb_closure, None, start_index=0)
        return x.unsqueeze(1)

    def step_with_forget_gate(self, state):
        """Run Phi once and also return the forget gate of that step, or None."""
        self._forget_logits = []
        self._capture_gates = True
        try:
            next_state = self(state)
        finally:
            self._capture_gates = False
        if not self._forget_logits:
            return next_state, None
        return next_state, torch.sigmoid(sum(self._forget_logits))

    def decode(self, state):
        """Binary prediction per cell from the recurrent x, as the CA readout does."""
        logits = self.model.lm_head(self.model.transformer.ln_f(state[:, 0]))
        return logits.argmax(dim=-1)


def orthonormal_tangents(batch, dim, num_vectors, *, generator, dtype, device): # random orthonormal starting directions for the error vectors; they are pushed through the Jacobian alongside the rollout and never added to the state
    """p random orthonormal columns per example: shape [B, dim, p]."""
    gaussian = torch.randn(batch, dim, num_vectors, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(gaussian)
    return q.to(dtype=dtype, device=device)


def jvp_columns(step, state, tangents, *, source=None, target=None):
    """Apply the Jacobian of `step` at `state` to every column of `tangents`.

    `state` is [B, S, T, D] and `tangents` is [B, n, p]. With `source` set,
    the columns live in that state component only (other components get zero
    tangent); with `target` set, only that component of the output tangent is
    returned. Both None gives the full Jacobian. Returns [B, n_out, p].

    The p tangents are evaluated by replicating every example p times. The
    examples do not interact inside the block, so this equals p separate JVPs.
    """
    batch, components, cells, width = state.shape
    num_vectors = tangents.shape[-1]
    replicated = state.repeat_interleave(num_vectors, dim=0)
    columns = tangents.transpose(1, 2).reshape(batch * num_vectors, -1)
    if source is None:
        tangent_state = columns.reshape(batch * num_vectors, components, cells, width)
    else:
        tangent_state = torch.zeros_like(replicated)
        tangent_state[:, source] = columns.reshape(batch * num_vectors, cells, width)
    with _math_attention():
        _, output_tangent = torch.func.jvp(step, (replicated,), (tangent_state,)) #jacaobian of step evaluated at replicated times tangent state
    if target is not None:
        output_tangent = output_tangent[:, target]
    output_tangent = output_tangent.reshape(batch, num_vectors, -1)
    return output_tangent.transpose(1, 2)


def benettin_update(images):
    """QR re-orthonormalisation: new orthonormal tangents and log|R_ii|."""
    q, r = torch.linalg.qr(images)
    log_growth = torch.log(torch.diagonal(r, dim1=-2, dim2=-1).abs())
    return q, log_growth


def token_uniformity(vectors, cells, width):
    """Fraction of each vector's energy in its component shared by all cells.

    `vectors` is [B, cells * width]. 1 means the same perturbation at every
    cell (common mode); about 1 / cells means no shared component.
    """
    per_cell = vectors.reshape(vectors.shape[0], cells, width)
    shared = per_cell.mean(dim=1, keepdim=True).expand_as(per_cell)
    return shared.pow(2).sum(dim=(1, 2)) / per_cell.pow(2).sum(dim=(1, 2))


def remove_token_shift(vectors, components, cells, width):
    """Project out per-token hidden-feature offsets from full-state vectors.

    `vectors` is [B, components * cells * width, p]. For every token the mean
    over its `width` hidden features of the x part is subtracted; the cell
    part (component 1) is returned unchanged.
    """
    batch, _, num_vectors = vectors.shape
    per_feature = vectors.reshape(batch, components, cells, width, num_vectors).clone()
    per_feature[:, 0] -= per_feature[:, 0].mean(dim=2, keepdim=True)
    return per_feature.reshape(batch, -1, num_vectors)


def token_shift_energy_fraction(vectors, components, cells, width):
    """Fraction of each full-state vector's squared norm that is per-token x offset.

    `vectors` is [B, components * cells * width]; the denominator includes the
    cell part for LSTM-UT.
    """
    per_feature = vectors.reshape(vectors.shape[0], components, cells, width)
    offset_energy = width * per_feature[:, 0].mean(dim=2).pow(2).sum(dim=1)
    return offset_energy / vectors.pow(2).sum(dim=1)


def full_jacobian(step, state, *, chunk_size=512):
    """Dense one-step Jacobian for a single example: [n, n] with n = S*T*D."""
    if state.shape[0] != 1:
        raise ValueError("full_jacobian expects a single example.")
    dim = state[0].numel()
    identity = torch.eye(dim, dtype=state.dtype, device=state.device)
    columns = []
    for start in range(0, dim, chunk_size):
        basis = identity[:, start:start + chunk_size].unsqueeze(0)
        columns.append(jvp_columns(step, state, basis)[0])
    return torch.cat(columns, dim=1)


def vjp_columns(step, state, cotangents):
    """Apply the transposed Jacobian of `step` at `state` to every column.

    The reverse-mode counterpart of `jvp_columns` (full state, no
    source/target selection): `cotangents` is [B, n, p], returns [B, n, p].
    """
    batch, components, cells, width = state.shape
    num_vectors = cotangents.shape[-1]
    replicated = state.repeat_interleave(num_vectors, dim=0)
    output_cotangent = cotangents.transpose(1, 2).reshape(
        batch * num_vectors, components, cells, width
    )
    with _math_attention():
        _, pullback = torch.func.vjp(step, replicated)
        (input_cotangent,) = pullback(output_cotangent)
    return input_cotangent.reshape(batch, num_vectors, -1).transpose(1, 2)


def iterative_jacobian_check(
    step,
    state,
    *,
    num_singular_values=8,
    num_eigenvalues=6,
    tol=1e-6,
    maxiter=None,
    seed=0,
):
    """Top singular values and spectral radius of one example's Jacobian.

    J is never formed: ARPACK only sees J v (JVP) and J^T u (VJP). Products
    run in the model's dtype on its device; ARPACK works in float64 on the
    CPU. A solver that does not converge yields NaN and converged=False.
    """
    # Imported here so runs without Jacobian checks do not need SciPy.
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, svds

    if state.shape[0] != 1:
        raise ValueError("iterative_jacobian_check expects a single example.")
    dim = state[0].numel()
    products = 0

    def as_column(vector):
        column = torch.as_tensor(np.ascontiguousarray(vector).ravel(), device=state.device)
        return column.to(dtype=state.dtype).reshape(1, dim, 1)

    def matvec(vector):
        nonlocal products
        products += 1
        return jvp_columns(step, state, as_column(vector))[0, :, 0].double().cpu().numpy()

    def rmatvec(vector):
        nonlocal products
        products += 1
        return vjp_columns(step, state, as_column(vector))[0, :, 0].double().cpu().numpy()

    rng = np.random.default_rng(seed)
    converged = True
    try:
        singular_values = svds(
            LinearOperator((dim, dim), matvec=matvec, rmatvec=rmatvec, dtype=np.float64),
            k=num_singular_values, which="LM", tol=tol, maxiter=maxiter,
            v0=rng.standard_normal(dim), return_singular_vectors=False,
        )
        log_singular_values = np.log(np.sort(singular_values)[::-1]).tolist()
    except ArpackNoConvergence:
        converged = False
        log_singular_values = [float("nan")] * num_singular_values
    try:
        eigenvalues = eigs(
            LinearOperator((dim, dim), matvec=matvec, dtype=np.float64),
            k=num_eigenvalues, which="LM", tol=tol, maxiter=maxiter,
            v0=rng.standard_normal(dim), return_eigenvectors=False,
        )
        log_spectral_radius = float(np.log(np.abs(eigenvalues).max()))
    except ArpackNoConvergence:
        converged = False
        log_spectral_radius = float("nan")
    return {
        "log_top_singular_values": log_singular_values,
        "log_spectral_radius": log_spectral_radius,
        "method": "iterative",
        "converged": converged,
        "jacobian_products": products,
    }


def rollout_lyapunov(
    step,
    rows,
    *,
    max_depth,
    num_vectors,
    tangent_seed,
    jacobian_check_depths=(),
    jacobian_check_rows=0,
):
    """Free rollout from `rows` with Benettin tangents; returns numpy arrays.

    Step t (1-based) maps s_{t-1} to s_t. Every growth rate recorded at t is
    measured with the Jacobian at s_{t-1}; accuracy and gate statistics are
    those of s_t.
    """
    generator = torch.Generator().manual_seed(int(tangent_seed))
    rows = rows.to(next(step.model.parameters()).device)
    with torch.no_grad():
        state = step.initial_state(rows)
    batch, components, cells, width = state.shape
    dtype, device = state.dtype, state.device
    full_dim = components * cells * width
    block_dim = cells * width

    def new_tangents(dim):
        return orthonormal_tangents(
            batch, dim, num_vectors, generator=generator, dtype=dtype, device=device
        )

    def remove_shift(vectors):
        return remove_token_shift(vectors, components, cells, width)

    # Each tangent set is propagated by its own product of Jacobians. Entries
    # are (tangents, source, target, remove_shift). The shift-quotient set is
    # drawn last so the other sets keep the random draws they had before it
    # was added.
    tangent_sets = {"full": (new_tangents(full_dim), None, None, False)}
    if step.has_cell:
        tangent_sets["x_to_x"] = (new_tangents(block_dim), 0, 0, False)
        tangent_sets["cell_to_cell"] = (new_tangents(block_dim), 1, 1, False)
    shift_free, _ = torch.linalg.qr(remove_shift(new_tangents(full_dim)))
    tangent_sets["shift_quotient"] = (shift_free, None, None, True)

    records = {f"{name}_log_growth": [] for name in tangent_sets}
    records.update(
        radial_log_growth=[],
        x_norm=[],
        top_vector_x_uniformity=[],
        top_vector_shift_fraction=[],
        cell_accuracy=[],
        exact_accuracy=[],
    )
    if step.has_cell:
        records.update(
            cell_norm=[],
            top_vector_cell_uniformity=[],
            cell_abs_mean=[],
            forget_mean=[],
            forget_retaining_fraction=[],
        )
    jacobian_checks = []
    check_depths = set(int(depth) for depth in jacobian_check_depths)

    for depth in range(1, max_depth + 1):
        for name, (tangents, source, target, shift_removed) in tangent_sets.items():
            images = jvp_columns(step, state, tangents, source=source, target=target)
            if shift_removed:
                images = remove_shift(images)
            tangents, log_growth = benettin_update(images)
            tangent_sets[name] = (tangents, source, target, shift_removed)
            records[f"{name}_log_growth"].append(log_growth)

        # Growth of a perturbation along x itself, which LayerNorm-based
        # readouts and blocks largely ignore.
        radial = torch.zeros(batch, full_dim, 1, dtype=dtype, device=device)
        x_flat = state[:, 0].reshape(batch, -1)
        radial[:, :block_dim, 0] = x_flat / x_flat.norm(dim=1, keepdim=True)
        records["radial_log_growth"].append(
            torch.log(jvp_columns(step, state, radial)[:, :, 0].norm(dim=1))
        )

        if depth in check_depths and jacobian_check_rows > 0:
            for row_index in range(min(jacobian_check_rows, batch)):
                check = iterative_jacobian_check(step, state[row_index:row_index + 1])
                jacobian_checks.append({
                    "depth": depth,
                    "row": row_index,
                    **check,
                    "full_local_log_growth_top": float(records["full_log_growth"][-1][row_index, 0]),
                })

        with torch.no_grad():
            state, forget_gate = step.step_with_forget_gate(state)
            predictions = step.decode(state)
        targets = apply_rule30(rows, steps=depth)
        correct = predictions == targets
        records["cell_accuracy"].append(correct.to(dtype).mean(dim=1))
        records["exact_accuracy"].append(correct.all(dim=1).to(dtype))
        records["x_norm"].append(state[:, 0].reshape(batch, -1).norm(dim=1))
        full_top = tangent_sets["full"][0][:, :, 0]
        records["top_vector_x_uniformity"].append(
            token_uniformity(full_top[:, :block_dim], cells, width)
        )
        # Near 1 means the raw leading direction is an invisible offset.
        records["top_vector_shift_fraction"].append(
            token_shift_energy_fraction(full_top, components, cells, width)
        )
        if step.has_cell:
            cell = state[:, 1]
            records["cell_norm"].append(cell.reshape(batch, -1).norm(dim=1))
            records["top_vector_cell_uniformity"].append(
                token_uniformity(full_top[:, block_dim:], cells, width)
            )
            records["cell_abs_mean"].append(cell.abs().mean(dim=(1, 2)))
            if forget_gate is None:
                nan = torch.full((batch,), float("nan"), dtype=dtype, device=device)
                records["forget_mean"].append(nan)
                records["forget_retaining_fraction"].append(nan)
            else:
                records["forget_mean"].append(forget_gate.mean(dim=(1, 2)))
                records["forget_retaining_fraction"].append(
                    (forget_gate > FORGET_RETENTION_THRESHOLD).to(dtype).mean(dim=(1, 2))
                )

    # Arrays are indexed [depth - 1, row, ...].
    arrays = {
        name: torch.stack(values).detach().cpu().numpy()
        for name, values in records.items()
    }
    return arrays, jacobian_checks


def summarise(arrays, windows, max_depth):
    """Window means over depths, then means over rows."""
    summary = {"windows": {}, "per_depth": {}}
    growth_names = [name for name in arrays if name.endswith("_log_growth") and name != "radial_log_growth"]
    for first, last in windows:
        last = min(last, max_depth)
        if first > last:
            continue
        span = slice(first - 1, last)
        window = {}
        for name in growth_names:
            # [rows, p]: the FTLE of each Benettin direction over the window.
            exponents = arrays[name][span].mean(axis=0)
            window[name.removesuffix("_log_growth")] = {
                "top_exponent_mean": float(exponents[:, 0].mean()),
                "top_exponent_std_over_rows": float(exponents[:, 0].std()),
                "sum_exponents_mean": float(exponents.sum(axis=1).mean()),
                "sorted_spectrum_mean": np.sort(exponents, axis=1)[:, ::-1].mean(axis=0).tolist(),
            }
        window["radial_log_growth_mean"] = float(arrays["radial_log_growth"][span].mean())
        window["exact_accuracy_mean"] = float(arrays["exact_accuracy"][span].mean())
        summary["windows"][f"{first}-{last}"] = window
    for name in ("cell_accuracy", "exact_accuracy", "x_norm", "top_vector_x_uniformity",
                 "top_vector_shift_fraction", "cell_norm", "cell_abs_mean", "forget_mean",
                 "forget_retaining_fraction"):
        if name in arrays:
            summary["per_depth"][name] = np.nanmean(arrays[name], axis=1).tolist()
    summary["per_depth"]["full_top_local_log_growth"] = arrays["full_log_growth"][:, :, 0].mean(axis=1).tolist()
    summary["per_depth"]["shift_quotient_top_local_log_growth"] = (
        arrays["shift_quotient_log_growth"][:, :, 0].mean(axis=1).tolist()
    )
    return summary


def test_rows(model_args, num_rows, num_cells):
    """The first `num_rows` final-test rows, generated exactly as make_ca_fixed_loaders does."""
    dataset = Rule30Dataset(
        num_samples=num_rows,
        num_cells=num_cells,
        steps=1,
        bernoulli_p=model_args.ca_bernoulli_p,
        seed=int(model_args.ca_test_seed) + int(num_cells),
    )
    return torch.stack([dataset[index]["input_id"] for index in range(num_rows)])


def load_model(source_run_dir, checkpoint_type, checkpoint_dir, device, dtype):
    manifest = read_validated_manifest(Path(source_run_dir) / MANIFEST_FILENAME)
    model_args, _ = resolve_source_args(manifest, device)
    if checkpoint_dir is None:
        checkpoint_dir = manifest["provenance"]["checkpoint_dir"]
    metadata, checkpoint_path = _read_checkpoint_metadata(Path(checkpoint_dir), checkpoint_type)
    model = models.make_model_from_args(model_args)
    load_checkpoint_weights(model, checkpoint_path, metadata, device)
    model.to(dtype=dtype)
    # Only state tangents are needed; this stops JVPs from also recording a
    # reverse-mode graph through the weights.
    model.requires_grad_(False)
    return model, model_args, metadata, checkpoint_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-type", default="best_extrapolation_unconstrained",
                        choices=AVAILABLE_CHECKPOINT_TYPES)
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Override manifest.provenance.checkpoint_dir (e.g. a local copy).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--num-rows", type=int, default=64)
    parser.add_argument("--num-cells", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=60)
    parser.add_argument("--num-vectors", type=int, default=32)
    parser.add_argument("--tangent-seed", type=int, default=0)
    parser.add_argument("--jacobian-check-depths", type=int, nargs="*", default=[])
    parser.add_argument("--jacobian-check-rows", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    dtype = getattr(torch, args.dtype)
    model, model_args, metadata, checkpoint_path = load_model(
        args.source_run_dir, args.checkpoint_type, args.checkpoint_dir, args.device, dtype
    )
    step = BlockStep(model)
    rows = test_rows(model_args, args.num_rows, args.num_cells)
    # The rollout also uses the math kernel, so the trajectory and its
    # Jacobians come from the same attention arithmetic.
    with _math_attention():
        arrays, jacobian_checks = rollout_lyapunov(
            step,
            rows,
            max_depth=args.max_depth,
            num_vectors=args.num_vectors,
            tangent_seed=args.tangent_seed,
            jacobian_check_depths=args.jacobian_check_depths,
            jacobian_check_rows=args.jacobian_check_rows,
        )
    summary = summarise(arrays, DEFAULT_WINDOWS, args.max_depth)
    summary["jacobian_checks"] = jacobian_checks
    summary["provenance"] = {
        "created_at": utc_now(),
        "source_run_dir": str(args.source_run_dir.resolve()),
        "model": model_args.model,
        "checkpoint_type": args.checkpoint_type,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(metadata["step"]),
        "test_seed": int(model_args.ca_test_seed) + int(args.num_cells),
        "arguments": {key: (str(value) if isinstance(value, Path) else value)
                      for key, value in vars(args).items()},
        "torch_version": torch.__version__,
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "lyapunov.npz", rows=rows.numpy(), **arrays)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["windows"], indent=2))


if __name__ == "__main__":
    main()
