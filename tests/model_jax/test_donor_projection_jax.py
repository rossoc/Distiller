# -*- coding: utf-8 -*-
"""Tests for model_jax/donor_projection.py — the Flax NNX port.

Two halves:

1. A port of ``tests/test_donor_projection.py``'s assertions, at the same
   tolerances. Same three claims: gather-then-project is algebraically
   identical to materialize-then-index; the donor tables are frozen and absent
   from trainable state; the PCA init is an orthonormal leading subspace.
2. Cross-framework parity — tier 1 of the validation strategy. The same donor
   tables and the same P/Q in both frameworks must produce matching outputs
   *and* matching gradients. This is the tightest tier (pure matmuls and
   gathers, no recurrence) and it is the reason this slice was ported first.

A toy 300 x 48 donor stands in for the real 262144 x 1536 tables — nothing
here downloads a model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from model_jax.donor_projection import (
    Donor,
    DonorTables,
    _project_table,
    _projected_rms,
    build_projections,
    cached_principal_basis,
    p_from_torch_embedding,
    principal_basis,
    q_from_torch_head,
)

V, D_DONOR, D_MODEL = 300, 48, 16


def _tables(seed: int = 0) -> DonorTables:
    """Toy donor tables, generated identically to the PyTorch test's."""
    generator = torch.Generator().manual_seed(seed)
    embed = (torch.randn(V, D_DONOR, generator=generator) * 0.03).numpy()
    head = (torch.randn(V, D_DONOR, generator=generator) * 0.05).numpy()
    return DonorTables("toy/donor", jnp.asarray(embed), jnp.asarray(head))


def _param_paths(module) -> set:
    """Paths of every nnx.Param in *module* — what the optimizer and orbax see."""
    return {path for path, _ in nnx.to_flat_state(nnx.state(module, nnx.Param))}


def _rngs(seed: int = 0) -> nnx.Rngs:
    return nnx.Rngs(params=seed)


def _build(**kw):
    kw.setdefault("cache_dir", None)
    return build_projections(_tables(), D_MODEL, _rngs(), **kw)


# ---------------------------------------------------------------------------
# Algebraic equivalence
# ---------------------------------------------------------------------------


def test_embedding_matches_indexing_the_materialized_table():
    emb, _ = _build()
    ids = jnp.asarray(np.random.default_rng(0).integers(0, V, (2, 7)))
    table = emb.materialize_weight(jnp.float32)
    assert np.allclose(emb(ids), table[ids], atol=1e-5)


def test_head_matches_the_materialized_small_head():
    _, head = _build()
    hidden = jnp.asarray(np.random.default_rng(1).normal(size=(2, 7, D_MODEL)), jnp.float32)
    small = head.materialize_weight(jnp.float32)
    assert np.allclose(head(hidden), hidden @ small.T, atol=1e-4)


def test_gathered_projection_gives_the_same_gradient_as_the_full_table():
    """dP must not depend on whether we gathered first or projected first."""
    emb, _ = _build()
    ids = jnp.asarray(np.random.default_rng(2).integers(0, V, (2, 7)))
    target = jnp.asarray(np.random.default_rng(3).normal(size=(2, 7, D_MODEL)), jnp.float32)

    def gathered_loss(module):
        return jnp.mean((module(ids) - target) ** 2)

    def full_loss(module):
        full = (module.donor.get_value() @ module.P.get_value()) * module.scale.get_value()
        return jnp.mean((full[ids] - target) ** 2)

    gathered = nnx.grad(gathered_loss)(emb)
    full = nnx.grad(full_loss)(emb)

    assert np.allclose(gathered["P"].get_value(), full["P"].get_value(), atol=1e-6)


# ---------------------------------------------------------------------------
# What is frozen, and what is not
# ---------------------------------------------------------------------------


def test_donor_tables_never_receive_a_gradient():
    emb, head = _build()
    ids = jnp.asarray(np.random.default_rng(4).integers(0, V, (2, 7)))

    def loss(emb_, head_):
        return jnp.mean(head_(emb_(ids)) ** 2)

    grads = nnx.grad(loss, argnums=(0, 1))(emb, head)

    for module_grad, name in zip(grads, ("P", "Q")):
        leaves = jax.tree.leaves(module_grad)
        assert leaves, "expected at least one gradient leaf"
        assert name in module_grad, f"{name} must receive a gradient"
        assert float(jnp.abs(module_grad[name].get_value()).sum()) > 0
        assert "scale" in module_grad
        # The donor table is a Donor, not an nnx.Param, so it is not part of
        # the differentiated state at all — it has no gradient entry to check.
        assert "donor" not in module_grad


def test_donor_tables_are_not_trainable_state():
    """805 MiB per table — trainable state must carry only P/Q and the gains."""
    emb, head = _build()
    assert _param_paths(emb) == {("P",), ("scale",)}
    assert _param_paths(head) == {("Q",), ("scale",)}
    # ...and the donor really is there, just under a different variable type.
    assert isinstance(emb.donor, Donor)
    assert {p for p, _ in nnx.to_flat_state(nnx.state(emb, Donor))} == {("donor",)}


def test_donor_is_excluded_from_a_param_only_checkpoint():
    """The concrete consequence: nnx.state(..., nnx.Param) carries no 805 MiB table."""
    emb, _ = _build()
    saved = nnx.state(emb, nnx.Param)
    total = sum(int(np.asarray(x).size) for x in jax.tree.leaves(saved))
    assert total == D_DONOR * D_MODEL + 1  # P plus the scalar gain, nothing else


def test_learn_scale_false_leaves_no_scalar_parameter():
    emb, head = _build(learn_scales=False)
    assert emb.scale is None and head.scale is None
    assert _param_paths(emb) == {("P",)}


def test_scales_are_initialised_to_normalise_their_output():
    """Embeddings start at unit RMS; logits start at O(1) for unit-RMS input."""
    emb, head = _build()
    embedded = emb(jnp.arange(V)[None, :])
    assert abs(float(jnp.sqrt(jnp.mean(embedded**2))) - 1.0) < 1e-3

    hidden = jnp.asarray(np.random.default_rng(5).normal(size=(64, D_MODEL)), jnp.float32)
    hidden = hidden / jnp.sqrt(jnp.mean(hidden**2, axis=-1, keepdims=True))
    assert 0.2 < float(jnp.std(head(hidden))) < 5.0


# ---------------------------------------------------------------------------
# PCA initialisation
# ---------------------------------------------------------------------------


def test_principal_basis_is_orthonormal():
    basis = principal_basis(_tables().embed, D_MODEL)
    assert basis.shape == (D_DONOR, D_MODEL)
    assert np.allclose(basis.T @ basis, np.eye(D_MODEL), atol=1e-4)


def test_principal_basis_beats_a_random_subspace_at_reconstruction():
    table = np.asarray(_tables().embed)
    pca = principal_basis(table, D_MODEL)
    random = np.linalg.qr(np.random.default_rng(6).normal(size=(D_DONOR, D_MODEL)))[0]

    err = lambda b: float(((table - table @ b @ b.T) ** 2).sum())  # noqa: E731
    assert err(pca) < err(random)


def test_principal_basis_columns_are_ordered_by_decreasing_energy():
    table = np.asarray(_tables().embed)
    basis = principal_basis(table, D_MODEL)
    energy = ((table @ basis) ** 2).sum(axis=0)
    assert np.all(energy[:-1] >= energy[1:])


def test_pca_basis_is_cached_to_disk_and_reused(tmp_path):
    table = _tables().embed
    first = cached_principal_basis(table, D_MODEL, "toy/donor", "embed", str(tmp_path))
    cached = list(tmp_path.iterdir())
    assert len(cached) == 1 and cached[0].name.endswith(f"pca{D_MODEL}.npz")

    second = cached_principal_basis(
        jnp.asarray(np.random.default_rng(7).normal(size=(V, D_DONOR)), jnp.float32),
        D_MODEL,
        "toy/donor",
        "embed",
        str(tmp_path),
    )
    assert np.array_equal(first, second)


def test_jax_cache_file_does_not_collide_with_the_pytorch_one(tmp_path):
    """Both frameworks share a projection_cache_dir; neither may read the other's."""
    torch.save(torch.zeros(D_DONOR, D_MODEL), tmp_path / "toy_donor.embed.pca16.pt")
    basis = cached_principal_basis(
        _tables().embed, D_MODEL, "toy/donor", "embed", str(tmp_path)
    )
    assert basis.shape == (D_DONOR, D_MODEL)
    assert np.abs(basis).sum() > 0  # not the zeros from the .pt file
    assert (tmp_path / "toy_donor.embed.pca16.npz").exists()
    assert (tmp_path / "toy_donor.embed.pca16.pt").exists()


def test_stale_cache_with_the_wrong_shape_is_recomputed(tmp_path):
    np.savez(tmp_path / "toy_donor.embed.pca16.npz", basis=np.zeros((D_DONOR, D_MODEL + 1)))
    basis = cached_principal_basis(
        _tables().embed, D_MODEL, "toy/donor", "embed", str(tmp_path)
    )
    assert basis.shape == (D_DONOR, D_MODEL)
    assert np.allclose(basis.T @ basis, np.eye(D_MODEL), atol=1e-4)


# ---------------------------------------------------------------------------
# Init dispatch and guardrails
# ---------------------------------------------------------------------------


def test_orthogonal_and_xavier_inits_are_accepted():
    for init in ("orthogonal", "xavier"):
        emb, head = _build(init=init)
        assert emb.P.value.shape == (D_DONOR, D_MODEL)
        assert head.Q.value.shape == (D_DONOR, D_MODEL)


def test_unknown_init_is_rejected_by_name():
    with pytest.raises(ValueError, match="svd"):
        _build(init="svd")


def test_widening_past_the_donor_is_rejected():
    with pytest.raises(ValueError, match="compresses"):
        build_projections(_tables(), D_DONOR + 1, _rngs(), cache_dir=None)


# ---------------------------------------------------------------------------
# Chunked full-table helpers
# ---------------------------------------------------------------------------


def test_chunked_helpers_match_their_one_shot_equivalents():
    table = _tables().embed
    projection = principal_basis(table, D_MODEL)

    assert np.allclose(
        _project_table(table, projection, jnp.float32), np.asarray(table) @ projection, atol=1e-5
    )
    expected = float(np.sqrt(np.mean((np.asarray(table) @ projection) ** 2)))
    assert abs(_projected_rms(table, projection) - expected) < 1e-5


def test_chunked_helpers_handle_a_table_shorter_than_one_chunk():
    """_CHUNK_ROWS is 16384; the toy table is 300 rows — one partial chunk."""
    table = _tables().embed[:5]
    projection = np.random.default_rng(8).normal(size=(D_DONOR, D_MODEL)).astype(np.float32)
    assert _project_table(table, projection, jnp.float32).shape == (5, D_MODEL)


# ---------------------------------------------------------------------------
# Cross-framework parity (tier 1: atol 1e-5 / 1e-6)
# ---------------------------------------------------------------------------


def _paired_modules():
    """The same donor tables and the same P/Q, built in both frameworks.

    PyTorch is the reference; its P/Q are transplanted into the JAX modules so
    the only thing under test is the arithmetic, not the initialisation.
    """
    from model.donor_projection import DonorTables as TorchTables
    from model.donor_projection import build_projections as torch_build

    generator = torch.Generator().manual_seed(0)
    embed_t = torch.randn(V, D_DONOR, generator=generator) * 0.03
    head_t = torch.randn(V, D_DONOR, generator=generator) * 0.05
    t_emb, t_head = torch_build(
        TorchTables("toy/donor", embed_t, head_t), D_MODEL, cache_dir=None
    )

    j_tables = DonorTables(
        "toy/donor", jnp.asarray(embed_t.numpy()), jnp.asarray(head_t.numpy())
    )
    j_emb, j_head = build_projections(j_tables, D_MODEL, _rngs(), cache_dir=None)

    # Transplant P/Q (and the gains) from PyTorch into JAX.
    j_emb.P[...] = p_from_torch_embedding(t_emb.proj.weight.detach())
    j_head.Q[...] = q_from_torch_head(t_head.proj.weight.detach())
    j_emb.scale[...] = jnp.asarray(float(t_emb.scale.detach()), jnp.float32)
    j_head.scale[...] = jnp.asarray(float(t_head.scale.detach()), jnp.float32)
    return (t_emb, t_head), (j_emb, j_head)


def test_cross_framework_embedding_outputs_match():
    (t_emb, _), (j_emb, _) = _paired_modules()
    ids_np = np.random.default_rng(9).integers(0, V, (2, 7))

    torch_out = t_emb(torch.as_tensor(ids_np)).detach().numpy()
    jax_out = np.asarray(j_emb(jnp.asarray(ids_np)))
    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_cross_framework_head_outputs_match():
    (_, t_head), (_, j_head) = _paired_modules()
    hidden = np.random.default_rng(10).normal(size=(2, 7, D_MODEL)).astype(np.float32)

    torch_out = t_head(torch.as_tensor(hidden)).detach().numpy()
    jax_out = np.asarray(j_head(jnp.asarray(hidden)))
    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_cross_framework_gradients_match():
    """dP/dQ must agree, not just the forward pass — this is what training sees."""
    (t_emb, t_head), (j_emb, j_head) = _paired_modules()
    ids_np = np.random.default_rng(11).integers(0, V, (2, 7))

    t_head(t_emb(torch.as_tensor(ids_np))).pow(2).mean().backward()
    torch_dp = t_emb.proj.weight.grad.detach().numpy().T  # Pᵀ grad -> P grad
    torch_dq = t_head.proj.weight.grad.detach().numpy()

    ids = jnp.asarray(ids_np)

    def loss(emb_, head_):
        return jnp.mean(head_(emb_(ids)) ** 2)

    g_emb, g_head = nnx.grad(loss, argnums=(0, 1))(j_emb, j_head)

    assert np.allclose(torch_dp, np.asarray(g_emb["P"].get_value()), atol=1e-5)
    assert np.allclose(torch_dq, np.asarray(g_head["Q"].get_value()), atol=1e-5)


def test_cross_framework_pca_bases_span_the_same_subspace():
    """Eigenvector signs are arbitrary, so compare the subspace, not the entries."""
    from model.donor_projection import principal_basis as torch_basis

    generator = torch.Generator().manual_seed(0)
    embed_t = torch.randn(V, D_DONOR, generator=generator) * 0.03

    t_b = torch_basis(embed_t, D_MODEL).detach().numpy()
    j_b = principal_basis(jnp.asarray(embed_t.numpy()), D_MODEL)

    # Sign-invariant: |t_bᵀ j_b| must be (close to) the identity.
    overlap = np.abs(t_b.T @ j_b)
    assert np.allclose(overlap, np.eye(D_MODEL), atol=1e-3)
