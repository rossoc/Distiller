# -*- coding: utf-8 -*-
"""Tests for model/donor_projection.py — the frozen-donor projections.

The whole construction rests on three claims, and these tests pin all three:

1. Gathering donor rows and projecting them is *identical* to materializing
   the small table and indexing it (embedding), and up-projecting into donor
   space then applying the donor head is *identical* to applying the
   materialized small head (output).
2. The donor tables never receive a gradient and never enter a checkpoint,
   while P/Q always do.
3. The PCA initialisation is an orthonormal basis and really is the leading
   subspace.

A toy 300 x 48 donor stands in for the real 262144 x 1536 tables — nothing
here downloads a model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.donor_projection import (
    DonorTables,
    ProjectedEmbedding,
    ProjectedLMHead,
    _project_table,
    _projected_rms,
    build_projections,
    cached_principal_basis,
    principal_basis,
)

V, D_DONOR, D_MODEL = 300, 48, 16


def _tables(seed: int = 0) -> DonorTables:
    generator = torch.Generator().manual_seed(seed)
    embed = torch.randn(V, D_DONOR, generator=generator) * 0.03
    head = torch.randn(V, D_DONOR, generator=generator) * 0.05
    return DonorTables("toy/donor", embed, head)


# ---------------------------------------------------------------------------
# Algebraic equivalence
# ---------------------------------------------------------------------------


def test_embedding_matches_indexing_the_materialized_table():
    emb, _ = build_projections(_tables(), D_MODEL, cache_dir=None)
    ids = torch.randint(0, V, (2, 7))
    table = emb.materialize_weight(torch.float32)
    assert torch.allclose(emb(ids), F.embedding(ids, table), atol=1e-5)


def test_head_matches_the_materialized_small_head():
    _, head = build_projections(_tables(), D_MODEL, cache_dir=None)
    hidden = torch.randn(2, 7, D_MODEL)
    small = head.materialize_weight(torch.float32)
    assert torch.allclose(head(hidden), F.linear(hidden, small), atol=1e-4)


def test_gathered_projection_gives_the_same_gradient_as_the_full_table():
    """dP must not depend on whether we gathered first or projected first."""
    tables = _tables()
    emb, _ = build_projections(tables, D_MODEL, cache_dir=None)
    ids = torch.randint(0, V, (2, 7))
    target = torch.randn(2, 7, D_MODEL)

    (emb(ids) - target).pow(2).mean().backward()
    gathered = emb.proj.weight.grad.clone()

    emb.proj.weight.grad = None
    full = (emb.donor @ emb.proj.weight.T) * emb.scale
    (F.embedding(ids, full) - target).pow(2).mean().backward()

    assert torch.allclose(gathered, emb.proj.weight.grad, atol=1e-6)


# ---------------------------------------------------------------------------
# What is frozen, and what is not
# ---------------------------------------------------------------------------


def test_donor_tables_never_receive_a_gradient():
    emb, head = build_projections(_tables(), D_MODEL, cache_dir=None)
    ids = torch.randint(0, V, (2, 7))

    head(emb(ids)).pow(2).mean().backward()

    assert emb.donor.grad is None and head.donor.grad is None
    assert not emb.donor.requires_grad and not head.donor.requires_grad
    for module in (emb, head):
        assert module.proj.weight.grad.abs().sum() > 0
        assert module.scale.grad is not None


def test_donor_tables_stay_out_of_the_state_dict():
    """805 MiB per table — a checkpoint must carry only P, Q and the gains."""
    emb, head = build_projections(_tables(), D_MODEL, cache_dir=None)
    assert set(emb.state_dict()) == {"proj.weight", "scale"}
    assert set(head.state_dict()) == {"proj.weight", "scale"}


def test_learn_scale_false_leaves_no_scalar_parameter():
    emb, head = build_projections(
        _tables(), D_MODEL, learn_scales=False, cache_dir=None
    )
    assert emb.scale is None and head.scale is None
    assert set(emb.state_dict()) == {"proj.weight"}


def test_scales_are_initialised_to_normalise_their_output():
    """Embeddings start at unit RMS; logits start at O(1) for unit-RMS input."""
    emb, head = build_projections(_tables(), D_MODEL, cache_dir=None)
    with torch.no_grad():
        embedded = emb(torch.arange(V).unsqueeze(0))
    assert abs(float(embedded.pow(2).mean().sqrt()) - 1.0) < 1e-3

    hidden = torch.randn(64, D_MODEL)
    hidden = hidden / hidden.pow(2).mean(-1, keepdim=True).sqrt()
    with torch.no_grad():
        assert 0.2 < float(head(hidden).std()) < 5.0


# ---------------------------------------------------------------------------
# The lm_head's .weight shim
# ---------------------------------------------------------------------------


def test_head_exposes_a_weight_with_the_compute_dtype():
    """Mamba2ForCausalLM.forward reads lm_head.weight.dtype to cast into."""
    _, head = build_projections(_tables(), D_MODEL, cache_dir=None)
    head = head.to(torch.float16)
    assert head.weight.dtype == torch.float16
    assert head.weight.shape == (D_DONOR, D_MODEL)


# ---------------------------------------------------------------------------
# PCA initialisation
# ---------------------------------------------------------------------------


def test_principal_basis_is_orthonormal():
    basis = principal_basis(_tables().embed, D_MODEL)
    assert basis.shape == (D_DONOR, D_MODEL)
    assert torch.allclose(basis.T @ basis, torch.eye(D_MODEL), atol=1e-4)


def test_principal_basis_beats_a_random_subspace_at_reconstruction():
    table = _tables().embed
    pca = principal_basis(table, D_MODEL)
    random = torch.linalg.qr(torch.randn(D_DONOR, D_MODEL))[0]

    err = lambda b: float((table - table @ b @ b.T).pow(2).sum())  # noqa: E731
    assert err(pca) < err(random)


def test_principal_basis_columns_are_ordered_by_decreasing_energy():
    table = _tables().embed
    basis = principal_basis(table, D_MODEL)
    energy = (table @ basis).pow(2).sum(dim=0)
    assert torch.all(energy[:-1] >= energy[1:])


def test_pca_basis_is_cached_to_disk_and_reused(tmp_path):
    table = _tables().embed
    first = cached_principal_basis(table, D_MODEL, "toy/donor", "embed", str(tmp_path))
    cached = list(tmp_path.iterdir())
    assert len(cached) == 1 and cached[0].name.endswith(f"pca{D_MODEL}.pt")

    # Reuse must not depend on the table any more: a different table with the
    # same (donor, table, out_dim) key returns the cached basis unchanged.
    second = cached_principal_basis(
        torch.randn(V, D_DONOR), D_MODEL, "toy/donor", "embed", str(tmp_path)
    )
    assert torch.equal(first, second)


def test_stale_cache_with_the_wrong_shape_is_recomputed(tmp_path):
    table = _tables().embed
    path = tmp_path / "toy_donor.embed.pca16.pt"
    torch.save(torch.zeros(D_DONOR, D_MODEL + 1), path)

    basis = cached_principal_basis(
        table, D_MODEL, "toy/donor", "embed", str(tmp_path)
    )
    assert basis.shape == (D_DONOR, D_MODEL)
    assert torch.allclose(basis.T @ basis, torch.eye(D_MODEL), atol=1e-4)


# ---------------------------------------------------------------------------
# Init dispatch and guardrails
# ---------------------------------------------------------------------------


def test_orthogonal_and_xavier_inits_are_accepted():
    for init in ("orthogonal", "xavier"):
        emb, head = build_projections(_tables(), D_MODEL, init=init, cache_dir=None)
        assert emb.proj.weight.shape == (D_MODEL, D_DONOR)
        assert head.proj.weight.shape == (D_DONOR, D_MODEL)


def test_unknown_init_is_rejected_by_name():
    try:
        build_projections(_tables(), D_MODEL, init="svd", cache_dir=None)
    except ValueError as exc:
        assert "svd" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown init")


def test_widening_past_the_donor_is_rejected():
    """d_model > d_donor is not a compression — say so instead of silently failing."""
    try:
        build_projections(_tables(), D_DONOR + 1, cache_dir=None)
    except ValueError as exc:
        assert "compresses" in str(exc)
    else:
        raise AssertionError("expected ValueError for d_model > d_donor")


# ---------------------------------------------------------------------------
# Chunked full-table helpers
# ---------------------------------------------------------------------------


def test_chunked_helpers_match_their_one_shot_equivalents():
    table = _tables().embed
    projection = principal_basis(table, D_MODEL)

    assert torch.allclose(
        _project_table(table, projection, torch.float32), table @ projection, atol=1e-5
    )
    expected = float((table @ projection).pow(2).mean().sqrt())
    assert abs(_projected_rms(table, projection) - expected) < 1e-5


def test_chunked_helpers_handle_a_table_shorter_than_one_chunk():
    """_CHUNK_ROWS is 16384; the toy table is 300 rows — one partial chunk."""
    table = _tables().embed[:5]
    projection = torch.randn(D_DONOR, D_MODEL)
    assert _project_table(table, projection, torch.float32).shape == (5, D_MODEL)
