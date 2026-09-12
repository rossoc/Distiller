# -*- coding: utf-8 -*-
"""Tests for model_jax/mimir_mamba2.py — the assembled JAX model.

A port of ``tests/test_mimir_mamba2.py``'s model-level assertions (config
validation, gradient reachability, chunked == one-shot, the degenerate-batch
guard, the optimiser grouping), plus tier-3 cross-framework loss parity against
the PyTorch module.

A toy donor stands in for the real 262144 x 1536 tables — nothing downloads.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from model_jax.donor_projection import DonorTables
from model_jax.mimir_mamba2 import (
    IGNORE_INDEX,
    MimirMamba2Model,
    chunked_loss,
    guard_nonfinite_loss,
    param_group,
)

VOCAB, D_DONOR = 96, 48
BATCH, SEQ = 2, 13


def _tables(seed: int = 0) -> DonorTables:
    g = torch.Generator().manual_seed(seed)
    return DonorTables(
        "toy/donor",
        jnp.asarray((torch.randn(VOCAB, D_DONOR, generator=g) * 0.03).numpy()),
        jnp.asarray((torch.randn(VOCAB, D_DONOR, generator=g) * 0.05).numpy()),
    )


def _model(**overrides) -> MimirMamba2Model:
    kwargs = dict(
        d_model=32,
        num_hidden_layers=2,
        state_size=8,
        head_dim=16,
        chunk_size=8,
        projection_cache_dir=None,
        loss_chunk_tokens=8,
    )
    kwargs.update(overrides)
    return MimirMamba2Model(_tables(), rngs=nnx.Rngs(params=0), **kwargs)


def _batch(seed: int = 0, masked_tail: int = 0):
    rng = np.random.default_rng(seed)
    ids = jnp.asarray(rng.integers(0, VOCAB, (BATCH, SEQ)))
    labels = np.asarray(rng.integers(0, VOCAB, (BATCH, SEQ)))
    if masked_tail:
        labels[:, -masked_tail:] = IGNORE_INDEX
    return ids, jnp.asarray(labels)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_projections_are_wired_into_the_backbone_and_head():
    model = _model()
    assert model.projected_embedding.num_embeddings == VOCAB
    assert model.projected_lm_head.vocab_size == VOCAB
    assert model.cfg.vocab_size == VOCAB
    assert model.cfg.hidden_size == 32
    assert model.cfg.num_heads == (2 * 32) // 16


def test_indivisible_head_dim_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="divisible"):
        _model(head_dim=24)


def test_donor_tables_are_absent_from_trainable_state():
    """The two 805 MiB tables must never reach the optimiser or a checkpoint."""
    model = _model()
    params = nnx.state(model, nnx.Param)
    flat = nnx.to_flat_state(params)

    paths = {"/".join(map(str, p)) for p, _ in flat}
    assert not any("donor" in p for p in paths), paths
    # Nothing with a donor table's shape may be in there either — a table that
    # leaked in under some other name would still cost 805 MiB a checkpoint.
    assert not any(
        var.get_value().shape == (VOCAB, D_DONOR) for _, var in flat
    ), paths

    # The donor tables really do exist on the model, just not as Params.
    from model_jax.donor_projection import Donor

    donors = nnx.to_flat_state(nnx.state(model, Donor))
    assert len(donors) == 2
    assert all(var.get_value().shape == (VOCAB, D_DONOR) for _, var in donors)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_backward_reaches_the_projections_and_the_blocks_but_not_the_donor():
    model = _model()
    ids, labels = _batch()

    grads = nnx.grad(lambda m: m.loss(ids, labels))(model)
    flat = dict(nnx.to_flat_state(grads))
    paths = {"/".join(map(str, p)) for p in flat}

    assert any(p.endswith("embeddings/P") for p in paths)
    assert any(p.startswith("lm_head/Q") for p in paths)
    assert any("mixer" in p for p in paths)
    assert not any("donor" in p for p in paths)

    dead = [
        "/".join(map(str, p))
        for p, v in flat.items()
        if float(jnp.abs(v.get_value()).sum()) == 0.0
    ]
    assert not dead, f"parameters with zero gradient: {dead}"


# ---------------------------------------------------------------------------
# The chunked loss
# ---------------------------------------------------------------------------


def test_chunked_loss_matches_the_one_shot_loss():
    """The chunking is an exact algebraic identity, not an approximation."""
    model = _model()
    ids, labels = _batch(1)
    hidden = model.hidden_states(ids)

    losses, grads = {}, {}
    for chunk in (0, 16, 5):
        losses[chunk] = float(chunked_loss(model.lm_head, hidden, labels, chunk))
        g = nnx.grad(lambda h, c=chunk: chunked_loss(h, hidden, labels, c))(model.lm_head)
        grads[chunk] = {"/".join(map(str, p)): v.get_value() for p, v in nnx.to_flat_state(g)}

    for chunk in (16, 5):
        assert losses[chunk] == pytest.approx(losses[0], abs=1e-5)
        assert set(grads[chunk]) == set(grads[0])
        for name, grad in grads[chunk].items():
            assert np.allclose(grad, grads[0][name], atol=1e-5), name


def test_a_chunk_size_that_does_not_divide_the_sequence_is_handled():
    """The padded tail must contribute nothing — chunk counts are static."""
    model = _model()
    ids, labels = _batch(2)
    hidden = model.hidden_states(ids)
    total = BATCH * SEQ - 1
    one_shot = float(chunked_loss(model.lm_head, hidden, labels, 0))
    for chunk in (6, 7, 24):
        assert total % chunk != 0, f"chunk {chunk} needs a ragged final slice"
        assert float(
            chunked_loss(model.lm_head, hidden, labels, chunk)
        ) == pytest.approx(one_shot, abs=1e-5), chunk


def test_cross_row_boundary_is_masked():
    """Row i's last position must not be trained to predict row i+1's first."""
    model = _model()
    ids, labels = _batch(3)
    hidden = model.hidden_states(ids)

    baseline = float(chunked_loss(model.lm_head, hidden, labels, 8))
    # Changing only the first token of row 1 changes the boundary target, which
    # is masked — so the loss must not move.
    perturbed = labels.at[1, 0].set((int(labels[1, 0]) + 1) % VOCAB)
    assert float(chunked_loss(model.lm_head, hidden, perturbed, 8)) == pytest.approx(
        baseline, abs=1e-6
    )


def test_fully_masked_batch_yields_a_finite_zero_loss():
    """Every label masked: no valid tokens, so a finite 0, not a nan."""
    model = _model()
    ids = jnp.asarray(np.random.default_rng(4).integers(0, VOCAB, (BATCH, SEQ)))
    labels = jnp.full((BATCH, SEQ), IGNORE_INDEX)

    loss = model.loss(ids, labels)
    assert jnp.isfinite(loss) and float(loss) == 0.0

    grads = nnx.grad(lambda m: m.loss(ids, labels))(model)
    for path, var in nnx.to_flat_state(grads):
        g = np.asarray(var.get_value())
        assert np.all(np.isfinite(g)), path
        assert np.abs(g).sum() == 0.0, path


def test_guard_replaces_a_non_finite_loss_with_zero():
    assert float(guard_nonfinite_loss(jnp.asarray(jnp.nan))) == 0.0
    assert float(guard_nonfinite_loss(jnp.asarray(jnp.inf))) == 0.0
    assert float(guard_nonfinite_loss(jnp.asarray(2.5))) == 2.5


def test_loss_is_jittable_and_shape_stable():
    """No data-dependent branches — the whole reason the loss was restructured."""
    model = _model()
    graphdef, state = nnx.split(model)

    @jax.jit
    def step(state, ids, labels):
        return nnx.merge(graphdef, state).loss(ids, labels)

    ids, labels = _batch(5, masked_tail=3)
    first = float(step(state, ids, labels))
    # A second call with the same shapes must not retrace, and must agree.
    ids2, labels2 = _batch(6, masked_tail=3)
    assert np.isfinite(first)
    assert np.isfinite(float(step(state, ids2, labels2)))
    assert step._cache_size() == 1


# ---------------------------------------------------------------------------
# argmax helper
# ---------------------------------------------------------------------------


def test_argmax_token_ids_matches_a_full_argmax():
    model = _model()
    ids, _ = _batch(7)
    chunked = np.asarray(model.argmax_token_ids(ids))
    full = np.asarray(jnp.argmax(model(ids), axis=-1))
    assert chunked.shape == (BATCH, SEQ)
    assert np.array_equal(chunked, full)


# ---------------------------------------------------------------------------
# Optimiser grouping
# ---------------------------------------------------------------------------


def test_param_groups_split_projections_from_body_and_decay_from_no_decay():
    model = _model()
    groups = {}
    for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        groups.setdefault(param_group(path), []).append("/".join(map(str, path)))

    assert set(groups) == {"proj_decay", "proj_no_decay", "body_decay", "body_no_decay"}
    assert any(p.endswith("embeddings/P") for p in groups["proj_decay"])
    assert any(p.startswith("lm_head/Q") for p in groups["proj_decay"])
    # Scalar gains, A_log, D, dt_bias and every norm weight avoid decay.
    assert all("scale" in p for p in groups["proj_no_decay"])
    for name in ("A_log", "D", "dt_bias"):
        assert any(p.endswith(name) for p in groups["body_no_decay"]), name
    assert all("norm" not in p.lower() for p in groups["body_decay"])


# ---------------------------------------------------------------------------
# Tier 3: cross-framework loss parity
# ---------------------------------------------------------------------------


def test_loss_matches_the_pytorch_module():
    """Same donor, same P/Q, same blocks, same batch -> same loss and dP/dQ.

    Tier 3 compounds tier 2's backbone error through the head and the
    cross-entropy, so the tolerance is looser than the backbone's own (~1e-6)
    but far tighter than the fused-vs-fallback budget the PyTorch suite uses
    for its own within-framework check.
    """
    import model.mimir_mamba2 as torch_mod
    from model.donor_projection import DonorTables as TorchTables
    from model_jax.donor_projection import p_from_torch_embedding, q_from_torch_head
    from model_jax.mamba2_backbone import load_from_torch

    g = torch.Generator().manual_seed(0)
    embed_t = torch.randn(VOCAB, D_DONOR, generator=g) * 0.03
    head_t = torch.randn(VOCAB, D_DONOR, generator=g) * 0.05
    torch_tables = TorchTables("toy/donor", embed_t, head_t)

    class _Tok:
        pad_token_id, eos_token_id, bos_token_id = 0, 1, 2

    torch_mod.load_tokenizer = lambda *a, **k: _Tok()
    torch_mod.load_donor_tables = lambda *a, **k: torch_tables

    torch.manual_seed(0)
    t_module = torch_mod.MimirMamba2Module(
        donor_model_id="toy/donor",
        dtype="fp32",
        d_model=32,
        num_hidden_layers=2,
        state_size=8,
        head_dim=16,
        chunk_size=8,
        projection_cache_dir=None,
        loss_chunk_tokens=8,
        gradient_checkpointing=False,
    )
    t_module.eval()

    j_model = _model(gradient_checkpointing=False)
    load_from_torch(j_model.backbone, t_module.model.backbone)
    j_model.projected_embedding.P[...] = p_from_torch_embedding(
        t_module.projected_embedding.proj.weight.detach()
    )
    j_model.projected_lm_head.Q[...] = q_from_torch_head(
        t_module.projected_lm_head.proj.weight.detach()
    )
    j_model.projected_embedding.scale[...] = jnp.asarray(
        float(t_module.projected_embedding.scale.detach()), jnp.float32
    )
    j_model.projected_lm_head.scale[...] = jnp.asarray(
        float(t_module.projected_lm_head.scale.detach()), jnp.float32
    )

    ids, labels = _batch(8, masked_tail=2)
    ids_t = torch.as_tensor(np.asarray(ids), dtype=torch.long)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    t_loss, _ = t_module.forward(ids_t, None, labels_t)
    j_loss = j_model.loss(ids, labels)

    print(f"\nloss parity: torch {float(t_loss):.8f}  jax {float(j_loss):.8f}")
    assert float(j_loss) == pytest.approx(float(t_loss), rel=1e-4)

    # ...and the gradients the optimiser would actually see.
    t_loss.backward()
    grads = nnx.grad(lambda m: m.loss(ids, labels))(j_model)
    flat = dict(nnx.to_flat_state(grads))

    torch_dp = t_module.projected_embedding.proj.weight.grad.detach().numpy().T
    torch_dq = t_module.projected_lm_head.proj.weight.grad.detach().numpy()
    jax_dp = np.asarray(flat[("backbone", "embeddings", "P")].get_value())
    jax_dq = np.asarray(flat[("lm_head", "Q")].get_value())

    for name, a, b in (("dP", jax_dp, torch_dp), ("dQ", jax_dq, torch_dq)):
        rel = np.abs(a - b).max() / max(np.abs(b).max(), 1e-6)
        print(f"{name}: max rel {rel:.3e}")
        assert rel < 1e-3, f"{name} relative diff {rel:.4f}"
