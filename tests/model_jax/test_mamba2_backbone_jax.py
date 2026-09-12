# -*- coding: utf-8 -*-
"""Tests for the hand-rolled Flax NNX Mamba2 backbone.

The centrepiece is :func:`test_backbone_hidden_states_match_pytorch` — tier 2
of the cross-framework validation strategy, and the single highest-value check
in the port. Everything else in the JAX path is built on top of the backbone,
so a subtle recurrence bug here would surface downstream as "training doesn't
converge as well", which is exactly the kind of failure that is expensive to
diagnose later and cheap to catch now.

The comparison is against ``transformers``' *pure-PyTorch* Mamba2 path, which
is what this project actually runs — ``tests/conftest.py`` blocks
mamba-ssm/causal-conv1d suite-wide, and neither they nor the ``kernels``
package are installed.

Tolerance: the two implementations are independent, so bit-identity is not the
target. These run at float32 with matched weights and matched inputs, which is
the tightest setting available; the numbers below were measured, not guessed
(see ``test_parity_headroom`` for the recorded margin).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from model_jax.mamba2_backbone import (
    Mamba2Backbone,
    config_from_torch,
    load_from_torch,
)
from model_jax.mamba2_block import Mamba2Config

# Toy geometry. chunk_size deliberately does not divide seq_len, so the
# padding/trim path in the chunked scan is exercised on every test.
VOCAB, D_MODEL, LAYERS = 64, 32, 2
STATE, EXPAND, HEAD_DIM, GROUPS, CONV_K, CHUNK = 8, 2, 16, 1, 4, 8
BATCH, SEQ = 2, 13


class PlainEmbedding(nnx.Module):
    """Stand-in for ProjectedEmbedding — the backbone only needs `ids -> vectors`."""

    def __init__(self, vocab: int, dim: int):
        self.table = nnx.Param(jnp.zeros((vocab, dim), jnp.float32))

    def __call__(self, ids):
        return self.table.get_value()[ids]


def _torch_model():
    from transformers import Mamba2Config as TorchConfig
    from transformers import Mamba2ForCausalLM

    torch.manual_seed(0)
    cfg = TorchConfig(
        vocab_size=VOCAB,
        hidden_size=D_MODEL,
        num_hidden_layers=LAYERS,
        num_heads=(EXPAND * D_MODEL) // HEAD_DIM,
        head_dim=HEAD_DIM,
        state_size=STATE,
        expand=EXPAND,
        n_groups=GROUPS,
        conv_kernel=CONV_K,
        chunk_size=CHUNK,
        tie_word_embeddings=False,
        use_cache=False,
    )
    model = Mamba2ForCausalLM(cfg).to(torch.float32).eval()
    return model


def _paired(gradient_checkpointing: bool = False):
    """A torch Mamba2 and a JAX backbone carrying exactly its weights."""
    torch_model = _torch_model()
    cfg = config_from_torch(torch_model.config)

    embeddings = PlainEmbedding(VOCAB, D_MODEL)
    backbone = Mamba2Backbone(
        cfg,
        embeddings,
        rngs=nnx.Rngs(params=0),
        gradient_checkpointing=gradient_checkpointing,
    )
    load_from_torch(backbone, torch_model.backbone)
    embeddings.table[...] = jnp.asarray(
        torch_model.backbone.embeddings.weight.detach().numpy()
    )
    return torch_model, backbone


def _ids(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, VOCAB, (BATCH, SEQ))


# ---------------------------------------------------------------------------
# Config validation — the same guardrail as the PyTorch _build_config
# ---------------------------------------------------------------------------


def test_num_heads_is_derived_from_expand_and_head_dim():
    cfg = Mamba2Config.create(
        vocab_size=VOCAB, d_model=64, num_hidden_layers=1, state_size=8,
        expand=2, head_dim=16, n_groups=1, conv_kernel=4, chunk_size=8,
        initializer_range=0.02,
    )
    assert cfg.num_heads == (2 * 64) // 16
    assert cfg.intermediate_size == 128
    assert cfg.conv_dim == 128 + 2 * 1 * 8


def test_indivisible_head_dim_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="divisible"):
        Mamba2Config.create(
            vocab_size=VOCAB, d_model=32, num_hidden_layers=1, state_size=8,
            expand=2, head_dim=24, n_groups=1, conv_kernel=4, chunk_size=8,
            initializer_range=0.02,
        )


# ---------------------------------------------------------------------------
# Shapes and gradient reachability
# ---------------------------------------------------------------------------


def test_forward_shape():
    _, backbone = _paired()
    out = backbone(jnp.asarray(_ids()))
    assert out.shape == (BATCH, SEQ, D_MODEL)
    assert np.all(np.isfinite(np.asarray(out)))


def test_every_block_parameter_receives_a_gradient():
    """A dead parameter is a silent bug — every leaf must be reachable."""
    _, backbone = _paired()
    ids = jnp.asarray(_ids())

    grads = nnx.grad(lambda m: jnp.sum(m(ids) ** 2))(backbone)
    flat = nnx.to_flat_state(grads)
    assert flat, "expected gradients"

    dead = [
        "/".join(map(str, path))
        for path, var in flat
        if float(jnp.abs(var.get_value()).sum()) == 0.0
    ]
    assert not dead, f"parameters with zero gradient: {dead}"


def test_gradient_checkpointing_does_not_change_the_result():
    """nnx.remat is a memory strategy, not a numerical one."""
    ids = jnp.asarray(_ids())
    _, plain = _paired(gradient_checkpointing=False)
    _, remat = _paired(gradient_checkpointing=True)

    assert np.allclose(np.asarray(plain(ids)), np.asarray(remat(ids)), atol=1e-6)

    loss = lambda m: jnp.sum(m(ids) ** 2)  # noqa: E731
    g_plain = nnx.to_flat_state(nnx.grad(loss)(plain))
    g_remat = dict(nnx.to_flat_state(nnx.grad(loss)(remat)))
    for path, var in g_plain:
        assert np.allclose(
            np.asarray(var.get_value()),
            np.asarray(g_remat[path].get_value()),
            atol=1e-5,
        ), path


# ---------------------------------------------------------------------------
# Tier 2: cross-framework parity against the PyTorch fallback
# ---------------------------------------------------------------------------


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))


def test_backbone_hidden_states_match_pytorch():
    """The Phase B gate: matched weights, matched input, same hidden states."""
    torch_model, backbone = _paired()
    ids_np = _ids()

    with torch.no_grad():
        torch_out = torch_model.backbone(
            input_ids=torch.as_tensor(ids_np), use_cache=False, return_dict=True
        )[0].numpy()

    jax_out = np.asarray(backbone(jnp.asarray(ids_np)))

    assert jax_out.shape == torch_out.shape
    assert np.allclose(jax_out, torch_out, atol=1e-4, rtol=1e-3), (
        f"max abs {np.abs(jax_out - torch_out).max():.3e}, "
        f"max rel {_max_rel(jax_out, torch_out):.3e}"
    )


def test_every_layers_hidden_state_matches_pytorch():
    """Layer by layer, so a mismatch localises instead of only showing at the end."""
    torch_model, backbone = _paired()
    ids_np = _ids(1)

    with torch.no_grad():
        torch_out = torch_model.backbone(
            input_ids=torch.as_tensor(ids_np),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    torch_hidden = [h.numpy() for h in torch_out.hidden_states]

    _, jax_hidden = backbone(jnp.asarray(ids_np), output_hidden_states=True)
    jax_hidden = [np.asarray(h) for h in jax_hidden]

    assert len(jax_hidden) == len(torch_hidden)
    for i, (j, t) in enumerate(zip(jax_hidden, torch_hidden)):
        assert np.allclose(j, t, atol=1e-4, rtol=1e-3), (
            f"layer {i}: max abs {np.abs(j - t).max():.3e}, max rel {_max_rel(j, t):.3e}"
        )


def test_attention_mask_zeroes_padded_positions_like_pytorch():
    """Padding handling is its own source of drift — pin it separately."""
    torch_model, backbone = _paired()
    ids_np = _ids(2)
    mask_np = np.ones((BATCH, SEQ), dtype=np.int32)
    mask_np[0, -4:] = 0  # a padded tail on the first row

    with torch.no_grad():
        torch_out = torch_model.backbone(
            input_ids=torch.as_tensor(ids_np),
            attention_mask=torch.as_tensor(mask_np),
            use_cache=False,
            return_dict=True,
        )[0].numpy()

    jax_out = np.asarray(
        backbone(jnp.asarray(ids_np), attention_mask=jnp.asarray(mask_np))
    )
    assert np.allclose(jax_out, torch_out, atol=1e-4, rtol=1e-3), (
        f"max abs {np.abs(jax_out - torch_out).max():.3e}"
    )


def test_chunk_size_is_not_vestigial():
    """chunk_size is a live Optuna axis; changing it must not change the result.

    The chunked scan is an exact reformulation of the recurrence, so different
    block lengths are a memory/compute tradeoff and nothing else. If this ever
    fails, the scan's chunk-boundary handling is wrong — and the Optuna search
    over chunk_size would be optimising a numerical artefact.
    """
    ids = jnp.asarray(_ids(3))
    outputs = []
    for chunk in (4, 8, 16, SEQ):
        torch_model = _torch_model()
        cfg = config_from_torch(torch_model.config)
        cfg = Mamba2Config(**{**cfg.__dict__, "chunk_size": chunk})
        embeddings = PlainEmbedding(VOCAB, D_MODEL)
        backbone = Mamba2Backbone(cfg, embeddings, rngs=nnx.Rngs(params=0),
                                  gradient_checkpointing=False)
        load_from_torch(backbone, torch_model.backbone)
        embeddings.table[...] = jnp.asarray(
            torch_model.backbone.embeddings.weight.detach().numpy()
        )
        outputs.append(np.asarray(backbone(ids)))

    for i, out in enumerate(outputs[1:], start=1):
        assert np.allclose(out, outputs[0], atol=1e-4), f"chunk variant {i} disagrees"


def test_parity_headroom():
    """Record the actual margin, so the tolerance above is evidence-based."""
    torch_model, backbone = _paired()
    ids_np = _ids(4)
    with torch.no_grad():
        t = torch_model.backbone(
            input_ids=torch.as_tensor(ids_np), use_cache=False, return_dict=True
        )[0].numpy()
    j = np.asarray(backbone(jnp.asarray(ids_np)))
    abs_err, rel_err = float(np.abs(j - t).max()), _max_rel(j, t)
    print(f"\nbackbone parity: max abs {abs_err:.3e}, max rel {rel_err:.3e}")
    # An order of magnitude of headroom under the asserted tolerance.
    assert abs_err < 1e-5, abs_err
