# -*- coding: utf-8 -*-
"""Mamba2 + frozen-donor projections, in Flax NNX.

The JAX counterpart of ``model.mimir_mamba2.MimirMamba2Module``, minus
Lightning: this is a plain ``nnx.Module`` holding the model and its loss, and
``src/train_jax.py`` owns the optimisation. Same three ideas as the PyTorch
version (keep the donor's tables, drop attention for recurrence, shrink the
width) — see that module's docstring.

    input_ids -> ProjectedEmbedding -> Mamba2 backbone -> ProjectedLMHead -> logits
                 E_donor[ids] @ P       N SSM blocks       (h @ Qᵀ) @ W_donorᵀ
                 frozen · 0.8M train     ~20M train         frozen · 0.8M train

The one part that is *not* a line-by-line port is the chunked loss — see
:func:`chunked_loss` for why it cannot be, and what replaced each branch.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from model_jax.donor_projection import (
    DonorTables,
    ProjectedEmbedding,
    ProjectedLMHead,
    build_projections,
)
from model_jax.mamba2_backbone import Mamba2Backbone
from model_jax.mamba2_block import Mamba2Config

log = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _chunk_cross_entropy(
    head: ProjectedLMHead, hidden_chunk: jnp.ndarray, labels_chunk: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Summed (not averaged) CE for one slice, plus that slice's valid count.

    ``reduction="sum"`` so the caller can divide by the *batch-wide* count of
    unmasked tokens once; per-chunk means would weight a short trailing chunk
    as heavily as a full one.

    optax has no ``ignore_index``, so the ``-100`` masking is explicit: clamp
    the index (a negative label would gather a clamped row, silently, in JAX)
    and then zero the term. The float32 cast is explicit rather than left to a
    precision policy, so the loss is computed identically at every dtype — the
    same reason the PyTorch version calls ``.float()``.
    """
    logits = head(hidden_chunk).astype(jnp.float32)
    valid = labels_chunk != IGNORE_INDEX
    safe_labels = jnp.where(valid, labels_chunk, 0)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(log_probs, safe_labels[:, None], axis=-1)[:, 0]
    return -jnp.sum(jnp.where(valid, picked, 0.0)), jnp.sum(valid)


def chunked_loss(
    head: ProjectedLMHead,
    hidden_states: jnp.ndarray,
    labels: jnp.ndarray,
    chunk: int,
    remat: bool = True,
) -> jnp.ndarray:
    """Causal-LM cross-entropy over the donor-sized vocabulary, in slices.

    Same contract as the PyTorch ``_chunked_loss``: shift by one, mask the
    cross-row boundary, accumulate summed CE over ``chunk``-sized slices, and
    divide once by the batch-wide valid-token count. Each slice is wrapped in
    ``jax.checkpoint`` so the ``[chunk, 262144]`` logits are recomputed in
    backward rather than retained — the whole point of chunking.

    The slices are driven through ``jax.lax.scan``, not a Python ``for`` loop:
    with a full-length batch (e.g. batch=64, seq=256) there can be 200+
    chunks, and a Python loop traces a *separate* copy of the
    checkpoint+matmul+softmax subgraph per chunk instead of one subgraph run
    repeatedly. XLA's rematerialization pass then has an enormous, mostly
    duplicate HLO graph to schedule and does not reliably free one chunk's
    ``[chunk, 262144]`` logits before the next chunk's copy is live, so peak
    memory scales with chunk *count*, not just chunk *size* — confirmed by
    OOMs that shrank only when ``chunk`` (not ``batch``) shrank. ``scan``
    compiles the step once and reuses its buffers turn by turn, which is what
    "process the vocab in bounded-size slices" was supposed to buy in the
    first place.

    **Three branches of the PyTorch version could not survive ``jax.jit``**,
    because all three test traced values:

    * ``counts_list = counts.tolist()`` — a host sync, impossible under trace.
    * ``if n_valid == 0: return ...`` — replaced by dividing by
      ``max(n_valid, 1)``; every term is already masked to zero in that case,
      so the result is a finite zero with zero gradient, which is exactly what
      the early return produced.
    * ``if counts_list[i] == 0: continue`` — the all-padding-chunk skip.
      Dropped: a traced count cannot gate a Python ``continue``. The cost is
      real (an all-padding slice now does its matmul and contributes zero
      instead of being skipped) and is the one place this port is slower than
      PyTorch by construction. It only bites on batches with a long padded
      tail; the padding ladder in ``train_jax`` keeps that bounded.

    The sequence is padded up to a whole number of chunks so the chunk count is
    static at trace time — under ``jit`` it must be, and a ragged final chunk
    would otherwise make every distinct sequence length a fresh trace.
    """
    batch, seq_len, d_model = hidden_states.shape

    flat_hidden = hidden_states.reshape(-1, d_model)[:-1]
    flat_labels = labels.reshape(-1)[1:]
    if batch > 1:
        # The last position of row i predicts the first of row i+1 after the
        # flatten — mask those cross-row boundaries.
        row_end = jnp.arange(1, batch) * seq_len - 1
        flat_labels = flat_labels.at[row_end].set(IGNORE_INDEX)

    total_tokens = flat_labels.shape[0]
    if not chunk:
        loss_sum, n_valid = _chunk_cross_entropy(head, flat_hidden, flat_labels)
        return loss_sum / jnp.maximum(n_valid, 1)

    n_chunks = (total_tokens + chunk - 1) // chunk
    pad = n_chunks * chunk - total_tokens
    if pad:
        flat_hidden = jnp.pad(flat_hidden, ((0, pad), (0, 0)))
        flat_labels = jnp.pad(
            flat_labels, (0, pad), constant_values=IGNORE_INDEX
        )

    step = jax.checkpoint(_chunk_cross_entropy, static_argnums=()) if remat else _chunk_cross_entropy

    hidden_chunks = flat_hidden.reshape(n_chunks, chunk, d_model)
    label_chunks = flat_labels.reshape(n_chunks, chunk)

    def scan_body(carry, xs):
        loss_sum, n_valid = carry
        hidden_slice, labels_slice = xs
        chunk_loss, chunk_valid = step(head, hidden_slice, labels_slice)
        return (loss_sum + chunk_loss, n_valid + chunk_valid.astype(jnp.int32)), None

    init = (jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32))
    (loss_sum, n_valid), _ = jax.lax.scan(
        scan_body, init, (hidden_chunks, label_chunks)
    )

    return loss_sum / jnp.maximum(n_valid, 1).astype(jnp.float32)


def guard_nonfinite_loss(loss: jnp.ndarray) -> jnp.ndarray:
    """Replace a non-finite loss with a finite, zero-gradient 0.

    Degenerate batch: every label masked to ``-100``, so the mean divides by
    zero valid tokens. :func:`chunked_loss` already short-circuits that case
    (it divides by ``max(n_valid, 1)``), so this only fires on a genuine
    non-finite value from elsewhere — which it is also the right response to,
    since one nan otherwise poisons the whole epoch's averaged loss and every
    downstream consumer of it (checkpoint selection, Optuna pruning, W&B).

    The PyTorch version uses ``torch.where`` over a Python ``if`` specifically
    to avoid a blocking GPU->CPU sync on every step. That motivation does not
    exist here — under trace there is no host sync to avoid — but ``jnp.where``
    is still the right shape, because a Python ``if`` on a traced value is not
    expressible at all.

    Caveat, stated rather than papered over: like the PyTorch version, this
    zeroes the gradient *contribution* of the guarded value, but a nan produced
    further upstream can still reach parameter gradients via ``0 * nan``. It is
    a reporting guard, not a nan-sanitiser, in both frameworks.
    """
    return jnp.where(jnp.isfinite(loss), loss, 0.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MimirMamba2Model(nnx.Module):
    """Mamba2 backbone with donor-projected input embeddings and output head.

    Args mirror ``MimirMamba2Module``'s, minus the Lightning/optimiser ones
    (which ``train_jax.py`` owns) and minus ``trust_remote_code``/tokenizer
    loading (which ``scripts/convert_donor_checkpoint.py`` did once, offline).
    """

    def __init__(
        self,
        tables: DonorTables,
        *,
        rngs: nnx.Rngs,
        d_model: int = 512,
        num_hidden_layers: int = 12,
        state_size: int = 64,
        expand: int = 2,
        head_dim: int = 64,
        n_groups: int = 1,
        conv_kernel: int = 4,
        chunk_size: int = 32,
        initializer_range: float = 0.02,
        projection_init: str = "pca",
        learn_projection_scales: bool = True,
        projection_cache_dir: Optional[str] = "outputs/donor_cache",
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = True,
        loss_chunk_tokens: int = 512,
    ) -> None:
        self.cfg = Mamba2Config.create(
            vocab_size=tables.vocab_size,
            d_model=d_model,
            num_hidden_layers=num_hidden_layers,
            state_size=state_size,
            expand=expand,
            head_dim=head_dim,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
            initializer_range=initializer_range,
        )
        self.loss_chunk_tokens = loss_chunk_tokens
        self.freeze_backbone = freeze_backbone

        embedding, head = build_projections(
            tables,
            d_model=d_model,
            rngs=rngs,
            init=projection_init,
            learn_scales=learn_projection_scales,
            cache_dir=projection_cache_dir,
        )
        self.backbone = Mamba2Backbone(
            self.cfg,
            embedding,
            rngs=rngs,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.lm_head = head

        self._log_parameter_budget(tables.vocab_size, tables.d_donor, d_model)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def projected_embedding(self) -> ProjectedEmbedding:
        return self.backbone.embeddings

    @property
    def projected_lm_head(self) -> ProjectedLMHead:
        return self.lm_head

    def _log_parameter_budget(self, vocab_size: int, d_donor: int, d_model: int) -> None:
        params = nnx.state(self, nnx.Param)
        total = sum(int(x.size) for x in jax.tree.leaves(params))
        proj = sum(
            int(x.size)
            for path, var in nnx.to_flat_state(params)
            if path[0] == "lm_head" or path[:3] == ("backbone", "embeddings", "P")
            or path[:2] == ("backbone", "embeddings")
            for x in [var.get_value()]
        )
        log.info(
            "Trainable: %.1fM (%.1fM Mamba2 blocks + %.1fM projections) | "
            "frozen donor tables: %.0fM values | "
            "standalone export would be %.0fM params (vs %.0fM for the donor)",
            total / 1e6,
            (total - proj) / 1e6,
            proj / 1e6,
            2 * vocab_size * d_donor / 1e6,
            (total - proj + 2 * vocab_size * d_model) / 1e6,
            2 * vocab_size * d_donor / 1e6,
        )

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def hidden_states(
        self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        return self.backbone(input_ids, attention_mask=attention_mask)

    def __call__(
        self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Full logits. Materializes ``[batch, seq, 262144]`` — eval only."""
        return self.lm_head(self.hidden_states(input_ids, attention_mask))

    def loss(
        self,
        input_ids: jnp.ndarray,
        labels: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Training/eval loss.

        Calls the backbone directly for hidden states rather than a combined
        forward-with-loss entrypoint — exactly mirroring how the PyTorch
        ``forward()`` calls ``self.model.backbone(...)`` when the chunked loss
        is enabled, because materializing the full logits is the cost chunking
        exists to avoid.
        """
        hidden = self.hidden_states(input_ids, attention_mask)
        return guard_nonfinite_loss(
            chunked_loss(self.lm_head, hidden, labels, self.loss_chunk_tokens)
        )

    def argmax_token_ids(
        self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Teacher-forced greedy prediction: ``[batch, seq]`` of argmax ids.

        Applies the head in ``loss_chunk_tokens``-sized slices and reduces
        immediately, so peak memory is chunk-sized rather than
        ``[batch, seq, 262144]``. Bit-identical to argmaxing the whole thing.
        """
        hidden = self.hidden_states(input_ids, attention_mask)
        chunk = self.loss_chunk_tokens
        if not chunk:
            return jnp.argmax(self.lm_head(hidden), axis=-1)

        batch, seq, d_model = hidden.shape
        flat = hidden.reshape(-1, d_model)
        total = flat.shape[0]
        pad = (-total) % chunk
        if pad:
            flat = jnp.pad(flat, ((0, pad), (0, 0)))
        parts = [
            jnp.argmax(self.lm_head(flat[s : s + chunk]), axis=-1)
            for s in range(0, flat.shape[0], chunk)
        ]
        return jnp.concatenate(parts)[:total].reshape(batch, seq)


# ---------------------------------------------------------------------------
# Parameter grouping (shared with train_jax's optimiser construction)
# ---------------------------------------------------------------------------

NO_DECAY_LEAVES = {"bias", "a_log", "d", "dt_bias", "conv_bias", "scale"}


def param_group(path: tuple) -> str:
    """Classify one parameter path into one of the four optimiser groups.

    The JAX counterpart of ``configure_optimizers``' loop over
    ``named_parameters()``: ``{proj, body} x {decay, no_decay}``, with the same
    leaf-name rule (matched against the *leaf* so Mamba2's single-letter SSM
    parameters ``A_log``/``D`` and its ``dt_bias`` avoid decay without a
    substring rule catching unrelated names).
    """
    prefix = (
        "proj"
        if path[0] == "lm_head" or path[:2] == ("backbone", "embeddings")
        else "body"
    )
    leaf = str(path[-1]).lower()
    joined = "/".join(str(p) for p in path).lower()
    no_decay = leaf in NO_DECAY_LEAVES or "norm" in joined
    return f"{prefix}_{'no_decay' if no_decay else 'decay'}"
