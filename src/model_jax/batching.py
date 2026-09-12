# -*- coding: utf-8 -*-
"""torch DataLoader batch -> jnp arrays, at fixed shapes.

This module exists because of a JAX-specific constraint the PyTorch pipeline
never had to care about, and it is the one place the data path genuinely had to
change during the port.

``DistillerDataModule`` pads each batch to *its own* longest sequence
(``lit_datamodule._pad_batch`` / ``_collate_batch``) — deliberately, so a split
with one very long row does not force every batch to that length. Under
``jax.jit`` that is expensive in a way it is not under eager PyTorch: every
distinct ``(batch, seq_len)`` pair is a fresh trace *and* a fresh XLA
compilation of the whole train step. With per-batch dynamic padding, sequence
length varies on nearly every batch, so a single epoch can trigger hundreds of
recompiles, each of which costs far more than the step it enables. The eval
loaders make it worse: they do not ``drop_last``, so the final batch has a
different leading dimension too.

The fix is a **bucket ladder**: round each batch's sequence length up to the
next rung, and pad the batch dimension to the loader's nominal size. Padding is
semantically free — ``input_ids`` take the pad token, ``labels`` take ``-100``
(so the loss already ignores them, via the same masking the PyTorch path uses),
and ``attention_mask`` takes 0 (so the mixer zeroes those positions). The cost
is the wasted compute on padding; the ladder is geometric so that is bounded by
the rung spacing, and the number of distinct compiled shapes is bounded by the
number of rungs.

The dataloaders themselves are untouched — this is a boundary adapter, not a
data-pipeline rewrite.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)

IGNORE_INDEX = -100


def bucket_ladder(max_length: int, min_rung: int = 64, growth: float = 1.5) -> List[int]:
    """Geometric ladder of sequence lengths, ending exactly at *max_length*.

    Geometric rather than linear so the number of compiled shapes grows with
    ``log(max_length)`` while the worst-case padding waste stays a fixed
    fraction (``1 - 1/growth``, so ~33% at the default) of a rung.
    """
    if max_length <= min_rung:
        return [max_length]
    rungs, rung = [], min_rung
    while rung < max_length:
        rungs.append(int(rung))
        rung = max(rung + 1, int(rung * growth))
    rungs.append(int(max_length))
    return sorted(set(rungs))


def pick_rung(length: int, ladder: Iterable[int]) -> int:
    """Smallest rung >= *length* (the top rung if *length* overshoots it)."""
    for rung in ladder:
        if rung >= length:
            return rung
    return max(ladder)


def to_jax_batch(
    batch: Dict[str, "object"],
    ladder: List[int],
    batch_size: int,
    pad_token_id: int,
) -> Dict[str, jnp.ndarray]:
    """Convert one torch batch to fixed-shape jnp arrays.

    Returns ``input_ids``/``labels``/``attention_mask``, each padded to
    ``(batch_size, rung)``. Rows added to reach *batch_size* are fully masked,
    so they contribute no loss and no gradient — the same mechanism that
    already handles a short final batch's padding inside a row.
    """
    input_ids = np.asarray(batch["input_ids"])
    labels = np.asarray(batch["labels"])
    mask = batch.get("attention_mask")
    mask = np.ones_like(input_ids) if mask is None else np.asarray(mask)

    rows, length = input_ids.shape
    rung = pick_rung(length, ladder)
    pad_rows, pad_cols = batch_size - rows, rung - length
    if pad_rows < 0 or pad_cols < 0:
        raise ValueError(
            f"batch {input_ids.shape} exceeds the target shape "
            f"({batch_size}, {rung}); widen the ladder or the batch size"
        )

    pads = ((0, pad_rows), (0, pad_cols))
    return {
        "input_ids": jnp.asarray(
            np.pad(input_ids, pads, constant_values=pad_token_id), jnp.int32
        ),
        "labels": jnp.asarray(
            np.pad(labels, pads, constant_values=IGNORE_INDEX), jnp.int32
        ),
        "attention_mask": jnp.asarray(np.pad(mask, pads, constant_values=0), jnp.int32),
    }


def describe_ladder(ladder: List[int], batch_size: int) -> str:
    return (
        f"{len(ladder)} sequence buckets {ladder} at batch {batch_size} — "
        f"at most {len(ladder)} compiled train-step shapes per fold"
    )
