# -*- coding: utf-8 -*-
"""MoMos: weight compression by dictionary learning. See ``SPEC.md``.

This package currently implements Phases A, B and C — flat/global tiling,
dictionary state, the static-mosaic micro-loop, and dynamic swapping (the
exact neighbour graph plus the micro-loop's step D). Lifecycle (Phase D) and
the Mamba2 backbone integration (Phase E) are not implemented.
"""

from model_jax.momos.maintenance import neighbour_graph
from model_jax.momos.metrics import bytes_per_weight, live_motifs, swap_rate, usage_entropy
from model_jax.momos.state import (
    DENSE_BYTES_PER_WEIGHT,
    MosaicConfig,
    MosaicState,
    asymptotic_bytes_per_weight,
    dictionary_bytes,
    init,
    mosaic_dtype,
    trainable,
)
from model_jax.momos.tiling import (
    ParamLayout,
    block_tensor_ids,
    default_include,
    flatten_params,
    from_blocks,
    include_everything,
    to_blocks,
    unflatten_params,
)
from model_jax.momos.train_step import reconstruct, train_step

__all__ = [
    "DENSE_BYTES_PER_WEIGHT",
    "MosaicConfig",
    "MosaicState",
    "ParamLayout",
    "asymptotic_bytes_per_weight",
    "block_tensor_ids",
    "bytes_per_weight",
    "default_include",
    "dictionary_bytes",
    "flatten_params",
    "from_blocks",
    "include_everything",
    "init",
    "live_motifs",
    "mosaic_dtype",
    "neighbour_graph",
    "reconstruct",
    "swap_rate",
    "to_blocks",
    "trainable",
    "train_step",
    "unflatten_params",
    "usage_entropy",
]
