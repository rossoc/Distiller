# -*- coding: utf-8 -*-
"""MoMos: weight compression by dictionary learning. See ``SPEC.md``.

This package implements Phases A-E: flat/global tiling, dictionary state,
the static-mosaic micro-loop, dynamic swapping (exact neighbour graph plus
the micro-loop's step D), lifecycle (drop/merge), and — via
``model_jax.momos.integration`` — wiring the dictionary onto a real
``nnx.Module`` (the Mamba2 backbone) rather than only the toy SSM harness.
"""

from model_jax.momos.codebook import (
    assign_and_null_test,
    codebook_spread,
    spherical_kmeans,
    update_codebook,
)
from model_jax.momos.drift import (
    DriftState,
    accumulate,
    adaptive_scale,
    cohort_indices,
    init_drift_state,
    pick_coprime,
)
from model_jax.momos.integration import ModelBundle, init_bundle, merged_model
from model_jax.momos.integration import step as integration_step
from model_jax.momos.maintenance import neighbour_graph
from model_jax.momos.metrics import (
    bytes_per_weight,
    bytes_per_weight_with_drift,
    jump_eligible_rate,
    live_motifs,
    matched_rate,
    swap_rate,
    usage_entropy,
)
from model_jax.momos.reassign import WindowMetrics, macro_reassign
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
    "DriftState",
    "ModelBundle",
    "MosaicConfig",
    "MosaicState",
    "ParamLayout",
    "WindowMetrics",
    "accumulate",
    "adaptive_scale",
    "assign_and_null_test",
    "asymptotic_bytes_per_weight",
    "block_tensor_ids",
    "bytes_per_weight",
    "bytes_per_weight_with_drift",
    "codebook_spread",
    "cohort_indices",
    "default_include",
    "dictionary_bytes",
    "flatten_params",
    "from_blocks",
    "include_everything",
    "init",
    "init_bundle",
    "init_drift_state",
    "integration_step",
    "jump_eligible_rate",
    "live_motifs",
    "macro_reassign",
    "matched_rate",
    "merged_model",
    "mosaic_dtype",
    "neighbour_graph",
    "pick_coprime",
    "reconstruct",
    "spherical_kmeans",
    "swap_rate",
    "to_blocks",
    "trainable",
    "train_step",
    "unflatten_params",
    "update_codebook",
    "usage_entropy",
]
