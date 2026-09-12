# -*- coding: utf-8 -*-
"""SPEC.md §2's compression ledger, checked number-for-number against the table.

| S | K     | mosaic dtype | bytes/weight | dictionary (motifs+Adam) |
|---|-------|--------------|--------------|---------------------------|
| 1 | 256   | uint8        | 1.00         | 3 KB                      |
| 2 | 256   | uint8        | 0.50         | 6 KB                      |
| 2 | 4096  | uint16       | 1.00         | 96 KB                     |
| 4 | 4096  | uint16       | 0.50         | 192 KB                    |
| 4 | 65536 | uint16       | 0.50         | 3 MB                      |

The table's "bytes/weight" column is the *asymptotic* value (N -> infinity,
where the dictionary's fixed cost is negligible); ``asymptotic_bytes_per_weight``
reproduces it directly, while ``bytes_per_weight`` reports the honest, finite-N
number the actual regime (10^6-10^9) collapses onto it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from model_jax.momos.state import (
    DENSE_BYTES_PER_WEIGHT,
    MosaicConfig,
    MosaicState,
    asymptotic_bytes_per_weight,
    bytes_per_weight,
    dictionary_bytes,
    mosaic_dtype,
)
from model_jax.momos.tiling import ParamLayout


# ---------------------------------------------------------------------------
# The mosaic dtype boundary — "this IS the compression ratio"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "K,expected_itemsize",
    [(1, 1), (256, 1), (257, 2), (65536, 2), (65537, 4), (10**9, 4)],
)
def test_mosaic_dtype_boundaries(K, expected_itemsize):
    assert mosaic_dtype(K).itemsize == expected_itemsize


def test_mosaic_dtype_never_silently_widens_to_a_signed_or_wider_type():
    """A uint32 mosaic at S=1 is zero compression (SPEC.md §2) — this asserts
    the three real choices are exactly uint8/uint16/uint32, never int32/int64."""
    assert mosaic_dtype(256) == np.dtype(np.uint8)
    assert mosaic_dtype(65536) == np.dtype(np.uint16)
    assert mosaic_dtype(65537) == np.dtype(np.uint32)


def test_state_construction_rejects_a_mismatched_mosaic_dtype():
    """SPEC.md §2: 'Assert this at construction.'"""
    layout = ParamLayout(paths=(), shapes=(), dtypes=(), offsets=(), n_values=0,
                          tensor_id=np.zeros((0,), np.int32), excluded=())
    cfg = MosaicConfig(S=1, K=300, scale_mode="none")  # K=300 demands uint16
    with pytest.raises(AssertionError):
        MosaicState(
            motifs=jnp.zeros((300, 1)),
            mosaic=jnp.zeros((10,), dtype=jnp.uint8),  # wrong: uint8 only covers K<=256
            active=jnp.ones((300,), dtype=bool),
            scales=None,
            opt_state=(),
            layout=layout,
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# The ledger table itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "S,K,expected_bpw,expected_dict_bytes",
    [
        (1, 256, 1.00, 3 * 1024),
        (2, 256, 0.50, 6 * 1024),
        (2, 4096, 1.00, 96 * 1024),
        (4, 4096, 0.50, 192 * 1024),
        (4, 65536, 0.50, 3 * 1024 * 1024),
    ],
)
def test_ledger_matches_spec_table(S, K, expected_bpw, expected_dict_bytes):
    cfg = MosaicConfig(S=S, K=K, scale_mode="none")
    assert asymptotic_bytes_per_weight(cfg) == pytest.approx(expected_bpw)
    assert dictionary_bytes(cfg) == expected_dict_bytes


@pytest.mark.parametrize("S,K", [(1, 256), (2, 256), (2, 4096), (4, 4096), (4, 65536)])
def test_finite_n_ledger_converges_to_the_asymptote_at_large_n(S, K):
    """SPEC.md §1's regime tops out at N=10^9; even there the dictionary (at
    most 3 MB, for K=65536) is not literally zero relative to it, so this
    checks convergence — a loosening tolerance as N grows — rather than an
    exact match at one arbitrarily "large" N.
    """
    cfg = MosaicConfig(S=S, K=K, scale_mode="none")
    asymptote = asymptotic_bytes_per_weight(cfg)

    smaller, _ = bytes_per_weight(N=10**6, cfg=cfg)
    larger, ratio = bytes_per_weight(N=10**9, cfg=cfg)
    assert abs(larger - asymptote) < abs(smaller - asymptote), "not converging as N grows"
    assert larger == pytest.approx(asymptote, rel=1e-2)
    assert ratio == pytest.approx(DENSE_BYTES_PER_WEIGHT / larger)


def test_finite_n_ledger_is_honest_about_small_n():
    """At toy-task N the dictionary is NOT negligible — SPEC.md's table implicitly
    assumes the 10^6-10^9 regime, and reporting only the asymptote at small N
    would overstate the win. A large dictionary on a tiny model can cost MORE
    than dense fp32 training; this is the case that shows it."""
    cfg = MosaicConfig(S=4, K=65536, scale_mode="none")
    N = 56_000  # toy_tasks.SSMRegressor's rough parameter count
    per_weight, ratio = bytes_per_weight(N=N, cfg=cfg)
    assert per_weight > asymptotic_bytes_per_weight(cfg)
    assert ratio < 1.0, "a 3 MB dictionary swamps a 56k-parameter model"


def test_bytes_per_weight_rejects_nonpositive_n():
    cfg = MosaicConfig(S=1, K=256)
    with pytest.raises(ValueError):
        bytes_per_weight(N=0, cfg=cfg)


# ---------------------------------------------------------------------------
# scale_mode="per_tensor" adds a few hundred floats, not a few hundred KB
# ---------------------------------------------------------------------------


def test_per_tensor_scales_cost_is_negligible_next_to_the_mosaic():
    cfg = MosaicConfig(S=1, K=256, scale_mode="per_tensor")
    n_tensors = 400  # SPEC.md §3.1: "a few hundred floats total, i.e. free"
    extra = dictionary_bytes(cfg, n_tensors) - dictionary_bytes(cfg, 0)
    assert extra == n_tensors * 4 * 3
    per_weight, _ = bytes_per_weight(N=10**7, cfg=cfg, n_tensors=n_tensors)
    per_weight_no_scales, _ = bytes_per_weight(N=10**7, cfg=cfg, n_tensors=0)
    assert per_weight == pytest.approx(per_weight_no_scales, rel=1e-3)


def test_mosaic_config_rejects_invalid_s_k_and_scale_mode():
    with pytest.raises(ValueError):
        MosaicConfig(S=3, K=256)
    with pytest.raises(ValueError):
        MosaicConfig(S=1, K=0)
    with pytest.raises(ValueError):
        MosaicConfig(S=1, K=256, scale_mode="global")
