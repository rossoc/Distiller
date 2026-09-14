# -*- coding: utf-8 -*-
"""Tests for ``momos.tiling``: the flatten/unflatten and block round trips.

Phase A's gate (SPEC.md §8): ``unflatten(flatten(t)) == t`` exactly, including
dtype, and ``bytes_per_weight`` matches the §2 table (covered in
``test_compression.py``). This file also checks the two things the round trip
alone would not catch: that the default include predicate actually excludes
what SPEC.md §3 says it should, and that ``block_tensor_ids`` attributes every
block to the right tensor — the piece ``scale_mode="per_tensor"`` depends on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from model_jax.momos import tiling


class _Norm(nnx.Module):
    def __init__(self, n: int = 4):
        self.weight = nnx.Param(jnp.full((n,), 2.0))


class _Toy(nnx.Module):
    """Two mosaicked tensors (16 + 4 values) plus five excluded-by-default leaves."""

    def __init__(self, rngs: nnx.Rngs):
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (4, 4)))
        self.bias = nnx.Param(jnp.zeros((4,), dtype=jnp.bfloat16))
        self.norm = _Norm()
        self.norm_f = _Norm()
        self.A_log = nnx.Param(jnp.array([0.1, 0.2]))
        self.D = nnx.Param(jnp.array([1.0, 1.0]))
        self.dt_bias = nnx.Param(jnp.array([0.0, 0.0]))


def _param_state() -> nnx.State:
    return nnx.state(_Toy(nnx.Rngs(params=0)), nnx.Param)


def _as_dict(state: nnx.State) -> dict:
    return {path: variable.get_value() for path, variable in nnx.to_flat_state(state)}


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


def test_round_trip_is_exact_with_include_everything():
    tree = _param_state()
    flat, layout = tiling.flatten_params(tree, include=tiling.include_everything)
    restored = _as_dict(tiling.unflatten_params(flat, layout))
    original = _as_dict(tree)

    assert restored.keys() == original.keys()
    for path, value in original.items():
        got = restored[path]
        assert got.dtype == value.dtype, path
        assert np.array_equal(np.asarray(got), np.asarray(value)), path


def test_round_trip_preserves_excluded_leaves_untouched():
    """Default include drops norm/SSM-scalar leaves from the flat vector, but
    they must still come back out of unflatten — SPEC.md's round-trip
    requirement is stated over the whole tree, not just the mosaicked part."""
    tree = _param_state()
    flat, layout = tiling.flatten_params(tree, include=tiling.default_include)
    restored = _as_dict(tiling.unflatten_params(flat, layout))
    original = _as_dict(tree)

    assert restored.keys() == original.keys()
    for path, value in original.items():
        assert np.array_equal(np.asarray(restored[path]), np.asarray(value)), path


def test_default_include_excludes_exactly_the_documented_leaves():
    tree = _param_state()
    _, layout = tiling.flatten_params(tree, include=tiling.default_include)

    included = set(layout.paths)
    excluded = {path for path, _ in layout.excluded}

    assert included == {("kernel",), ("bias",)}
    assert excluded == {
        ("norm", "weight"),
        ("norm_f", "weight"),
        ("A_log",),
        ("D",),
        ("dt_bias",),
    }
    assert layout.n_values == 16 + 4
    assert layout.n_tensors == 2


def test_default_include_excludes_norm_f_weight_when_nested():
    """The real Mamba2 model nests ``norm_f`` under ``backbone`` (SPEC_PHASE_D_
    DEFECTS.md-style regression: the exact-path check only matched the
    top-level ``("norm_f", "weight")`` case)."""
    assert tiling.default_include(("backbone", "norm_f", "weight")) is False
    assert tiling.default_include(("norm_f", "weight")) is False
    assert tiling.default_include(("backbone", "layers", 0, "norm", "weight")) is False
    assert tiling.default_include(("backbone", "layers", 0, "mixer", "out_proj")) is True


def test_include_everything_drops_nothing():
    tree = _param_state()
    _, layout = tiling.flatten_params(tree, include=tiling.include_everything)
    assert layout.excluded == ()
    assert layout.n_tensors == 7


# ---------------------------------------------------------------------------
# Blocks: padding and trimming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,S", [(20, 1), (20, 2), (20, 4), (7, 4), (1, 4), (0, 4)])
def test_to_blocks_pads_to_a_multiple_of_s(n, S):
    flat = jnp.arange(n, dtype=jnp.float32)
    blocks = tiling.to_blocks(flat, S)
    M = -(-n // S)  # ceil division
    assert blocks.shape == (M, S)
    # The real values must appear, in order, at the front; padding is zero.
    recovered = blocks.reshape(-1)
    assert np.array_equal(np.asarray(recovered[:n]), np.asarray(flat))
    assert np.all(np.asarray(recovered[n:]) == 0.0)


@pytest.mark.parametrize("n,S", [(20, 1), (20, 2), (20, 4), (7, 4), (1, 4)])
def test_from_blocks_is_the_exact_inverse_of_to_blocks(n, S):
    flat = jax.random.normal(jax.random.key(0), (n,))
    blocks = tiling.to_blocks(flat, S)
    restored = tiling.from_blocks(blocks, n)
    assert restored.shape == (n,)
    assert np.array_equal(np.asarray(restored), np.asarray(flat))


def test_flatten_then_block_then_unflatten_round_trips_a_whole_tree():
    tree = _param_state()
    flat, layout = tiling.flatten_params(tree, include=tiling.include_everything)
    for S in (1, 2, 4):
        blocks = tiling.to_blocks(flat, S)
        restored_flat = tiling.from_blocks(blocks, layout.n_values)
        restored = _as_dict(tiling.unflatten_params(restored_flat, layout))
        for path, value in _as_dict(tree).items():
            assert np.array_equal(np.asarray(restored[path]), np.asarray(value)), (S, path)


# ---------------------------------------------------------------------------
# Per-block tensor identity (scale_mode="per_tensor")
# ---------------------------------------------------------------------------


def test_block_tensor_ids_are_exact_when_tensors_align_to_s():
    """kernel (16 values) and bias (4 values) are both multiples of S=4, so
    every block belongs entirely to one tensor — checked directly against
    ``layout.tensor_id`` rather than assuming a particular flatten order."""
    tree = _param_state()
    _, layout = tiling.flatten_params(tree, include=tiling.default_include)
    S = 4
    M = -(-layout.n_values // S)
    ids = np.asarray(tiling.block_tensor_ids(layout, S, M))

    for block in range(M):
        start = block * S
        owners = set(layout.tensor_id[start : start + S].tolist())
        assert len(owners) == 1, "block straddles two tensors despite S-aligned sizes"
        assert ids[block] == owners.pop()


def test_block_tensor_ids_with_no_included_leaves_is_well_defined():
    tree = _param_state()
    _, layout = tiling.flatten_params(tree, include=lambda path: False)
    ids = tiling.block_tensor_ids(layout, 4, 3)
    assert ids.shape == (3,)
