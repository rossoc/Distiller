# -*- coding: utf-8 -*-
"""Flat, global tiling of a parameter tree into ``(M, S)`` blocks.

Every mosaicked weight in the network — regardless of which tensor or layer it
came from — ends up at some position in one contiguous vector. That is the
mechanism that makes "one global dictionary" possible at all: a block's motif
assignment carries no memory of which tensor it was cut from, so the same K
motifs can be reused across every tensor in the model. See SPEC.md §3.

Two independent concerns, four functions:

* :func:`flatten_params` / :func:`unflatten_params` — the tree <-> flat vector
  round trip. Exact, including dtype, and independent of ``S``.
* :func:`to_blocks` / :func:`from_blocks` — the flat vector <-> ``(M, S)``
  round trip. Exact once the caller supplies the original value count, so the
  zero-padding added to reach a multiple of ``S`` is trimmed away.

Splitting them this way means a single ``flatten_params`` call can be re-tiled
at several ``S`` for a capacity sweep without re-walking the parameter tree.

**Per-tensor padding, not one trailing pad.** Each included leaf is padded to
a multiple of ``S`` *before* concatenation, not after — so the global padding
this produces is always zero (a concatenation of multiples of ``S`` is itself
a multiple of ``S``) and, more importantly, no block ever straddles two
tensors. That is what makes ``ParamLayout.tensor_id`` (and therefore
``scale_mode="per_tensor"``) well-defined for every block rather than just
"well-defined except at a few boundaries". SPEC.md's tiling paragraph reads as
one global pad at the end; padding per leaf is equivalent for the exact-round
-trip requirement (the total is still "zero-padded to a multiple of S") and is
strictly better for the scale machinery, so that is what this implements.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from flax import nnx

Path = Tuple  # a tuple of str/int components, as returned by nnx.to_flat_state
IncludePredicate = Callable[[Path], bool]


# ---------------------------------------------------------------------------
# What gets mosaicked
# ---------------------------------------------------------------------------

# Per-head SSM scalars: tiny (one scalar per head, not per weight) and highly
# scale-sensitive — A_log in particular sets the recurrence's decay rate, so
# quantising it onto a shared dictionary would directly perturb dynamics
# rather than just representational capacity.
_EXCLUDED_LEAF_NAMES = frozenset({"A_log", "D", "dt_bias"})


def include_everything(path: Path) -> bool:
    """The ablation predicate: mosaic literally everything, no exceptions."""
    return True


def default_include(path: Path) -> bool:
    """Exclude 1-D norm weights and per-head SSM scalars; mosaic the rest.

    Matches SPEC.md §3's default: ``*/norm/weight``, ``norm_f/weight``,
    ``A_log``, ``D``, ``dt_bias``. Everything these exclude is a rounding
    error in parameter count next to the tensors that dominate model size
    (embeddings, in/out projections), so excluding them costs essentially no
    compression while sidestepping the scale-sensitivity that would otherwise
    make them the first thing to break under a shared dictionary.
    """
    leaf = path[-1]
    if leaf in _EXCLUDED_LEAF_NAMES:
        return False
    if leaf == "weight" and len(path) >= 2 and path[-2] == "norm":
        return False
    if leaf == "weight" and len(path) >= 2 and path[-2] == "norm_f":
        return False
    return True


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParamLayout:
    """Static, host-side record of how a parameter tree maps onto the flat vector.

    Everything here is plain Python/NumPy, never a traced JAX value: per
    SPEC.md §4 this travels alongside ``MosaicState`` as a compile-time
    constant, not as part of the pytree that gets differentiated or jitted.

    ``excluded`` holds the leaves ``include`` rejected, verbatim, so that
    :func:`unflatten_params` can hand back a tree structurally identical to
    the one :func:`flatten_params` consumed. In phases A/B those leaves are
    simply frozen at their initial value — nothing in this phase's micro-loop
    trains them. (A later phase that wants to keep training the excluded
    leaves densely, alongside the dictionary, would do so with a second,
    ordinary optimiser over exactly this ``excluded`` set.)
    """

    paths: Tuple[Path, ...]
    shapes: Tuple[Tuple[int, ...], ...]
    dtypes: Tuple[np.dtype, ...]
    offsets: Tuple[int, ...]
    n_values: int
    tensor_id: np.ndarray  # (n_values,) int32 — which leaf (index into `paths`) owns each value
    excluded: Tuple[Tuple[Path, jnp.ndarray], ...]

    @property
    def n_tensors(self) -> int:
        return len(self.paths)


# ---------------------------------------------------------------------------
# Tree <-> flat vector
# ---------------------------------------------------------------------------


def flatten_params(
    tree: nnx.State, include: IncludePredicate = default_include
) -> Tuple[jnp.ndarray, ParamLayout]:
    """Concatenate every included leaf of *tree* into one 1-D vector.

    *tree* is whatever ``nnx.state(model, nnx.Param)`` produces: a nested
    ``nnx.State`` whose leaves are ``nnx.Param`` variables. Walked with
    ``nnx.to_flat_state`` rather than raw ``jax.tree_util``, so paths are
    plain ``('backbone', 'layers', 0, 'norm', 'weight')``-style tuples — the
    same convention ``mimir_mamba2.param_group`` already uses — instead of
    ``jax.tree_util``'s ``DictKey``/``GetAttrKey`` wrappers.
    """
    paths: list = []
    shapes: list = []
    dtypes: list = []
    offsets: list = []
    chunks: list = []
    tensor_ids: list = []
    excluded: list = []

    offset = 0
    for path, variable in nnx.to_flat_state(tree):
        value = variable.get_value()
        if not include(path):
            excluded.append((path, value))
            continue

        leaf_id = len(paths)
        paths.append(path)
        shapes.append(tuple(int(d) for d in value.shape))
        dtypes.append(value.dtype)
        offsets.append(offset)

        raveled = jnp.reshape(value, (-1,))
        chunks.append(raveled)
        tensor_ids.append(np.full((raveled.shape[0],), leaf_id, dtype=np.int32))
        offset += raveled.shape[0]

    flat = jnp.concatenate(chunks) if chunks else jnp.zeros((0,), jnp.float32)
    tensor_id = np.concatenate(tensor_ids) if tensor_ids else np.zeros((0,), dtype=np.int32)

    layout = ParamLayout(
        paths=tuple(paths),
        shapes=tuple(shapes),
        dtypes=tuple(dtypes),
        offsets=tuple(offsets),
        n_values=offset,
        tensor_id=tensor_id,
        excluded=tuple(excluded),
    )
    return flat, layout


def unflatten_params(
    flat: jnp.ndarray,
    layout: ParamLayout,
    excluded_values: Optional[Sequence[jnp.ndarray]] = None,
) -> nnx.State:
    """Exact inverse of :func:`flatten_params`: reassemble the parameter tree.

    Casts each leaf back to its recorded dtype explicitly rather than relying
    on ``flat``'s dtype to already be right — ``jnp.concatenate`` promotes
    mixed-dtype leaves to a common type, so without this cast a tree that
    started with (say) a bfloat16 leaf would come back float32. Reshape/cast
    do no arithmetic, so this is exact regardless.

    ``excluded_values``, if given, overrides ``layout.excluded``'s frozen
    initial values, one-for-one in the same order — for a caller (e.g.
    ``model_jax.momos.integration``) that trains the excluded leaves with an
    ordinary optimiser alongside the dictionary, rather than leaving them
    frozen at init (Phases A-D's behaviour, and still the default here: with
    ``excluded_values=None`` this is bit-for-bit identical to before).
    """
    if excluded_values is None:
        items = [(path, nnx.Param(jnp.asarray(value))) for path, value in layout.excluded]
    else:
        items = [
            (path, nnx.Param(jnp.asarray(value)))
            for (path, _), value in zip(layout.excluded, excluded_values)
        ]
    for path, shape, dtype, offset in zip(
        layout.paths, layout.shapes, layout.dtypes, layout.offsets
    ):
        size = int(np.prod(shape)) if shape else 1
        value = jnp.reshape(flat[offset : offset + size], shape).astype(dtype)
        items.append((path, nnx.Param(value)))
    return nnx.statelib.from_flat_state(items)


# ---------------------------------------------------------------------------
# Flat vector <-> (M, S) blocks
# ---------------------------------------------------------------------------


def to_blocks(flat: jnp.ndarray, S: int) -> jnp.ndarray:
    """Zero-pad *flat* to a multiple of ``S`` and reshape to ``(M, S)``."""
    n = int(flat.shape[0])
    pad = (-n) % S
    if pad:
        flat = jnp.concatenate([flat, jnp.zeros((pad,), dtype=flat.dtype)])
    return flat.reshape(-1, S)


def from_blocks(blocks: jnp.ndarray, n_values: int) -> jnp.ndarray:
    """Flatten ``(M, S)`` blocks and trim the zero-padding :func:`to_blocks` added."""
    return blocks.reshape(-1)[:n_values]


# ---------------------------------------------------------------------------
# Per-block tensor identity (scale_mode="per_tensor" support)
# ---------------------------------------------------------------------------


def block_tensor_ids(layout: ParamLayout, S: int, M: int) -> jnp.ndarray:
    """Which tensor (index into ``layout.paths``) each block's first value belongs to.

    Only meaningful for ``scale_mode="per_tensor"``. Because every leaf is
    padded to a multiple of ``S`` before concatenation (see the module
    docstring), no real block ever straddles two tensors — this is exact, not
    an approximation, for every block built entirely from real values. Blocks
    built entirely from the trailing global pad (past ``n_values``, which can
    only happen if the total value count itself is not a multiple of ``S``)
    are attributed to the last tensor; harmless, since ``from_blocks`` trims
    that padding before it reaches anywhere the scale is applied to a value
    that matters.
    """
    if layout.n_tensors == 0:
        return jnp.zeros((M,), dtype=jnp.int32)
    padded = np.full((M * S,), layout.tensor_id[-1] if layout.n_values else 0, dtype=np.int32)
    padded[: layout.n_values] = layout.tensor_id
    return jnp.asarray(padded.reshape(M, S)[:, 0])
