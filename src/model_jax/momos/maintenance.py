# -*- coding: utf-8 -*-
"""The neighbour graph for Phase C's swap proposal (SPEC.md §6.1).

Every ``cfg.maintenance_every`` steps, an external training loop is expected
to rebuild this graph from the live ``motifs``/``active`` arrays and attach it
to ``MosaicState.graph`` (e.g. via ``dataclasses.replace``). Building it is
*not* ``train_step``'s job — the graph is read-only from the micro-loop's
point of view (SPEC.md §5 step D just does ``nbrs = state.graph[cur]``) and is
deliberately kept stale between maintenance steps rather than recomputed every
training step, since exact all-pairs distances at every one of the many
micro-loop steps between two maintenance steps would be wasted work against a
dictionary that has barely moved.

**Exact, not LSH.** SPEC.md §6.1 measured sign-random-projection LSH directly
at K=512 and found it useless in this regime: 0.2% / 7.0% / 26.2% top-1
agreement with exact nearest-neighbour at S=1/2/4. The reason is structural,
not a tuning problem: at S=1 an L2-normalised motif is exactly ±1, so a single
random hyperplane's sign carries at most one bit of information about it, and
most of that bit is thrown away when hashing to a handful of buckets. Exact
all-pairs squared-L2 is the only approach that measures right in this regime,
and at the sizes here (K up to a few tens of thousands, S ≤ 4) it is cheap:
K=4096, S=4 is 67 Mflop total. Computing it in row blocks (rather than
materialising the full K×K matrix, ~67 MB at K=4096 in fp32) keeps memory
bounded without changing the result at all — see
``test_neighbour_graph_blockwise_matches_single_block``.

**Masking is by ``+inf`` on the distance, never by a 0/1 multiplicative
mask.** For a squared distance (always ≥ 0), multiplying a dead or
self-column by 0 makes it look like the *closest possible* point (distance
0) rather than making it disappear — the opposite of the intended effect.
``jnp.where(..., jnp.inf)`` is the only correct way to remove a candidate from
a nearest-neighbour search.

**No S=1 sort specialisation.** SPEC.md §6.1 offers one: for S=1, motifs sort
into a total order and each one's neighbours are adjacent in that order,
O(K log K) instead of this file's O(K^2/block_size · block_size) = O(K^2)
all-pairs pass. It is explicitly optional ("worth a specialisation if
profiling shows the graph costs anything"), and it is not as clean as it
first looks: the *k* nearest neighbours of a sorted value are not simply "k/2
either side" once dead motifs are excluded (a value near the low end of the
range, or surrounded by several dead neighbours, needs an asymmetric,
variable-width window), so a correct implementation is a real two-pointer
merge, not a slice. Skipped here: at this regime's K (a few thousand to
tens of thousands) the generic exact path above is already a few
milliseconds, profiling shows nothing to fix, and a second, only-for-S=1 code
path is exactly the kind of complexity SPEC.md's "build v1 simple; measure
first" guidance (§5, "Memory") argues against paying for pre-emptively.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def neighbour_graph(
    motifs: jnp.ndarray,
    active: jnp.ndarray,
    n_neighbors: int,
    block_size: int = 1024,
) -> jnp.ndarray:
    """Exact squared-L2 nearest neighbours for every motif, dead ones excluded.

    ``motifs``: ``(K, S)`` float. ``active``: ``(K,)`` bool. Returns
    ``(K, n_neighbors)`` int32 — row ``k`` holds the ``n_neighbors`` motif
    indices nearest to motif ``k`` by squared L2 distance, excluding ``k``
    itself and every inactive motif.

    Computed in blocks of ``block_size`` rows against the *full* motif matrix
    (so each block materialises only a ``(block_size, K)`` distance slab, never
    the whole ``(K, K)`` matrix — SPEC.md §6.1) via the standard expansion
    ``||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b``, which turns the all-pairs
    distance into one matmul per block plus cheap broadcasting.

    ``n_neighbors`` is clamped to ``K - 1`` (there cannot be more distinct
    neighbours than other motifs) so a small ``K`` in a test never needs the
    caller to special-case it.

    Only ``jax.lax.top_k`` is asked to pick *finite* winners: if fewer than
    ``n_neighbors`` motifs are active (excluding self), some returned slots
    for that row will point at an inactive motif despite the masking here.
    This is not a correctness gap in practice — SPEC.md §5 step D re-applies
    the identical ``active``-mask to whatever this returns before ever using
    it as a swap target, so a motif this function is forced to return for
    lack of any live alternative still can never be swapped onto (see
    ``test_train_step`` for the swap-time guard, which is what actually
    matters for the "no dead motif ever appears in the assignment" property).
    """
    K = int(motifs.shape[0])
    n_neighbors = min(int(n_neighbors), max(K - 1, 0))
    if n_neighbors <= 0:
        return jnp.zeros((K, 0), dtype=jnp.int32)

    motifs = motifs.astype(jnp.float32)
    sq_norms = jnp.sum(motifs * motifs, axis=-1)  # (K,)
    all_idx = jnp.arange(K)

    rows = []
    for start in range(0, K, block_size):
        stop = min(start + block_size, K)
        block = motifs[start:stop]  # (b, S)
        cross = block @ motifs.T  # (b, K)
        d = sq_norms[start:stop, None] + sq_norms[None, :] - 2.0 * cross
        d = jnp.maximum(d, 0.0)  # squared distances can't be negative; clip fp roundoff
        row_ids = all_idx[start:stop]
        d = jnp.where(row_ids[:, None] == all_idx[None, :], jnp.inf, d)  # mask self
        d = jnp.where(active[None, :], d, jnp.inf)  # mask dead columns — never promote them
        _, nearest = jax.lax.top_k(-d, n_neighbors)  # top_k of -d == smallest d
        rows.append(nearest)
    return jnp.concatenate(rows, axis=0).astype(jnp.int32)
