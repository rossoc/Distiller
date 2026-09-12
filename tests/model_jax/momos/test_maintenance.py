# -*- coding: utf-8 -*-
"""Tests for ``momos.maintenance.neighbour_graph`` (SPEC.md §6.1).

Three properties, each with its own test: the graph is *exact* (matches a
brute-force NumPy reference, not an approximation), it never returns a dead
motif when live alternatives exist, and computing it in row blocks gives the
same answer as computing it in one block (the whole point of ``block_size``
is to bound memory, not to change the result).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from model_jax.momos import maintenance


def _brute_force_neighbors(motifs: np.ndarray, active: np.ndarray, n_neighbors: int):
    """Reference implementation: full distance matrix, no blocking, plain NumPy."""
    diffs = motifs[:, None, :] - motifs[None, :, :]
    dist = np.sum(diffs**2, axis=-1)
    np.fill_diagonal(dist, np.inf)
    dist[:, ~active] = np.inf
    return np.argsort(dist, axis=1)[:, :n_neighbors]


def test_neighbour_graph_matches_brute_force_exactly():
    """The core correctness gate: exact squared-L2 NN, not an approximation.

    Compared as a *set* per row rather than an ordered array — nothing in
    SPEC.md's ``neighbour_graph`` contract promises a particular ordering
    among the ``n_neighbors`` winners, only that they are the ``n_neighbors``
    nearest. Continuous random motifs make an exact-distance tie between the
    graph's boundary and the reference's essentially impossible, so a set
    comparison is not accidentally hiding an off-by-one.
    """
    K, S, n_neighbors = 12, 3, 4
    motifs = np.asarray(jax.random.normal(jax.random.key(0), (K, S)))
    active = np.ones((K,), dtype=bool)

    graph = maintenance.neighbour_graph(jnp.asarray(motifs), jnp.asarray(active), n_neighbors)
    ref = _brute_force_neighbors(motifs, active, n_neighbors)

    assert graph.shape == (K, n_neighbors)
    assert graph.dtype == np.int32
    for k in range(K):
        assert set(np.asarray(graph[k]).tolist()) == set(ref[k].tolist()), f"row {k}"


def test_neighbour_graph_never_returns_a_dead_motif():
    """Construct: the nearest motif to index 0 is dead; live ones are far.

    If masking were done by multiplying the distance by ``active`` (the bug
    SPEC.md §6.1 warns against — a 0/1 mask on a *distance* makes a dead
    motif look like distance 0, i.e. the closest possible point), index 1
    would win every row it's compared against. It must never appear.
    """
    motifs = jnp.array([[0.0], [0.0001], [5.0], [-6.0], [20.0]])
    active = jnp.array([True, False, True, True, True])

    graph = maintenance.neighbour_graph(motifs, active, n_neighbors=2)

    assert not bool(jnp.any(graph == 1)), "dead motif (index 1) appears in the graph"
    # And the test has teeth: the two nearest *live* motifs to index 0 really
    # are 2 (distance 25) and 3 (distance 36), well ahead of 4 (distance 400)
    # and of the dead motif 1 (distance ~1e-8) that must be excluded.
    assert set(np.asarray(graph[0]).tolist()) == {2, 3}


def test_neighbour_graph_blockwise_matches_single_block():
    """``block_size`` bounds memory; it must not change the answer."""
    K, S, n_neighbors = 20, 2, 3
    motifs = jax.random.normal(jax.random.key(3), (K, S))
    active = jnp.ones((K,), dtype=bool)

    one_block = maintenance.neighbour_graph(motifs, active, n_neighbors, block_size=1024)
    many_blocks = maintenance.neighbour_graph(motifs, active, n_neighbors, block_size=3)

    np.testing.assert_array_equal(np.asarray(one_block), np.asarray(many_blocks))


def test_neighbour_graph_clamps_n_neighbors_to_k_minus_one():
    """A small K must not require the caller to know K ahead of time."""
    motifs = jnp.array([[0.0], [1.0], [2.0]])
    active = jnp.ones((3,), dtype=bool)

    graph = maintenance.neighbour_graph(motifs, active, n_neighbors=10)

    assert graph.shape == (3, 2)  # clamped to K - 1
    for k in range(3):
        assert set(np.asarray(graph[k]).tolist()) == {0, 1, 2} - {k}


def test_neighbour_graph_zero_neighbors_returns_empty_columns():
    motifs = jax.random.normal(jax.random.key(4), (5, 2))
    active = jnp.ones((5,), dtype=bool)
    graph = maintenance.neighbour_graph(motifs, active, n_neighbors=0)
    assert graph.shape == (5, 0)
