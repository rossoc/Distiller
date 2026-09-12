#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase C's gate (SPEC.md §8): at matched K, does dynamic swapping beat the
static Phase B mosaic?

    python scripts/momos_phase_c.py
    python scripts/momos_phase_c.py --k 1024 4096 --s 2 4
    python scripts/momos_phase_c.py --steps 400 --subset-size 128

For every (S, K) cell this trains the *same* toy model, from the *same* seed,
for the *same* number of steps, twice: once with ``cfg.subset_size=0`` (step D
disabled — Phase B's static mosaic) and once with swapping turned on (step D
enabled, with the neighbour graph rebuilt from the live dictionary every
``--maintenance-every`` steps). Everything else — learning rate, batch,
architecture, random seed — is identical between the two runs, so any
difference in final loss is attributable to step D alone, not to a
confound. The dense baseline (an ordinary ``nnx.Optimizer`` on the same
architecture) is trained once, since it does not depend on ``S``/``K`` at all.

SPEC.md's own words for the gate this script exists to run: "at matched K,
dynamic beats static. If not, stop — the swap criterion is not earning its
complexity." This script prints what it measures either way; it does not
retry with different hyperparameters until the gate passes.

**On ``--margin``'s scale.** The swap criterion compares a candidate
neighbour's distance to ``w_prop`` against ``d_cur = ||update||^2`` — the
*squared magnitude of one optimiser step* (see ``train_step.py``'s Correction
docstring for why it is this and not the spec's literal, always-zero
formula). Adam normalises its step size to roughly the learning rate, so
``d_cur`` sits close to ``learning_rate**2`` regardless of gradient scale —
measured directly at this script's defaults, ``d_cur`` medians ~3.6e-5 at
``learning_rate=6e-3``. A margin at or above that (the first value tried
here, ``1e-4``) makes ``d_cur - margin`` negative for essentially every
block, which no non-negative squared distance can ever beat — swap rate is
then exactly and permanently zero, not "low". ``--margin`` must be kept
comfortably below ``learning_rate**2`` for step D to do anything at all.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402

from model_jax import toy_tasks  # noqa: E402
from model_jax.momos import maintenance, tiling  # noqa: E402
from model_jax.momos.state import MosaicConfig, MosaicState  # noqa: E402
from model_jax.momos.state import init as momos_init  # noqa: E402
from model_jax.momos.train_step import reconstruct, train_step  # noqa: E402

MODEL_KWARGS = dict(d_model=32, num_layers=2, state_size=16, chunk_size=8)


def _build_model(task: str, seed: int) -> nnx.Module:
    return toy_tasks.build(task, rngs=nnx.Rngs(params=seed), **MODEL_KWARGS)


def train_dense(
    task: str, steps: int, batch: int, seq_len: int, seed: int, learning_rate: float
) -> float:
    """Ordinary dense training of the same architecture — computed once,
    since neither ``S`` nor ``K`` (both MoMos-only concepts) affects it."""
    model = _build_model(task, seed)
    tx = optax.adam(learning_rate)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    make_batch = toy_tasks.TASKS[task]

    @nnx.jit
    def step(model, optimizer, x, y, w):
        loss, grads = nnx.value_and_grad(lambda m: m.loss(x, y, w))(model)
        optimizer.update(model, grads)
        return loss

    key = jax.random.key(seed)
    for _ in range(steps):
        key, sub = jax.random.split(key)
        x, y, w = make_batch(sub, batch, seq_len)
        step(model, optimizer, x, y, w)

    key, sub = jax.random.split(key)
    x, y, w = make_batch(sub, 256, seq_len)
    return float(model.loss(x, y, w))


def train_momos(
    task: str,
    S: int,
    K: int,
    steps: int,
    batch: int,
    seq_len: int,
    seed: int,
    learning_rate: float,
    subset_size: int,
    margin: float,
    n_neighbors: int,
    maintenance_every: int,
) -> tuple:
    """One Phase C cell: train with step D either on (``subset_size > 0``) or
    off (``subset_size == 0``, Phase B's exact behaviour). Returns
    ``(eval_loss, swap_rate_history)`` — the latter empty when swapping is
    off.

    Mirrors ``momos_capacity.py``'s jit strategy: ``MosaicState`` is not a
    registered pytree (its ``layout``/``cfg`` are static, host-side data), so
    the hot loop closes over those two and jits only the genuinely traced
    arrays — including ``graph``, which changes *value* every
    ``maintenance_every`` steps but never changes shape, so re-passing a
    freshly rebuilt graph into the same compiled ``jit_core`` costs nothing
    beyond the graph rebuild itself (a handful of milliseconds — SPEC.md
    §6.1's 67 Mflop estimate at the largest cell used here).
    """
    model = _build_model(task, seed)
    graphdef, param_state = nnx.split(model, nnx.Param)
    flat, layout = tiling.flatten_params(param_state, include=tiling.default_include)

    cfg = MosaicConfig(
        S=S, K=K, scale_mode="none",
        subset_size=subset_size, margin=margin, n_neighbors=n_neighbors,
    )
    tx = optax.adam(learning_rate)
    state = momos_init(flat, layout, cfg, tx, jax.random.key(seed))
    make_batch = toy_tasks.TASKS[task]

    def loss_fn(params, batch_xyw):
        x, y, w = batch_xyw
        return nnx.merge(graphdef, params).loss(x, y, w)

    def core(motifs, mosaic, active, scales, opt_state, graph, rng, x, y, w):
        s = MosaicState(
            motifs=motifs, mosaic=mosaic, active=active, scales=scales,
            opt_state=opt_state, layout=layout, cfg=cfg, graph=graph,
        )
        new_s, loss, swap_rate = train_step(s, (x, y, w), rng, loss_fn, tx)
        fields = (new_s.motifs, new_s.mosaic, new_s.active, new_s.scales, new_s.opt_state)
        return fields, loss, swap_rate

    jit_core = jax.jit(core)

    graph = None
    if subset_size > 0:
        graph = maintenance.neighbour_graph(state.motifs, state.active, n_neighbors)

    fields = (state.motifs, state.mosaic, state.active, state.scales, state.opt_state)
    key = jax.random.key(seed + 1)
    swap_history = []
    for step in range(steps):
        key, sub, step_rng = jax.random.split(key, 3)
        x, y, w = make_batch(sub, batch, seq_len)
        fields, _, swap_rate = jit_core(*fields, graph, step_rng, x, y, w)
        if subset_size > 0:
            swap_history.append(float(swap_rate))
            if maintenance_every > 0 and (step + 1) % maintenance_every == 0:
                graph = maintenance.neighbour_graph(fields[0], fields[2], n_neighbors)

    final_state = MosaicState(
        motifs=fields[0], mosaic=fields[1], active=fields[2], scales=fields[3],
        opt_state=fields[4], layout=layout, cfg=cfg, graph=graph,
    )
    key, sub = jax.random.split(key)
    x, y, w = make_batch(sub, 256, seq_len)
    eval_params = reconstruct(final_state)
    eval_loss = float(nnx.merge(graphdef, eval_params).loss(x, y, w))
    return eval_loss, swap_history


def _bucket_means(values, n_buckets: int = 8):
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    edges = np.linspace(0, len(arr), n_buckets + 1).astype(int)
    return [float(arr[edges[i]:edges[i + 1]].mean()) for i in range(n_buckets)
            if edges[i + 1] > edges[i]]


def run_gate(
    task: str,
    S_values,
    K_values,
    steps: int,
    batch: int,
    seq_len: int,
    seed: int,
    learning_rate: float,
    subset_size: int,
    margin: float,
    n_neighbors: int,
    maintenance_every: int,
) -> None:
    baseline_mse = toy_tasks.baseline_mse(task, jax.random.key(seed + 999), seq_len=seq_len)
    dense_loss = train_dense(task, steps, batch, seq_len, seed, learning_rate)

    print(f"task              : {task}")
    print(f"model             : {MODEL_KWARGS}")
    print(f"steps             : {steps}  (batch {batch}, seq_len {seq_len}, seed {seed})")
    print(f"subset_size       : {subset_size}   margin: {margin}   "
          f"n_neighbors: {n_neighbors}   maintenance_every: {maintenance_every}")
    print(f"constant baseline : {baseline_mse:.4f}  (best constant predictor)")
    print(f"dense training    : {dense_loss:.5f}  ({dense_loss / baseline_mse:.1%} of baseline)")
    print()

    header = (
        f"{'S':>2} {'K':>6} {'static':>10} {'dynamic':>10} {'dyn/static':>10} "
        f"{'vs dense':>9} {'swap early':>10} {'swap late':>10}  {'gate'}"
    )
    print(header)
    print("-" * len(header))

    passed, failed = 0, 0
    representative = None  # (S, K, swap_history) for the largest K, kept for the curve below
    start = time.time()
    for S in S_values:
        for K in K_values:
            static_loss, _ = train_momos(
                task, S, K, steps, batch, seq_len, seed, learning_rate,
                subset_size=0, margin=margin, n_neighbors=n_neighbors,
                maintenance_every=maintenance_every,
            )
            dynamic_loss, swap_history = train_momos(
                task, S, K, steps, batch, seq_len, seed, learning_rate,
                subset_size=subset_size, margin=margin, n_neighbors=n_neighbors,
                maintenance_every=maintenance_every,
            )
            ratio = dynamic_loss / static_loss
            gate_ok = dynamic_loss < static_loss
            passed, failed = (passed + 1, failed) if gate_ok else (passed, failed + 1)

            n = len(swap_history)
            early = float(np.mean(swap_history[: max(n // 5, 1)])) if n else float("nan")
            late = float(np.mean(swap_history[-max(n // 5, 1):])) if n else float("nan")

            print(
                f"{S:>2} {K:>6} {static_loss:>10.5f} {dynamic_loss:>10.5f} "
                f"{ratio:>9.2f}x {dynamic_loss / dense_loss:>8.2f}x "
                f"{early:>10.3f} {late:>10.3f}  {'PASS' if gate_ok else 'FAIL'}"
            )

            if representative is None or K >= representative[1]:
                representative = (S, K, swap_history)
    elapsed = time.time() - start

    print()
    print(f"gate: {passed}/{passed + failed} cells had dynamic beat static "
          f"(dynamic loss < static loss)")
    if failed:
        print(
            "NOT a universal pass. SPEC.md §8's Phase C gate is per-config, and a "
            "negative result here is the falsification the gate is designed to catch — "
            "see the script's final printed verdict / the report for the reading."
        )

    if representative is not None:
        S, K, swap_history = representative
        buckets = _bucket_means(swap_history)
        print()
        print(f"swap rate over training, S={S} K={K} (mean per decile of steps):")
        if buckets:
            print("  " + "  ".join(f"{b:.3f}" for b in buckets))
        else:
            print("  (no swaps recorded — subset_size may be 0)")

    print()
    print(f"wall clock: {elapsed:.1f}s on {jax.devices()[0].platform}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="cumsum", choices=sorted(toy_tasks.TASKS))
    parser.add_argument("--s", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--k", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=6e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=64,
                         help="blocks examined for reassignment per step (Phase C's dynamic run)")
    parser.add_argument(
        "--margin", type=float, default=1e-6,
        help="hysteresis threshold; measured empirically to sit near "
             "(learning_rate)^2 in scale (a single Adam step's squared magnitude), "
             "so --margin should shrink with --learning-rate or swaps stop firing "
             "at all — see the report for how this was found.",
    )
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--maintenance-every", type=int, default=20,
                         help="steps between neighbour-graph rebuilds")
    args = parser.parse_args()

    run_gate(
        task=args.task, S_values=args.s, K_values=args.k, steps=args.steps,
        batch=args.batch, seq_len=args.seq_len, seed=args.seed,
        learning_rate=args.learning_rate, subset_size=args.subset_size,
        margin=args.margin, n_neighbors=args.n_neighbors,
        maintenance_every=args.maintenance_every,
    )


if __name__ == "__main__":
    main()
