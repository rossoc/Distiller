#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep (S, K, scale_mode) for MoMos on a toy task; print a capacity curve.

    python scripts/momos_capacity.py                      # cumsum, default grid
    python scripts/momos_capacity.py --task adding
    python scripts/momos_capacity.py --k 256 1024 --s 1 2

This is Phase B's gate (SPEC.md §8): final loss vs K at each S, in both
``scale_mode``s, against the dense baseline that trains the *same* model
(same architecture, same steps, same data) with an ordinary ``nnx.Optimizer``
instead of a dictionary. The toy task is ``model_jax.toy_tasks`` — a ~56k
-parameter SSM regressor with an established dense baseline — chosen
specifically so this runs in well under a minute on a CPU (SPEC.md §9).

Every MoMos config also reports ``bytes_per_weight`` at the model's *actual*
parameter count, not the asymptotic (N -> infinity) number SPEC.md §2's table
quotes: at this toy scale the fixed-size dictionary is not negligible, and
reporting only the asymptote would overstate the win. Both numbers are
printed so the difference is visible rather than papered over.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402

from model_jax import toy_tasks  # noqa: E402
from model_jax.momos import tiling  # noqa: E402
from model_jax.momos.state import (  # noqa: E402
    MosaicConfig,
    MosaicState,
    asymptotic_bytes_per_weight,
)
from model_jax.momos.state import bytes_per_weight as ledger_bytes_per_weight  # noqa: E402
from model_jax.momos.state import init as momos_init  # noqa: E402
from model_jax.momos.train_step import reconstruct, train_step  # noqa: E402

MODEL_KWARGS = dict(d_model=32, num_layers=2, state_size=16, chunk_size=8)


def _build_model(task: str, seed: int) -> nnx.Module:
    return toy_tasks.build(task, rngs=nnx.Rngs(params=seed), **MODEL_KWARGS)


def train_dense(
    task: str, steps: int, batch: int, seq_len: int, seed: int, learning_rate: float
) -> float:
    """Ordinary dense training of the same architecture — the number every
    MoMos cell in the capacity table is measured against."""
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
    scale_mode: str,
    steps: int,
    batch: int,
    seq_len: int,
    seed: int,
    learning_rate: float,
) -> tuple:
    """One capacity-curve cell. Returns ``(eval_loss, bytes_per_weight, ratio_vs_dense)``.

    ``MosaicState`` is not registered as a JAX pytree (see state.py) — its
    static fields (``layout``, ``cfg``) do not belong in a traced argument
    list at all. So the hot loop here jits a closure over exactly those two
    static objects plus the loss function and optimiser, taking only the
    genuinely traced arrays as arguments. That is what makes this run at
    roughly the same per-step cost as the ``@nnx.jit``-decorated dense loop
    above rather than paying Python dispatch overhead on every array op.
    """
    model = _build_model(task, seed)
    graphdef, param_state = nnx.split(model, nnx.Param)
    flat, layout = tiling.flatten_params(param_state, include=tiling.default_include)

    cfg = MosaicConfig(S=S, K=K, scale_mode=scale_mode)
    tx = optax.adam(learning_rate)
    state = momos_init(flat, layout, cfg, tx, jax.random.key(seed))
    make_batch = toy_tasks.TASKS[task]

    def loss_fn(params, batch_xyw):
        x, y, w = batch_xyw
        return nnx.merge(graphdef, params).loss(x, y, w)

    def core(motifs, mosaic, active, scales, opt_state, rng, x, y, w):
        s = MosaicState(
            motifs=motifs, mosaic=mosaic, active=active, scales=scales,
            opt_state=opt_state, layout=layout, cfg=cfg,
        )
        new_s, loss, _ = train_step(s, (x, y, w), rng, loss_fn, tx)
        return (new_s.motifs, new_s.mosaic, new_s.active, new_s.scales, new_s.opt_state), loss

    jit_core = jax.jit(core)

    fields = (state.motifs, state.mosaic, state.active, state.scales, state.opt_state)
    key = jax.random.key(seed + 1)
    for _ in range(steps):
        key, sub, step_rng = jax.random.split(key, 3)
        x, y, w = make_batch(sub, batch, seq_len)
        fields, _ = jit_core(*fields, step_rng, x, y, w)

    final_state = MosaicState(
        motifs=fields[0], mosaic=fields[1], active=fields[2], scales=fields[3],
        opt_state=fields[4], layout=layout, cfg=cfg,
    )
    key, sub = jax.random.split(key)
    x, y, w = make_batch(sub, 256, seq_len)
    eval_params = reconstruct(final_state)
    eval_loss = float(nnx.merge(graphdef, eval_params).loss(x, y, w))

    n_tensors = layout.n_tensors if scale_mode == "per_tensor" else 0
    per_weight, ratio = ledger_bytes_per_weight(layout.n_values, cfg, n_tensors)
    return eval_loss, per_weight, ratio, layout.n_values


def run_sweep(
    task: str,
    S_values,
    K_values,
    scale_modes,
    steps: int,
    batch: int,
    seq_len: int,
    seed: int,
    learning_rate: float,
) -> None:
    baseline_dense = train_dense(task, steps, batch, seq_len, seed, learning_rate)
    baseline_mse = toy_tasks.baseline_mse(task, jax.random.key(seed + 999), seq_len=seq_len)

    print(f"task              : {task}")
    print(f"model             : {MODEL_KWARGS}")
    print(f"steps             : {steps}  (batch {batch}, seq_len {seq_len})")
    print(f"constant baseline : {baseline_mse:.4f}  (best constant predictor)")
    print(f"dense training    : {baseline_dense:.5f}  "
          f"({baseline_dense / baseline_mse:.1%} of constant baseline)")
    print("dense bytes/weight: 12.00 (4 param + 4 Adam m + 4 Adam v)")
    print()

    header = (
        f"{'S':>2} {'K':>6} {'scale_mode':>11} {'eval loss':>10} "
        f"{'vs dense':>9} {'B/weight':>9} {'vs dense B/w':>12} {'asymptotic B/w':>14}"
    )
    print(header)
    print("-" * len(header))

    start = time.time()
    for S in S_values:
        for K in K_values:
            for scale_mode in scale_modes:
                loss, per_weight, ratio, N = train_momos(
                    task, S, K, scale_mode, steps, batch, seq_len, seed, learning_rate
                )
                cfg = MosaicConfig(S=S, K=K, scale_mode=scale_mode)
                asymptote = asymptotic_bytes_per_weight(cfg)
                print(
                    f"{S:>2} {K:>6} {scale_mode:>11} {loss:>10.5f} "
                    f"{loss / baseline_dense:>8.2f}x {per_weight:>9.3f} "
                    f"{ratio:>11.2f}x {asymptote:>14.3f}"
                )
    elapsed = time.time() - start

    print()
    print(f"N (mosaicked values) : {N:,}  "
          f"(dictionary overhead is NOT negligible at this scale — see the")
    print("                        'B/weight' vs 'asymptotic B/w' gap above)")
    print(f"wall clock            : {elapsed:.1f}s on {jax.devices()[0].platform}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="cumsum", choices=sorted(toy_tasks.TASKS))
    parser.add_argument("--s", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--k", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument(
        "--scale-modes", nargs="+", default=["none", "per_tensor"],
        choices=["none", "per_tensor"],
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=6e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_sweep(
        task=args.task, S_values=args.s, K_values=args.k, scale_modes=args.scale_modes,
        steps=args.steps, batch=args.batch, seq_len=args.seq_len, seed=args.seed,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
