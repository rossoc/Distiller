#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the tiny SSM regressor on a synthetic task. Runs on a CPU in seconds.

    python scripts/ssm_adding_problem.py                       # running sum
    python scripts/ssm_adding_problem.py --task adding         # classic adding problem
    python scripts/ssm_adding_problem.py --seq-len 128 --steps 3000

This is the cheapest end-to-end check that the hand-rolled Mamba2 recurrence
actually works: no donor checkpoint, no GPU, no dataset. If the scan is broken
the loss sits at the constant-predictor baseline and never moves.

Read the output as: final MSE well under the baseline = the recurrence carries
state correctly. Final MSE at the baseline = the model is predicting the mean,
i.e. learning nothing from the sequence.
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


def train(
    task: str = "cumsum",
    steps: int = 2000,
    batch: int = 64,
    seq_len: int = 64,
    d_model: int = 64,
    num_layers: int = 2,
    chunk_size: int = 16,
    learning_rate: float = 3e-3,
    seed: int = 0,
    log_every: int = 100,
    verbose: bool = True,
):
    """Train to convergence and return ``(final_mse, baseline_mse, history)``."""
    model = toy_tasks.build(
        task,
        rngs=nnx.Rngs(params=seed),
        d_model=d_model,
        num_layers=num_layers,
        chunk_size=chunk_size,
    )
    n_params = model.parameter_count()
    baseline = toy_tasks.baseline_mse(task, jax.random.key(seed + 777), seq_len=seq_len)

    if verbose:
        print(f"task        : {task}")
        print(f"parameters  : {n_params:,}")
        print(f"sequence    : {seq_len} steps, chunk_size {chunk_size} "
              f"({seq_len // chunk_size} chunks -> inter-chunk recurrence exercised)")
        print(f"baseline MSE: {baseline:.4f}  (best constant predictor)")
        print()

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=learning_rate,
        warmup_steps=max(steps // 20, 1), decay_steps=steps, end_value=0.0,
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=0.0))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    make_batch = toy_tasks.TASKS[task]

    @nnx.jit
    def train_step(model, optimizer, x, y, w):
        loss, grads = nnx.value_and_grad(lambda m: m.loss(x, y, w))(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_loss(model, x, y, w):
        return model.loss(x, y, w)

    key = jax.random.key(seed)
    history = []
    start = time.time()

    for step in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        x, y, w = make_batch(subkey, batch, seq_len)
        loss = float(train_step(model, optimizer, x, y, w))
        history.append(loss)

        if verbose and (step % log_every == 0 or step == 1):
            recent = float(np.mean(history[-log_every:]))
            bar = "#" * int(40 * max(0.0, 1.0 - recent / max(baseline, 1e-9)))
            print(f"  step {step:>5}  loss {recent:.5f}  "
                  f"{recent / baseline:6.1%} of baseline  {bar}")

    # Held-out evaluation on freshly generated data the model never trained on.
    key, subkey = jax.random.split(key)
    x, y, w = make_batch(subkey, 512, seq_len)
    final = float(eval_loss(model, x, y, w))
    elapsed = time.time() - start

    if verbose:
        print()
        print(f"held-out MSE: {final:.5f}   ({final / baseline:.2%} of baseline)")
        print(f"improvement : {baseline / max(final, 1e-12):.0f}x better than "
              f"predicting the mean")
        print(f"wall clock  : {elapsed:.1f}s on {jax.devices()[0].platform}")
        verdict = (
            "PASS — the recurrence carries state"
            if final < 0.1 * baseline
            else "FAIL — loss is at the constant-predictor baseline"
        )
        print(f"verdict     : {verdict}")

    return final, baseline, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="cumsum", choices=sorted(toy_tasks.TASKS))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    final, baseline, _ = train(
        task=args.task, steps=args.steps, batch=args.batch, seq_len=args.seq_len,
        d_model=args.d_model, num_layers=args.num_layers, chunk_size=args.chunk_size,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    sys.exit(0 if final < 0.1 * baseline else 1)


if __name__ == "__main__":
    main()
