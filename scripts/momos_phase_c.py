#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MoMos reassignment gate: compare mosaic reassignment methods on a toy SSM.

Every method is a Hydra config group in ``src/config/momos/``; ``gate.arms``
picks which ones run and in what column order, with the first arm as the
control the ratio columns are measured against.

    python scripts/momos_phase_c.py
    python scripts/momos_phase_c.py 'gate.s=[2]' 'gate.k=[1024]' gate.steps=600
    python scripts/momos_phase_c.py 'gate.arms=[static,single,drift]'
    python scripts/momos_phase_c.py momos=drift momos.eval_window=5

| arm    | description                                                   |
|--------|---------------------------------------------------------------|
| static | Phase B, no reassignment (subset_size=0, cohort_frac=0)       |
| single | Phase C, per-step swap on g_blocks — the adopted method       |
| drift  | Phase C2, windowed drift accumulation; parked, off by default |

Regime requirements (SPEC.md §8, SPEC_PHASE_C2.md §8):
- Model with >= 1M mosaicked values: toy_tasks.build(..., d_model=256, num_layers=4)
  giving ~1.63M mosaicked parameters.
- Skip any cell with K >= M/10 (degenerate/out of regime).
- Skip any cell whose compressed footprint reaches dense fp32 (SPEC.md §2.1);
  below the crossover N the dictionary does not amortise.
- Print K/M and the compression ledger as columns so the regime is visible.
- Report real experimental measurements without tuning until it passes.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from flax import nnx  # noqa: E402

from model_jax import toy_tasks  # noqa: E402
from model_jax.momos import maintenance, metrics, tiling  # noqa: E402
from model_jax.momos.drift import cohort_indices, init_drift_state  # noqa: E402
from model_jax.momos.lifecycle import lifecycle_pass  # noqa: E402
from model_jax.momos.reassign import macro_reassign  # noqa: E402
from model_jax.momos.metrics import drift_buffer_bytes  # noqa: E402
from model_jax.momos.state import DENSE_BYTES_PER_WEIGHT  # noqa: E402
from model_jax.momos.state import MosaicConfig, MosaicState  # noqa: E402
from model_jax.momos.state import asymptotic_bytes_per_weight  # noqa: E402
from model_jax.momos.state import bytes_per_weight as ledger_bytes_per_weight  # noqa: E402
from model_jax.momos.state import init as momos_init  # noqa: E402
from model_jax.momos.train_step import reconstruct, train_step  # noqa: E402

# Target regime: >= 1M parameters (~1.63M mosaicked)
MODEL_KWARGS = dict(d_model=256, num_layers=4, state_size=16, chunk_size=8)


def _build_model(task: str, seed: int) -> nnx.Module:
    return toy_tasks.build(task, rngs=nnx.Rngs(params=seed), **MODEL_KWARGS)


def train_dense(
    task: str, steps: int, batch: int, seq_len: int, seed: int, learning_rate: float
) -> float:
    """Ordinary dense training baseline on the ~1.63M parameter architecture."""
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


_MOSAIC_CONFIG_FIELDS = {f.name for f in dataclass_fields(MosaicConfig)}


def _mosaic_config(
    method: DictConfig,
    *,
    S: int,
    K: int,
    learning_rate: float,
    reserve_zero_motif: Optional[bool] = None,
) -> MosaicConfig:
    """Build a MosaicConfig from one ``momos/*.yaml`` group.

    Only keys that are real MosaicConfig fields are forwarded, so a yaml may
    carry presentation/runner knobs (``method``, ``maintenance_every``)
    alongside the dataclass's own without a filter list here going stale every
    time SPEC adds a knob. ``S``/``K`` come from the gate sweep and ``base_lr``
    from the run's learning rate, so all three override whatever the file says.
    """
    raw: Dict[str, Any] = OmegaConf.to_container(method, resolve=True)  # type: ignore[assignment]
    kwargs = {k: v for k, v in raw.items() if k in _MOSAIC_CONFIG_FIELDS}
    kwargs.update(S=S, K=K, base_lr=learning_rate)
    if reserve_zero_motif is not None:
        kwargs["reserve_zero_motif"] = reserve_zero_motif
    return MosaicConfig(**kwargs)


class MomosResult(tuple):
    """6-tuple return for train_momos with extra usage_0 attributes for P1."""
    usage_0: int
    usage_0_frac: float

    def __new__(cls, values, usage_0: int = 0, usage_0_frac: float = 0.0):
        t = super().__new__(cls, values)
        t.usage_0 = usage_0
        t.usage_0_frac = usage_0_frac
        return t


def train_momos(
    task: str,
    S: int,
    K: int,
    method: "DictConfig",
    steps: int,
    batch: int,
    seq_len: int,
    seed: int,
    learning_rate: float,
    reserve_zero_motif: Optional[bool] = None,
) -> Any:
    """Train one arm, whose knobs come entirely from a ``momos/*.yaml`` group.

    ``method`` is one composed config from ``src/config/momos/`` — its
    ``method:`` field names the arm and the rest are MosaicConfig knobs. The
    branch below is on that name rather than on a hardcoded arm list, so a new
    method is a new yaml file, not an edit here.

    Returns:
        (eval_loss, swap_history, spread_history, final_live, final_entropy, lifecycle_history)
        as a MomosResult with .usage_0 and .usage_0_frac attributes.
    """
    model = _build_model(task, seed)
    graphdef, param_state = nnx.split(model, nnx.Param)
    flat, layout = tiling.flatten_params(param_state, include=tiling.default_include)
    M = math.ceil(layout.n_values / S)
    cfg = _mosaic_config(
        method, S=S, K=K, learning_rate=learning_rate, reserve_zero_motif=reserve_zero_motif
    )
    arm = str(method.method)
    maint_every = int(method.maintenance_every)
    C = max(1, int(cfg.cohort_frac * M))

    tx = optax.adam(learning_rate)
    state = momos_init(flat, layout, cfg, tx, jax.random.key(seed))
    make_batch = toy_tasks.TASKS[task]

    def loss_fn(params, batch_xyw):
        x, y, w = batch_xyw
        return nnx.merge(graphdef, params).loss(x, y, w)

    swap_history: List[float] = []
    spread_history: List[float] = []
    lifecycle_history: List[Any] = []

    if arm == "drift":

        def core_drift(motifs, mosaic, active, scales, opt_state, drift, window, rng, x, y, w):
            s = MosaicState(
                motifs=motifs,
                mosaic=mosaic,
                active=active,
                scales=scales,
                opt_state=opt_state,
                layout=layout,
                cfg=cfg,
                graph=None,
            )
            new_s, loss, _, new_drift = train_step(
                s, (x, y, w), rng, loss_fn, tx, drift=drift, window=window
            )
            fields = (new_s.motifs, new_s.mosaic, new_s.active, new_s.scales, new_s.opt_state)
            return fields, loss, new_drift

        jit_core_drift = jax.jit(core_drift)

        graph = maintenance.neighbour_graph(state.motifs, state.active, cfg.n_neighbors)
        drift_st = init_drift_state(cfg, M, rng=jax.random.key(seed + 2))
        drift = drift_st.drift
        codebook_dirs = drift_st.codebook_dirs
        window = 0

        fields = (state.motifs, state.mosaic, state.active, state.scales, state.opt_state)
        key = jax.random.key(seed + 1)

        for step in range(steps):
            key, sub, step_rng = jax.random.split(key, 3)
            x, y, w = make_batch(sub, batch, seq_len)
            fields, _, drift = jit_core_drift(
                *fields, drift, window, step_rng, x, y, w
            )

            if (step + 1) % cfg.eval_window == 0:
                cohort = cohort_indices(window, M, C, cfg.A, cfg.B)
                new_mosaic, codebook_dirs, win_metrics = macro_reassign(
                    drift=drift,
                    cohort=cohort,
                    motifs=fields[0],
                    mosaic=fields[1],
                    active=fields[2],
                    graph=graph,
                    codebook_dirs=codebook_dirs,
                    window=window,
                    cfg=cfg,
                )
                fields = (fields[0], new_mosaic, fields[2], fields[3], fields[4])
                swap_history.append(float(win_metrics.swap_rate))
                spread_history.append(float(win_metrics.codebook_spread))

                # Reset drift buffer and increment window
                drift = jnp.zeros((C, cfg.S), dtype=jnp.float32)
                window += 1

            # Graph rebuild is on its own cadence, NOT nested inside the
            # window-end branch: nesting it silently skipped every rebuild
            # whenever eval_window did not divide maintenance_every.
            if maint_every > 0 and (step + 1) % maint_every == 0:
                graph = maintenance.neighbour_graph(fields[0], fields[2], cfg.n_neighbors)

    else:

        def core_single(motifs, mosaic, active, scales, opt_state, graph, rng, x, y, w):
            s = MosaicState(
                motifs=motifs,
                mosaic=mosaic,
                active=active,
                scales=scales,
                opt_state=opt_state,
                layout=layout,
                cfg=cfg,
                graph=graph,
            )
            new_s, loss, swap_rate = train_step(s, (x, y, w), rng, loss_fn, tx)
            fields = (new_s.motifs, new_s.mosaic, new_s.active, new_s.scales, new_s.opt_state)
            return fields, loss, swap_rate

        jit_core_single = jax.jit(core_single)

        graph = None
        if cfg.subset_size > 0 and arm in ("single", "lifecycle"):
            graph = maintenance.neighbour_graph(state.motifs, state.active, cfg.n_neighbors)

        fields = (state.motifs, state.mosaic, state.active, state.scales, state.opt_state)
        key = jax.random.key(seed + 1)

        for step in range(steps):
            key, sub, step_rng = jax.random.split(key, 3)
            x, y, w = make_batch(sub, batch, seq_len)
            fields, _, swap_rate = jit_core_single(*fields, graph, step_rng, x, y, w)

            if arm in ("single", "lifecycle"):
                swap_history.append(float(swap_rate))
                if maint_every > 0 and (step + 1) % maint_every == 0:
                    maint_pass = (step + 1) // maint_every
                    if (
                        arm == "lifecycle"
                        and cfg.lifecycle_every > 0
                        and (maint_pass % cfg.lifecycle_every == 0)
                    ):
                        cur_state = MosaicState(
                            motifs=fields[0],
                            mosaic=fields[1],
                            active=fields[2],
                            scales=fields[3],
                            opt_state=fields[4],
                            layout=layout,
                            cfg=cfg,
                            graph=graph,
                        )
                        cur_state, pass_metrics = lifecycle_pass(cur_state, step=step)
                        fields = (
                            cur_state.motifs,
                            cur_state.mosaic,
                            cur_state.active,
                            cur_state.scales,
                            cur_state.opt_state,
                        )
                        lifecycle_history.append(pass_metrics)
                    graph = maintenance.neighbour_graph(fields[0], fields[2], cfg.n_neighbors)

    final_state = MosaicState(
        motifs=fields[0],
        mosaic=fields[1],
        active=fields[2],
        scales=fields[3],
        opt_state=fields[4],
        layout=layout,
        cfg=cfg,
        graph=None,
    )
    key, sub = jax.random.split(key)
    x, y, w = make_batch(sub, 256, seq_len)
    eval_params = reconstruct(final_state)
    eval_loss = float(nnx.merge(graphdef, eval_params).loss(x, y, w))
    final_live = metrics.live_motifs(final_state)
    final_entropy = metrics.usage_entropy(final_state)
    u0_count, u0_frac = metrics.zero_motif_usage(final_state)
    return MomosResult(
        (eval_loss, swap_history, spread_history, final_live, final_entropy, lifecycle_history),
        usage_0=u0_count,
        usage_0_frac=u0_frac,
    )


def _bucket_means(values: List[float], n_buckets: int = 8) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    edges = np.linspace(0, len(arr), n_buckets + 1).astype(int)
    return [
        float(arr[edges[i] : edges[i + 1]].mean())
        for i in range(n_buckets)
        if edges[i + 1] > edges[i]
    ]


def _cell_ledger(
    N: int, S: int, K: int, cohort_frac: float
) -> Tuple[float, float, float, float, float]:
    """Persistent-memory ledger for one (S, K) cell (SPEC.md §2, §2.1).

    Reuses the ledger helpers rather than restating the arithmetic: the
    static/single arms pay mosaic + dictionary, and the drift arm additionally
    pays the (C, S) fp32 drift buffer of SPEC_PHASE_C2.md §3.

    Returns ``(bw_base, ratio_base, bw_drift, ratio_drift, bw_asymptotic)``,
    where each ``bw_*`` is bytes/weight and each ``ratio_*`` is the factor
    against dense fp32 training (12 B/wt).
    """
    cfg = MosaicConfig(S=S, K=K, scale_mode="none", cohort_frac=cohort_frac)
    # scale_mode="none" for every arm, so n_tensors contributes nothing.
    bw_base, ratio_base = ledger_bytes_per_weight(N, cfg, 0)
    M = math.ceil(N / S)
    drift_b = drift_buffer_bytes(M, S, cohort_frac) if cohort_frac > 0 else 0
    bw_drift = bw_base + drift_b / N
    return bw_base, ratio_base, bw_drift, DENSE_BYTES_PER_WEIGHT / bw_drift, asymptotic_bytes_per_weight(cfg)

def run_gate(gate: DictConfig, methods: Dict[str, DictConfig]) -> None:
    """Compare every method in ``gate.arms`` across the (S, K) sweep.

    The first arm is the control that the ratio columns are taken against;
    `gate/default.yaml` documents that `static` should stay in that slot.
    """
    task = str(gate.task)
    steps, batch, seq_len = int(gate.steps), int(gate.batch), int(gate.seq_len)
    seed, learning_rate = int(gate.seed), float(gate.learning_rate)
    seeds: List[int] = (
        [int(s) for s in gate.seeds]
        if "seeds" in gate and gate.seeds is not None
        else [seed]
    )
    if not seeds:
        seeds = [seed]
    arms: List[str] = [str(a) for a in gate.arms]
    control = arms[0]

    sample_model = _build_model(task, seeds[0])
    _, sample_params = nnx.split(sample_model, nnx.Param)
    _, sample_layout = tiling.flatten_params(
        sample_params, include=tiling.default_include
    )
    n_mosaicked = sample_layout.n_values

    baseline_mse = toy_tasks.baseline_mse(
        task, jax.random.key(seeds[0] + 999), seq_len=seq_len
    )
    if len(seeds) > 1:
        dense_losses = [
            train_dense(task, steps, batch, seq_len, s, learning_rate) for s in seeds
        ]
        dense_loss = float(np.mean(dense_losses))
    else:
        dense_loss = train_dense(task, steps, batch, seq_len, seeds[0], learning_rate)

    # The widest drift buffer among the participating arms; 0 when none of them
    # accumulates drift, in which case the buffer columns are dropped entirely
    # rather than printed as a column of zeros.
    max_cohort_frac = max(float(methods[a].cohort_frac) for a in arms)
    show_drift_cols = max_cohort_frac > 0

    print("=" * 80)
    print("MoMos reassignment gate".center(80))
    print("=" * 80)
    print(f"task              : {task}")
    print(f"model             : {MODEL_KWARGS}")
    print(f"mosaicked params  : {n_mosaicked:,} values")
    print(
        f"steps             : {steps}  (batch {batch}, seq_len {seq_len}, "
        f"{'seeds ' + str(seeds) if len(seeds) > 1 else 'seed ' + str(seed)})"
    )
    print(f"learning_rate     : {learning_rate}")
    print(f"arms              : {', '.join(arms)}  (control: {control})")
    for a in arms:
        knobs = OmegaConf.to_container(methods[a], resolve=True)
        knobs.pop("method", None)
        print(f"  {a:<15} : " + ", ".join(f"{k}={v}" for k, v in knobs.items()))
    print(f"constant baseline : {baseline_mse:.4f}")
    print(f"dense training    : {dense_loss:.5f}  ({dense_loss / baseline_mse:.1%} of baseline)")
    print()

    print(
        "  ledger (SPEC.md §2): B/wt = mosaic+dictionary bytes per weight; "
        "mem_x = factor vs dense\n"
        f"  fp32 ({DENSE_BYTES_PER_WEIGHT:.0f} B/wt). B/wt_inf is the N->inf asymptote. "
        "loss/dns is a LOSS ratio, not memory."
    )

    gate_rzm = bool(gate.get("reserve_zero_motif", False))
    show_u0 = (
        gate_rzm
        or bool(gate.get("show_u0", False))
        or any(bool(methods[a].get("reserve_zero_motif", False)) for a in arms)
    )

    header = f"{'S':>2} {'K':>5} {'K/M':>7} {'B/wt':>6} {'mem_x':>6}"
    if show_drift_cols:
        header += f" {'Bwt_dft':>7} {'mem_x':>6}"
    header += f" {'B/wt_inf':>8}"
    for a in arms:
        header += f" {a[:8]:>8} {a[:3]+'_lv':>6} {a[:4]+'_h':>6}"
        if show_u0:
            header += f" {a[:3]+'_u0':>6}"
    for a in arms[1:]:
        header += f" {(a[:5] + '/ctl'):>9}"
    header += f" {'loss/dns':>8}  {'best':<12}"
    print(header)
    print("-" * len(header))

    f1_reports: List[str] = []
    wins = {a: 0 for a in arms[1:]}
    total_evaluated = 0
    trace_data = []
    start_time = time.time()

    for S in [int(v) for v in gate.s]:
        M = math.ceil(n_mosaicked / S)
        for K in [int(v) for v in gate.k]:
            km_ratio = K / M
            bw_base, mem_base, bw_drift, mem_drift, bw_asym = _cell_ledger(
                n_mosaicked, S, K, max_cohort_frac
            )

            # Regime requirement (SPEC_PHASE_C2.md §8): skip any cell with K >= M/10
            if K >= M / 10:
                print(
                    f"{S:>2} {K:>5} {km_ratio:>7.2%} "
                    f"SKIPPED (K >= M/10: degenerate / out of regime)",
                    flush=True,
                )
                continue

            # Compression requirement (SPEC.md §2.1): below the crossover N the
            # dictionary does not amortise and the compressed model's persistent
            # memory meets or exceeds dense fp32 — measuring there is meaningless.
            if bw_drift >= DENSE_BYTES_PER_WEIGHT:
                print(
                    f"{S:>2} {K:>5} {km_ratio:>7.2%} "
                    f"SKIPPED (no compression: {bw_drift:.2f} B/wt >= "
                    f"{DENSE_BYTES_PER_WEIGHT:.1f} dense; "
                    f"crossover needs N > {12 * K * S / (12 - bw_asym):.0f}, have {n_mosaicked})",
                    flush=True,
                )
                continue

            total_evaluated += 1
            print(
                f"Running cell S={S} K={K} (K/M={km_ratio:.2%}, "
                f"{bw_base:.3f} B/wt = {mem_base:.1f}x dense, "
                f"{'seeds ' + str(seeds) if len(seeds) > 1 else 'seed ' + str(seed)})...",
                flush=True,
            )

            per_seed_results: Dict[int, Dict[str, Any]] = {}
            for s in seeds:
                per_seed_results[s] = {}
                for a in arms:
                    arm_rzm = gate_rzm or bool(methods[a].get("reserve_zero_motif", False))
                    res = train_momos(
                        task,
                        S,
                        K,
                        methods[a],
                        steps,
                        batch,
                        seq_len,
                        s,
                        learning_rate,
                        reserve_zero_motif=arm_rzm,
                    )
                    per_seed_results[s][a] = res

            losses: Dict[str, float] = {}
            loss_spreads: Dict[str, float] = {}
            swaps: Dict[str, List[float]] = {}
            spreads: Dict[str, List[float]] = {}
            lives: Dict[str, int] = {}
            entropies: Dict[str, float] = {}
            lifecycles: Dict[str, List[Any]] = {}
            u0_counts: Dict[str, int] = {}
            u0_fracs: Dict[str, float] = {}

            for a in arms:
                arm_losses = [per_seed_results[s][a][0] for s in seeds]
                losses[a] = float(np.mean(arm_losses))
                loss_spreads[a] = float(np.max(arm_losses) - np.min(arm_losses))
                lives[a] = int(round(np.mean([per_seed_results[s][a][3] for s in seeds])))
                entropies[a] = float(np.mean([per_seed_results[s][a][4] for s in seeds]))
                u0_counts[a] = int(round(np.mean([per_seed_results[s][a].usage_0 for s in seeds])))
                u0_fracs[a] = float(np.mean([per_seed_results[s][a].usage_0_frac for s in seeds]))
                swaps[a] = per_seed_results[seeds[0]][a][1]
                spreads[a] = per_seed_results[seeds[0]][a][2]
                lifecycles[a] = per_seed_results[seeds[0]][a][5]

            if len(seeds) > 1:
                for s in seeds:
                    s_detail = ", ".join(
                        f"{a}={per_seed_results[s][a][0]:.5f}" for a in arms
                    )
                    print(f"  seed {s}: {s_detail}", flush=True)
                sp_detail = ", ".join(
                    f"{a}={loss_spreads[a]:.5f}" for a in arms
                )
                print(f"  spread: {sp_detail}", flush=True)

                if "lifecycle" in arms and "single" in arms:
                    gap = abs(losses["lifecycle"] - losses["single"])
                    max_sp = max(loss_spreads["lifecycle"], loss_spreads["single"])
                    if max_sp > gap:
                        v = f"INCONCLUSIVE (spread {max_sp:.5f} > gap {gap:.5f})"
                    elif losses["lifecycle"] < losses["single"]:
                        v = f"PASS (gap {gap:.5f} > spread {max_sp:.5f}, ratio {losses['lifecycle']/losses['single']:.2f}x)"
                    else:
                        v = f"FAIL (single beats lifecycle by {gap:.5f} > spread {max_sp:.5f})"
                    f1_reports.append(
                        f"  S={S} K={K}: {v}\n"
                        f"    single:    mean={losses['single']:.5f} spread={loss_spreads['single']:.5f}\n"
                        f"    lifecycle: mean={losses['lifecycle']:.5f} spread={loss_spreads['lifecycle']:.5f}"
                    )

            best = min(arms, key=lambda a: losses[a])
            for a in arms[1:]:
                if losses[a] < losses[control]:
                    wins[a] += 1

            row = f"{S:>2} {K:>5} {km_ratio:>7.2%} {bw_base:>6.3f} {mem_base:>5.1f}x"
            if show_drift_cols:
                row += f" {bw_drift:>7.3f} {mem_drift:>5.1f}x"
            row += f" {bw_asym:>8.3f}"
            for a in arms:
                row += f" {losses[a]:>8.5f} {lives[a]:>6d} {entropies[a]:>6.2f}"
                if show_u0:
                    row += f" {u0_fracs[a]:>6.1%}"
            for a in arms[1:]:
                row += f" {losses[a] / losses[control]:>8.2f}x"
            row += f" {losses[best] / dense_loss:>7.2f}x  {best:<12}"
            print(row, flush=True)

            trace_data.append((S, K, swaps, spreads, lifecycles, u0_counts, u0_fracs))

    elapsed = time.time() - start_time
    print()
    print("-" * 80)
    print("Summary:")
    for a in arms[1:]:
        print(f"  {a} beat {control} : {wins[a]}/{total_evaluated} cells")
    print(f"  Wall-clock time   : {elapsed:.1f}s on {jax.devices()[0].platform}")
    if f1_reports:
        print("-" * 80)
        print("F1 Seed Replication Report (SPEC_PHASE_D_DEFECTS.md §F1):")
        for rep in f1_reports:
            print(rep)
    print("-" * 80, flush=True)

    for item in trace_data:
        S, K, swaps, spreads, lifecycles = item[:5]
        u0_c = item[5] if len(item) > 5 else {}
        u0_f = item[6] if len(item) > 6 else {}
        print(f"\n[Trace S={S}, K={K}]")
        for a in arms:
            if a in u0_c:
                print(f"  {a:<9} usage_0   : {u0_c[a]} ({u0_f[a]:.2%})")
            if swaps.get(a):
                vals = swaps[a]
                # Per-step arms produce one value per step; windowed arms produce
                # one per window, which is already short enough to print raw.
                shown = _bucket_means(vals, n_buckets=6) if len(vals) > 12 else vals
                print(f"  {a:<9} swap_rate : " + "  ".join(f"{b:.3f}" for b in shown))
            if spreads.get(a):
                print(f"  {a:<9} spread    : " + "  ".join(f"{b:.3f}" for b in spreads[a]))
            if lifecycles.get(a):
                for idx, lm in enumerate(lifecycles[a]):
                    print(
                        f"  {a:<9} pass {idx + 1:<2}   : "
                        f"merged={lm.n_merged:<3} dropped={lm.n_dropped:<3} "
                        f"live={lm.live_before}->{lm.live_after} "
                        f"ent={lm.entropy_before:.2f}->{lm.entropy_after:.2f} "
                        f"scale={lm.merge_scale:.4e} thr={lm.merge_thr:.4e} "
                        f"clamped={lm.floor_clamped}"
                    )
    sys.stdout.flush()


CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "src" / "config")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_momos")
def main(cfg: DictConfig) -> None:
    # Each arm reads its own momos/*.yaml. The arm that matches the selected
    # `momos` group uses the composed cfg.momos instead of the file, so CLI
    # overrides (`momos=drift momos.eval_window=5`) actually reach the run.
    methods: Dict[str, DictConfig] = {}
    for arm in [str(a) for a in cfg.gate.arms]:
        if str(cfg.momos.method) == arm:
            methods[arm] = cfg.momos
        else:
            path = Path(CONFIG_DIR) / "momos" / f"{arm}.yaml"
            if not path.exists():
                raise FileNotFoundError(
                    f"gate.arms lists {arm!r} but {path} does not exist. "
                    f"Available: {sorted(q.stem for q in path.parent.glob('*.yaml'))}"
                )
            methods[arm] = OmegaConf.load(path)

    run_gate(cfg.gate, methods)


if __name__ == "__main__":
    main()
