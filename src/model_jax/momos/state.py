# -*- coding: utf-8 -*-
"""The MoMos dictionary's state, its dtype-driven compression ledger, and init.

``MosaicState`` holds exactly what SPEC.md §4 lists: the dictionary
(``motifs``), the compressed model (``mosaic``), which motifs are alive
(``active`` — always all-True until Phase D adds drop/revive), the optional
per-tensor scales, the motif-level optimiser state, and the static
:class:`~model_jax.momos.tiling.ParamLayout`. Phase C adds one field beyond
what SPEC.md §4's table shows: ``graph``, the ``(K, n_neighbors)`` neighbour
graph that ``train_step``'s step D reads swap candidates from
(:func:`model_jax.momos.maintenance.neighbour_graph` builds it). It defaults
to ``None`` because building it is a *maintenance*-loop responsibility, run
every ``cfg.maintenance_every`` steps against the live dictionary — not
``init``'s, which only ever sees the dictionary at step 0 — so a fresh state
is always expected to have its graph attached externally, via
``dataclasses.replace``, once the caller wants swapping.

The one thing worth internalising before touching any of this: **the mosaic's
dtype is the entire compression story.** Everything else here — motifs, Adam
moments, scales — is a fixed-size dictionary that does not grow with the
model. Only the mosaic scales with ``N``, so its per-element width in bytes is
what SPEC.md §2's "bytes/weight" column actually measures.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from model_jax.momos.tiling import ParamLayout, to_blocks

# Dense fp32 training: 4 bytes for the parameter itself + 4 for Adam's first
# moment + 4 for its second moment. SPEC.md §2's baseline for every ratio.
DENSE_BYTES_PER_WEIGHT = 12.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MosaicConfig:
    """The knobs that are fixed for the lifetime of one dictionary.

    ``subset_size``, ``margin`` and ``n_neighbors`` are Phase C's step D
    (SPEC.md §5) — a random subset of blocks proposes reassignment to a
    neighbouring motif each step, using the optimiser's real update. They
    default to values that make step D a no-op (``subset_size=0``), so every
    Phase B config literal (``MosaicConfig(S=..., K=..., scale_mode=...)``)
    keeps its exact Phase B behaviour without change — swapping is opt-in.
    Deliberately still excludes Phase D's lifecycle knobs (drop/merge/revive
    thresholds, maintenance cadence): those do not exist yet, and adding
    fields nothing reads is exactly the kind of premature structure the
    "build v1 simple" guidance warns against.
    """

    S: int
    K: int
    scale_mode: str = "none"  # "none" | "per_tensor" — SPEC.md §3.1
    subset_size: int = 0  # blocks examined for reassignment per step; 0 disables step D
    margin: float = 0.0  # hysteresis: swap only if strictly better by this much
    n_neighbors: int = 8  # neighbour-graph out-degree per motif (SPEC.md §6.1)

    # Phase C2 additions (SPEC_PHASE_C2.md §2):
    cohort_frac: float = 0.01  # fraction of M accumulating drift at a time; 0 disables reassignment
    eval_window: int = 50  # steps per accumulation window
    k_delta: int = 16  # codebook directions (clamped to 2 for S=1)
    dir_ema: float = 0.9  # codebook direction EMA
    tau_sim: float = 0.5  # min cosine to count as matching a direction
    jump_frac_init: float = 0.05  # fraction of cohort eligible to jump, window 0
    jump_frac_min: float = 0.005  # minimum fraction of cohort eligible to jump
    jump_frac_decay: float = 0.98  # eligibility decay per window
    base_lr: float = 1e-3  # base learning rate for adaptive scale
    beta2: float = 0.999  # Adam beta2 for bias correction
    kmeans_iters: int = 4  # iterations for spherical k-means
    A: int = 0  # coprime multiplier for cohort generation; 0 -> pick at init
    B: int = 0  # offset for cohort generation

    def __post_init__(self) -> None:
        if self.S not in (1, 2, 4):
            raise ValueError(f"S must be 1, 2, or 4 (SPEC.md §1); got {self.S}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.scale_mode not in ("none", "per_tensor"):
            raise ValueError(
                f"Unknown scale_mode {self.scale_mode!r} — expected 'none' or 'per_tensor'."
            )
        if self.subset_size < 0:
            raise ValueError(f"subset_size must be >= 0, got {self.subset_size}")
        if self.margin < 0:
            raise ValueError(f"margin must be >= 0, got {self.margin}")
        if self.n_neighbors < 0:
            raise ValueError(f"n_neighbors must be >= 0, got {self.n_neighbors}")

        # Phase C2 validations (SPEC_PHASE_C2.md §2):
        if self.S == 1 and self.k_delta != 2:
            import logging
            logging.getLogger(__name__).info(
                f"S=1: clamping k_delta from {self.k_delta} to 2 (unit vectors in 1-D are {{+1, -1}})"
            )
            object.__setattr__(self, "k_delta", 2)

        if self.k_delta < 2:
            raise ValueError(f"k_delta must be >= 2, got {self.k_delta}")
        if not (0.0 < self.tau_sim < 1.0):
            raise ValueError(f"tau_sim must be in (0, 1), got {self.tau_sim}")
        if not (0.0 <= self.cohort_frac <= 1.0):
            raise ValueError(f"cohort_frac must be in [0, 1], got {self.cohort_frac}")
        if self.eval_window <= 0:
            raise ValueError(f"eval_window must be > 0, got {self.eval_window}")
        if not (0.0 <= self.dir_ema <= 1.0):
            raise ValueError(f"dir_ema must be in [0, 1], got {self.dir_ema}")
        if not (0.0 < self.jump_frac_init <= 1.0):
            raise ValueError(f"jump_frac_init must be in (0, 1], got {self.jump_frac_init}")
        if not (0.0 <= self.jump_frac_min <= 1.0):
            raise ValueError(f"jump_frac_min must be in [0, 1], got {self.jump_frac_min}")
        if not (0.0 < self.jump_frac_decay <= 1.0):
            raise ValueError(f"jump_frac_decay must be in (0, 1], got {self.jump_frac_decay}")
        if self.kmeans_iters <= 0:
            raise ValueError(f"kmeans_iters must be > 0, got {self.kmeans_iters}")


def mosaic_dtype(K: int) -> np.dtype:
    """The mosaic's storage dtype, chosen purely from the dictionary size.

    SPEC.md §2: this choice *is* the compression ratio. A uint32 mosaic at
    S=1 costs 4 bytes/weight — identical to a dense fp32 parameter, i.e. zero
    compression — so this is asserted at every construction site rather than
    left to whatever dtype ``jnp.zeros``/``astype`` would pick by default.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if K <= 256:
        return np.dtype(np.uint8)
    if K <= 65536:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _assert_mosaic_dtype(mosaic: jnp.ndarray, K: int) -> None:
    expected = mosaic_dtype(K)
    if mosaic.dtype != expected:
        raise AssertionError(
            f"mosaic dtype {mosaic.dtype} does not match the dtype K={K} demands "
            f"({expected}) — a silent widening here destroys the compression ratio."
        )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MosaicState:
    motifs: jnp.ndarray  # (K, S) float32 — the only large trainable tensor
    mosaic: jnp.ndarray  # (M,) uint8/uint16/uint32 — the compressed model
    active: jnp.ndarray  # (K,) bool
    scales: Optional[jnp.ndarray]  # (n_tensors,) float32, or None
    opt_state: optax.OptState  # sized for (K,S) + scales, never (M,S)
    layout: ParamLayout  # static, host-side
    cfg: MosaicConfig  # static, host-side
    graph: Optional[jnp.ndarray] = None  # (K, n_neighbors) int32, or None

    def __post_init__(self) -> None:
        _assert_mosaic_dtype(self.mosaic, self.cfg.K)


def trainable(state: MosaicState) -> dict:
    """The pytree the motif-level optimiser actually owns — never (M, S)."""
    return {"motifs": state.motifs, "scales": state.scales}


def init(
    flat: jnp.ndarray,
    layout: ParamLayout,
    cfg: MosaicConfig,
    optimizer: optax.GradientTransformation,
    rng: jax.Array,
) -> MosaicState:
    """Build the initial dictionary and a random mosaic over it.

    Motif *values* are seeded from real blocks of ``flat`` (sampled without
    replacement when ``K <= M``, with replacement otherwise) rather than from
    noise — a cheap, one-pick version of k-means++'s seeding step, without the
    Lloyd iterations. This matters most for ``scale_mode="none"``: nothing
    else pulls a tensor's values onto a shared scale, so starting the
    dictionary at the scale of the actual weights (instead of, say, unit
    Gaussian noise) is the difference between training from a plausible point
    and training from one that is orders of magnitude off for most tensors.

    The mosaic itself — which block points at which motif — is assigned
    uniformly at random, per SPEC.md Phase B ("random init assignment (or
    nearest-motif from a k-means++ style seed)"). The nearest-motif variant is
    a strict improvement left for later: it requires an (M, K) distance
    computation that is exactly the kind of thing worth adding once a
    capacity curve shows random assignment is the bottleneck, not before.
    """
    blocks = to_blocks(flat, cfg.S)
    M = int(blocks.shape[0])
    dtype = mosaic_dtype(cfg.K)

    if cfg.A == 0:
        from model_jax.momos.drift import pick_coprime

        cfg = dataclasses.replace(cfg, A=pick_coprime(M))

    rng_seed, rng_mosaic = jax.random.split(rng)
    if M == 0:
        motifs = jnp.zeros((cfg.K, cfg.S), jnp.float32)
    else:
        idx = jax.random.choice(rng_seed, M, (cfg.K,), replace=cfg.K > M)
        motifs = blocks[idx].astype(jnp.float32)

    mosaic = jax.random.randint(rng_mosaic, (M,), 0, cfg.K).astype(dtype)
    active = jnp.ones((cfg.K,), dtype=bool)
    scales = (
        jnp.ones((layout.n_tensors,), jnp.float32) if cfg.scale_mode == "per_tensor" else None
    )

    opt_state = optimizer.init({"motifs": motifs, "scales": scales})
    return MosaicState(
        motifs=motifs,
        mosaic=mosaic,
        active=active,
        scales=scales,
        opt_state=opt_state,
        layout=layout,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# The compression ledger — SPEC.md §2
# ---------------------------------------------------------------------------


def dictionary_bytes(cfg: MosaicConfig, n_tensors: int = 0) -> int:
    """Fixed cost of the dictionary: motif value + Adam m + Adam v, per scalar.

    ``K*S`` motif scalars at 3 fp32 slots each, plus (if per-tensor scaling is
    on) the same 3 slots for each of the ``n_tensors`` scales. Reproduces
    SPEC.md §2's "dictionary (motifs+Adam)" column exactly (e.g. K=4096, S=4
    -> 4096*4*12 = 196608 B = 192 KB).
    """
    total = cfg.K * cfg.S * 4 * 3
    if cfg.scale_mode == "per_tensor":
        total += n_tensors * 4 * 3
    return total


def asymptotic_bytes_per_weight(cfg: MosaicConfig) -> float:
    """bytes/weight as N -> infinity, where the dictionary's fixed cost vanishes.

    This is the number SPEC.md §2's table actually reports: with N in the
    10^6-10^9 regime the dictionary (a few KB to a few MB) is negligible next
    to the mosaic, so bytes/weight collapses to ``mosaic_itemsize / S``.
    """
    return mosaic_dtype(cfg.K).itemsize / cfg.S


def bytes_per_weight(N: int, cfg: MosaicConfig, n_tensors: int = 0) -> Tuple[float, float]:
    """Actual bytes/weight at a *finite* N, and its ratio against dense fp32.

    At toy-task N (10^4-10^5) the dictionary is not negligible the way it is
    at the regime's intended scale — this is the honest, non-asymptotic
    number, which is why it can come out worse than dense for a small model
    with a large dictionary. Report both this and
    :func:`asymptotic_bytes_per_weight`; conflating them overstates the win
    at small N and understates it at large N.
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    M = math.ceil(N / cfg.S)
    mosaic_bytes = M * mosaic_dtype(cfg.K).itemsize
    total = mosaic_bytes + dictionary_bytes(cfg, n_tensors)
    per_weight = total / N
    return per_weight, DENSE_BYTES_PER_WEIGHT / per_weight
