# -*- coding: utf-8 -*-
"""Phase E: wiring MoMos onto a real ``nnx.Module`` (SPEC.md §8, Phase E).

Everything in ``train_step.py`` was built and gated against
``model_jax.toy_tasks``' 56k/1.6M-parameter SSM regressor — a single flat
params tree with no frozen buffers. A real model has two things that
harness never exercised:

1. **Non-``nnx.Param`` state that must survive the round trip.**
   ``model_jax.mimir_mamba2.MimirMamba2Model`` holds its (805 MiB) donor
   tables as ``Donor`` variables specifically so they are *not*
   ``nnx.Param`` — SPEC.md never has to think about them, but reconstructing
   a callable model needs them merged back in every step.
2. **Trainable leaves the dictionary must not touch.** ``tiling.
   default_include`` already excludes Mamba2's per-head SSM scalars
   (``A_log``, ``D``, ``dt_bias``) and norm weights, for the scale-sensitivity
   reason documented there — but for phases A-D those leaves were simply
   *frozen at init* (``tiling.py``'s own docstring says so). That is fine for
   a norm weight the whole ablation never needed to move; it is not fine for
   ``A_log`` on a real SSM, which sets the recurrence's decay rate and has to
   train. This module gives those leaves an ordinary optimiser, updated in
   the *same* backward pass as the dictionary (one forward/backward, not
   two — see :func:`step`), so enabling MoMos does not change what the
   excluded leaves do.

``ModelBundle`` is the real-model analogue of ``MosaicState`` — everything
:func:`step` and :func:`merged_model` need, held together so the caller
threads one object through the training loop instead of five.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from model_jax.momos import tiling
from model_jax.momos.state import MosaicConfig, MosaicState
from model_jax.momos.state import init as mosaic_init
from model_jax.momos.state import trainable
from model_jax.momos.train_step import _dense_blocks, _propose_swaps, reconstruct

# (model, batch) -> scalar loss — the real-model analogue of
# ``train_step.LossFn``, which takes a bare params tree; this takes the
# merged model itself since the model owns ``.loss(...)``.
ModelLossFn = Callable[[nnx.Module, Any], jnp.ndarray]


@dataclasses.dataclass
class ModelBundle:
    """Everything needed to run and train a MoMos-quantised real model.

    Two different things are both "not part of the dictionary", and this
    dataclass keeps them apart because they are handled completely
    differently:

    * ``rest`` — non-``nnx.Param`` state (e.g. frozen ``Donor`` tables).
      Never touched by anything here; carried through ``nnx.split``/
      ``nnx.merge`` unchanged, every step.
    * ``excluded`` / ``excluded_opt_state`` — the ``nnx.Param`` leaves
      ``tiling``'s include predicate rejected (SSM scalars, norms). These
      *do* train, via ``excluded_optimizer``, just not through the
      dictionary. ``excluded`` is a tuple aligned with
      ``mosaic.layout.excluded``'s order (fixed at :func:`init_bundle` time).
    """

    graphdef: Any
    rest: Any
    mosaic: MosaicState
    excluded: Tuple[jnp.ndarray, ...]
    excluded_opt_state: Any


def init_bundle(
    model: nnx.Module,
    cfg: MosaicConfig,
    dict_optimizer: optax.GradientTransformation,
    excluded_optimizer: optax.GradientTransformation,
    rng: jax.Array,
    include: tiling.IncludePredicate = tiling.default_include,
) -> ModelBundle:
    """Split a real model into (frozen buffers, dictionary, excluded leaves).

    ``nnx.split(model, nnx.Param, ...)``'s ``...`` filter is what makes this
    safe for a model with buffers ``tiling`` has never seen: it catches
    *everything* that is not ``nnx.Param`` (donor tables included) into
    ``rest``, rather than silently dropping it — dropping it would make
    ``nnx.merge`` fail, or worse, succeed with the wrong donor tables.
    """
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    flat, layout = tiling.flatten_params(params, include=include)
    mosaic = mosaic_init(flat, layout, cfg, dict_optimizer, rng)
    excluded_values = tuple(jnp.asarray(value) for _, value in layout.excluded)
    excluded_opt_state = excluded_optimizer.init(excluded_values)
    return ModelBundle(
        graphdef=graphdef,
        rest=rest,
        mosaic=mosaic,
        excluded=excluded_values,
        excluded_opt_state=excluded_opt_state,
    )


def merged_model(bundle: ModelBundle, gather_dtype: Optional[jnp.dtype] = None) -> nnx.Module:
    """Reconstruct the dense, callable model (SPEC.md §5's ``reconstruct``,
    generalised to also restore the excluded leaves' *trained* values and the
    frozen ``rest`` state a bare ``reconstruct()`` call knows nothing about).
    """
    params = reconstruct(bundle.mosaic, gather_dtype=gather_dtype, excluded_values=bundle.excluded)
    return nnx.merge(bundle.graphdef, params, bundle.rest)


def step(
    bundle: ModelBundle,
    batch: Dict[str, jnp.ndarray],
    rng: jax.Array,
    loss_fn: ModelLossFn,
    dict_optimizer: optax.GradientTransformation,
    excluded_optimizer: optax.GradientTransformation,
) -> Tuple[ModelBundle, jnp.ndarray, jnp.ndarray]:
    """One MoMos micro-loop step against a real model.

    Mirrors ``train_step.train_step``'s steps A-D exactly (gather before
    grad, aggregate once with ``segment_sum``, one motif-level optimiser
    step, then propose swaps from that step's real update — see
    ``train_step.py``'s module docstring for why each of those orderings is
    load-bearing) with one addition: ``loss_of`` takes the excluded leaves as
    a *third* differentiable argument, so ``jax.value_and_grad`` returns their
    gradient from the same forward/backward pass that produces the
    dictionary's — training them costs nothing beyond what training the
    dictionary alone would already cost.
    """
    mstate = bundle.mosaic
    layout, cfg = mstate.layout, mstate.cfg
    gathered = mstate.motifs[mstate.mosaic]

    def loss_of(blocks, scales, excluded_values):
        dense = _dense_blocks(blocks, scales, layout, cfg.scale_mode)
        flat = tiling.from_blocks(dense, layout.n_values)
        params = tiling.unflatten_params(flat, layout, excluded_values=excluded_values)
        model = nnx.merge(bundle.graphdef, params, bundle.rest)
        return loss_fn(model, batch)

    loss, (g_blocks, g_scales, g_excluded) = jax.value_and_grad(loss_of, argnums=(0, 1, 2))(
        gathered, mstate.scales, bundle.excluded
    )

    # Aggregate exactly once, usage-normalised — see train_step.train_step.
    K = mstate.motifs.shape[0]
    seg = mstate.mosaic.astype(jnp.int32)
    counts = jax.ops.segment_sum(jnp.ones_like(seg, dtype=jnp.float32), seg, K)
    g_motifs = jax.ops.segment_sum(g_blocks, seg, K) / jnp.maximum(counts, 1.0)[:, None]
    g_motifs = jnp.where(mstate.active[:, None], g_motifs, 0.0)
    if cfg.reserve_zero_motif:
        g_motifs = g_motifs.at[0].set(0.0)

    dict_params = trainable(mstate)
    dict_grads = {"motifs": g_motifs, "scales": g_scales}
    dict_updates, dict_opt_state = dict_optimizer.update(dict_grads, mstate.opt_state, dict_params)
    if cfg.reserve_zero_motif:
        dict_updates["motifs"] = dict_updates["motifs"].at[0].set(0.0)
    dict_new = optax.apply_updates(dict_params, dict_updates)
    motifs, scales = dict_new["motifs"], dict_new["scales"]
    if cfg.reserve_zero_motif:
        motifs = motifs.at[0].set(0.0)

    excluded_updates, excluded_opt_state = excluded_optimizer.update(
        g_excluded, bundle.excluded_opt_state, bundle.excluded
    )
    excluded = optax.apply_updates(bundle.excluded, excluded_updates)

    mosaic_idx, swap_rate = _propose_swaps(
        mstate, cfg, motifs, dict_updates["motifs"], g_motifs, g_blocks, rng
    )

    new_mosaic_state = dataclasses.replace(
        mstate, motifs=motifs, scales=scales, opt_state=dict_opt_state, mosaic=mosaic_idx
    )
    new_bundle = dataclasses.replace(
        bundle, mosaic=new_mosaic_state, excluded=excluded, excluded_opt_state=excluded_opt_state
    )
    return new_bundle, loss, swap_rate


# ---------------------------------------------------------------------------
# jit wrapping — for a real training loop (src/train_jax.py)
# ---------------------------------------------------------------------------
#
# ``MosaicState``/``ModelBundle`` are plain dataclasses, not registered JAX
# pytrees: ``ParamLayout`` holds a ``numpy`` array, which is unhashable, and a
# registered pytree's static/aux data must be hashable, so registering them
# is not an option, not just an omission. ``scripts/momos_phase_c.py`` hits
# the same wall and its answer — close ``layout``/``cfg`` over from the
# enclosing scope, and jit a function whose actual arguments are only array
# leaves (see its ``core_drift``/``core_single``) — is the established idiom
# this project already uses. The two functions below package that idiom once
# so a caller (``train_jax.py``) does not have to hand-roll it again.


def make_jit_step(
    layout,
    cfg: MosaicConfig,
    graphdef: Any,
    loss_fn: ModelLossFn,
    dict_optimizer: optax.GradientTransformation,
    excluded_optimizer: optax.GradientTransformation,
):
    """Build a ``jax.jit``-compiled version of :func:`step`.

    Returns a function ``core(motifs, mosaic, active, scales, opt_state,
    excluded, excluded_opt_state, rest, rng, batch) -> (motifs, mosaic,
    active, scales, opt_state, excluded, excluded_opt_state, rest, loss,
    swap_rate)`` — every argument and return value a plain array or a
    pytree of arrays, so ``jax.jit`` can trace it directly. The caller keeps
    ``layout``/``cfg``/``graphdef`` (unchanging for the run) on the side and
    threads only this tuple through the training loop.
    """

    def core(motifs, mosaic, active, scales, opt_state, excluded, excluded_opt_state, rest, rng, batch):
        bundle = ModelBundle(
            graphdef=graphdef,
            rest=rest,
            mosaic=MosaicState(
                motifs=motifs, mosaic=mosaic, active=active, scales=scales,
                opt_state=opt_state, layout=layout, cfg=cfg, graph=None,
            ),
            excluded=excluded,
            excluded_opt_state=excluded_opt_state,
        )
        new_bundle, loss, swap_rate = step(bundle, batch, rng, loss_fn, dict_optimizer, excluded_optimizer)
        m = new_bundle.mosaic
        return (
            m.motifs, m.mosaic, m.active, m.scales, m.opt_state,
            new_bundle.excluded, new_bundle.excluded_opt_state, new_bundle.rest,
            loss, swap_rate,
        )

    return jax.jit(core)


def make_jit_eval(layout, cfg: MosaicConfig, graphdef: Any, loss_fn: ModelLossFn):
    """The evaluation-side counterpart of :func:`make_jit_step`: rebuild the
    dense model from a bundle's array leaves and return its loss, jitted."""

    def core(motifs, mosaic, active, scales, excluded, rest, batch):
        bundle = ModelBundle(
            graphdef=graphdef,
            rest=rest,
            mosaic=MosaicState(
                motifs=motifs, mosaic=mosaic, active=active, scales=scales,
                opt_state=None, layout=layout, cfg=cfg, graph=None,
            ),
            excluded=excluded,
            excluded_opt_state=None,
        )
        return loss_fn(merged_model(bundle), batch)

    return jax.jit(core)


def bundle_arrays(bundle: ModelBundle) -> Tuple[Any, ...]:
    """The array leaves :func:`make_jit_step` and :func:`make_jit_eval` take
    and return, unpacked from a :class:`ModelBundle` once (e.g. right after
    :func:`init_bundle`) so the training loop can thread the tuple through
    instead of re-deriving it every step."""
    m = bundle.mosaic
    return (m.motifs, m.mosaic, m.active, m.scales, m.opt_state, bundle.excluded, bundle.excluded_opt_state, bundle.rest)


def bundle_from_arrays(arrays: Tuple[Any, ...], *, layout, cfg: MosaicConfig, graphdef: Any) -> ModelBundle:
    """Inverse of :func:`bundle_arrays` — for checkpointing or a final
    ``merged_model`` call outside the jitted loop."""
    motifs, mosaic, active, scales, opt_state, excluded, excluded_opt_state, rest = arrays
    return ModelBundle(
        graphdef=graphdef,
        rest=rest,
        mosaic=MosaicState(
            motifs=motifs, mosaic=mosaic, active=active, scales=scales,
            opt_state=opt_state, layout=layout, cfg=cfg, graph=None,
        ),
        excluded=excluded,
        excluded_opt_state=excluded_opt_state,
    )
