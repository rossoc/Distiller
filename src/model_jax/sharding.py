# -*- coding: utf-8 -*-
"""Mesh / sharding helpers for data-parallel training.

The motivating reason for the JAX port is TPU and multi-device access, so this
is the piece that makes one code path run unchanged on a laptop CPU, a single
GPU, and a TPU pod slice.

The sharding plan, and why:

* **Donor tables (~402M values each, ~805 MB at bf16): replicated.** Two of
  them is ~1.6 GB, which fits comfortably beside everything else in a modern
  TPU chip's HBM, and replication keeps both the 262144-row embedding gather
  and the head matmul fully device-local. Sharding them would buy memory this
  model does not need and cost an all-to-all on every step.
* **Trainable params (~20M backbone + ~1.6M projections): replicated.**
  At this parameter count FSDP-style parameter sharding has nothing to offer;
  the gradients all-reduce automatically under ``jax.jit``'s SPMD lowering.
* **Batch: sharded over the ``data`` axis.**

A mesh of one device makes every collective a no-op, so the same jitted step
runs unmodified everywhere — there is no separate single-device path to keep in
sync.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

log = logging.getLogger(__name__)

DATA_AXIS = "data"


def make_mesh(shape: Optional[Sequence[int]] = None) -> Mesh:
    """Build a 1-D data-parallel mesh over the available devices.

    *shape* of ``None`` (or ``[-1]``) uses every device. An explicit shape is
    validated against the device count so a typo fails here rather than as a
    confusing reshape error inside the first jitted call.
    """
    devices = jax.devices()
    if shape is None or tuple(shape) in ((-1,), (0,)):
        n = len(devices)
    else:
        n = int(shape[0])
        if n > len(devices):
            raise ValueError(
                f"mesh_shape asks for {n} devices but only {len(devices)} "
                f"are visible: {devices}"
            )
    # numpy, not jnp: Mesh wants an object-dtype array of Device handles, and
    # jnp.asarray rejects them as non-numeric.
    mesh = Mesh(np.array(devices[:n], dtype=object).reshape(n), (DATA_AXIS,))
    log.info(
        "Mesh: %d device(s) on axis %r (%s)",
        n,
        DATA_AXIS,
        devices[0].platform,
    )
    return mesh


def batch_sharding(mesh: Mesh) -> NamedSharding:
    """Shard the leading (batch) axis across the mesh; replicate the rest."""
    return NamedSharding(mesh, P(DATA_AXIS, None))


def replicated(mesh: Mesh) -> NamedSharding:
    """Replicate a whole array on every device — params, optimizer state, donor."""
    return NamedSharding(mesh, P())


def shard_batch(mesh: Mesh, batch: dict) -> dict:
    """Place a host batch onto the mesh, sharded on the batch axis.

    The batch size must be divisible by the mesh size; ``batching.to_jax_batch``
    already pads every batch to a fixed leading dimension, so this is a
    configuration check rather than a per-batch hazard.
    """
    n = mesh.devices.size
    for name, array in batch.items():
        if array.shape[0] % n:
            raise ValueError(
                f"batch axis {array.shape[0]} of {name!r} is not divisible by "
                f"the {n}-device mesh; pick a batch size that is"
            )
    sharding = batch_sharding(mesh)
    return {k: jax.device_put(v, sharding) for k, v in batch.items()}


def replicate_state(mesh: Mesh, state):
    """Put every leaf of a pytree on every device."""
    return jax.device_put(state, replicated(mesh))
