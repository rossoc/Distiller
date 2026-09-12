# -*- coding: utf-8 -*-
"""Tests for model_jax/sharding.py.

CI has one device, so these pin the *single-device degenerate case* — which is
the case that actually matters for correctness. The whole design claim is that a
mesh of one makes every collective a no-op, so the same jitted step runs
unmodified on a laptop, one GPU, and a TPU slice; if the one-device path needed
special-casing, that claim would be false and the multi-device path would be a
separate, untested code path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_jax import sharding as sh


def test_default_mesh_uses_every_visible_device():
    mesh = sh.make_mesh(None)
    assert mesh.devices.size == len(jax.devices())
    assert mesh.axis_names == (sh.DATA_AXIS,)


def test_explicit_mesh_shape_is_validated_against_the_device_count():
    """A typo must fail here, not as a reshape error inside the first jitted call."""
    with pytest.raises(ValueError, match="only .* are visible"):
        sh.make_mesh([len(jax.devices()) + 99])


def test_batch_is_sharded_on_the_leading_axis():
    mesh = sh.make_mesh(None)
    batch = {
        "input_ids": jnp.ones((4, 8), jnp.int32),
        "labels": jnp.ones((4, 8), jnp.int32),
    }
    out = sh.shard_batch(mesh, batch)

    for name, array in out.items():
        assert array.shape == batch[name].shape
        assert array.sharding.spec == jax.sharding.PartitionSpec(sh.DATA_AXIS, None)


def test_an_indivisible_batch_is_rejected_with_a_useful_message():
    """Caught at placement time rather than as a silent reshape later."""
    mesh = sh.make_mesh(None)
    if mesh.devices.size == 1:
        pytest.skip("every batch size divides a 1-device mesh")
    with pytest.raises(ValueError, match="not divisible"):
        sh.shard_batch(mesh, {"input_ids": jnp.ones((mesh.devices.size + 1, 4))})


def test_replicated_state_round_trips_values():
    mesh = sh.make_mesh(None)
    state = {"w": jnp.arange(6.0).reshape(2, 3)}
    placed = sh.replicate_state(mesh, state)
    assert np.allclose(np.asarray(placed["w"]), np.asarray(state["w"]))
    assert placed["w"].sharding.spec == jax.sharding.PartitionSpec()


def test_sharded_and_unsharded_steps_agree_on_one_device():
    """The claim under test: a mesh of one changes results not at all."""
    mesh = sh.make_mesh(None)

    @jax.jit
    def step(batch):
        return jnp.sum(batch["x"] ** 2)

    batch = {"x": jnp.asarray(np.random.default_rng(0).normal(size=(4, 8)), jnp.float32)}
    plain = float(step(batch))
    sharded = float(step(sh.shard_batch(mesh, batch)))
    assert plain == pytest.approx(sharded, rel=1e-7)
