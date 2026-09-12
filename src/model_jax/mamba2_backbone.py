# -*- coding: utf-8 -*-
"""Mamba2 backbone: embeddings -> N blocks -> final norm.

Mirrors ``transformers.models.mamba2.modeling_mamba2.Mamba2Model``, minus the
cache (training never uses it) and minus generation.

Two things worth knowing:

* **Gradient checkpointing is per block**, driven by the same
  ``gradient_checkpointing`` config key the PyTorch path uses
  (``src/config/model/mamba2.yaml``, applied there via
  ``model.gradient_checkpointing_enable()``). Without it the chunked scan's
  intra-chunk ``[b, h, nc, c, c]`` tensors are retained for every layer's
  backward pass; ``nnx.remat`` recomputes each block instead, which is what
  makes the production geometry fit. This was missing from the original plan
  (which remat'd only the loss chunks) and is not optional at these shapes.
* **:func:`load_from_torch` transplants weights** from a live
  ``Mamba2ForCausalLM``. It exists for the cross-framework parity gate
  (tier 2), which is the highest-value check in the whole port, and is reused
  by ``export.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import nnx

from model_jax.mamba2_block import Mamba2Block, Mamba2Config, Mamba2RMSNorm

log = logging.getLogger(__name__)


class Mamba2Backbone(nnx.Module):
    """Stack of :class:`Mamba2Block` with a final norm.

    ``embeddings`` is injected rather than constructed: the whole point of this
    project is that it is a :class:`~model_jax.donor_projection.ProjectedEmbedding`,
    not an ``nnx.Embed``. Anything callable on ``[batch, seq]`` int ids works.
    """

    def __init__(
        self,
        cfg: Mamba2Config,
        embeddings: nnx.Module,
        *,
        rngs: nnx.Rngs,
        gradient_checkpointing: bool = True,
        param_dtype=jnp.float32,
    ):
        self.cfg = cfg
        self.embeddings = embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nnx.List(
            [
                Mamba2Block(cfg, i, rngs=rngs, param_dtype=param_dtype)
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm_f = Mamba2RMSNorm(
            cfg.hidden_size, eps=cfg.layer_norm_epsilon, param_dtype=param_dtype
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_hidden_states: bool = False,
    ):
        hidden_states = self.embeddings(input_ids)
        # Collected per block and once after norm_f — deliberately *not*
        # including the embedding output, matching what HF's Mamba2Model puts
        # in `output.hidden_states`, so the two line up index for index in the
        # layer-by-layer parity test.
        all_hidden = [] if output_hidden_states else None

        apply_block = _remat_block if self.gradient_checkpointing else _plain_block
        for block in self.layers:
            hidden_states = apply_block(block, hidden_states, attention_mask)
            if output_hidden_states:
                all_hidden.append(hidden_states)

        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden.append(hidden_states)
            return hidden_states, all_hidden
        return hidden_states


def _plain_block(block, hidden_states, attention_mask):
    return block(hidden_states, attention_mask=attention_mask)


# Recompute the block during backward instead of retaining its activations.
# `static_argnums` is not needed: attention_mask is an array (or None, which
# nnx.remat handles as an empty subtree).
_remat_block = nnx.remat(_plain_block)


# ---------------------------------------------------------------------------
# Cross-framework weight transplant
# ---------------------------------------------------------------------------


def _t(array) -> np.ndarray:
    """torch tensor -> float32 numpy, detached."""
    return np.asarray(array.detach().to("cpu").float().numpy(), dtype=np.float32)


def config_from_torch(torch_config) -> Mamba2Config:
    """Build a :class:`Mamba2Config` from a ``transformers.Mamba2Config``."""
    return Mamba2Config(
        vocab_size=torch_config.vocab_size,
        hidden_size=torch_config.hidden_size,
        num_hidden_layers=torch_config.num_hidden_layers,
        num_heads=torch_config.num_heads,
        head_dim=torch_config.head_dim,
        state_size=torch_config.state_size,
        expand=torch_config.expand,
        n_groups=torch_config.n_groups,
        conv_kernel=torch_config.conv_kernel,
        chunk_size=torch_config.chunk_size,
        initializer_range=torch_config.initializer_range,
        layer_norm_epsilon=torch_config.layer_norm_epsilon,
        use_bias=torch_config.use_bias,
        use_conv_bias=torch_config.use_conv_bias,
        residual_in_fp32=torch_config.residual_in_fp32,
        time_step_min=torch_config.time_step_min,
        time_step_max=torch_config.time_step_max,
        time_step_floor=torch_config.time_step_floor,
        time_step_limit=tuple(torch_config.time_step_limit),
    )


def load_from_torch(backbone: Mamba2Backbone, torch_backbone) -> Mamba2Backbone:
    """Copy a PyTorch ``Mamba2Model``'s block weights into *backbone*, in place.

    Only the blocks and the final norm — the embeddings are this project's
    projected ones and are transplanted separately (see
    ``donor_projection.p_from_torch_embedding``).

    Layout differences handled here, and nowhere else:

    * dense kernels: torch ``[out, in]`` -> jax ``[in, out]``
    * depthwise conv: torch ``[channels, 1, kernel]`` -> jax ``[kernel, 1, channels]``
    """
    for i, block in enumerate(backbone.layers):
        src = torch_backbone.layers[i]
        block.norm.weight[...] = jnp.asarray(_t(src.norm.weight))

        mixer, s_mixer = block.mixer, src.mixer
        mixer.in_proj[...] = jnp.asarray(_t(s_mixer.in_proj.weight).T)
        if mixer.in_proj_bias is not None:
            mixer.in_proj_bias[...] = jnp.asarray(_t(s_mixer.in_proj.bias))

        mixer.conv1d.kernel[...] = jnp.asarray(
            np.transpose(_t(s_mixer.conv1d.weight), (2, 1, 0))
        )
        if s_mixer.conv1d.bias is not None:
            mixer.conv1d.bias[...] = jnp.asarray(_t(s_mixer.conv1d.bias))

        mixer.A_log[...] = jnp.asarray(_t(s_mixer.A_log))
        mixer.D[...] = jnp.asarray(_t(s_mixer.D))
        mixer.dt_bias[...] = jnp.asarray(_t(s_mixer.dt_bias))
        mixer.norm.weight[...] = jnp.asarray(_t(s_mixer.norm.weight))

        mixer.out_proj[...] = jnp.asarray(_t(s_mixer.out_proj.weight).T)
        if mixer.out_proj_bias is not None:
            mixer.out_proj_bias[...] = jnp.asarray(_t(s_mixer.out_proj.bias))

    backbone.norm_f.weight[...] = jnp.asarray(_t(torch_backbone.norm_f.weight))
    return backbone
