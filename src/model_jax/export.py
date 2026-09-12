# -*- coding: utf-8 -*-
"""Collapse the projections and export a standalone model.

The JAX counterpart of ``MimirMamba2Module.export_standalone``, and the step the
whole design builds towards: apply the *current* ``P`` and ``Q`` to the frozen
donor tables and write out concrete ``[vocab_size, d_model]`` tables. The result
has no donor, no projections, and no custom modules — and is ~3x smaller than
the donor it was distilled from.

It exports to a **PyTorch** ``Mamba2ForCausalLM`` rather than a JAX artifact,
deliberately. Orbax checkpoints and PyTorch ``state_dict``s are not
interchangeable — a hard format break called out as a risk in the port plan —
and everything downstream of training in this repo (``src/predict.py``,
``scripts/eval_all_checkpoints.py``, anyone loading the model with plain
``transformers``) speaks PyTorch. Exporting across the break is what keeps the
JAX path from being a dead end, and it means a JAX-trained model drops into the
existing evaluation and deployment tooling unchanged.

    from model_jax.export import export_to_torch
    torch_model = export_to_torch(jax_model)
    torch_model.save_pretrained("outputs/exported")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def torch_config_from(cfg, pad_token_id: Optional[int] = None,
                      bos_token_id: Optional[int] = None,
                      eos_token_id: Optional[int] = None):
    """Build a ``transformers.Mamba2Config`` matching this project's config."""
    from transformers import Mamba2Config

    return Mamba2Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        state_size=cfg.state_size,
        expand=cfg.expand,
        n_groups=cfg.n_groups,
        conv_kernel=cfg.conv_kernel,
        chunk_size=cfg.chunk_size,
        initializer_range=cfg.initializer_range,
        layer_norm_epsilon=cfg.layer_norm_epsilon,
        use_bias=cfg.use_bias,
        use_conv_bias=cfg.use_conv_bias,
        residual_in_fp32=cfg.residual_in_fp32,
        # The projections replace both tables, so there is nothing to tie —
        # and tying would try to share two differently-shaped tensors.
        tie_word_embeddings=False,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        use_cache=True,
    )


def export_to_torch(
    model,
    dtype: str = "float32",
    pad_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
):
    """A stock ``Mamba2ForCausalLM`` producing the same logits as *model*.

    The projections are collapsed into concrete tables via
    ``materialize_weight`` (a chunked sweep over the donor, so the
    ``[262144, 1536]`` intermediate is never held whole), and the block weights
    are transposed back into torch's ``[out, in]`` convention — the inverse of
    ``mamba2_backbone.load_from_torch``.
    """
    import torch
    from transformers import Mamba2ForCausalLM

    cfg = model.cfg
    exported = Mamba2ForCausalLM(
        torch_config_from(cfg, pad_token_id, bos_token_id, eos_token_id)
    )

    with torch.no_grad():
        for i, block in enumerate(model.backbone.layers):
            dst = exported.backbone.layers[i]
            dst.norm.weight.copy_(torch.as_tensor(_np(block.norm.weight.get_value())))

            mixer, d_mixer = block.mixer, dst.mixer
            d_mixer.in_proj.weight.copy_(
                torch.as_tensor(_np(mixer.in_proj.get_value()).T)
            )
            if mixer.in_proj_bias is not None:
                d_mixer.in_proj.bias.copy_(
                    torch.as_tensor(_np(mixer.in_proj_bias.get_value()))
                )

            # jax [kernel, 1, channels] -> torch [channels, 1, kernel]
            d_mixer.conv1d.weight.copy_(
                torch.as_tensor(
                    np.transpose(_np(mixer.conv1d.kernel.get_value()), (2, 1, 0))
                )
            )
            if mixer.conv1d.bias is not None:
                d_mixer.conv1d.bias.copy_(
                    torch.as_tensor(_np(mixer.conv1d.bias.get_value()))
                )

            d_mixer.A_log.copy_(torch.as_tensor(_np(mixer.A_log.get_value())))
            d_mixer.D.copy_(torch.as_tensor(_np(mixer.D.get_value())))
            d_mixer.dt_bias.copy_(torch.as_tensor(_np(mixer.dt_bias.get_value())))
            d_mixer.norm.weight.copy_(
                torch.as_tensor(_np(mixer.norm.weight.get_value()))
            )
            d_mixer.out_proj.weight.copy_(
                torch.as_tensor(_np(mixer.out_proj.get_value()).T)
            )
            if mixer.out_proj_bias is not None:
                d_mixer.out_proj.bias.copy_(
                    torch.as_tensor(_np(mixer.out_proj_bias.get_value()))
                )

        exported.backbone.norm_f.weight.copy_(
            torch.as_tensor(_np(model.backbone.norm_f.weight.get_value()))
        )

        # "Recompute the embeddings with the layer that just got updated",
        # made explicit — the scalar gains fold in, so the map stays linear.
        import jax.numpy as jnp

        embed = _np(model.projected_embedding.materialize_weight(jnp.float32))
        head = _np(model.projected_lm_head.materialize_weight(jnp.float32))
        exported.backbone.embeddings.weight.copy_(torch.as_tensor(embed))
        exported.lm_head.weight.copy_(torch.as_tensor(head))

    return exported.to(getattr(torch, dtype)).eval()


def save_pretrained(model, save_path: str, tokenizer_dir: Optional[str] = None) -> None:
    """Write the collapsed standalone model (+ tokenizer) to *save_path*.

    Loadable by anyone with plain ``transformers`` — no donor required, no JAX
    required. Unlike the PyTorch path this does not additionally write
    ``donor_projections.pt``: resuming JAX training uses the orbax checkpoint
    that ``train_jax`` already keeps, and mixing the two serialisation formats
    in one directory would invite exactly the confusion the format break
    already threatens.
    """
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    export_to_torch(model).save_pretrained(str(path))

    if tokenizer_dir:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(str(tokenizer_dir)).save_pretrained(str(path))
    log.info("Saved standalone Mamba2 model to %s", path)
