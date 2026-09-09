# -*- coding: utf-8 -*-
"""Mamba2 Lightning module with frozen-donor embedding / head projections.

A much smaller replacement for full DFM-Mimir fine-tuning
(``model.dfm_mimir``). Three ideas, in the order they matter:

1. **Keep the donor's embeddings.** DFM-Mimir's lexical knowledge of a
   low-resource language lives in its 262144 x 1536 input table and its
   equally large output head — ~805M of its ~1B parameters. Both are kept
   frozen and reused verbatim; only a ``[1536, 512]`` projection of each is
   learned (786K parameters apiece). See ``model.donor_projection``.

2. **Drop attention for recurrence.** Normalized span extraction needs to
   find a handful of keywords, not to reason over them, so the sequence
   mixer is a stack of Mamba2 (SSM) blocks — a linear-time recurrent model —
   trained conventionally by backprop.

3. **Shrink the width.** ``d_model`` drops 1536 -> 512, so every Mamba2 block
   is ~9x cheaper than a DFM-Mimir block of the same shape would be.

Layout, with defaults (``d_model=512``, 12 layers)::

    input_ids ─▶ ProjectedEmbedding ─▶ Mamba2 backbone ─▶ ProjectedLMHead ─▶ logits
                 E_donor[ids] @ P        12 SSM blocks       (h @ Qᵀ) @ W_donorᵀ
                 frozen · 0.8M train     ~20M train          frozen · 0.8M train

Tokenization is the donor's, unchanged, and still lives in
``lit_datamodule.py`` — token ids have to index the donor tables, so the two
models are not just compatible but required to share a tokenizer.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (
    Mamba2Config,
    Mamba2ForCausalLM,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from model.hf_compat import DTYPE_MAP, load_tokenizer  # noqa: F401  (side effects + shared helpers)
from model.donor_projection import (
    ProjectedEmbedding,
    ProjectedLMHead,
    build_projections,
    load_donor_tables,
)

log = logging.getLogger(__name__)

# Filename used by save_pretrained/from_pretrained for the trainable
# projections. They are excluded from the exported standalone model (which
# carries concrete, already-projected tables instead), but are needed to
# resume training or to re-derive the tables from a different donor revision.
_PROJECTIONS_FILE = "donor_projections.pt"
_PROJECTIONS_META = "donor_projections.json"

_WARNED_SLOW_SCAN = False


def _warn_if_slow_scan_path() -> None:
    """Note once whether Mamba2 is running on its pure-PyTorch scan.

    Without ``mamba-ssm``/``causal-conv1d`` installed, transformers falls back
    to a correct but unfused scan that materializes
    ``[batch, seq, chunk_size, heads, state]`` float32 tensors per layer. That
    fallback — not the model's 22M parameters — is what sets the batch size
    this trains at, so it is worth saying out loud rather than leaving it to be
    rediscovered from an OOM traceback.
    """
    global _WARNED_SLOW_SCAN
    if _WARNED_SLOW_SCAN:
        return
    _WARNED_SLOW_SCAN = True
    if importlib.util.find_spec("mamba_ssm") and importlib.util.find_spec("causal_conv1d"):
        return
    log.info(
        "mamba-ssm / causal-conv1d are not installed, so Mamba2 runs its "
        "pure-PyTorch scan. It is numerically correct but far heavier on "
        "memory; if training OOMs, lower model.chunk_size (or install the "
        "kernels: pip install mamba-ssm causal-conv1d)."
    )


class MimirMamba2Module(L.LightningModule):
    """Lightning wrapper around ``Mamba2ForCausalLM`` with projected tables.

    Args:
        donor_model_id: Model whose embedding table, output head, and
            tokenizer are reused. Its weights are never updated.
        d_model: Width of the small model. Must not exceed the donor's width.
        num_hidden_layers: Number of Mamba2 blocks.
        state_size / expand / head_dim / n_groups / conv_kernel / chunk_size:
            Mamba2 mixer geometry. ``expand * d_model`` must be divisible by
            ``head_dim`` — the constructor checks and says so if it is not.
            ``chunk_size`` is the SSM scan's block length; without the fused
            kernels it also sets the size of the scan's
            ``[batch, seq, chunk_size, heads, state]`` float32 temporaries, so
            it is a memory knob as much as an accuracy one.
        initializer_range: Std of the normal init for the Mamba2 blocks.
            Mamba2Config defaults to 0.1, which is wide for a 512-unit
            model trained from scratch; 0.02 is the usual choice at this
            width and is what this module defaults to.
        projection_init: ``pca`` (default), ``orthogonal``, or ``xavier``.
            ``pca`` starts each projection at the donor table's own leading
            principal subspace, which is the best linear compression that
            ``d_model`` dimensions can hold.
        learn_projection_scales: Learn one scalar gain per projection,
            initialised so embeddings start at unit RMS and logits at O(1).
        projection_lr_mult: LR multiplier for ``P``/``Q`` relative to the
            Mamba2 body. The projections sit at both ends of the network and
            see very differently-scaled gradients from the blocks between
            them, so this is worth having as a knob; 1.0 keeps one LR.
        projection_cache_dir: Where to cache PCA bases across runs. ``null``
            disables caching and recomputes them every time.
        freeze_backbone: Train only the projections, leaving the Mamba2 blocks
            at their initialisation. Diagnostic, not a normal training mode.
        gradient_checkpointing: Recompute each Mamba2 block during backward
            instead of holding its activations. Costs ~20% compute and is
            close to mandatory without the fused kernels, whose pure-PyTorch
            fallback keeps several large float32 scan intermediates per layer.
        loss_chunk_tokens: Positions per chunk in the chunked cross-entropy
            (see ``_chunked_loss``). 0 falls back to HF's one-shot loss, which
            needs a ``[batch, seq, 262144]`` float32 logit tensor and its
            gradient resident at once — 2 GiB per 8x256 batch, before the
            softmax's own temporaries.
    """

    def __init__(
        self,
        donor_model_id: str = "danish-foundation-models/DFM-Mimir",
        trust_remote_code: bool = True,
        dtype: str = "bf16",
        d_model: int = 512,
        num_hidden_layers: int = 12,
        state_size: int = 64,
        expand: int = 2,
        head_dim: int = 64,
        n_groups: int = 1,
        conv_kernel: int = 4,
        chunk_size: int = 32,
        initializer_range: float = 0.02,
        projection_init: str = "pca",
        learn_projection_scales: bool = True,
        projection_lr_mult: float = 1.0,
        projection_cache_dir: Optional[str] = "outputs/donor_cache",
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = True,
        loss_chunk_tokens: int = 512,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        lr_scheduler: str = "cosine",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)

        # --------------------------------------------------------------
        # Tokenizer — the donor's, necessarily: token ids index its tables.
        # --------------------------------------------------------------
        self.tokenizer = load_tokenizer(donor_model_id, trust_remote_code)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # --------------------------------------------------------------
        # Frozen donor tables + the learned projections of them.
        # --------------------------------------------------------------
        tables = load_donor_tables(
            donor_model_id,
            trust_remote_code=trust_remote_code,
            dtype=self.torch_dtype,
        )
        embedding, head = build_projections(
            tables,
            d_model=d_model,
            init=projection_init,
            learn_scales=learn_projection_scales,
            cache_dir=projection_cache_dir,
        )

        # --------------------------------------------------------------
        # Mamba2 backbone.
        # --------------------------------------------------------------
        self.model = Mamba2ForCausalLM(
            self._build_config(
                vocab_size=tables.vocab_size,
                d_model=d_model,
                num_hidden_layers=num_hidden_layers,
                state_size=state_size,
                expand=expand,
                head_dim=head_dim,
                n_groups=n_groups,
                conv_kernel=conv_kernel,
                chunk_size=chunk_size,
                initializer_range=initializer_range,
            )
        )
        # Swap the stock nn.Embedding / nn.Linear for the projected views.
        # Done *after* construction so Mamba2's own post_init weight
        # initialisation runs over the blocks (which want it) and not over the
        # projections (which are already initialised from the donor).
        self.model.set_input_embeddings(embedding)
        self.model.lm_head = head
        self.model.train()

        # Note what is NOT done here: the module is not cast to torch_dtype.
        # Only the frozen donor tables carry it (they are where 805M of the
        # values are, and 1.6 GiB vs 3.2 GiB is worth caring about); the 22M
        # trainable parameters stay float32 and let Lightning's precision
        # plugin decide. Under `bf16-mixed` that is the whole point — float32
        # master weights, bf16 matmuls — and casting them here would silently
        # turn it into bf16-true with autocast overhead on top. Under
        # `bf16-true` Lightning casts the module itself, so this is still
        # right. The projections' forward passes cast between the two dtypes.

        # The KV/SSM cache is only needed for generation; forward() never
        # passes use_cache, so leaving it on would allocate a cache every
        # training step for nothing. generate() re-enables it.
        self.model.config.use_cache = False

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if freeze_backbone:
            for name, param in self.model.backbone.named_parameters():
                if not name.startswith("embeddings."):
                    param.requires_grad_(False)
            log.warning(
                "freeze_backbone=True — only the donor projections will train."
            )

        _warn_if_slow_scan_path()
        self._log_parameter_budget(tables.vocab_size, tables.d_donor, d_model)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_config(
        self,
        vocab_size: int,
        d_model: int,
        num_hidden_layers: int,
        state_size: int,
        expand: int,
        head_dim: int,
        n_groups: int,
        conv_kernel: int,
        chunk_size: int,
        initializer_range: float,
    ) -> Mamba2Config:
        """Assemble the Mamba2 config, checking the one non-obvious constraint.

        Mamba2 splits its ``expand * hidden_size`` inner width into
        ``num_heads`` heads of ``head_dim`` each, and derives ``num_heads``
        from that division. A non-integer result fails deep inside the mixer
        with a shape error that says nothing about which knob is wrong, so
        catch it here.
        """
        inner = expand * d_model
        if inner % head_dim != 0:
            raise ValueError(
                f"expand*d_model = {expand}*{d_model} = {inner} is not divisible "
                f"by head_dim={head_dim}; Mamba2 needs an integer head count. "
                f"Pick a head_dim that divides {inner}."
            )
        num_heads = inner // head_dim

        return Mamba2Config(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=num_hidden_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            expand=expand,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
            initializer_range=initializer_range,
            # The projections replace both tables, so there is nothing to tie
            # — and tying would try to share two differently-shaped tensors.
            tie_word_embeddings=False,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=True,
        )

    def _log_parameter_budget(
        self, vocab_size: int, d_donor: int, d_model: int
    ) -> None:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.model.buffers())
        backbone = sum(
            p.numel()
            for n, p in self.model.backbone.named_parameters()
            if p.requires_grad and not n.startswith("embeddings.")
        )
        log.info(
            "Trainable: %.1fM (%.1fM Mamba2 blocks + %.1fM projections) | "
            "frozen donor tables: %.0fM values | "
            "standalone export would be %.0fM params (vs %.0fM for the donor)",
            trainable / 1e6,
            backbone / 1e6,
            (trainable - backbone) / 1e6,
            frozen / 1e6,
            (backbone + 2 * vocab_size * d_model) / 1e6,
            2 * vocab_size * d_donor / 1e6,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def projected_embedding(self) -> ProjectedEmbedding:
        return self.model.get_input_embeddings()

    @property
    def projected_lm_head(self) -> ProjectedLMHead:
        return self.model.lm_head

    # ------------------------------------------------------------------
    # Forward / training / eval steps
    # ------------------------------------------------------------------

    def _chunked_loss(
        self, hidden_states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Causal-LM cross-entropy over the donor-sized vocabulary, in slices.

        The donor's vocabulary is 262144 tokens wide, so a single ``[B, T, V]``
        float32 logit tensor for an 8x256 batch is 2 GiB — and the backward
        pass needs its gradient alongside it. That, not the 21M-parameter body,
        is what decides the batch size this model can train at.

        So the head is applied to ``loss_chunk_tokens`` positions at a time,
        each chunk wrapped in ``torch.utils.checkpoint``: the forward pass
        keeps only the chunk's hidden states, and its logits are recomputed
        (and freed again) one chunk at a time during backward. Peak logit
        memory becomes a function of the chunk size instead of the batch size,
        which is what lets a 16x256 batch fit in 15 GiB.

        Numerically this is the same mean-over-unmasked-tokens cross-entropy
        HF computes, on the same one-position shift: ``logits[t]`` is scored
        against ``labels[t+1]``.
        """
        d_model = hidden_states.shape[-1]
        flat_hidden = hidden_states[:, :-1].reshape(-1, d_model)
        flat_labels = labels[:, 1:].reshape(-1)

        n_valid = int((flat_labels != -100).sum())
        if n_valid == 0:
            # Every label masked — see the guard in forward() for when that
            # happens. Zero, but reached *through* hidden_states so the result
            # still carries a grad_fn: Lightning calls .backward() on whatever
            # training_step returns, and a bare zeros() leaf would raise
            # "element 0 of tensors does not require grad". Multiplying by 0
            # gives every parameter a zero gradient, i.e. the batch is skipped
            # rather than allowed to corrupt the update.
            return (hidden_states.sum() * 0.0).float()

        chunk = self.hparams.loss_chunk_tokens
        total = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        for start in range(0, flat_hidden.shape[0], chunk):
            labels_chunk = flat_labels[start : start + chunk]
            if not bool((labels_chunk != -100).any()):
                continue  # all-padding slice: contributes nothing, costs 0
            hidden_chunk = flat_hidden[start : start + chunk]
            total = total + checkpoint(
                self._chunk_cross_entropy,
                hidden_chunk,
                labels_chunk,
                use_reentrant=False,
            )
        return total / n_valid

    def _chunk_cross_entropy(
        self, hidden_chunk: torch.Tensor, labels_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Summed (not averaged) CE for one slice — the recomputed inner step.

        ``reduction="sum"`` so the caller can divide by the *batch-wide* count
        of unmasked tokens once; per-chunk means would weight a short trailing
        chunk as heavily as a full one.

        The float32 cast is explicit rather than left to autocast so the loss
        is computed identically under ``bf16-true`` (no autocast) and
        ``bf16-mixed``.
        """
        logits = self.model.lm_head(hidden_chunk).float()
        return F.cross_entropy(
            logits, labels_chunk, ignore_index=-100, reduction="sum"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run a forward pass through the causal LM.

        Returns ``(loss, logits)``. With ``labels`` and the chunked loss
        enabled, ``logits`` is None — materializing them is exactly the cost
        the chunking exists to avoid, and the training/validation/test steps
        only ever read the loss. Callers that need logits (``predict.py``)
        invoke ``self.model`` directly, under ``no_grad`` and at eval batch
        sizes.
        """
        if labels is not None and self.hparams.loss_chunk_tokens:
            hidden_states = self.model.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )[0]
            loss, logits = self._chunked_loss(hidden_states, labels), None
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss, logits = outputs.loss, outputs.logits

        if loss is not None and not torch.isfinite(loss):
            # Degenerate batch: every example's labels got fully masked to
            # -100 (e.g. the prompt alone already filled max_length before
            # truncation left room for any output tokens). HF's
            # CrossEntropyLoss(ignore_index=-100, reduction="mean") then
            # divides by zero valid tokens -> nan, silently (no exception).
            # (_chunked_loss short-circuits that case itself, so this guard
            # only ever fires on the HF path — or on a genuine nan from
            # somewhere else, which it is also the right response to.)
            # Left alone, that one nan poisons the *whole* epoch's averaged
            # train/eval loss (mean of anything containing nan is nan) and
            # every downstream consumer of it (checkpoint selection, Optuna
            # pruning/reporting, W&B) — so surface a finite 0 instead. A
            # fresh zero leaf (not wired into the autograd graph) also means
            # this batch contributes no gradient, i.e. it's skipped for
            # training rather than corrupting the model with a nan update.
            log.warning(
                "Non-finite loss (%s) from a batch with no valid (non -100) "
                "label tokens — likely max_length too short for these "
                "examples' prompts. Reporting loss=0 for this batch instead "
                "of letting it poison the epoch average.",
                loss.item(),
            )
            loss = torch.zeros(
                (), dtype=loss.dtype, device=loss.device, requires_grad=loss.requires_grad
            )

        return loss, logits

    def _log(self, name: str, value, **kwargs: Any) -> None:
        # Route to the configured logger only when one is actually attached;
        # otherwise stay silent (avoids Lightning's "no logger configured"
        # warning). This makes W&B logging work without spamming when disabled.
        use_logger = bool(
            getattr(self, "trainer", None) and getattr(self.trainer, "logger", None)
        )
        self.log(name, value, **{**kwargs, "logger": use_logger})

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _ = self.forward(
            batch["input_ids"], batch.get("attention_mask"), batch["labels"]
        )
        self._log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _ = self.forward(
            batch["input_ids"], batch.get("attention_mask"), batch["labels"]
        )
        self._log("eval_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _ = self.forward(
            batch["input_ids"], batch.get("attention_mask"), batch["labels"]
        )
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        """Four parameter groups: {projection, body} x {decay, no-decay}.

        The frozen donor tables are buffers, not parameters, so they cannot
        reach the optimizer by construction — no filtering needed to keep them
        out. What *does* reach it is ``P``, ``Q``, the two scalar gains, and
        the Mamba2 blocks.
        """
        # Matched against the *leaf* name so Mamba2's single-letter SSM
        # parameters (A_log, D) and its dt_bias are excluded from weight decay
        # without a substring rule accidentally catching unrelated names.
        no_decay_leaves = {"bias", "a_log", "d", "dt_bias", "conv_bias", "scale"}
        base_lr = self.hparams.learning_rate
        proj_lr = base_lr * self.hparams.projection_lr_mult

        groups: Dict[str, List[torch.Tensor]] = {
            "proj_decay": [], "proj_no_decay": [], "body_decay": [], "body_no_decay": []
        }
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # The two donor projections and their scalar gains live under
            # these prefixes; everything else is a Mamba2 block.
            prefix = (
                "proj"
                if name.startswith(("backbone.embeddings.", "lm_head."))
                else "body"
            )
            leaf = name.rsplit(".", 1)[-1].lower()
            no_decay = leaf in no_decay_leaves or "norm" in name.lower()
            groups[f"{prefix}_{'no_decay' if no_decay else 'decay'}"].append(param)

        weight_decay = self.hparams.weight_decay
        optimizer = torch.optim.AdamW(
            [
                {"params": groups["proj_decay"], "weight_decay": weight_decay, "lr": proj_lr},
                {"params": groups["proj_no_decay"], "weight_decay": 0.0, "lr": proj_lr},
                {"params": groups["body_decay"], "weight_decay": weight_decay, "lr": base_lr},
                {"params": groups["body_no_decay"], "weight_decay": 0.0, "lr": base_lr},
            ],
            lr=base_lr,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        # NOTE: total_steps is used as num_training_steps for BOTH the warmup
        # ramp and the decay phase (rather than only decaying over
        # total_steps - warmup_steps), so the LR actually ramps from ~0 up to
        # `learning_rate` over the first `warmup_steps` steps before decaying
        # — get_*_schedule_with_warmup handles both phases internally.
        schedule_fns = {
            "cosine": get_cosine_schedule_with_warmup,
            "linear": get_linear_schedule_with_warmup,
        }
        schedule_fn = schedule_fns.get(self.hparams.lr_scheduler)
        scheduler = (
            schedule_fn(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            if schedule_fn is not None
            else None
        )

        result: Dict[str, Any] = {"optimizer": optimizer}
        if scheduler is not None:
            result["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        return result

    # ------------------------------------------------------------------
    # Prediction / generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def argmax_token_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced greedy prediction: ``[batch, seq]`` of argmax ids.

        This is what held-out scoring needs, and it is the one place the
        262144-wide logits would otherwise be materialized in full — 6.4 GiB
        for a 16x256 batch across the bf16 head output and HF's float32 cast
        of it, which OOMs where training (with its chunked loss) does not.

        Only the argmax survives each position, so the head is applied in
        ``loss_chunk_tokens``-sized slices and reduced immediately: peak
        memory becomes chunk-sized, and the result is bit-identical to
        argmaxing the whole thing at once.
        """
        self.eval()
        hidden_states = self.model.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )[0]

        chunk = self.hparams.loss_chunk_tokens
        if not chunk:
            return self.model.lm_head(hidden_states).argmax(dim=-1)

        batch, seq, d_model = hidden_states.shape
        flat = hidden_states.reshape(-1, d_model)
        predicted = torch.empty(
            flat.shape[0], dtype=torch.long, device=hidden_states.device
        )
        for start in range(0, flat.shape[0], chunk):
            stop = start + chunk
            predicted[start:stop] = self.model.lm_head(flat[start:stop]).argmax(dim=-1)
        return predicted.view(batch, seq)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a completion for a single prompt string."""
        self.eval()
        device = next(self.model.parameters()).device

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Training leaves use_cache off; generation is the one place the SSM
        # recurrent state cache actually pays for itself.
        was_cached = self.model.config.use_cache
        self.model.config.use_cache = True
        try:
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
        finally:
            self.model.config.use_cache = was_cached

        generated = gen_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Export / load
    # ------------------------------------------------------------------

    @torch.no_grad()
    def export_standalone(self, dtype: Optional[torch.dtype] = None) -> Mamba2ForCausalLM:
        """Collapse the projections into a plain, self-contained Mamba2 model.

        This is the step the whole design builds towards: apply the *current*
        ``P`` and ``Q`` to the frozen donor tables and write out the resulting
        concrete ``[vocab_size, d_model]`` embedding and head. The returned
        model is a stock ``Mamba2ForCausalLM`` — no donor, no projections, no
        custom modules — that produces the same logits as this one, and is
        ~3x smaller than the donor it was distilled from.
        """
        dtype = dtype or self.torch_dtype
        exported = Mamba2ForCausalLM(copy.deepcopy(self.model.config))

        embed_weight = self.projected_embedding.materialize_weight(torch.float32)
        head_weight = self.projected_lm_head.materialize_weight(torch.float32)

        backbone_state = {
            k: v
            for k, v in self.model.backbone.state_dict().items()
            if not k.startswith("embeddings.")
        }
        exported.backbone.load_state_dict(backbone_state, strict=False)
        exported.backbone.embeddings.weight.copy_(embed_weight)
        exported.lm_head.weight.copy_(head_weight)
        return exported.to(dtype).eval()

    def save_pretrained(self, save_path: str) -> None:
        """Save the standalone small model + tokenizer + the projections.

        Two artifacts, deliberately:

        * ``save_path/`` is a stock HuggingFace ``Mamba2ForCausalLM`` with the
          projections already collapsed into concrete tables — loadable by
          anyone with plain ``transformers``, no donor required.
        * ``donor_projections.pt`` keeps ``P``/``Q`` (plus the donor id) so
          training can be resumed, or the tables re-derived, later.
        """
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        self.export_standalone().save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

        torch.save(
            {
                "embedding_proj": self.projected_embedding.state_dict(),
                "lm_head_proj": self.projected_lm_head.state_dict(),
            },
            path / _PROJECTIONS_FILE,
        )
        with open(path / _PROJECTIONS_META, "w") as f:
            json.dump(
                {
                    "donor_model_id": self.hparams.donor_model_id,
                    "d_model": self.hparams.d_model,
                    "projection_init": self.hparams.projection_init,
                    "learn_projection_scales": self.hparams.learn_projection_scales,
                },
                f,
                indent=2,
            )
        log.info("Saved standalone Mamba2 model + projections to %s", path)

    @classmethod
    def from_pretrained(
        cls,
        save_path: str,
        donor_model_id: Optional[str] = None,
        trust_remote_code: bool = True,
        dtype: str = "bf16",
        **kwargs: Any,
    ) -> "MimirMamba2Module":
        """Rebuild a trained module from a ``save_pretrained`` directory.

        The Mamba2 blocks come from the saved standalone model; the
        projections come from ``donor_projections.pt`` and are re-attached to
        freshly-loaded donor tables. Loading this way (rather than using the
        standalone model directly) keeps the module differentiable end-to-end
        through ``P``/``Q``, so a checkpoint can be trained further.
        """
        path = Path(save_path)
        meta_path = path / _PROJECTIONS_META
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        donor_model_id = donor_model_id or meta.get("donor_model_id")
        if donor_model_id is None:
            raise ValueError(
                f"{path} has no {_PROJECTIONS_META} recording its donor, and no "
                "donor_model_id was passed — cannot rebuild the frozen tables."
            )

        saved = Mamba2ForCausalLM.from_pretrained(str(path), dtype=DTYPE_MAP.get(dtype))
        config = saved.config

        instance = cls(
            donor_model_id=donor_model_id,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            d_model=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            state_size=config.state_size,
            expand=config.expand,
            head_dim=config.head_dim,
            n_groups=config.n_groups,
            conv_kernel=config.conv_kernel,
            chunk_size=config.chunk_size,
            initializer_range=config.initializer_range,
            projection_init=meta.get("projection_init", "pca"),
            learn_projection_scales=meta.get("learn_projection_scales", True),
            **kwargs,
        )

        backbone_state = {
            k: v
            for k, v in saved.backbone.state_dict().items()
            if not k.startswith("embeddings.")
        }
        instance.model.backbone.load_state_dict(backbone_state, strict=False)

        projections_path = path / _PROJECTIONS_FILE
        if projections_path.exists():
            blobs = torch.load(projections_path, map_location="cpu")
            # strict=False: the donor buffer is non-persistent and therefore
            # absent from the saved state — it was just rebuilt from the donor.
            instance.projected_embedding.load_state_dict(
                blobs["embedding_proj"], strict=False
            )
            instance.projected_lm_head.load_state_dict(
                blobs["lm_head_proj"], strict=False
            )
        else:
            log.warning(
                "%s not found in %s — the Mamba2 blocks were restored but the "
                "projections fall back to their %s initialisation.",
                _PROJECTIONS_FILE,
                path,
                meta.get("projection_init", "pca"),
            )

        instance.model = instance.model.to(instance.torch_dtype)
        instance.model.eval()
        # Own the device placement, as predict.py's loader expects: unlike the
        # DFM-Mimir path this does not go through accelerate's device_map, so
        # nothing else will move it.
        if torch.cuda.is_available():
            instance.model = instance.model.cuda()
        del saved
        return instance
