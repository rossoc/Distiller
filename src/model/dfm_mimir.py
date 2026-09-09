# -*- coding: utf-8 -*-
"""DFM-Mimir Lightning module for autoregressive full fine-tuning.

Wraps a HuggingFace causal LM (DFM-Mimir by default) in a LightningModule.
Tokenization lives in ``lit_datamodule.py`` — this module only owns the model,
optimizer, scheduler, and generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# HF compatibility shims (Hub-noise loggers + the transformers>=5.3
# TokenizersBackend kwarg regression), the shared dtype map, and the
# pad-token-aware tokenizer loader live in one place so this module and
# model.mimir_mamba2 cannot drift apart on them.
from model.hf_compat import DTYPE_MAP, load_tokenizer

log = logging.getLogger(__name__)


class DFMMimirModule(L.LightningModule):
    """Lightning wrapper around a HF causal LM (DFM-Mimir by default)."""

    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = True,
        dtype: str = "bf16",
        learning_rate: float = 2.0e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        lr_scheduler: str = "cosine",
        hrm_cycles: Optional[Dict[str, int]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)

        self.tokenizer = load_tokenizer(model_id, trust_remote_code)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            dtype=self.torch_dtype,
            # NOTE: no device_map="auto". With Lightning we let the Trainer own
            # device placement (model.to(cuda)) on real tensors. device_map="auto"
            # can shard the model onto `meta` tensors when VRAM is tight across
            # repeated loads in one process (e.g. Optuna trials), which then
            # crashes model.to() with "Cannot copy out of meta tensor; no data!".
        )
        self.model.train()

        # Disable the causal-LM KV cache during training: it is only needed for
        # generation (self.model.forward is called without use_cache, so it
        # would otherwise allocate the cache every step and waste memory/compute).
        # The HRM recurrent hidden-state machinery is independent of this.
        try:
            self.model.config.use_cache = False
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Optional HRM recurrent-cycle reduction (memory / speed).
        #
        # DFM-Mimir (model_type `hrm_text`) unrolls H_cycles x L_cycles inner
        # recurrent steps per forward pass, each holding full-sequence
        # activations through the backward pass. With the default
        # H_cycles=2 / L_cycles=3 the 16 layers are partitioned into 8
        # stack-passes, which dominates GPU memory on a ~15 GiB card.
        #
        # Passing `hrm_cycles: {H_cycles: 1, L_cycles: 1}` cuts the unroll to
        # 2 stack-passes (~4x less activation memory). We override the loaded
        # config *after* from_pretrained (so weights load correctly under the
        # original partition) and refresh the model's cached
        # `L_bp_cycles_padded`, which is built once at __init__ from config.
        # ------------------------------------------------------------------
        if hrm_cycles:
            _h = hrm_cycles.get("H_cycles", self.model.config.H_cycles)
            _l = hrm_cycles.get("L_cycles", self.model.config.L_cycles)
            self.model.config.H_cycles = int(_h if _h is not None else self.model.config.H_cycles)
            self.model.config.L_cycles = int(_l if _l is not None else self.model.config.L_cycles)
            raw_bp = list(getattr(self.model.config, "L_bp_cycles", []) or [])
            self.model.L_bp_cycles_padded = [1] * max(
                0, self.model.config.H_cycles - len(raw_bp)
            ) + raw_bp

        # Gradient checkpointing trades ~20% compute for large activation
        # memory savings — important given the recurrent unroll above.
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # ------------------------------------------------------------------
    # Forward / training / eval steps
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Run a forward pass through the causal LM.

        Returns (loss, logits). If labels is None, returns (None, logits).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss
        logits = outputs.logits

        if loss is not None and not torch.isfinite(loss):
            # Degenerate batch: every example's labels got fully masked to
            # -100 (e.g. the prompt alone already filled max_length before
            # truncation left room for any output tokens). HF's
            # CrossEntropyLoss(ignore_index=-100, reduction="mean") then
            # divides by zero valid tokens -> nan, silently (no exception).
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
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"]

        loss, _ = self.forward(input_ids, attention_mask, labels)
        self._log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"]

        loss, _ = self.forward(input_ids, attention_mask, labels)
        self._log(
            "eval_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"]

        loss, _ = self.forward(input_ids, attention_mask, labels)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False
        )
        return loss

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        no_decay = ["bias", "layer_norm", "ln", "norm"]
        param_dict = dict(self.model.named_parameters())

        decay_params: List[torch.Tensor] = []
        no_decay_params: List[torch.Tensor] = []
        for name, param in param_dict.items():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
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

        The shared contract ``predict.py`` scores against — see
        ``MimirMamba2Module.argmax_token_ids``, which computes the same thing
        without materializing the full vocabulary of logits.
        """
        self.eval()
        _, logits = self.forward(input_ids, attention_mask)
        return logits.argmax(dim=-1)

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
        device = self.model.device

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

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
        generated = gen_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def save_pretrained(self, save_path: str) -> None:
        """Save model + tokenizer to *save_path*."""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        save_path: str,
        trust_remote_code: bool = True,
        dtype: str = "bf16",
        **kwargs: Any,
    ) -> "DFMMimirModule":
        """Load a trained DFMMimirModule from disk."""
        torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        tokenizer = load_tokenizer(save_path, trust_remote_code)

        model = AutoModelForCausalLM.from_pretrained(
            save_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            device_map="auto",
        )
        model.eval()

        instance = cls(
            model_id=model_id,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            **kwargs,
        )
        instance.model = model
        instance.tokenizer = tokenizer
        instance.pad_token_id = tokenizer.pad_token_id
        instance.eos_token_id = tokenizer.eos_token_id
        return instance
