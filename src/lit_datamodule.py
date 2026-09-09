# -*- coding: utf-8 -*-
"""K-fold LightningDataModule for autoregressive DFM-Mimir fine-tuning.

Owns:
- Data loading via ``data.loader`` utilities
- Train/test row split (held-out, never touched in CV)
- K-fold index computation (deterministic, seeded)
- Tokenization (moved here from DFMMimirModule)
- train_dataloader / val_dataloader / test_dataloader

Usage mirrors NEM's ``lit_datamodule.py``: ``train.py::run_fold`` builds one
instance per fold. Set ``fold=-1`` to build a datamodule that covers ALL
training folds at once plus the full train/test sample sets without
requiring a fold index — used by ``cv.py`` for CV summary stats and by
``predict.py`` for the held-out test set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import polars as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from data.loader import (
    build_samples,
    build_samples_by_row,
    kfold_indices,
    read_ground_truth,
    split_rows_by_max_length,
    train_test_row_split,
)

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenization helper (NEM-style: datamodule owns tokenization)
# ---------------------------------------------------------------------------

def _tokenize_pairs(
    tokenizer,
    pad_token_id: int,
    inputs: List[str],
    outputs: List[str],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize (input, output) pairs for causal LM training.

    The single copy of this logic — imported by both ``train.py`` and
    ``predict.py`` so training and evaluation tokenize identically.
    """
    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    attention_list: List[torch.Tensor] = []

    for inp, out in zip(inputs, outputs):
        full_text = f"{inp}{tokenizer.eos_token}{out}{tokenizer.eos_token}"
        enc = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        ids = enc["input_ids"]

        inp_enc = tokenizer(
            inp,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        inp_len = len(inp_enc["input_ids"])

        labels = ids[:]
        for i in range(min(inp_len + 1, len(labels))):
            labels[i] = -100

        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        # Built from the real (pre-pad) length, not from comparing token ids
        # against pad_token_id — when the tokenizer has no dedicated pad
        # token, pad_token_id falls back to eos_token_id, and eos_token
        # appears inside every real sequence (as the input/output separator
        # and terminator). Masking by id would zero out those real positions
        # too, not just the padded tail.
        attention_list.append(torch.ones(len(ids), dtype=torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_list, batch_first=True, padding_value=0
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Tokenized dataset (Lightning-style, replaces the copy in train.py / predict.py)
# ---------------------------------------------------------------------------

class _TokenizedDataset(Dataset):
    """Wraps tokenized inputs/labels for the Lightning DataLoader."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------

class DistillerDataModule(L.LightningDataModule):
    """K-fold cross-validation LightningDataModule for DFM-Mimir.

    Two-stage split (like NEM's ``lit_datamodule.py``):
      1. ``test_frac`` of ROWS carved off as held-out test (seeded by
         ``split_seed``) — never touched during CV.
      2. Remaining rows split into ``n_folds`` folds (seeded by ``fold_seed``);
         the ``fold``-th chunk is validation, the rest trains.

    When ``fold=-1``, the datamodule prepares ALL training folds at once and
    exposes them via ``folds`` — used by the single-CV path.

    Args:
        cfg: Full Hydra DictConfig (data/model/training/optuna sections used).
        fold: Which fold (0..n_folds-1) is validation.  ``-1`` = all folds.
        runtime: Optional DataLoader runtime dict (workers, pinning).
    """

    def __init__(
        self,
        cfg: DictConfig,
        fold: int = 0,
        runtime: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self.runtime = runtime or {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        }

        # Data specs
        self.xlsx_path = cfg.data.xlsx_path
        self.sheet_name = cfg.data.sheet_name
        self.source_columns = cfg.data.source_columns
        self.target_columns = cfg.data.target_columns
        self.unknown_token = cfg.data.unknown_token
        self.test_frac = cfg.data.test_frac
        self.split_seed = cfg.data.split_seed
        self.max_length = cfg.data.max_length
        self.fold_seed = cfg.data.fold_seed
        # False (default) = "S_text L_text <field>?" (causal-attention models
        # like DFM-Mimir, which can attend back over the whole prompt at
        # generation time). True = "<field>? S_text L_text" (recurrent
        # models like Mamba2, which need the field name folded into their
        # running state *before* scanning the source text). See
        # data.loader.build_samples_by_row and config/data/rnn.yaml.
        self.prompt_first = cfg.data.get("prompt_first", False)
        self.n_folds = cfg.optuna.n_folds

        # Model ref, for tokenizer access. Any of the model modules will
        # do: this only ever reads `.tokenizer` and `.pad_token_id`, which
        # every one of them exposes.
        self.module: Optional[L.LightningModule] = None

        # State
        self._train_df: Optional[pl.DataFrame] = None
        self._test_df: Optional[pl.DataFrame] = None
        self._folds: Optional[List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]] = None
        self._train_samples: Optional[List[Dict[str, str]]] = None
        self._test_samples: Optional[List[Dict[str, str]]] = None

        # Per-fold datasets (populated by setup)
        self.train_dataset: Optional[_TokenizedDataset] = None
        self.val_dataset: Optional[_TokenizedDataset] = None
        self.test_dataset: Optional[_TokenizedDataset] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load XLSX and return (train_rows, test_rows) split."""
        if self._train_df is not None:
            return self._train_df, self._test_df
        df = read_ground_truth(self.xlsx_path, self.sheet_name)
        train_df, test_df = train_test_row_split(
            df, self.test_frac, self.split_seed
        )
        self._train_df = train_df.clone()
        self._test_df = test_df.clone()
        return train_df, test_df

    def _build_samples(
        self, df: pl.DataFrame
    ) -> List[Dict[str, str]]:
        """Build training samples from a DataFrame."""
        return build_samples(
            df, self.source_columns, self.target_columns, self.unknown_token,
            self.prompt_first,
        )

    def _long_row_idx(self, train_df: pl.DataFrame) -> Optional[np.ndarray]:
        """Row indices (within *train_df*) too long for ``max_length``.

        These are passed as ``forced_val_idx`` to ``kfold_indices`` so they
        always land in validation (across every fold) and are never trained
        on: a row whose samples exceed ``max_length`` gets truncated down to
        a fully-masked (all -100) label, which trains on nothing useful and
        destabilizes the loss (see model.dfm_mimir.forward's non-finite-loss
        guard). Keeping them in validation instead also gives visibility into
        how the model behaves on this OOD-length data it never trained on.

        Requires a tokenizer, so returns None (no forced split — plain
        K-fold rotation) when no module is attached yet, e.g. the
        stats-only ``DistillerDataModule(cfg, fold=-1)`` in cv.py, which
        doesn't need or use this datamodule's fold splits at all.
        """
        if self.module is None:
            return None
        rows_samples = build_samples_by_row(
            train_df, self.source_columns, self.target_columns, self.unknown_token,
            self.prompt_first,
        )
        short_idx, long_idx = split_rows_by_max_length(
            rows_samples, self.module.tokenizer, self.max_length
        )
        if len(long_idx):
            log.info(
                "%d/%d training rows exceed max_length=%d and are forced into "
                "validation only (never trained on) in every fold.",
                len(long_idx),
                len(short_idx) + len(long_idx),
                self.max_length,
            )
        return long_idx

    def _tokenize(
        self, samples: List[Dict[str, str]]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize samples using the module's tokenizer."""
        if self.module is None:
            raise RuntimeError("module not set — call set_module() first")
        texts = [s["input"] for s in samples]
        outputs = [s["output"] for s in samples]
        return _tokenize_pairs(
            self.module.tokenizer,
            self.module.pad_token_id,
            texts,
            outputs,
            self.max_length,
        )

    def _make_dataset(
        self, samples: List[Dict[str, str]]
    ) -> _TokenizedDataset:
        """Tokenize samples and wrap in a _TokenizedDataset."""
        tok = self._tokenize(samples)
        return _TokenizedDataset(
            tok["input_ids"], tok["labels"], tok["attention_mask"]
        )

    def _build_dataloader(
        self, dataset: Dataset, shuffle: bool = False
    ) -> DataLoader:
        kwargs: Dict[str, Any] = {
            "batch_size": self.cfg.training.batch_size,
            "shuffle": shuffle,
            "num_workers": int(self.runtime["num_workers"]),
            "pin_memory": bool(self.runtime["pin_memory"]),
            # Only the shuffled (train) loader drops a ragged last batch —
            # keeps the effective batch size (and grad-accumulation math)
            # consistent across steps; eval loaders keep every sample.
            "drop_last": shuffle,
        }
        # Use eval batch size (eval_batch_multiplier * train) for non-shuffled
        # (val/test) loaders. Defaults to 1 (same as train) — val/test batches
        # keep every sample (no drop_last) and, for val, include every
        # forced-long (max_length-sized) row, so a >1 multiplier here is much
        # more OOM-prone than the same multiplier would be on train.
        if not shuffle:
            eval_batch_multiplier = self.cfg.training.get("eval_batch_multiplier", 1)
            kwargs["batch_size"] = eval_batch_multiplier * self.cfg.training.batch_size
        if kwargs["num_workers"] > 0:
            kwargs["persistent_workers"] = bool(
                self.runtime["persistent_workers"]
            )
            if self.runtime["prefetch_factor"] is not None:
                kwargs["prefetch_factor"] = int(
                    self.runtime["prefetch_factor"]
                )
        return DataLoader(dataset, **kwargs)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Build samples (and, once a module is attached, tokenized datasets)
        for this fold.

        Tokenization requires a tokenizer, so ``_TokenizedDataset``s are only
        built when ``set_module()`` was called first; sample lists
        (``get_folds()``/``get_train_samples()``/``get_test_samples()``) are
        always available after ``setup()``, model or no model — this lets
        callers use the datamodule purely for sample/stat access (e.g. CV
        summary counts) without loading a model at all.
        """
        if self._folds is not None:
            return  # already set up

        train_df, test_df = self._load_data()

        fold_indices = kfold_indices(
            len(train_df),
            self.n_folds,
            self.fold_seed,
            forced_val_idx=self._long_row_idx(train_df),
        )

        if self.fold == -1:
            # All-fold mode: build every fold's samples
            self._folds = []
            for _, (train_idx, val_idx) in enumerate(fold_indices):
                ft = self._build_samples(train_df.gather(train_idx).clone())
                fv = self._build_samples(train_df.gather(val_idx).clone())
                self._folds.append((ft, fv))
        else:
            # Single-fold mode
            train_idx, val_idx = fold_indices[self.fold]
            self._folds = [
                (
                    self._build_samples(train_df.gather(train_idx).clone()),
                    self._build_samples(train_df.gather(val_idx).clone()),
                )
            ]

        # Full-train and held-out test samples (used for summary stats and
        # by the test dataloader), independent of which fold is selected.
        self._train_samples = self._build_samples(train_df)
        self._test_samples = self._build_samples(test_df)

        # Tokenized datasets need a tokenizer — only build them once a
        # module is attached (set_module() called before setup()).
        if self.module is not None:
            self.test_dataset = self._make_dataset(self._test_samples)
            if self.fold != -1:
                train_samples, val_samples = self._folds[0]
                self.train_dataset = self._make_dataset(train_samples)
                self.val_dataset = self._make_dataset(val_samples)

    # ------------------------------------------------------------------
    # Module wiring
    # ------------------------------------------------------------------

    def set_module(self, module: L.LightningModule) -> None:
        """Attach the model module so we can tokenize and generate."""
        self.module = module

    # ------------------------------------------------------------------
    # DataLoader accessors
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("setup() not called")
        if self.fold == -1:
            raise RuntimeError(
                "fold=-1 mode: use folds directly, not train_dataloader()"
            )
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("setup() not called")
        if self.fold == -1:
            raise RuntimeError(
                "fold=-1 mode: use folds directly, not val_dataloader()"
            )
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("setup() not called")
        return self._build_dataloader(self.test_dataset, shuffle=False)

    # ------------------------------------------------------------------
    # Fold accessors (for all-fold mode and Optuna)
    # ------------------------------------------------------------------

    def get_folds(self) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
        """Return list of (train_samples, val_samples) for every fold.

        Only valid after setup() and when fold=-1, or when called before
        single-fold setup (the list always has at least one entry).
        """
        if self._folds is None:
            raise RuntimeError("setup() not called")
        return self._folds

    def get_train_samples(self) -> List[Dict[str, str]]:
        """Full training-set samples (all training rows)."""
        if self._train_samples is None:
            raise RuntimeError("setup() not called")
        return self._train_samples

    def get_test_samples(self) -> List[Dict[str, str]]:
        """Held-out test-set samples."""
        if self._test_samples is None:
            raise RuntimeError("setup() not called")
        return self._test_samples

    def get_fold(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Return the (train, val) samples for the current fold.

        For fold=-1 this returns the first fold (callers should use
        ``get_folds()`` and iterate).
        """
        folds = self.get_folds()
        if self.fold == -1:
            return folds[0]
        return folds[self.fold]
