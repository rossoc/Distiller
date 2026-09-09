"""
Data loading and K-fold CV split for autoregressive DFM-Mimir fine-tuning.

Design:
- One training sample per (row, target_column).
- Input  = S_TEXT + L_TEXT + <field_name>          (rich prompt)
- Output = extracted field value, or "unknown" if empty/missing.
- Train/test split on ROWS first (held-out test set never touched in CV).
- Training rows split deterministically into K folds.
"""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

# fastexcel sometimes can't infer a dtype for (mostly-empty / mixed) text
# columns and falls back to string — exactly what we want for span extraction.
# Silence the per-column noise.
logging.getLogger("fastexcel.types.dtype").setLevel(logging.ERROR)
# Benign polars 1.x notice about from_arrow's future return type.
warnings.filterwarnings(
    "ignore",
    message="from_arrow.*will return a Series instead of a DataFrame",
    category=FutureWarning,
)

# ---------------------------------------------------------------------------
# Sample creation
# ---------------------------------------------------------------------------

def build_samples_by_row(
    df: pl.DataFrame,
    source_columns: List[str],
    target_columns: List[str],
    unknown_token: str = "unknown",
    prompt_first: bool = False,
) -> List[List[Dict[str, str]]]:
    """Like ``build_samples``, but grouped one list-of-samples per DataFrame
    row (in row order) instead of a single flat list.

    Used where same-row samples need to be kept together — e.g. deciding
    whether a *row* is too long for ``max_length`` (see
    ``split_rows_by_max_length``) so all of its target-column samples move as
    one unit, consistent with ``kfold_indices`` splitting at the row level to
    avoid the same source text appearing in both a fold's train and val split.

    Args:
        prompt_first: if False (default), input = "{source_text} <{col}>?" —
            the field marker comes last, right before the model starts
            generating, which is the natural framing for a causal-attention
            model (DFM-Mimir) that can attend back over the whole prompt at
            generation time. If True, input = "<{col}>? {source_text}" — the
            field marker comes first, so a recurrent model (Mamba2) knows
            what to extract *before* it scans the source text, letting it
            build that into its running state as it reads instead of having
            to re-derive it from a state that already collapsed the text.
    """
    rows_samples: List[List[Dict[str, str]]] = []
    for row in df.iter_rows(named=True):
        src_parts = [
            ""
            if (v := row[c]) is None or (isinstance(v, float) and np.isnan(v))
            else str(v).strip()
            for c in source_columns
            if c in df.columns
        ]
        source_text = " ".join(p for p in src_parts if p)

        row_samples: List[Dict[str, str]] = []
        if source_text:
            for col in target_columns:
                if col not in df.columns:
                    continue
                val = row[col]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    output = unknown_token
                else:
                    s = str(val).strip()
                    output = s if s else unknown_token

                field_marker = f"<{col}>?"
                row_samples.append({
                    "input": (
                        f"{field_marker} {source_text}"
                        if prompt_first
                        else f"{source_text} {field_marker}"
                    ),
                    "output": output,
                })
        rows_samples.append(row_samples)
    return rows_samples


def build_samples(
    df: pl.DataFrame,
    source_columns: List[str],
    target_columns: List[str],
    unknown_token: str = "unknown",
    prompt_first: bool = False,
) -> List[Dict[str, str]]:
    """Create training samples from a Polars DataFrame.

    Each row × target_column → one sample:
        input  = "S_text L_text <field_name>"   (or, if prompt_first,
                  "<field_name> S_text L_text" — see build_samples_by_row)
        output = "<extracted_value>"  (or unknown_token if empty)

    Args:
        df: Polars DataFrame (from read_ground_truth).
        source_columns: e.g. ["S_text", "L_text"].
        target_columns: list of field names to extract.
        unknown_token: string to use for missing values.
        prompt_first: see build_samples_by_row.

    Returns:
        List of {"input": str, "output": str} dicts.
    """
    rows_samples = build_samples_by_row(
        df, source_columns, target_columns, unknown_token, prompt_first
    )
    return [s for row_samples in rows_samples for s in row_samples]


def split_rows_by_max_length(
    rows_samples: List[List[Dict[str, str]]],
    tokenizer: Any,
    max_length: int,
    eos_token: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partition row indices into (short_idx, long_idx) by tokenized length.

    A row is "long" if ANY of its target-column samples, framed exactly as
    training does (``"{input}{eos}{output}{eos}"``, see
    ``lit_datamodule._tokenize_pairs``), tokenizes to more than *max_length*
    tokens — i.e. it would be truncated. Rows are kept together (rather than
    splitting per-sample) since all of a row's samples share the same source
    text and ``kfold_indices`` splits at the row level.

    Row order is preserved within each returned array (both ascending), so
    callers get a stable, deterministic partition.
    """
    eos_token = eos_token if eos_token is not None else tokenizer.eos_token
    short_idx: List[int] = []
    long_idx: List[int] = []
    for i, samples in enumerate(rows_samples):
        is_long = False
        for s in samples:
            full_text = f"{s['input']}{eos_token}{s['output']}{eos_token}"
            n_tokens = len(tokenizer(full_text, truncation=False)["input_ids"])
            if n_tokens > max_length:
                is_long = True
                break
        (long_idx if is_long else short_idx).append(i)
    return np.array(short_idx, dtype=np.int64), np.array(long_idx, dtype=np.int64)


_FIELD_MARKER_RE = re.compile(r"<([^<>]+)>\?")


def count_by_target(
    samples: List[Dict[str, str]],
) -> Dict[str, int]:
    """Count samples per target field (the field name is inside <...>? in input).

    The marker can sit at either end of ``input`` — the tail, as
    ``build_samples`` normally writes it, or the head, when built with
    ``prompt_first=True`` — so it's located by regex rather than assumed to
    be the last whitespace-delimited token.
    """
    counts: Dict[str, int] = defaultdict(int)
    for s in samples:
        m = _FIELD_MARKER_RE.search(s["input"])
        field = m.group(1) if m else s["input"].rsplit(" ", 1)[-1].strip("<>?")
        counts[field] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextPairDataset(Dataset):
    """Holds (input, output) string pairs for autoregressive training."""

    def __init__(self, samples: List[Dict[str, str]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Deterministic train/test row split
# ---------------------------------------------------------------------------

def train_test_row_split(
    df: pl.DataFrame,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministically split ROWS into (train, held_out_test).

    The split is based on a stable hash of the row index + seed, so it is
    identical across runs.
    """
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return df.gather(train_idx).clone(), df.gather(test_idx).clone()


# ---------------------------------------------------------------------------
# Deterministic K-fold indices (on row indices within the training set)
# ---------------------------------------------------------------------------

def kfold_indices(
    n_rows: int,
    n_folds: int = 5,
    seed: int = 42,
    forced_val_idx: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return K (train_idx, val_idx) folds of *row indices within the training set*.

    Deterministic: same (n_rows, n_folds, seed) → same folds.
    Uses a stable shuffle via numpy PCG64.

    *forced_val_idx*, if given, are row indices excluded from the K-fold
    rotation entirely and instead appended to EVERY fold's ``val_idx`` (never
    ``train_idx``) — e.g. rows too long for ``max_length`` (see
    ``split_rows_by_max_length``), which we always want to validate/inspect
    on but never train on. The rotation below only distributes the remaining
    rows, so fold sizes are computed over ``n_rows - len(forced_val_idx)``.
    """
    forced_val_idx = (
        np.asarray(forced_val_idx, dtype=np.int64)
        if forced_val_idx is not None
        else np.array([], dtype=np.int64)
    )
    forced_set = set(forced_val_idx.tolist())
    indices = np.array([i for i in range(n_rows) if i not in forced_set], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    n_rotate = len(indices)
    fold_size = n_rotate // n_folds
    remainder = n_rotate % n_folds
    start = 0
    for f in range(n_folds):
        end = start + fold_size + (1 if f < remainder else 0)
        val_idx = np.sort(np.concatenate([indices[start:end], forced_val_idx]))
        train_idx = np.sort(np.concatenate([indices[:start], indices[end:]]))
        folds.append((train_idx, val_idx))
        start = end
    return folds


# ---------------------------------------------------------------------------
# Ground truth XLSX loader
# ---------------------------------------------------------------------------

def read_ground_truth(
    xlsx_path: str,
    sheet_name: str = "Ground Truth",
) -> pl.DataFrame:
    """Load the Ground Truth sheet from an XLSX file as a Polars DataFrame."""
    return pl.read_excel(xlsx_path, sheet_name=sheet_name)


# ---------------------------------------------------------------------------
# Full data pipeline
# ---------------------------------------------------------------------------

def build_full_pipeline(
    df: pl.DataFrame,
    source_columns: List[str],
    target_columns: List[str],
    unknown_token: str = "unknown",
    test_frac: float = 0.15,
    split_seed: int = 42,
    n_folds: int = 5,
    fold_seed: int = 42,
    prompt_first: bool = False,
) -> Dict[str, Any]:
    """Run the full data preparation pipeline and return everything needed for training.

    Returns a dict with:
        - "train_df": training rows (DataFrame)
        - "test_df":  held-out test rows (DataFrame)
        - "train_samples": samples built from train_df
        - "test_samples":  samples built from test_df
        - "folds": list of (train_fold_samples, val_fold_samples) per K-fold
        - "stats": summary statistics
    """
    # 1. Split rows
    train_df, test_df = train_test_row_split(df, test_frac, split_seed)

    # 2. Build samples
    train_samples = build_samples(
        train_df, source_columns, target_columns, unknown_token, prompt_first
    )
    test_samples = build_samples(
        test_df, source_columns, target_columns, unknown_token, prompt_first
    )

    # 3. K-fold on training samples (by index within train_df)
    fold_indices = kfold_indices(len(train_df), n_folds, fold_seed)
    folds: List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]] = []
    for train_idx, val_idx in fold_indices:
        fold_train_samples = build_samples(
            train_df.gather(train_idx).clone(),
            source_columns, target_columns, unknown_token, prompt_first,
        )
        fold_val_samples = build_samples(
            train_df.gather(val_idx).clone(),
            source_columns, target_columns, unknown_token, prompt_first,
        )
        folds.append((fold_train_samples, fold_val_samples))

    # 4. Stats
    stats = {
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_train_samples": len(train_samples),
        "n_test_samples": len(test_samples),
        "n_folds": n_folds,
        "train_per_fold": [
            (len(ft), len(fv)) for ft, fv in folds
        ],
        "target_counts_train": count_by_target(train_samples),
        "target_counts_test": count_by_target(test_samples),
    }

    return {
        "train_df": train_df,
        "test_df": test_df,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "folds": folds,
        "stats": stats,
    }
