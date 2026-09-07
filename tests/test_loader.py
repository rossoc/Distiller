# -*- coding: utf-8 -*-
"""Tests for data.loader — pure DataFrame/sample-building logic, no file I/O."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from data.loader import (
    build_full_pipeline,
    build_samples,
    build_samples_by_row,
    count_by_target,
    kfold_indices,
    split_rows_by_max_length,
    train_test_row_split,
)


# ---------------------------------------------------------------------------
# build_samples
# ---------------------------------------------------------------------------


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "S_text": ["hello", None, "world"],
            "L_text": ["foo", "bar", None],
            "Pieces1": ["3", None, ""],
            "Pieces2": [None, "5", "7"],
        }
    )


def test_build_samples_one_row_per_target_column():
    df = _sample_df()
    samples = build_samples(df, ["S_text", "L_text"], ["Pieces1", "Pieces2"])
    # 3 rows x 2 target columns = 6 samples, in row-major order.
    assert len(samples) == 6
    assert [s["input"].rsplit(" ", 1)[-1] for s in samples] == [
        "<Pieces1>?",
        "<Pieces2>?",
        "<Pieces1>?",
        "<Pieces2>?",
        "<Pieces1>?",
        "<Pieces2>?",
    ]


def test_build_samples_source_none_and_nan_are_dropped_not_stringified():
    df = pl.DataFrame({"S_text": [None], "L_text": ["ok"], "T": ["v"]})
    samples = build_samples(df, ["S_text", "L_text"], ["T"])
    # None must not be stringified as the literal "None".
    assert samples[0]["input"] == "ok <T>?"


def test_build_samples_source_float_nan_is_dropped():
    df = pl.DataFrame({"S_text": [float("nan")], "L_text": ["ok"], "T": ["v"]})
    samples = build_samples(df, ["S_text", "L_text"], ["T"])
    assert samples[0]["input"] == "ok <T>?"


def test_build_samples_missing_target_value_uses_unknown_token():
    df = pl.DataFrame({"S_text": ["hi"], "T": [None]})
    samples = build_samples(df, ["S_text"], ["T"], unknown_token="unknown")
    assert samples[0]["output"] == "unknown"


def test_build_samples_empty_string_target_uses_unknown_token():
    df = pl.DataFrame({"S_text": ["hi"], "T": [""]})
    samples = build_samples(df, ["S_text"], ["T"], unknown_token="unknown")
    assert samples[0]["output"] == "unknown"


def test_build_samples_custom_unknown_token():
    df = pl.DataFrame({"S_text": ["hi"], "T": [None]})
    samples = build_samples(df, ["S_text"], ["T"], unknown_token="N/A")
    assert samples[0]["output"] == "N/A"


def test_build_samples_skips_rows_with_no_source_text():
    df = pl.DataFrame({"S_text": [None], "T": ["v"]})
    samples = build_samples(df, ["S_text"], ["T"])
    assert samples == []


def test_build_samples_target_column_present_value_stripped():
    df = pl.DataFrame({"S_text": ["hi"], "T": ["  spaced  "]})
    samples = build_samples(df, ["S_text"], ["T"])
    assert samples[0]["output"] == "spaced"


def test_build_samples_ignores_unknown_target_columns():
    df = pl.DataFrame({"S_text": ["hi"], "T": ["v"]})
    samples = build_samples(df, ["S_text"], ["T", "DoesNotExist"])
    assert len(samples) == 1


# ---------------------------------------------------------------------------
# build_samples_by_row
# ---------------------------------------------------------------------------


def test_build_samples_by_row_groups_by_row_and_flattens_like_build_samples():
    df = _sample_df()
    rows_samples = build_samples_by_row(df, ["S_text", "L_text"], ["Pieces1", "Pieces2"])
    assert len(rows_samples) == len(df)  # one entry per row
    assert [len(rs) for rs in rows_samples] == [2, 2, 2]  # 2 target columns each
    flat = [s for rs in rows_samples for s in rs]
    assert flat == build_samples(df, ["S_text", "L_text"], ["Pieces1", "Pieces2"])


def test_build_samples_by_row_empty_list_for_row_with_no_source_text():
    df = pl.DataFrame({"S_text": ["hi", None], "T": ["v", "v2"]})
    rows_samples = build_samples_by_row(df, ["S_text"], ["T"])
    assert [len(rs) for rs in rows_samples] == [1, 0]


# ---------------------------------------------------------------------------
# split_rows_by_max_length
# ---------------------------------------------------------------------------


class _WhitespaceTokenizer:
    """Deterministic whitespace tokenizer — enough to exercise length-based
    row splitting without a real HF tokenizer."""

    eos_token = "<eos>"

    def __call__(self, text, truncation=False):
        return {"input_ids": text.split()}


def test_split_rows_by_max_length_separates_short_and_long_rows():
    tok = _WhitespaceTokenizer()
    rows_samples = [
        [{"input": "a b c", "output": "d"}],  # a b c <eos> d <eos> = 6 tokens
        [{"input": "a b c d e f g h", "output": "i"}],  # 11 tokens
    ]
    short_idx, long_idx = split_rows_by_max_length(rows_samples, tok, max_length=6)
    assert short_idx.tolist() == [0]
    assert long_idx.tolist() == [1]


def test_split_rows_by_max_length_row_is_long_if_any_sample_is():
    tok = _WhitespaceTokenizer()
    rows_samples = [
        [
            {"input": "a", "output": "b"},  # short
            {"input": "a b c d e f g h", "output": "i"},  # long
        ],
    ]
    short_idx, long_idx = split_rows_by_max_length(rows_samples, tok, max_length=6)
    assert short_idx.tolist() == []
    assert long_idx.tolist() == [0]


def test_split_rows_by_max_length_empty_row_samples_is_short():
    tok = _WhitespaceTokenizer()
    short_idx, long_idx = split_rows_by_max_length([[]], tok, max_length=6)
    assert short_idx.tolist() == [0]
    assert long_idx.tolist() == []


# ---------------------------------------------------------------------------
# count_by_target
# ---------------------------------------------------------------------------


def test_count_by_target():
    samples = [
        {"input": "x <A>?", "output": "1"},
        {"input": "x <A>?", "output": "2"},
        {"input": "x <B>?", "output": "3"},
    ]
    assert count_by_target(samples) == {"A": 2, "B": 1}


def test_count_by_target_empty():
    assert count_by_target([]) == {}


# ---------------------------------------------------------------------------
# kfold_indices
# ---------------------------------------------------------------------------


def test_kfold_indices_full_coverage_no_overlap():
    n_rows, n_folds = 23, 5
    folds = kfold_indices(n_rows, n_folds, seed=0)
    assert len(folds) == n_folds
    seen = set()
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))
        assert len(train_idx) + len(val_idx) == n_rows
        seen.update(val_idx.tolist())
    # Every row index appears in exactly one fold's validation set.
    assert seen == set(range(n_rows))


def test_kfold_indices_remainder_distributed_across_first_folds():
    # 23 rows / 5 folds = 4 remainder 3 -> first 3 folds get 5, rest get 4.
    folds = kfold_indices(23, 5, seed=0)
    val_sizes = [len(val_idx) for _, val_idx in folds]
    assert sorted(val_sizes, reverse=True) == [5, 5, 5, 4, 4]


def test_kfold_indices_deterministic():
    folds_a = kfold_indices(30, 4, seed=7)
    folds_b = kfold_indices(30, 4, seed=7)
    for (ta, va), (tb, vb) in zip(folds_a, folds_b):
        assert np.array_equal(ta, tb)
        assert np.array_equal(va, vb)


def test_kfold_indices_different_seed_differs():
    folds_a = kfold_indices(30, 4, seed=1)
    folds_b = kfold_indices(30, 4, seed=2)
    assert not all(
        np.array_equal(va, vb) for (_, va), (_, vb) in zip(folds_a, folds_b)
    )


def test_kfold_indices_forced_val_idx_always_in_val_never_in_train():
    forced = np.array([2, 7])
    folds = kfold_indices(20, 4, seed=0, forced_val_idx=forced)
    for train_idx, val_idx in folds:
        assert set(forced).issubset(set(val_idx.tolist()))
        assert set(forced).isdisjoint(set(train_idx.tolist()))
        # Full coverage still holds: every row is train or val, never neither.
        assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(20))


def test_kfold_indices_forced_val_idx_fold_sizes_computed_over_remainder():
    # 18 rotating rows / 3 folds = 6 each; 2 forced rows on top of every val.
    folds = kfold_indices(20, 3, seed=0, forced_val_idx=np.array([0, 1]))
    for _, val_idx in folds:
        assert len(val_idx) == 6 + 2


def test_kfold_indices_no_forced_val_idx_matches_previous_behavior():
    # forced_val_idx omitted/None must behave exactly like plain kfold_indices.
    folds_none = kfold_indices(23, 5, seed=0)
    folds_empty = kfold_indices(23, 5, seed=0, forced_val_idx=np.array([]))
    for (ta, va), (tb, vb) in zip(folds_none, folds_empty):
        assert np.array_equal(np.sort(ta), np.sort(tb))
        assert np.array_equal(np.sort(va), np.sort(vb))


# ---------------------------------------------------------------------------
# train_test_row_split
# ---------------------------------------------------------------------------


def test_train_test_row_split_no_leakage_and_full_coverage():
    df = pl.DataFrame({"id": list(range(50))})
    train_df, test_df = train_test_row_split(df, test_frac=0.2, seed=3)
    train_ids = set(train_df["id"].to_list())
    test_ids = set(test_df["id"].to_list())
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(range(50))
    assert len(test_df) == 10


def test_train_test_row_split_deterministic():
    df = pl.DataFrame({"id": list(range(50))})
    train_a, test_a = train_test_row_split(df, test_frac=0.2, seed=3)
    train_b, test_b = train_test_row_split(df, test_frac=0.2, seed=3)
    assert train_a["id"].to_list() == train_b["id"].to_list()
    assert test_a["id"].to_list() == test_b["id"].to_list()


def test_train_test_row_split_at_least_one_test_row():
    df = pl.DataFrame({"id": list(range(3))})
    _, test_df = train_test_row_split(df, test_frac=0.01, seed=0)
    assert len(test_df) >= 1


# ---------------------------------------------------------------------------
# build_full_pipeline
# ---------------------------------------------------------------------------


def test_build_full_pipeline_shapes():
    df = pl.DataFrame(
        {
            "S_text": [f"s{i}" for i in range(20)],
            "T": [str(i) for i in range(20)],
        }
    )
    result = build_full_pipeline(
        df,
        source_columns=["S_text"],
        target_columns=["T"],
        test_frac=0.2,
        n_folds=4,
    )
    assert set(result.keys()) == {
        "train_df",
        "test_df",
        "train_samples",
        "test_samples",
        "folds",
        "stats",
    }
    assert len(result["folds"]) == 4
    assert result["stats"]["n_train_rows"] + result["stats"]["n_test_rows"] == 20
    # 1 target column -> one sample per row.
    assert result["stats"]["n_train_samples"] == result["stats"]["n_train_rows"]
