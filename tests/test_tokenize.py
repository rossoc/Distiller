# -*- coding: utf-8 -*-
"""Tests for _tokenize_pairs (lit_datamodule.py — the single copy, also used
by predict.py).

Uses a small hand-rolled fake tokenizer instead of a real HF one: no small
HF tokenizer is cached locally for this project's model, and a fake is
simpler and fully deterministic anyway.
"""

from __future__ import annotations

from typing import Dict

import torch

from lit_datamodule import _tokenize_pairs


class FakeTokenizer:
    """Deterministic whitespace tokenizer.

    ``eos_token`` is wrapped in spaces so whitespace-splitting treats it as
    its own token even when concatenated directly onto adjacent text (as
    ``_tokenize_pairs`` does: ``f"{inp}{eos_token}{out}{eos_token}"``) —
    mirroring how a real tokenizer emits a dedicated EOS token regardless of
    surrounding text.
    """

    def __init__(self) -> None:
        self.eos_token = " <eos> "
        self._vocab: Dict[str, int] = {}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab)
        return self._vocab[tok]

    @property
    def eos_token_id(self) -> int:
        return self._id("<eos>")

    def __call__(self, text, max_length=None, truncation=False, padding=False, return_tensors=None):
        ids = [self._id(tok) for tok in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_tokenize_pairs_masks_prompt_and_separator_only():
    tok = FakeTokenizer()
    pad_id = 999  # distinct from any real token id, no aliasing in this test

    out = _tokenize_pairs(tok, pad_id, ["a b c"], ["d e"], max_length=50)

    # full sequence: a b c <eos> d e <eos>  (7 tokens, no padding needed)
    labels = out["labels"][0].tolist()
    input_ids = out["input_ids"][0].tolist()
    assert len(input_ids) == 7

    # Prompt ("a b c") plus the separating <eos> (4 tokens) must be masked;
    # the answer + trailing <eos> (3 tokens) must be left as real labels.
    assert labels[:4] == [-100, -100, -100, -100]
    assert labels[4:] == input_ids[4:]
    assert -100 not in labels[4:]


def test_tokenize_pairs_attention_mask_survives_pad_eos_aliasing():
    """Regression test: when pad_token_id == eos_token_id, the attention
    mask must still be 1 on every real position (including interior EOS
    tokens) and 0 only on the padded tail — not derived by comparing token
    ids against pad_token_id, which would zero out real EOS positions too.
    """
    tok = FakeTokenizer()
    pad_id = tok.eos_token_id  # alias: no dedicated pad token

    # Short sample: "a b" <eos> "c" <eos>  -> 5 real tokens.
    # Long sample:  "a b c d e f g" <eos> "h" <eos> -> 10 real tokens.
    out = _tokenize_pairs(
        tok,
        pad_id,
        ["a b", "a b c d e f g"],
        ["c", "h"],
        max_length=50,
    )

    attention_mask = out["attention_mask"]
    input_ids = out["input_ids"]
    assert attention_mask.shape[1] == 10  # padded to the longer sample

    # Short sample: first 5 positions real (1s), last 5 padded (0s) — even
    # though one of the first 5 positions is itself an <eos> token whose id
    # equals pad_id.
    assert attention_mask[0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert (input_ids[0][5:] == pad_id).all()

    # Long sample has no padding at all.
    assert attention_mask[1].tolist() == [1] * 10


def test_tokenize_pairs_truncates_to_max_length():
    tok = FakeTokenizer()
    out = _tokenize_pairs(tok, 0, ["a b c d e"], ["f g h"], max_length=4)
    assert out["input_ids"].shape[1] == 4
    assert out["labels"].shape[1] == 4
    assert out["attention_mask"].shape[1] == 4
