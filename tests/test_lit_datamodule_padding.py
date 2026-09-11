# -*- coding: utf-8 -*-
"""Tests for per-batch dynamic padding in lit_datamodule.py (performance
review item #5).

Previously every sample in a split was padded up front to that split's
single longest sequence (``_tokenize_pairs``, still kept for callers that
want one padded batch out of a whole sample list). Now ``_TokenizedDataset``
stores unpadded per-sample tensors and ``_collate_batch`` — the DataLoader's
``collate_fn`` — pads each batch to only its own longest sequence.
"""

from __future__ import annotations

import torch

from lit_datamodule import (
    _TokenizedDataset,
    _collate_batch,
    _tokenize_pair,
    _tokenize_pairs,
)
from tests.test_tokenize import FakeTokenizer


def _texts():
    # Deliberately uneven lengths: the third sample is much longer than the
    # first two, so a batch of just the first two must not pay for it.
    inputs = ["a b", "a b c", "a b c d e f g h i j"]
    outputs = ["x", "y y", "z"]
    return inputs, outputs


def test_collate_batch_pads_to_the_batch_own_max_not_the_split_max():
    tok = FakeTokenizer()
    pad_id = tok.eos_token_id
    inputs, outputs = _texts()

    samples = [
        _tokenize_pair(tok, inp, out, max_length=50)
        for inp, out in zip(inputs, outputs)
    ]

    # A batch of just the two short samples...
    short_batch = _collate_batch(samples[:2], pad_token_id=pad_id)
    # ...must be padded to their own longest length, not sample[2]'s (which
    # is much longer) — the whole point of per-batch over dataset-wide padding.
    own_max = max(len(samples[0][0]), len(samples[1][0]))
    long_len = len(samples[2][0])
    assert short_batch["input_ids"].shape[1] == own_max
    assert own_max < long_len


def test_dataset_and_collate_round_trip_matches_tokenize_pair():
    tok = FakeTokenizer()
    pad_id = tok.eos_token_id
    inputs, outputs = _texts()

    input_ids_list, labels_list = [], []
    for inp, out in zip(inputs, outputs):
        ids, labels = _tokenize_pair(tok, inp, out, max_length=50)
        input_ids_list.append(ids)
        labels_list.append(labels)

    dataset = _TokenizedDataset(input_ids_list, labels_list)
    assert len(dataset) == 3
    batch = _collate_batch([dataset[0], dataset[1]], pad_token_id=pad_id)

    # Unpadded row 0 must appear verbatim in the padded batch (real content
    # untouched by padding, only the tail changes).
    row0_len = len(input_ids_list[0])
    assert batch["input_ids"][0, :row0_len].tolist() == input_ids_list[0].tolist()
    assert batch["labels"][0, :row0_len].tolist() == labels_list[0].tolist()
    assert batch["attention_mask"][0, :row0_len].tolist() == [1] * row0_len
    if batch["input_ids"].shape[1] > row0_len:
        assert batch["attention_mask"][0, row0_len:].tolist() == [0] * (
            batch["input_ids"].shape[1] - row0_len
        )


def test_per_batch_padding_yields_identical_loss_to_dataset_wide_padding():
    """The actual regression guard: a batch that does NOT contain the
    split's longest row must produce the identical model loss whether it
    was padded per-batch (new) or sliced out of a dataset-wide padded split
    (old) — padding more than necessary must be a pure no-op on the result.
    """
    import model.mimir_mamba2 as mamba2_mod
    from model.donor_projection import DonorTables

    VOCAB, D_DONOR = 320, 64

    class _FakeHFTok:
        pad_token = "<pad>"
        eos_token = "</s>"
        pad_token_id = 0
        eos_token_id = 3
        bos_token_id = 2

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    gen = torch.Generator().manual_seed(0)
    tables = DonorTables(
        "toy/donor",
        torch.randn(VOCAB, D_DONOR, generator=gen) * 0.03,
        torch.randn(VOCAB, D_DONOR, generator=gen) * 0.05,
    )
    original_load_tokenizer = mamba2_mod.load_tokenizer
    original_load_donor = mamba2_mod.load_donor_tables
    mamba2_mod.load_tokenizer = lambda *a, **k: _FakeHFTok()
    mamba2_mod.load_donor_tables = lambda *a, **k: tables
    try:
        module = mamba2_mod.MimirMamba2Module(
            donor_model_id="toy/donor",
            dtype="fp32",
            d_model=32,
            num_hidden_layers=2,
            state_size=8,
            head_dim=16,
            chunk_size=8,
            projection_cache_dir=None,
        )
    finally:
        mamba2_mod.load_tokenizer = original_load_tokenizer
        mamba2_mod.load_donor_tables = original_load_donor
    module.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (2, 6))
    labels = ids.clone()
    labels[:, :1] = -100
    # Row 1 is shorter (simulated by masking its tail with -100/pad — the
    # dataset-wide-padded "old" tensor already has trailing pad here; the
    # "new" per-batch tensor is trimmed to this batch's own max, which for
    # a 2-row batch that already IS its own max is identical shape. To
    # actually exercise "doesn't hit the split's max", pad an extra 4
    # dataset-wide-only columns onto the OLD tensors that a longer row
    # elsewhere in the split would have forced, and confirm the NEW
    # (untrimmed) shorter tensors give the same loss.
    old_extra_pad = 4
    old_input_ids = torch.nn.functional.pad(
        ids, (0, old_extra_pad), value=module.pad_token_id
    )
    old_labels = torch.nn.functional.pad(labels, (0, old_extra_pad), value=-100)
    old_mask = torch.nn.functional.pad(
        torch.ones_like(ids), (0, old_extra_pad), value=0
    )

    new_input_ids, new_labels, new_mask = ids, labels, torch.ones_like(ids)

    old_loss, _ = module(old_input_ids, old_mask, old_labels)
    new_loss, _ = module(new_input_ids, new_mask, new_labels)

    assert torch.allclose(old_loss, new_loss, atol=1e-6)
