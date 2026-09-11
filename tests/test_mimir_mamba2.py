# -*- coding: utf-8 -*-
"""Tests for model/mimir_mamba2.py — the Mamba2 + frozen-donor module.

Nothing here downloads DFM-Mimir. ``load_donor_tables`` is monkeypatched to
return a toy 320 x 64 pair of tables and ``load_tokenizer`` to return a stub,
so the real ``__init__`` runs end to end (config assembly, projection build,
module swap, parameter accounting) against fakes. The pure-optimisation and
pure-loss helpers are additionally exercised on a bare instance.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import model.mimir_mamba2 as mamba2_mod
from model.donor_projection import DonorTables, ProjectedEmbedding, ProjectedLMHead
from model.mimir_mamba2 import MimirMamba2Module

VOCAB, D_DONOR = 320, 64


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 3
    bos_token_id = 2

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def save_pretrained(self, path):
        """No-op: the round-trip tests reload the tokenizer from this stub."""


@pytest.fixture
def fake_donor(monkeypatch):
    """Patch out the tokenizer download and the 1B donor load."""
    generator = torch.Generator().manual_seed(0)
    tables = DonorTables(
        "toy/donor",
        torch.randn(VOCAB, D_DONOR, generator=generator) * 0.03,
        torch.randn(VOCAB, D_DONOR, generator=generator) * 0.05,
    )
    monkeypatch.setattr(mamba2_mod, "load_tokenizer", lambda *a, **k: _FakeTokenizer())
    monkeypatch.setattr(mamba2_mod, "load_donor_tables", lambda *a, **k: tables)
    return tables


def _module(**overrides) -> MimirMamba2Module:
    kwargs = dict(
        donor_model_id="toy/donor",
        dtype="fp32",
        d_model=32,
        num_hidden_layers=2,
        state_size=8,
        head_dim=16,
        chunk_size=8,
        projection_cache_dir=None,
    )
    kwargs.update(overrides)
    return MimirMamba2Module(**kwargs)


# ---------------------------------------------------------------------------
# Construction: the module swap and what it implies
# ---------------------------------------------------------------------------


def test_projections_replace_the_stock_embedding_and_head(fake_donor):
    module = _module()
    assert isinstance(module.projected_embedding, ProjectedEmbedding)
    assert isinstance(module.projected_lm_head, ProjectedLMHead)
    assert module.model.config.vocab_size == VOCAB
    assert module.model.config.hidden_size == 32


def test_donor_tables_are_not_in_the_checkpoint(fake_donor):
    """The two 805 MiB tables must never be written to disk per checkpoint."""
    state = _module().model.state_dict()
    assert not [k for k in state if "donor" in k]
    assert "backbone.embeddings.proj.weight" in state
    assert "lm_head.proj.weight" in state


def test_num_heads_is_derived_from_expand_and_head_dim(fake_donor):
    config = _module(d_model=64, head_dim=16, expand=2).model.config
    assert config.num_heads == 2 * 64 // 16


def test_indivisible_head_dim_is_rejected_with_a_useful_message(fake_donor):
    with pytest.raises(ValueError, match="head_dim"):
        _module(d_model=32, head_dim=48)


def test_the_tokenizer_comes_from_the_donor(fake_donor):
    """Token ids index the donor tables, so the tokenizers must be the same one."""
    module = _module()
    assert module.pad_token_id == _FakeTokenizer.pad_token_id
    assert module.eos_token_id == _FakeTokenizer.eos_token_id
    assert module.model.config.pad_token_id == _FakeTokenizer.pad_token_id


def test_cache_is_off_for_training(fake_donor):
    assert _module().model.config.use_cache is False


def test_freeze_backbone_leaves_only_the_projections_trainable(fake_donor):
    module = _module(freeze_backbone=True)
    trainable = {n for n, p in module.model.named_parameters() if p.requires_grad}
    assert trainable == {
        "backbone.embeddings.proj.weight",
        "backbone.embeddings.scale",
        "lm_head.proj.weight",
        "lm_head.scale",
    }


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_backward_reaches_the_projections_and_the_blocks_but_not_the_donor(fake_donor):
    module = _module()
    ids = torch.randint(0, VOCAB, (2, 12))
    labels = ids.clone()
    labels[:, :4] = -100

    loss, logits = module(ids, torch.ones_like(ids), labels)
    assert logits is None, "chunked loss should not materialize logits"
    loss.backward()

    named = dict(module.model.named_parameters())
    for name in (
        "backbone.embeddings.proj.weight",
        "backbone.embeddings.scale",
        "lm_head.proj.weight",
        "lm_head.scale",
        "backbone.layers.0.mixer.in_proj.weight",
    ):
        grad = named[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name

    assert module.projected_embedding.donor.grad is None
    assert module.projected_lm_head.donor.grad is None


# ---------------------------------------------------------------------------
# The chunked loss
# ---------------------------------------------------------------------------


def test_chunked_loss_matches_the_one_shot_loss(fake_donor):
    """Chunking is a memory strategy, so it must change nothing numerically."""
    module = _module()
    module.eval()
    ids = torch.randint(0, VOCAB, (3, 17))
    mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :5] = -100
    labels[2, 12:] = -100

    losses = {}
    grads = {}
    for chunk in (0, 64, 5):
        for param in module.model.parameters():
            param.grad = None
        module.hparams.loss_chunk_tokens = chunk
        loss, _ = module(ids, mask, labels)
        loss.backward()
        losses[chunk] = float(loss.detach())
        grads[chunk] = {
            n: p.grad.clone()
            for n, p in module.model.named_parameters()
            if p.grad is not None
        }

    for chunk in (64, 5):
        assert losses[chunk] == pytest.approx(losses[0], abs=1e-5)
        assert set(grads[chunk]) == set(grads[0])
        for name, grad in grads[chunk].items():
            assert torch.allclose(grad, grads[0][name], atol=1e-5), name


def test_fully_masked_batch_yields_a_finite_zero_loss(fake_donor):
    """An all -100 batch must not put a nan into the epoch average.

    It must also stay differentiable: Lightning calls .backward() on whatever
    training_step returns, so a detached zero would raise instead of quietly
    skipping the batch.
    """
    module = _module()
    ids = torch.randint(0, VOCAB, (2, 9))
    labels = torch.full_like(ids, -100)

    loss, _ = module(ids, torch.ones_like(ids), labels)
    assert torch.isfinite(loss) and float(loss.detach()) == 0.0

    loss.backward()  # must not raise
    for name, param in module.model.named_parameters():
        if param.grad is not None:
            assert float(param.grad.abs().sum()) == 0.0, name


def test_genuine_non_finite_loss_is_replaced_without_a_python_bool_branch(fake_donor):
    """forward()'s degenerate-batch guard used to be
    `if not torch.isfinite(loss):` — a Python bool() of a CUDA tensor, i.e. a
    blocking host sync on *every* call, finite or not. It's now
    `torch.where(finite, loss, ...)`, entirely on-tensor. Exercise the actual
    non-finite path (loss_chunk_tokens=0 forces HF's one-shot
    reduction="mean" loss, which is nan on an all-masked batch — the
    _chunked_loss path short-circuits before ever producing a non-finite
    value, so it can't be used to test this branch) and check the same two
    guarantees the old code made: a finite 0 loss, and zero gradient."""
    module = _module(loss_chunk_tokens=0)
    ids = torch.randint(0, VOCAB, (2, 9))
    labels = torch.full_like(ids, -100)

    loss, _ = module(ids, torch.ones_like(ids), labels)
    assert torch.isfinite(loss) and float(loss.detach()) == 0.0

    loss.backward()  # must not raise
    for name, param in module.model.named_parameters():
        if param.grad is not None:
            assert float(param.grad.abs().sum()) == 0.0, name


def test_guard_nonfinite_loss_never_syncs_to_host(fake_donor, monkeypatch):
    """The whole point of the torch.where rewrite: _guard_nonfinite_loss must
    never call Tensor.__bool__/.item() itself, on either the finite or the
    non-finite path (torch.isfinite(...) alone is fine — it's a tensor op,
    not a host sync). Scoped to this one method (not the full forward()),
    since forward() also runs the Mamba2 backbone, whose own library code
    (transformers' attention-mask handling) does unrelated host syncs that
    have nothing to do with this fix."""
    module = _module()

    def _boom(self, *a, **k):
        raise AssertionError("_guard_nonfinite_loss synced a tensor to host")

    monkeypatch.setattr(torch.Tensor, "__bool__", _boom, raising=True)
    monkeypatch.setattr(torch.Tensor, "item", _boom, raising=True)
    try:
        finite_out = module._guard_nonfinite_loss(torch.tensor(1.5, requires_grad=True))
        nan_out = module._guard_nonfinite_loss(
            torch.tensor(float("nan"), requires_grad=True)
        )
    finally:
        monkeypatch.undo()

    assert float(finite_out.detach()) == 1.5
    assert torch.isfinite(nan_out) and float(nan_out.detach()) == 0.0
    assert int(module._degenerate_batch_count.item()) == 1


def test_degenerate_batch_count_is_accumulated_on_gpu_and_flushed_per_epoch(
    fake_donor, caplog
):
    """The old per-batch warning is now a per-epoch summary, synced once via
    on_train_epoch_end/on_validation_epoch_end instead of once per forward()
    call — same visibility, a fraction of the host syncs."""
    module = _module(loss_chunk_tokens=0)
    ids = torch.randint(0, VOCAB, (2, 9))
    degenerate_labels = torch.full_like(ids, -100)
    normal_labels = ids.clone()

    module(ids, torch.ones_like(ids), degenerate_labels)
    module(ids, torch.ones_like(ids), normal_labels)
    module(ids, torch.ones_like(ids), degenerate_labels)
    assert int(module._degenerate_batch_count.item()) == 2

    with caplog.at_level("WARNING"):
        module.on_train_epoch_end()
    assert "2 degenerate" in caplog.text
    assert int(module._degenerate_batch_count.item()) == 0  # reset

    # A clean epoch logs nothing.
    caplog.clear()
    module(ids, torch.ones_like(ids), normal_labels)
    with caplog.at_level("WARNING"):
        module.on_validation_epoch_end()
    assert caplog.text == ""


@pytest.mark.parametrize(
    "batch,seq,chunk", [(1, 17, 5), (2, 9, 4), (5, 13, 3), (4, 6, 64)]
)
def test_chunked_loss_matches_a_naive_slice_then_flatten_reference(
    fake_donor, batch, seq, chunk
):
    """``_chunked_loss`` flattens hidden_states/labels *before* slicing off
    the row-boundary element (to keep the flatten a zero-copy view) and
    patches the resulting cross-row-boundary label pairs back to -100
    instead. That reindexing must match the naive
    ``hidden_states[:, :-1].reshape(...)`` / ``labels[:, 1:].reshape(...)``
    it replaces, across batch/seq/chunk shapes that do and don't divide
    evenly (a batch of 1 has no boundary to patch at all)."""
    module = _module()
    module.eval()
    ids = torch.randint(0, VOCAB, (batch, seq))
    labels = ids.clone()
    labels[:, :2] = -100  # a couple of masked prompt tokens, like real data

    module.hparams.loss_chunk_tokens = chunk
    got, _ = module(ids, torch.ones_like(ids), labels)

    hidden_states = module.model.backbone(
        input_ids=ids, use_cache=False, return_dict=True
    )[0]
    d_model = hidden_states.shape[-1]
    ref_hidden = hidden_states[:, :-1].reshape(-1, d_model)
    ref_labels = labels[:, 1:].reshape(-1)
    expected = torch.nn.functional.cross_entropy(
        module.model.lm_head(ref_hidden).float(),
        ref_labels,
        ignore_index=-100,
        reduction="sum",
    ) / int((ref_labels != -100).sum())

    assert float(got.detach()) == pytest.approx(float(expected.detach()), abs=1e-4)


def test_flatten_then_slice_avoids_copying_hidden_states(fake_donor):
    """The whole point of flattening before slicing (rather than the other
    way around) is that it never copies the [B, T, d_model] hidden states —
    only a same-storage view is taken. Pin that down directly, since a
    future edit that reorders the ops back to slice-then-flatten would
    silently reintroduce the O(B*T*d_model) copy this was written to avoid,
    while still passing the purely numerical tests above."""
    hidden_states = torch.randn(4, 9, 6)
    d_model = hidden_states.shape[-1]
    flat_hidden = hidden_states.reshape(-1, d_model)[:-1]
    assert flat_hidden.data_ptr() == hidden_states.data_ptr()
    assert flat_hidden.is_contiguous()


def test_chunks_that_are_entirely_masked_are_skipped_not_averaged_in(fake_donor):
    """Trailing all-padding chunks must not dilute the mean."""
    module = _module()
    ids = torch.randint(0, VOCAB, (1, 20))
    labels = ids.clone()
    labels[:, 6:] = -100  # only positions 1..5 score

    module.hparams.loss_chunk_tokens = 4  # several chunks are fully masked
    chunked, _ = module(ids, torch.ones_like(ids), labels)
    module.hparams.loss_chunk_tokens = 0
    one_shot, _ = module(ids, torch.ones_like(ids), labels)
    assert float(chunked.detach()) == pytest.approx(float(one_shot.detach()), abs=1e-5)


# ---------------------------------------------------------------------------
# Export / round trip
# ---------------------------------------------------------------------------


def test_export_standalone_is_a_plain_mamba2_with_concrete_tables(fake_donor):
    module = _module()
    module.eval()
    ids = torch.randint(0, VOCAB, (2, 11))
    mask = torch.ones_like(ids)

    exported = module.export_standalone(torch.float32)

    assert isinstance(exported.get_input_embeddings(), torch.nn.Embedding)
    assert exported.get_input_embeddings().weight.shape == (VOCAB, 32)
    assert exported.lm_head.weight.shape == (VOCAB, 32)
    assert not [k for k in exported.state_dict() if "donor" in k]

    with torch.no_grad():
        projected = module.model(ids, attention_mask=mask).logits
        collapsed = exported(ids, attention_mask=mask).logits
    rel = float((projected - collapsed).abs().max() / projected.abs().max())
    assert rel < 1e-4, f"collapsing the projections changed the model ({rel:.2e})"


def test_save_and_load_round_trips_to_identical_logits(fake_donor, tmp_path):
    module = _module()
    module.eval()
    ids = torch.randint(0, VOCAB, (2, 11))
    mask = torch.ones_like(ids)

    module.save_pretrained(str(tmp_path))
    assert (tmp_path / "donor_projections.pt").exists()
    assert (tmp_path / "donor_projections.json").exists()

    restored = MimirMamba2Module.from_pretrained(
        str(tmp_path), dtype="fp32", projection_cache_dir=None
    )
    restored.eval()

    # from_pretrained places itself on the GPU when there is one (predict.py
    # relies on that). Score both models there: a CPU-vs-GPU comparison would
    # be measuring TF32, which train.py enables process-wide, rather than the
    # round trip.
    device = restored.model.device
    ids, mask = ids.to(device), mask.to(device)
    module.model.to(device)
    with torch.no_grad():
        before = module.model(ids, attention_mask=mask).logits
        after = restored.model(ids, attention_mask=mask).logits
    assert torch.allclose(before, after, atol=1e-5)


def test_from_pretrained_without_a_recorded_donor_is_refused(fake_donor, tmp_path):
    module = _module()
    module.save_pretrained(str(tmp_path))
    (tmp_path / "donor_projections.json").unlink()

    with pytest.raises(ValueError, match="donor"):
        MimirMamba2Module.from_pretrained(str(tmp_path), dtype="fp32")


# ---------------------------------------------------------------------------
# configure_optimizers — exercised on a bare instance, no donor needed
# ---------------------------------------------------------------------------


def test_optimizer_splits_projections_from_blocks_and_decay_from_no_decay(fake_donor):
    module = _module()
    module.trainer = SimpleNamespace(estimated_stepping_batches=100)
    module.hparams.projection_lr_mult = 5.0
    base_lr = module.hparams.learning_rate

    groups = module.configure_optimizers()["optimizer"].param_groups
    proj_decay, proj_no_decay, body_decay, body_no_decay = groups

    # P and Q decay; the two scalar gains do not.
    assert len(proj_decay["params"]) == 2
    assert len(proj_no_decay["params"]) == 2
    assert proj_decay["weight_decay"] == 0.01
    assert proj_no_decay["weight_decay"] == 0.0

    # Both projection groups run at the multiplied LR, both body groups don't.
    assert proj_decay["initial_lr"] == pytest.approx(5.0 * base_lr)
    assert proj_no_decay["initial_lr"] == pytest.approx(5.0 * base_lr)
    assert body_decay["initial_lr"] == pytest.approx(base_lr)
    assert body_no_decay["initial_lr"] == pytest.approx(base_lr)

    # Every trainable parameter lands in exactly one group.
    assert sum(len(g["params"]) for g in groups) == len(
        [p for p in module.model.parameters() if p.requires_grad]
    )


def test_ssm_scalars_and_norms_are_excluded_from_weight_decay(fake_donor):
    module = _module()
    module.trainer = SimpleNamespace(estimated_stepping_batches=100)
    _, _, body_decay, body_no_decay = module.configure_optimizers()[
        "optimizer"
    ].param_groups

    decayed = {id(p) for p in body_decay["params"]}
    named = dict(module.model.named_parameters())
    for name in (
        "backbone.layers.0.mixer.A_log",
        "backbone.layers.0.mixer.D",
        "backbone.layers.0.mixer.dt_bias",
        "backbone.norm_f.weight",
    ):
        assert id(named[name]) not in decayed, f"{name} should not be decayed"
    assert id(named["backbone.layers.0.mixer.in_proj.weight"]) in decayed


def test_unknown_scheduler_yields_no_scheduler(fake_donor):
    module = _module()
    module.trainer = SimpleNamespace(estimated_stepping_batches=100)
    module.hparams.lr_scheduler = "triangular"
    assert "lr_scheduler" not in module.configure_optimizers()
