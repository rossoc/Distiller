# -*- coding: utf-8 -*-
"""Phase E gate: MoMos on the real Mamba2 backbone (SPEC.md §8, Phase E).

    .venv/bin/python scripts/momos_phase_e.py

Trains the real ``model_jax.mimir_mamba2.MimirMamba2Model`` (the actual
Mamba2 backbone + donor-projection architecture, not the toy SSM harness
every earlier phase gate used) on the real district-heating ground-truth
data (``data/data_district_heating.xlsx``, the same file
``src/train.py``'s PyTorch path trains on) — dense vs MoMos, at matched
hyperparameters. Per the SPEC.md §8 gate: "loss curve tracks the dense JAX
path at matched hyperparameters, with measured bytes_per_weight."

**No real donor model is available in this environment** (``outputs/
donor_jax`` — the offline-converted donor tables + tokenizer — does not
exist here; building it requires downloading and converting a real donor
checkpoint over the network). So, exactly like
``tests/test_lit_datamodule_padding.py`` and ``tests/test_tokenize.py``
already do for this same reason, this script uses a small deterministic
fake tokenizer and randomly-initialised toy donor tables sized to its
vocabulary. **What is real:** the text, the Mamba2 backbone architecture and
its parameter geometry, the loss function, and the training loop. **What is
a stand-in:** the tokenizer and the donor tables' values (not their role in
the architecture). This is the same "real architecture, fake donor" pattern
``tests/model_jax/test_mimir_mamba2_jax.py`` already uses to test the model
without a network call — reused here for a training-dynamics comparison
rather than a single forward pass.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loader import build_samples, read_ground_truth  # noqa: E402
from lit_datamodule import _tokenize_pairs  # noqa: E402
from model_jax.donor_projection import DonorTables  # noqa: E402
from model_jax.mimir_mamba2 import MimirMamba2Model  # noqa: E402
from model_jax.momos import integration, state as momos_state  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("momos_phase_e")


class FakeTokenizer:
    """Deterministic whitespace tokenizer — see this module's docstring for
    why: no real donor tokenizer is available offline. Ported from
    ``tests/test_tokenize.py`` rather than imported, so this script has no
    test-suite dependency."""

    def __init__(self) -> None:
        self.eos_token = " <eos> "
        self.pad_token_id = None  # set below once <eos> exists
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_batches(
    n_samples: int, batch_size: int, max_length: int, seed: int = 0
) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
    """Real district-heating (input, output) pairs -> a handful of padded batches.

    Returns (batches, vocab_size, pad_token_id).
    """
    xlsx = REPO_ROOT / "data" / "data_district_heating.xlsx"
    df = read_ground_truth(str(xlsx), sheet_name="Ground Truth")
    samples = build_samples(
        df,
        source_columns=["S_text", "L_text"],
        target_columns=["Pieces1", "Manufacturer1", "SubType1", "HxType1", "NominelEffectEach1", "Year1"],
        unknown_token="unknown",
        prompt_first=True,
    )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))[:n_samples]
    chosen = [samples[i] for i in idx]
    log.info("Loaded %d real samples from %s (of %d total)", len(chosen), xlsx.name, len(samples))

    tok = FakeTokenizer()
    tok.pad_token_id = tok.eos_token_id  # fixes id 0, same aliasing test_tokenize.py exercises
    inputs = [s["input"] for s in chosen]
    outputs = [s["output"] for s in chosen]
    padded = _tokenize_pairs(tok, tok.pad_token_id, inputs, outputs, max_length=max_length)
    vocab_size = len(tok._vocab)

    n = padded["input_ids"].shape[0]
    batches = []
    for start in range(0, n - n % batch_size, batch_size):
        sl = slice(start, start + batch_size)
        batches.append(
            {
                "input_ids": jnp.asarray(padded["input_ids"][sl].numpy()),
                "labels": jnp.asarray(padded["labels"][sl].numpy()),
                "attention_mask": jnp.asarray(padded["attention_mask"][sl].numpy()),
            }
        )
    log.info(
        "Tokenized vocab=%d, %d batches of %d, seq_len=%d",
        vocab_size, len(batches), batch_size, padded["input_ids"].shape[1],
    )
    return batches, vocab_size, tok.pad_token_id


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(vocab_size: int, d_donor: int, seed: int, **kwargs) -> MimirMamba2Model:
    g = torch.Generator().manual_seed(seed)
    tables = DonorTables(
        "toy/donor",
        jnp.asarray((torch.randn(vocab_size, d_donor, generator=g) * 0.03).numpy()),
        jnp.asarray((torch.randn(vocab_size, d_donor, generator=g) * 0.05).numpy()),
    )
    return MimirMamba2Model(tables, rngs=nnx.Rngs(params=seed), **kwargs)


def loss_fn(model, batch):
    return model.loss(batch["input_ids"], batch["labels"], batch["attention_mask"])


# ---------------------------------------------------------------------------
# Training arms
# ---------------------------------------------------------------------------


def train_dense(model, batches, lr: float, n_steps: int) -> List[float]:
    tx = optax.adamw(lr)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def step(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        optimizer.update(model, grads)
        return loss

    losses = []
    for i in range(n_steps):
        losses.append(float(step(model, optimizer, batches[i % len(batches)])))
    return losses


def train_momos(model, batches, lr: float, n_steps: int, cfg: momos_state.MosaicConfig, seed: int, dict_lr_mult: float = 1.0) -> Tuple[List[float], integration.ModelBundle]:
    dict_opt = optax.adam(lr * dict_lr_mult)
    excl_opt = optax.adamw(lr)
    bundle = integration.init_bundle(model, cfg, dict_opt, excl_opt, jax.random.PRNGKey(seed))

    rng = jax.random.PRNGKey(seed + 1)
    losses = []
    for i in range(n_steps):
        rng, sub = jax.random.split(rng)
        bundle, loss, _swap = integration.step(
            bundle, batches[i % len(batches)], sub, loss_fn, dict_opt, excl_opt
        )
        losses.append(float(loss))
    return losses, bundle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    seed = 0
    d_model, num_layers, state_size, head_dim = 64, 4, 16, 16
    batch_size, max_length, n_batches_wanted = 8, 96, 48
    n_steps = 150
    lr = 3e-3
    S, K = 1, 4096

    batches, vocab_size, _pad_id = load_batches(
        n_samples=batch_size * n_batches_wanted, batch_size=batch_size, max_length=max_length, seed=seed
    )
    d_donor = 96
    model_kwargs = dict(
        d_model=d_model, num_hidden_layers=num_layers, state_size=state_size,
        head_dim=head_dim, chunk_size=16, projection_cache_dir=None, loss_chunk_tokens=128,
    )

    dense_model = build_model(vocab_size, d_donor, seed, **model_kwargs)
    n_params = sum(int(x.size) for x in jax.tree.leaves(nnx.state(dense_model, nnx.Param)))
    log.info("Model: %d trainable params (d_model=%d, layers=%d)", n_params, d_model, num_layers)

    t0 = time.perf_counter()
    dense_losses = train_dense(dense_model, batches, lr, n_steps)
    t_dense = time.perf_counter() - t0
    log.info("dense:  loss %.4f -> %.4f  (%.1fs)", dense_losses[0], dense_losses[-1], t_dense)

    momos_model = build_model(vocab_size, d_donor, seed, **model_kwargs)
    cfg = momos_state.MosaicConfig(S=S, K=K, scale_mode="none", subset_size=0)
    t0 = time.perf_counter()
    momos_losses, bundle = train_momos(momos_model, batches, lr, n_steps, cfg, seed, dict_lr_mult=1.0)
    t_momos = time.perf_counter() - t0
    log.info("momos:  loss %.4f -> %.4f  (%.1fs)", momos_losses[0], momos_losses[-1], t_momos)

    n_mosaicked = bundle.mosaic.layout.n_values
    bw, ratio = momos_state.bytes_per_weight(n_mosaicked, cfg, bundle.mosaic.layout.n_tensors)
    bw_inf = momos_state.asymptotic_bytes_per_weight(cfg)
    log.info(
        "ledger: %d/%d values mosaicked (S=%d K=%d) -> %.3f B/wt (%.1fx dense), asymptotic %.3f B/wt",
        n_mosaicked, n_params, S, K, bw, ratio, bw_inf,
    )

    log.info("-" * 70)
    log.info("Phase E gate (SPEC.md §8): loss curve vs dense, matched hyperparameters")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = min(n_steps - 1, int(frac * (n_steps - 1)))
        log.info("  step %3d:  dense=%.4f  momos=%.4f  ratio=%.3fx", i, dense_losses[i], momos_losses[i], momos_losses[i] / max(dense_losses[i], 1e-9))


if __name__ == "__main__":
    main()
