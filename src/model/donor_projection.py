# -*- coding: utf-8 -*-
"""Frozen-donor embedding / LM-head projections.

The idea: a ~1B causal LM (DFM-Mimir) spends most of its parameters on two
tables, both ``[vocab_size, d_donor]`` = ``[262144, 1536]``:

* ``embed_tokens.weight``  — 402M params
* ``lm_head.weight``       — 402M params

Together that is ~805M of DFM-Mimir's ~1B. Whatever lexical and morphological
knowledge a low-resource-language model has is concentrated there, so we keep
those tables verbatim (frozen, never a leaf of the optimizer) and learn only a
low-rank *view* of them:

    E_small = E_donor @ P            P: [d_donor, d_model]   (786K params)
    W_small = W_donor @ Q            Q: [d_donor, d_model]   (786K params)

``P``/``Q`` are the only trainable tensors here. Backprop flows *through* the
frozen table values into ``P``/``Q``; the tables themselves receive no update.
Because the projection is re-applied on every forward pass, the small model's
effective embedding matrix is always the product of the current ``P`` and the
donor table — "recompute the embeddings after each optimizer step" is
structurally guaranteed rather than a step someone has to remember to run.

Efficiency notes (both are exact algebraic identities, not approximations):

* Embeddings gather first, project second: ``E_donor[ids] @ P`` touches only
  the ``B*T`` rows in the batch instead of materializing all 262144 of them.
  Rows outside the batch contribute nothing to ``dP`` either way.
* The head projects *up* instead of materializing ``W_small``::

      h @ (W_donor @ Q)ᵀ  ==  (h @ Qᵀ) @ W_donorᵀ

  The right-hand side costs a ``[B,T,d_model] -> [B,T,d_donor]`` matmul plus
  the donor's own head, and never allocates the ``[262144, d_model]`` product
  (or its gradient) at all.

``materialize_weight()`` on either module builds the concrete small table when
you actually want one — for exporting a standalone checkpoint.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from model import hf_compat  # noqa: F401  (imported for its side-effect patch)

log = logging.getLogger(__name__)

# Rows processed per chunk whenever we sweep a full donor table. A float32
# copy of a [262144, 1536] table is 1.6 GiB; a 16K-row chunk is 100 MiB.
_CHUNK_ROWS = 16384

# Process-level cache of donor tables, keyed by (model id, dtype). A K-fold
# sweep calls run_fold() once per fold (and once per fold *per Optuna trial*)
# in a single process; without this every one of those would re-load the full
# 1B donor just to read two matrices out of it.
_DONOR_CACHE: Dict[Tuple[str, torch.dtype], "DonorTables"] = {}


def _accum_device() -> str:
    """Where to run the full-table sweeps (Gram, RMS, materialization)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class DonorTables:
    """The two frozen ``[vocab_size, d_donor]`` tables lifted off the donor."""

    __slots__ = ("model_id", "embed", "head")

    def __init__(self, model_id: str, embed: torch.Tensor, head: torch.Tensor) -> None:
        self.model_id = model_id
        self.embed = embed
        self.head = head

    @property
    def vocab_size(self) -> int:
        return int(self.embed.shape[0])

    @property
    def d_donor(self) -> int:
        return int(self.embed.shape[1])


def load_donor_tables(
    model_id: str,
    trust_remote_code: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    use_cache: bool = True,
) -> DonorTables:
    """Load the donor's input-embedding and output-head weights.

    The donor is loaded to CPU, the two tables are cloned out of it, and the
    model is dropped immediately — nothing but the two matrices is retained.
    Results are memoised per process (see ``_DONOR_CACHE``).

    The tables are kept in the *training* dtype (bf16 by default, 805 MiB
    each) rather than float32: every float32 computation over them below
    upcasts chunk by chunk, so the extra 1.6 GiB of resident memory would buy
    nothing.
    """
    key = (model_id, dtype)
    if use_cache and key in _DONOR_CACHE:
        return _DONOR_CACHE[key]

    log.info("Loading donor tables from %s (once per process)", model_id)
    donor = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )
    with torch.no_grad():
        embed = donor.get_input_embeddings().weight.detach().to("cpu", dtype).clone()
        out_embed = donor.get_output_embeddings()
        # A tied-embedding donor has no separate head — the embedding table
        # *is* the head. DFM-Mimir is untied, so this is just defensiveness.
        head = (
            embed.clone()
            if out_embed is None
            else out_embed.weight.detach().to("cpu", dtype).clone()
        )

    del donor, out_embed
    gc.collect()

    tables = DonorTables(model_id, embed, head)
    log.info(
        "Donor tables: vocab=%d d_donor=%d (%.0fM frozen values per table)",
        tables.vocab_size,
        tables.d_donor,
        tables.vocab_size * tables.d_donor / 1e6,
    )
    if use_cache:
        _DONOR_CACHE[key] = tables
    return tables


# ---------------------------------------------------------------------------
# Chunked full-table math
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _full_precision_matmul():
    """Run the enclosed float32 matmuls at full precision, not TF32.

    ``train.py`` turns TF32 on process-wide, which is the right trade for
    training: 10 mantissa bits for a large speedup on every step. It is the
    wrong trade for the sweeps in this module. They run once, so the speed is
    irrelevant, and their outputs are not activations but the projection basis
    and the exported model's actual weight tables — where TF32 costs about
    three decimal digits (measured: 1.5e-2 absolute error against a float64
    reference, versus 1.3e-5 at full precision).
    """
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _iter_row_chunks(matrix: torch.Tensor, device: str):
    """Yield ``(start, chunk)`` row-slices of *matrix*, cast to float32 on *device*.

    Shared stepping logic for the three full-table sweeps below, which
    otherwise differ only in what they do with each chunk.
    """
    for start in range(0, matrix.shape[0], _CHUNK_ROWS):
        yield start, matrix[start : start + _CHUNK_ROWS].to(device, torch.float32)


def _gram(matrix: torch.Tensor) -> torch.Tensor:
    """``matrix.T @ matrix`` in float32, accumulated over row chunks."""
    device = _accum_device()
    dim = matrix.shape[1]
    gram = torch.zeros((dim, dim), dtype=torch.float32, device=device)
    with _full_precision_matmul():
        for _, chunk in _iter_row_chunks(matrix, device):
            gram += chunk.T @ chunk
            del chunk
    return gram.cpu()


def _projected_rms(matrix: torch.Tensor, projection: torch.Tensor) -> float:
    """RMS over all entries of ``matrix @ projection``, without materializing it."""
    device = _accum_device()
    proj = projection.to(device, torch.float32)
    total = torch.zeros((), dtype=torch.float64, device=device)
    with _full_precision_matmul():
        for _, chunk in _iter_row_chunks(matrix, device):
            total += (chunk @ proj).pow(2).sum().double()
            del chunk
    count = matrix.shape[0] * projection.shape[1]
    return float((total / count).sqrt().cpu())


def _project_table(
    matrix: torch.Tensor, projection: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """``matrix @ projection`` in float32, assembled chunk by chunk on CPU."""
    device = _accum_device()
    proj = projection.to(device, torch.float32)
    out = torch.empty(
        (matrix.shape[0], projection.shape[1]), dtype=out_dtype, device="cpu"
    )
    with _full_precision_matmul():
        for start, chunk in _iter_row_chunks(matrix, device):
            stop = start + chunk.shape[0]
            out[start:stop] = (chunk @ proj).to(out_dtype).cpu()
            del chunk
    return out


# ---------------------------------------------------------------------------
# Projection initialisation
# ---------------------------------------------------------------------------


def principal_basis(matrix: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Top-``out_dim`` (uncentered) principal directions of ``matrix``.

    Returns ``[d_donor, out_dim]`` with orthonormal columns: the subspace
    minimising ``||M - M V Vᵀ||_F``, i.e. the best rank-``out_dim`` linear
    compression of the donor table under squared error. Starting ``P``/``Q``
    here means the small model begins with the most donor information the
    target width can hold, and training only has to re-shape that subspace
    rather than discover one from noise.

    Uncentered on purpose: subtracting the column mean would make the map
    affine, and the construction is specified as a *linear* projection of the
    donor table.
    """
    gram = _gram(matrix)
    # eigh returns ascending eigenvalues; take the trailing (largest) block
    # and flip it so column 0 is the leading direction.
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
    basis = eigenvectors[:, -out_dim:].flip(-1).to(torch.float32).contiguous()
    kept = float(eigenvalues[-out_dim:].clamp_min(0).sum())
    total = float(eigenvalues.clamp_min(0).sum())
    log.info(
        "PCA basis %d -> %d retains %.1f%% of the donor table's energy",
        matrix.shape[1],
        out_dim,
        100.0 * kept / total if total > 0 else float("nan"),
    )
    return basis


def _cache_path(cache_dir: Path, model_id: str, table: str, out_dim: int) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
    return cache_dir / f"{slug}.{table}.pca{out_dim}.pt"


def cached_principal_basis(
    matrix: torch.Tensor,
    out_dim: int,
    model_id: str,
    table: str,
    cache_dir: Optional[str],
) -> torch.Tensor:
    """``principal_basis`` memoised to disk.

    The eigendecomposition depends only on (donor, table, out_dim), so it is
    identical across folds, trials, and runs — but it costs a full pass over a
    402M-value table each time. The basis itself is tiny (``1536 x 512``
    float32 = 3 MiB), so caching it is nearly free.
    """
    if cache_dir is None:
        return principal_basis(matrix, out_dim)

    path = _cache_path(Path(cache_dir), model_id, table, out_dim)
    if path.exists():
        basis = torch.load(path, map_location="cpu")
        if tuple(basis.shape) == (matrix.shape[1], out_dim):
            log.info("Reusing cached PCA basis %s", path)
            return basis
        log.warning("Ignoring stale PCA cache %s (shape %s)", path, tuple(basis.shape))

    basis = principal_basis(matrix, out_dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis, path)
    log.info("Cached PCA basis to %s", path)
    return basis


def _init_projection(
    matrix: torch.Tensor,
    out_dim: int,
    init: str,
    model_id: str,
    table: str,
    cache_dir: Optional[str],
) -> torch.Tensor:
    """Build the ``[d_donor, out_dim]`` projection matrix for one donor table."""
    in_dim = int(matrix.shape[1])
    if out_dim > in_dim:
        raise ValueError(
            f"d_model={out_dim} exceeds the donor width d_donor={in_dim} — "
            "this construction compresses the donor tables, it cannot widen them."
        )
    if init == "pca":
        return cached_principal_basis(matrix, out_dim, model_id, table, cache_dir)

    weight = torch.empty((in_dim, out_dim), dtype=torch.float32)
    if init == "orthogonal":
        nn.init.orthogonal_(weight)
    elif init == "xavier":
        nn.init.xavier_uniform_(weight)
    else:
        raise ValueError(
            f"Unknown projection init {init!r} — "
            "expected 'pca', 'orthogonal', or 'xavier'."
        )
    return weight


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class ProjectedEmbedding(nn.Module):
    """Drop-in ``nn.Embedding`` whose table is a learned view of a frozen one.

    ``forward(ids)`` returns ``scale * (E_donor[ids] @ P)``. Only ``P`` (and
    the scalar) are parameters; ``E_donor`` is a non-persistent buffer, so it
    never lands in a checkpoint (805 MiB saved per checkpoint) and is rebuilt
    from the donor on load.

    Args:
        donor: Frozen ``[vocab_size, d_donor]`` donor embedding table.
        out_dim: Width of the small model (``d_model``).
        projection: ``[d_donor, out_dim]`` initial ``P``.
        learn_scale: Add a learnable scalar gain, initialised so the projected
            embeddings start at unit RMS. This keeps the small model's first
            block seeing sanely-scaled activations regardless of how the donor
            scaled its own table (DFM-Mimir, for one, carries a separate
            ``embedding_scale`` of ~39 in its config). The scalar folds into
            ``P`` on export, so the map stays exactly linear.
    """

    def __init__(
        self,
        donor: torch.Tensor,
        out_dim: int,
        projection: torch.Tensor,
        learn_scale: bool = True,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(donor.shape[0])
        self.embedding_dim = int(out_dim)
        self.d_donor = int(donor.shape[1])

        self.register_buffer("donor", donor, persistent=False)
        # nn.Linear(in, out).weight is [out, in], so it holds P transposed;
        # forward() then computes donor_rows @ P as a single matmul.
        self.proj = nn.Linear(self.d_donor, out_dim, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(projection.T)

        self.scale: Optional[nn.Parameter] = None
        if learn_scale:
            rms = _projected_rms(donor, projection)
            self.scale = nn.Parameter(
                torch.tensor(1.0 / rms if rms > 0 else 1.0, dtype=torch.float32)
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Gather-then-project: identical to indexing the materialized small
        # table, but only the rows actually in the batch are touched.
        #
        # The explicit cast is what lets the donor table stay in the training
        # dtype (bf16, 805 MiB) while the trainable projection stays float32
        # under mixed precision — the two operands would otherwise disagree.
        rows = F.embedding(input_ids, self.donor).to(self.proj.weight.dtype)
        out = self.proj(rows)
        return out if self.scale is None else out * self.scale.to(out.dtype)

    @torch.no_grad()
    def materialize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The concrete ``[vocab_size, d_model]`` small embedding table.

        This is the "recompute the embeddings with the layer that just got
        updated" step made explicit — used when exporting a standalone model.
        """
        scale = 1.0 if self.scale is None else float(self.scale)
        return _project_table(self.donor, self.proj.weight.T * scale, dtype)

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"d_donor={self.d_donor}, frozen_donor=True"
        )


class ProjectedLMHead(nn.Module):
    """Output head that is a learned linear view of the donor's frozen head.

    Computes ``scale * (h @ Qᵀ) @ W_donorᵀ``, which is exactly
    ``h @ (W_donor @ Q)ᵀ`` — the small head ``W_donor @ Q`` — without ever
    allocating that ``[vocab_size, d_model]`` product or its gradient.

    Exposes ``.weight`` because ``Mamba2ForCausalLM.forward`` casts its hidden
    states with ``hidden.to(self.lm_head.weight.dtype)``. We return the
    trainable projection's weight: it is the first tensor the hidden states
    actually meet, so its dtype is the correct cast target.
    """

    def __init__(
        self,
        donor: torch.Tensor,
        in_dim: int,
        projection: torch.Tensor,
        learn_scale: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = int(donor.shape[0])
        self.d_donor = int(donor.shape[1])
        self.in_features = int(in_dim)
        self.out_features = self.vocab_size

        self.register_buffer("donor", donor, persistent=False)
        # nn.Linear(in_dim, d_donor).weight is [d_donor, in_dim] — exactly Q.
        self.proj = nn.Linear(in_dim, self.d_donor, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(projection)

        self.scale: Optional[nn.Parameter] = None
        if learn_scale:
            # Target logits of order 1 for a unit-RMS hidden state: such a
            # vector has norm sqrt(in_dim), so logits land around
            # sqrt(in_dim) * rms(W_small). Without this the donor head's own
            # scale puts the initial cross-entropy in the hundreds and the
            # first few hundred steps go entirely into undoing that.
            rms = _projected_rms(donor, projection)
            denom = rms * math.sqrt(in_dim)
            self.scale = nn.Parameter(
                torch.tensor(1.0 / denom if denom > 0 else 1.0, dtype=torch.float32)
            )

    @property
    def weight(self) -> torch.Tensor:
        return self.proj.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Up-project into donor space first, then reuse the donor head. The
        # cast goes toward the donor's dtype, not away from it: casting a
        # [B, T, d_donor] activation is free next to casting a
        # [vocab_size, d_donor] table, and the donor dtype is the training
        # dtype anyway.
        up = self.proj(hidden_states.to(self.proj.weight.dtype))
        logits = F.linear(up.to(self.donor.dtype), self.donor)
        return logits if self.scale is None else logits * self.scale.to(logits.dtype)

    @torch.no_grad()
    def materialize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The concrete ``[vocab_size, d_model]`` small head, ``W_donor @ Q``."""
        scale = 1.0 if self.scale is None else float(self.scale)
        return _project_table(self.donor, self.proj.weight * scale, dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"d_donor={self.d_donor}, frozen_donor=True"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_projections(
    tables: DonorTables,
    d_model: int,
    init: str = "pca",
    learn_scales: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[ProjectedEmbedding, ProjectedLMHead]:
    """Build the embedding and head projections off one set of donor tables.

    The two get independent projections: DFM-Mimir is untied
    (``tie_word_embeddings: false``), so its input table and its output head
    span genuinely different subspaces and there is nothing to gain by forcing
    ``P == Q``.
    """
    embed_basis = _init_projection(
        tables.embed, d_model, init, tables.model_id, "embed", cache_dir
    )
    head_basis = _init_projection(
        tables.head, d_model, init, tables.model_id, "head", cache_dir
    )
    return (
        ProjectedEmbedding(tables.embed, d_model, embed_basis, learn_scale=learn_scales),
        ProjectedLMHead(tables.head, d_model, head_basis, learn_scale=learn_scales),
    )
