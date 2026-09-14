# -*- coding: utf-8 -*-
"""Frozen-donor embedding / LM-head projections, in Flax NNX.

A direct port of ``model.donor_projection`` (PyTorch). The construction is
unchanged — see that module's docstring for the *why*; this one documents only
what differs in JAX.

    E_small = E_donor @ P            P: [d_donor, d_model]
    W_small = W_donor @ Q            Q: [d_donor, d_model]

Three things are genuinely different from the PyTorch version:

1. **Freezing is a type, not a flag.** PyTorch marks the donor tables frozen
   twice over (``register_buffer(..., persistent=False)`` + no ``requires_grad``).
   Here they live in a :class:`Donor` variable, an ``nnx.Variable`` subclass
   that is deliberately *not* ``nnx.Param`` — so ``nnx.state(model, nnx.Param)``
   excludes them from the optax optimizer's pytree and from the orbax
   checkpoint by construction, with no filtering anywhere. Gradients still flow
   *through* the table values into ``P``/``Q``; the tables themselves are never
   a differentiation target because they are never in the differentiated state.

2. **Parameter layout follows the maths, not ``nn.Linear``.** PyTorch stores
   these inside ``nn.Linear`` modules, whose ``.weight`` is ``[out, in]`` — so
   ``ProjectedEmbedding.proj.weight`` is Pᵀ while ``ProjectedLMHead.proj.weight``
   is Q, an asymmetry that exists only because of how ``nn.Linear`` is shaped.
   Here both are stored as the mathematical ``[d_donor, d_model]``.
   :func:`p_from_torch_embedding` / :func:`q_from_torch_head` convert, and are
   what the cross-framework parity tests transplant weights through.

3. **The PCA cache is ``.npz``, in its own file.** The eigendecomposition is
   framework-independent, but the existing PyTorch cache is ``torch.save``d
   ``.pt``. Rather than change the format on the PyTorch path (which is
   working, and which this port deliberately does not touch), the JAX path
   writes ``{slug}.{table}.pca{out_dim}.npz`` beside it, under the same cache
   key. Both coexist in one ``projection_cache_dir``.

The Gram accumulation and the eigendecomposition run in NumPy, not JAX: the
PyTorch path accumulates the Gram in float32 and then eigendecomposes in
float64, and JAX silently truncates float64 to float32 unless ``jax_enable_x64``
is set process-wide. Setting a global flag to match one init-time computation
would be the tail wagging the dog — this is a once-per-(donor, table, d_model)
cost that happens outside ``jax.jit`` anyway, so NumPy is both exact and
simpler. The result is bit-comparable with the PyTorch path's basis.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

log = logging.getLogger(__name__)

# Rows processed per chunk whenever a full donor table is swept. Mirrors the
# PyTorch path's _CHUNK_ROWS so the two produce identical chunk boundaries
# (and therefore identical float32 accumulation order) in the Gram.
_CHUNK_ROWS = 16384


class Donor(nnx.Variable):
    """A frozen donor table.

    An ``nnx.Variable`` that is pointedly not an ``nnx.Param``. Everything that
    walks trainable state does so via ``nnx.state(model, nnx.Param)``, so a
    ``Donor`` is invisible to the optimizer, to the checkpointer, and to
    ``nnx.grad`` — the single mechanism standing in for PyTorch's
    ``persistent=False`` buffer *plus* ``requires_grad=False``.
    """


class DonorTables:
    """The two frozen ``[vocab_size, d_donor]`` tables lifted off the donor."""

    __slots__ = ("model_id", "embed", "head")

    def __init__(self, model_id: str, embed: jnp.ndarray, head: jnp.ndarray) -> None:
        self.model_id = model_id
        self.embed = embed
        self.head = head

    @property
    def vocab_size(self) -> int:
        return int(self.embed.shape[0])

    @property
    def d_donor(self) -> int:
        return int(self.embed.shape[1])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_DONOR_CACHE: dict[tuple[str, str], DonorTables] = {}


def _reinterpret_bfloat16(raw: np.ndarray) -> np.ndarray:
    """Recover ``ml_dtypes.bfloat16`` arrays ``np.savez`` mis-tagged as void.

    ``scripts/convert_donor_checkpoint.py`` saves bf16 tables via
    ``ml_dtypes.bfloat16`` so they survive the ``.npz`` round-trip without
    NumPy silently upcasting to fp32 (per that script's own comment) — but
    ``np.savez``/``np.load`` do not actually preserve that dtype's tag: the
    array round-trips as raw ``|V2`` void bytes, a generic 2-byte blob NumPy
    has no registered cast *from* (hence ``jnp.asarray(..., dtype=bfloat16)``
    failing with "No cast function available", not a dtype mismatch in the
    data itself). The bytes are exactly right, only the header's dtype tag
    is lost, so this reinterprets rather than casts — a zero-copy ``.view``,
    not a conversion. Every other dtype this pipeline saves (fp16, fp32) is
    NumPy-native and round-trips with its tag intact, so this only fires for
    the one case that needs it.
    """
    if raw.dtype.kind == "V" and raw.itemsize == 2:
        import ml_dtypes

        return raw.view(ml_dtypes.bfloat16)
    return raw


def load_donor_tables(
    donor_dir: str,
    dtype: jnp.dtype = jnp.bfloat16,
    use_cache: bool = True,
) -> DonorTables:
    """Load donor tables from a converted ``.npz`` directory.

    Unlike the PyTorch path, this never touches ``transformers`` — the donor is
    a custom ``trust_remote_code`` Hub model with no Flax-native counterpart, so
    it is converted once by ``scripts/convert_donor_checkpoint.py`` and read
    back here as plain arrays. That keeps ``torch``/``transformers`` off the JAX
    training critical path entirely.
    """
    key = (str(donor_dir), jnp.dtype(dtype).name)
    if use_cache and key in _DONOR_CACHE:
        return _DONOR_CACHE[key]

    path = Path(donor_dir)
    npz = path / "donor_tables.npz" if path.is_dir() else path
    if not npz.exists():
        raise FileNotFoundError(
            f"No converted donor tables at {npz}. Run:\n"
            f"  python scripts/convert_donor_checkpoint.py "
            f"--donor-model-id <hub-id> --out {path}"
        )

    with np.load(npz, allow_pickle=False) as blob:
        model_id = str(blob["model_id"]) if "model_id" in blob else str(path)
        embed = jnp.asarray(_reinterpret_bfloat16(blob["embed"]), dtype=dtype)
        head = jnp.asarray(_reinterpret_bfloat16(blob["head"]), dtype=dtype)

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
# Chunked full-table math (init-time, outside jit)
# ---------------------------------------------------------------------------


def _iter_row_chunks(matrix, chunk_rows: int = _CHUNK_ROWS):
    """Yield ``(start, chunk)`` float32 row-slices of *matrix*."""
    n = int(matrix.shape[0])
    for start in range(0, n, chunk_rows):
        yield start, np.asarray(matrix[start : start + chunk_rows], dtype=np.float32)


def _gram(matrix) -> np.ndarray:
    """``matrix.T @ matrix`` in float32, accumulated over row chunks."""
    dim = int(matrix.shape[1])
    gram = np.zeros((dim, dim), dtype=np.float32)
    for _, chunk in _iter_row_chunks(matrix):
        gram += chunk.T @ chunk
    return gram


def _projected_rms(matrix, projection) -> float:
    """RMS over all entries of ``matrix @ projection``, without materializing it."""
    proj = np.asarray(projection, dtype=np.float32)
    total = np.zeros((), dtype=np.float64)
    for _, chunk in _iter_row_chunks(matrix):
        total += np.square(chunk @ proj, dtype=np.float32).sum(dtype=np.float64)
    count = int(matrix.shape[0]) * int(proj.shape[1])
    return float(np.sqrt(total / count))


def _project_table(matrix, projection, out_dtype) -> jnp.ndarray:
    """``matrix @ projection`` in float32, assembled chunk by chunk."""
    proj = np.asarray(projection, dtype=np.float32)
    out = np.empty((int(matrix.shape[0]), int(proj.shape[1])), dtype=np.float32)
    for start, chunk in _iter_row_chunks(matrix):
        out[start : start + chunk.shape[0]] = chunk @ proj
    return jnp.asarray(out, dtype=out_dtype)


# ---------------------------------------------------------------------------
# Projection initialisation
# ---------------------------------------------------------------------------


def principal_basis(matrix, out_dim: int) -> np.ndarray:
    """Top-``out_dim`` (uncentered) principal directions of ``matrix``.

    Returns ``[d_donor, out_dim]`` with orthonormal columns. Uncentered on
    purpose — see the PyTorch docstring.

    ``numpy.linalg.eigh`` returns ascending eigenvalues, the same convention as
    ``torch.linalg.eigh``, so the "take the trailing (largest) block and flip"
    slice is identical. Note that eigenvector *sign* is arbitrary and differs
    between implementations (and between runs on different BLAS builds), so any
    comparison against another basis must be sign-invariant — compare the
    subspace (e.g. ``|V1ᵀ V2|``, or reconstruction error), never the entries.
    """
    gram = _gram(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(gram.astype(np.float64))
    basis = np.ascontiguousarray(
        eigenvectors[:, -out_dim:][:, ::-1].astype(np.float32)
    )
    kept = float(np.clip(eigenvalues[-out_dim:], 0.0, None).sum())
    total = float(np.clip(eigenvalues, 0.0, None).sum())
    log.info(
        "PCA basis %d -> %d retains %.1f%% of the donor table's energy",
        matrix.shape[1],
        out_dim,
        100.0 * kept / total if total > 0 else float("nan"),
    )
    return basis


def _cache_path(cache_dir: Path, model_id: str, table: str, out_dim: int) -> Path:
    """Cache filename. Same key as the PyTorch path, different extension.

    ``.npz`` rather than ``.pt`` so the JAX cache sits beside the PyTorch one in
    a shared ``projection_cache_dir`` without either needing to read the other's
    format (and without this port having to modify the PyTorch module).
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
    return cache_dir / f"{slug}.{table}.pca{out_dim}.npz"


def cached_principal_basis(
    matrix,
    out_dim: int,
    model_id: str,
    table: str,
    cache_dir: str | None,
) -> np.ndarray:
    """``principal_basis`` memoised to disk as ``.npz``."""
    if cache_dir is None:
        return principal_basis(matrix, out_dim)

    path = _cache_path(Path(cache_dir), model_id, table, out_dim)
    if path.exists():
        with np.load(path, allow_pickle=False) as blob:
            basis = blob["basis"]
        if tuple(basis.shape) == (int(matrix.shape[1]), out_dim):
            log.info("Reusing cached PCA basis %s", path)
            return basis
        log.warning("Ignoring stale PCA cache %s (shape %s)", path, tuple(basis.shape))

    basis = principal_basis(matrix, out_dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, basis=basis)
    log.info("Cached PCA basis to %s", path)
    return basis


def _init_projection(
    matrix,
    out_dim: int,
    init: str,
    model_id: str,
    table: str,
    cache_dir: Optional[str],
    rngs: nnx.Rngs,
) -> jnp.ndarray:
    """Build the ``[d_donor, out_dim]`` projection matrix for one donor table."""
    in_dim = int(matrix.shape[1])
    if out_dim > in_dim:
        raise ValueError(
            f"d_model={out_dim} exceeds the donor width d_donor={in_dim} — "
            "this construction compresses the donor tables, it cannot widen them."
        )
    if init == "pca":
        return jnp.asarray(
            cached_principal_basis(matrix, out_dim, model_id, table, cache_dir),
            dtype=jnp.float32,
        )

    key = rngs.params()
    if init == "orthogonal":
        return jax.nn.initializers.orthogonal()(key, (in_dim, out_dim), jnp.float32)
    if init == "xavier":
        return jax.nn.initializers.glorot_uniform()(key, (in_dim, out_dim), jnp.float32)
    raise ValueError(
        f"Unknown projection init {init!r} — expected 'pca', 'orthogonal', or 'xavier'."
    )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class ProjectedEmbedding(nnx.Module):
    """Embedding whose table is a learned view of a frozen donor table.

    ``__call__(ids)`` returns ``scale * (E_donor[ids] @ P)``. Only ``P`` (and
    the optional scalar) are ``nnx.Param``; the donor is a :class:`Donor`.
    """

    def __init__(
        self,
        donor: jnp.ndarray,
        out_dim: int,
        projection: jnp.ndarray,
        learn_scale: bool = True,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.num_embeddings = int(donor.shape[0])
        self.embedding_dim = int(out_dim)
        self.d_donor = int(donor.shape[1])
        self.param_dtype = param_dtype

        self.donor = Donor(donor)
        # Stored as the mathematical P: [d_donor, d_model]. The PyTorch module
        # holds Pᵀ only because nn.Linear.weight is [out, in].
        self.P = nnx.Param(jnp.asarray(projection, dtype=param_dtype))

        # Assigned in one statement: flax 0.12 fixes an attribute's data/static
        # status on first assignment, so pre-declaring `scale = None` and
        # filling in a Param later is rejected.
        if learn_scale:
            rms = _projected_rms(donor, projection)
            self.scale = nnx.Param(
                jnp.asarray(1.0 / rms if rms > 0 else 1.0, dtype=param_dtype)
            )
        else:
            self.scale = None

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        # Gather-then-project: identical to indexing the materialized small
        # table, but only the rows actually in the batch are touched. JAX's
        # advanced indexing has the same gather semantics as F.embedding.
        rows = self.donor.get_value()[input_ids].astype(self.P.get_value().dtype)
        out = rows @ self.P.get_value()
        if self.scale is None:
            return out
        return out * self.scale.get_value().astype(out.dtype)

    def materialize_weight(self, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
        """The concrete ``[vocab_size, d_model]`` small embedding table."""
        scale = 1.0 if self.scale is None else float(self.scale.get_value())
        return _project_table(self.donor.get_value(), np.asarray(self.P.get_value()) * scale, dtype)


class ProjectedLMHead(nnx.Module):
    """Output head that is a learned linear view of the donor's frozen head.

    Computes ``scale * (h @ Qᵀ) @ W_donorᵀ``, which is exactly
    ``h @ (W_donor @ Q)ᵀ`` — without ever allocating that
    ``[vocab_size, d_model]`` product or its gradient.
    """

    def __init__(
        self,
        donor: jnp.ndarray,
        in_dim: int,
        projection: jnp.ndarray,
        learn_scale: bool = True,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.vocab_size = int(donor.shape[0])
        self.d_donor = int(donor.shape[1])
        self.in_features = int(in_dim)
        self.out_features = self.vocab_size
        self.param_dtype = param_dtype

        self.donor = Donor(donor)
        # Q as [d_donor, d_model] — here that happens to coincide with what
        # nn.Linear(in_dim, d_donor).weight holds on the PyTorch side.
        self.Q = nnx.Param(jnp.asarray(projection, dtype=param_dtype))

        if learn_scale:
            rms = _projected_rms(donor, projection)
            denom = rms * math.sqrt(in_dim)
            self.scale = nnx.Param(
                jnp.asarray(1.0 / denom if denom > 0 else 1.0, dtype=param_dtype)
            )
        else:
            self.scale = None

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        up = hidden_states.astype(self.Q.get_value().dtype) @ self.Q.get_value().T
        logits = up.astype(self.donor.get_value().dtype) @ self.donor.get_value().T
        if self.scale is None:
            return logits
        return logits * self.scale.get_value().astype(logits.dtype)

    def materialize_weight(self, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
        """The concrete ``[vocab_size, d_model]`` small head, ``W_donor @ Q``."""
        scale = 1.0 if self.scale is None else float(self.scale.get_value())
        return _project_table(self.donor.get_value(), np.asarray(self.Q.get_value()) * scale, dtype)


# ---------------------------------------------------------------------------
# Cross-framework weight conversion (used by the parity tests and by export)
# ---------------------------------------------------------------------------


def p_from_torch_embedding(torch_proj_weight) -> jnp.ndarray:
    """``ProjectedEmbedding.proj.weight`` (PyTorch, ``[d_model, d_donor]`` = Pᵀ) -> P."""
    return jnp.asarray(np.asarray(torch_proj_weight, dtype=np.float32).T)


def q_from_torch_head(torch_proj_weight) -> jnp.ndarray:
    """``ProjectedLMHead.proj.weight`` (PyTorch, ``[d_donor, d_model]`` = Q) -> Q."""
    return jnp.asarray(np.asarray(torch_proj_weight, dtype=np.float32))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_projections(
    tables: DonorTables,
    d_model: int,
    rngs: nnx.Rngs,
    init: str = "pca",
    learn_scales: bool = True,
    cache_dir: str | None = None,
    param_dtype: jnp.dtype = jnp.float32,
) -> Tuple[ProjectedEmbedding, ProjectedLMHead]:
    """Build the embedding and head projections off one set of donor tables."""
    embed_basis = _init_projection(
        tables.embed, d_model, init, tables.model_id, "embed", cache_dir, rngs
    )
    head_basis = _init_projection(
        tables.head, d_model, init, tables.model_id, "head", cache_dir, rngs
    )
    return (
        ProjectedEmbedding(
            tables.embed,
            d_model,
            embed_basis,
            learn_scale=learn_scales,
            param_dtype=param_dtype,
        ),
        ProjectedLMHead(
            tables.head,
            d_model,
            head_basis,
            learn_scale=learn_scales,
            param_dtype=param_dtype,
        ),
    )
