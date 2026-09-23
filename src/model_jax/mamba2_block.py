# -*- coding: utf-8 -*-
"""Hand-rolled Mamba2 mixer and block, in Flax NNX.

Owned outright by this project — no external Mamba dependency of any kind.
Written against two references, read rather than imported:

* ``transformers.models.mamba2.modeling_mamba2`` as installed here (5.16.x).
  Note that the ``torch_forward``/``cuda_kernels_forward`` split older
  transformers had is **gone**: the mixer now has a single ``forward`` that
  calls module-level functions decorated
  ``@use_kernel_func_from_hub_with_fallback``, whose *bodies* are the pure
  PyTorch path. The one that matters here is ``mamba2_chunk_scan`` — with
  ``segment_sum``/``reshape_into_chunks``/``pad_tensor_by_size`` as helpers.
  That function is this file's line-by-line numerical reference, because it is
  what the project actually runs (``kernels``/``mamba-ssm``/``causal-conv1d``
  are all absent, and ``tests/conftest.py`` pins the fallback deliberately).
* ``jax-ml/bonsai``'s Mamba2, consulted for how a Flax-NNX implementation
  structures ``nnx.Conv(feature_group_count=...)`` and the chunked scan.

**The scan is chunked, not sequential.** The original plan proposed starting
with a sequential ``lax.scan`` recurrence and treating the chunked form as a
later performance upgrade. Three things argue against that ordering, and this
implementation takes the chunked form directly:

1. The PyTorch reference *is* chunked. A sequential port would have to be
   diffed against a differently-shaped algorithm, so any mismatch mixes
   "my recurrence is wrong" with "the two formulations disagree". Porting the
   same algorithm makes the parity gate a direct comparison.
2. Reverse-mode through a sequential ``lax.scan`` retains every step's SSM
   carry. At this project's production geometry (batch 4, 16 heads x 64
   head_dim x 64 state, seq 512, 12 layers) that is ~0.5 GB of float32
   residuals per layer, ~6 GB for the stack — before activations, before the
   loss. It would OOM on the box this is meant to run on.
3. ``chunk_size`` is a live Optuna search dimension
   (``src/config/optuna/mamba2.yaml`` sweeps ``[128, 256]``) and the PyTorch
   config documents it as "the SSM scan's block length". A sequential scan
   would make it a no-op, silently turning a searched axis into dead weight.
   Keeping the chunked form keeps the config honest across both backends.

Everything in here is shape-static, so it traces once per input shape under
``jax.jit``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

# jax.checkpoint's `static_argnames` is a no-op on the legacy remat path;
# mamba2_chunk_scan below relies on it (chunk_size/dt_softplus/dt_limit are
# passed as keywords, so static_argnums can't reach them either).
jax.config.update("jax_remat3", True)


@dataclass(frozen=True)
class Mamba2Config:
    """Mamba2 geometry, mirroring the subset of ``transformers.Mamba2Config``
    this project actually sets, plus the same validation.

    ``num_heads`` is derived, not configured — see :meth:`create`.
    """

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_heads: int
    head_dim: int
    state_size: int
    expand: int
    n_groups: int
    conv_kernel: int
    chunk_size: int
    initializer_range: float = 0.1
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = False
    use_conv_bias: bool = True
    residual_in_fp32: bool = True
    rescale_prenorm_residual: bool = True
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: tuple[float, float] = (0.0, float("inf"))

    @property
    def intermediate_size(self) -> int:
        return self.expand * self.hidden_size

    @property
    def conv_dim(self) -> int:
        return self.intermediate_size + 2 * self.n_groups * self.state_size

    @classmethod
    def create(
        cls,
        vocab_size: int,
        d_model: int,
        num_hidden_layers: int,
        state_size: int,
        expand: int,
        head_dim: int,
        n_groups: int,
        conv_kernel: int,
        chunk_size: int,
        initializer_range: float,
        **kwargs,
    ) -> "Mamba2Config":
        """Assemble the config, checking the one non-obvious constraint.

        Mamba2 splits its ``expand * hidden_size`` inner width into
        ``num_heads`` heads of ``head_dim`` each and derives ``num_heads`` from
        that division. A non-integer result fails deep inside the mixer with a
        shape error that says nothing about which knob is wrong, so catch it
        here — same message as the PyTorch path's ``_build_config``.
        """
        inner = expand * d_model
        if inner % head_dim != 0:
            raise ValueError(
                f"expand*d_model = {expand}*{d_model} = {inner} is not divisible "
                f"by head_dim={head_dim}; Mamba2 needs an integer head count. "
                f"Pick a head_dim that divides {inner}."
            )
        return cls(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=num_hidden_layers,
            num_heads=inner // head_dim,
            head_dim=head_dim,
            state_size=state_size,
            expand=expand,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
            initializer_range=initializer_range,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Chunked scan helpers — ports of modeling_mamba2.py's module-level functions
# ---------------------------------------------------------------------------


def pad_by_size(x: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Zero-pad *x* by ``pad_size`` at the end of the sequence axis (dim 1)."""
    if pad_size == 0:
        return x
    pads = [(0, 0)] * x.ndim
    pads[1] = (0, pad_size)
    return jnp.pad(x, pads)


def reshape_into_chunks(x: jnp.ndarray, pad_size: int, chunk_size: int) -> jnp.ndarray:
    """Pad on the sequence axis and split it into ``chunk_size``-long chunks."""
    x = pad_by_size(x, pad_size)
    return x.reshape(x.shape[0], -1, chunk_size, *x.shape[2:])


def segment_sum(x: jnp.ndarray) -> jnp.ndarray:
    """Lower-triangular cumulative sum"""
    chunk = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((chunk, chunk), dtype=bool))
    return jnp.where(mask, x_segsum, -jnp.inf)


@functools.partial(  # [opt 4] recompute activations on backward rather than storing them
    jax.checkpoint, static_argnames=("chunk_size", "dt_softplus", "dt_limit")
)
def mamba2_chunk_scan(
    hidden_states: jnp.ndarray,  # [b, l, h, p]
    dt: jnp.ndarray,  # [b, l, h]
    A: jnp.ndarray,  # [h]
    B: jnp.ndarray,  # [b, l, g, n]
    C: jnp.ndarray,  # [b, l, g, n]
    chunk_size: int,
    D: jnp.ndarray | None = None,  # [h]
    dt_bias: jnp.ndarray | None = None,  # [h]
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> jnp.ndarray:
    """The SSD (state-space duality) chunked scan. Returns [b, l, h, p]"""
    batch, seq_len, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[2]

    if dt_bias is not None:
        dt = dt + dt_bias.astype(dt.dtype)
    if dt_softplus:
        dt = jax.nn.softplus(dt)
    dt = jnp.clip(dt, dt_limit[0], dt_limit[1])

    hidden_states = hidden_states.astype(jnp.float32)
    dt = dt.astype(jnp.float32)
    # At n_groups == 1 (the only value this project uses) this broadcasts the
    # single B/C group across every head.
    repeat = num_heads // num_groups
    B = jnp.repeat(B.astype(jnp.float32), repeat, axis=2)
    C = jnp.repeat(C.astype(jnp.float32), repeat, axis=2)

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    d_residual = None
    if D is not None:
        d_residual = D.astype(jnp.float32)[..., None] * pad_by_size(
            hidden_states, pad_size
        )

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None]
    A = A.astype(jnp.float32) * dt

    hidden_states, A, B, C = (
        reshape_into_chunks(t, pad_size, chunk_size) for t in (hidden_states, A, B, C)
    )

    A = jnp.transpose(A, (0, 3, 1, 2))  # [b, h, nc, c]
    a_cumsum = jnp.cumsum(A, axis=-1)

    # 1. Intra-chunk outputs — [opt 2] fused einsum avoids two [b,nc,c,c,h] intermediates (G, M)
    L = jnp.exp(segment_sum(A))  # [b, h, nc, c, c]
    y_diag = jnp.einsum(  # [b, nc, c, h, p]
        "bqihn,bhqij,bqjhn,bqjhp->bqihp",
        C,
        L,
        B,
        hidden_states,
        preferred_element_type=jnp.float32,
    )

    # 2. Per-chunk state (right term of the low-rank off-diagonal factorization)
    decay_states = jnp.exp(a_cumsum[:, :, :, -1:] - a_cumsum)  # [b, h, nc, c]
    b_decay = (
        B * jnp.transpose(decay_states, (0, 2, 3, 1))[..., None]
    )  # [b, nc, c, h, n]
    states = jnp.einsum(  # [b, nc, h, p, n]
        "bqjhn,bqjhp->bqhpn",
        b_decay,
        hidden_states,
        preferred_element_type=jnp.float32,
    )

    # 3. Inter-chunk recurrence — [opt 1] associative scan: O(NC log NC) vs O(NC²)
    # Each chunk is an affine map h[t] = decay[t]*h[t-1] + state[t].
    # Two affine maps compose as: (d_b, s_b) ∘ (d_a, s_a) = (d_b*d_a, d_b*s_a + s_b).
    chunk_decay = jnp.exp(a_cumsum[:, :, :, -1])  # [b, h, nc]

    def compose(a, b):
        d_a, s_a = a
        d_b, s_b = b
        return d_b * d_a, d_b[..., None, None] * s_a + s_b

    decay_scan = jnp.transpose(chunk_decay, (2, 0, 1))  # [nc, b, h]
    states_scan = jnp.transpose(states, (1, 0, 2, 3, 4))  # [nc, b, h, p, n]
    _, cumstates = jax.lax.associative_scan(compose, (decay_scan, states_scan))
    # Shift right: chunk t needs the state accumulated *before* it (chunks 0..t-1)
    cumstates = jnp.concatenate([jnp.zeros_like(cumstates[:1]), cumstates[:-1]], axis=0)
    states = jnp.transpose(cumstates, (1, 0, 2, 3, 4))  # [b, nc, h, p, n]

    # 4. State -> output per chunk (left term of the factorization)
    state_decay_out = jnp.exp(a_cumsum)  # [b, h, nc, c]
    y_off = (
        jnp.einsum(  # [b, nc, c, h, p]
            "bqihn,bqhpn->bqihp",
            C,
            states,
            preferred_element_type=jnp.float32,
        )
        * jnp.transpose(state_decay_out, (0, 2, 3, 1))[..., None]
    )

    output = (y_diag + y_off).reshape(batch, -1, num_heads, head_dim)
    if d_residual is not None:
        output = output + d_residual
    if pad_size > 0:
        output = output[:, :seq_len]
    return output


def apply_mask_to_padding_states(
    hidden_states: jnp.ndarray, attention_mask: Optional[jnp.ndarray]
) -> jnp.ndarray:
    """Zero out padded positions — see state-spaces/mamba#66."""
    if attention_mask is None:
        return hidden_states
    return hidden_states * attention_mask[:, :, None].astype(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class Mamba2RMSNorm(nnx.Module):
    """Plain RMSNorm, upcasting to float32 for the reduction."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, param_dtype=jnp.float32):
        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=param_dtype))
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return self.weight.get_value() * x.astype(in_dtype)


class MambaRMSNormGated(nnx.Module):
    """RMSNorm with a SiLU gate applied *before* the reduction (norm_before_gate=False)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, param_dtype=jnp.float32):
        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=param_dtype))
        self.eps = eps

    def __call__(
        self, x: jnp.ndarray, gate: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        in_dtype = x.dtype
        x = x.astype(jnp.float32)
        if gate is not None:
            x = x * jax.nn.silu(gate.astype(jnp.float32))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return self.weight.get_value() * x.astype(in_dtype)


# ---------------------------------------------------------------------------
# Mixer
# ---------------------------------------------------------------------------


def _dt_bias_init(key, num_heads: int, cfg: Mamba2Config) -> jnp.ndarray:
    """Inverse-softplus of a log-uniform timestep, as HF's ``init_mamba2_weights``."""
    u = jax.random.uniform(key, (num_heads,), dtype=jnp.float32)
    dt = jnp.exp(
        u * (math.log(cfg.time_step_max) - math.log(cfg.time_step_min))
        + math.log(cfg.time_step_min)
    )
    dt = jnp.clip(dt, cfg.time_step_floor, None)
    # Inverse of softplus (pytorch/pytorch#72759).
    return dt + jnp.log(-jnp.expm1(-dt))


class Mamba2Mixer(nnx.Module):
    """One Mamba2 sequence mixer: gated projection -> causal conv -> SSD scan -> out.

    Parameter names follow HF's (``in_proj``, ``conv1d``, ``A_log``, ``D``,
    ``dt_bias``, ``norm``, ``out_proj``) so a PyTorch<->JAX transplant is a
    name lookup. Shapes are JAX-natural (``[in, out]`` for dense kernels rather
    than torch's ``[out, in]``); :func:`model_jax.mamba2_backbone.load_from_torch`
    owns the transposition.
    """

    def __init__(
        self,
        cfg: Mamba2Config,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
    ):
        self.cfg = cfg
        self.layer_idx = layer_idx

        inner = cfg.intermediate_size
        conv_dim = cfg.conv_dim
        proj_size = inner + conv_dim + cfg.num_heads

        # HF initializes Linear weights with its PreTrainedModel default
        # (normal, std=initializer_range) and conv1d/out_proj with
        # kaiming_uniform(a=sqrt(5)) — which for these shapes is
        # uniform(±1/sqrt(fan_in)).
        k_in, k_conv, k_out, k_dt = jax.random.split(rngs.params(), 4)

        self.in_proj = nnx.Param(
            jax.random.normal(k_in, (cfg.hidden_size, proj_size), jnp.float32)
            * cfg.initializer_range
        )
        self.in_proj_bias = (
            nnx.Param(jnp.zeros((proj_size,), jnp.float32)) if cfg.use_bias else None
        )

        # Causal depthwise conv. `feature_group_count=conv_dim` makes it
        # depthwise; left-only padding makes it causal, so (unlike torch's
        # symmetric `padding=K-1` plus a `[:, :, :L]` slice) no trim is needed.
        conv_bound = 1.0 / math.sqrt(cfg.conv_kernel)
        self.conv1d = nnx.Conv(
            in_features=conv_dim,
            out_features=conv_dim,
            kernel_size=(cfg.conv_kernel,),
            feature_group_count=conv_dim,
            padding=[(cfg.conv_kernel - 1, 0)],
            use_bias=cfg.use_conv_bias,
            kernel_init=lambda key, shape, dtype: jax.random.uniform(
                key, shape, dtype, -conv_bound, conv_bound
            ),
            bias_init=lambda key, shape, dtype: jnp.zeros(shape, dtype),
            param_dtype=param_dtype,
            rngs=nnx.Rngs(params=k_conv),
        )

        # S4D-real initialization; these are not discretized.
        self.A_log = nnx.Param(
            jnp.log(jnp.arange(1, cfg.num_heads + 1, dtype=jnp.float32))
        )
        self.D = nnx.Param(jnp.ones((cfg.num_heads,), jnp.float32))
        self.dt_bias = nnx.Param(_dt_bias_init(k_dt, cfg.num_heads, cfg))

        self.norm = MambaRMSNormGated(
            inner, eps=cfg.layer_norm_epsilon, param_dtype=param_dtype
        )

        out_bound = 1.0 / math.sqrt(inner)
        out_w = jax.random.uniform(
            k_out, (inner, cfg.hidden_size), jnp.float32, -out_bound, out_bound
        )
        if cfg.rescale_prenorm_residual:
            # GPT-2 residual scaling: 1/sqrt(N) for N residual layers.
            out_w = out_w / math.sqrt(cfg.num_hidden_layers)
        self.out_proj = nnx.Param(out_w)
        self.out_proj_bias = (
            nnx.Param(jnp.zeros((cfg.hidden_size,), jnp.float32))
            if cfg.use_bias
            else None
        )

    def __call__(
        self, hidden_states: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        cfg = self.cfg
        dtype = hidden_states.dtype
        inner, conv_dim = cfg.intermediate_size, cfg.conv_dim

        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected = hidden_states @ self.in_proj.get_value().astype(dtype)
        if self.in_proj_bias is not None:
            projected = projected + self.in_proj_bias.get_value().astype(dtype)

        A = -jnp.exp(self.A_log.get_value().astype(jnp.float32))

        gate = projected[..., :inner]
        hidden_states_B_C = projected[..., inner : inner + conv_dim]
        dt = projected[..., inner + conv_dim :]

        # 2. Convolution sequence transformation (causal, depthwise)
        hidden_states_B_C = jax.nn.silu(self.conv1d(hidden_states_B_C))

        # 3. SSM transformation
        hidden_states_B_C = apply_mask_to_padding_states(
            hidden_states_B_C, attention_mask
        )
        gn = cfg.n_groups * cfg.state_size
        x = hidden_states_B_C[..., :inner]
        B = hidden_states_B_C[..., inner : inner + gn]
        C = hidden_states_B_C[..., inner + gn : inner + 2 * gn]

        batch, seq_len = x.shape[0], x.shape[1]
        scan_output = mamba2_chunk_scan(
            x.reshape(batch, seq_len, cfg.num_heads, cfg.head_dim),
            dt,
            A,
            B.reshape(batch, seq_len, cfg.n_groups, cfg.state_size),
            C.reshape(batch, seq_len, cfg.n_groups, cfg.state_size),
            chunk_size=cfg.chunk_size,
            D=self.D.get_value(),
            dt_bias=self.dt_bias.get_value(),
            dt_softplus=True,
            dt_limit=cfg.time_step_limit,
        )
        scan_output = scan_output.reshape(batch, seq_len, -1)
        scan_output = self.norm(scan_output, gate)

        # 4. Final linear projection
        out = scan_output.astype(dtype) @ self.out_proj.get_value().astype(dtype)
        if self.out_proj_bias is not None:
            out = out + self.out_proj_bias.get_value().astype(dtype)
        return out


class Mamba2Block(nnx.Module):
    """Pre-norm residual block: ``h + mixer(norm(h))``."""

    def __init__(
        self,
        cfg: Mamba2Config,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
    ):
        self.cfg = cfg
        self.norm = Mamba2RMSNorm(
            cfg.hidden_size, eps=cfg.layer_norm_epsilon, param_dtype=param_dtype
        )
        self.mixer = Mamba2Mixer(cfg, layer_idx, rngs=rngs, param_dtype=param_dtype)

    def __call__(
        self, hidden_states: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.cfg.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)
        return residual + hidden_states
