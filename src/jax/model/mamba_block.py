from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class Mamba2Config:
    """Mamba2 geometry, mirroring the subset of ``transformers.Mamba2Config``
    this project actually sets, plus the same validation.
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
        """Assemble the config, checking the one non-obvious constraint."""
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


@jax.remat  # [opt 4] recompute activations on backward rather than storing them
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
        d_residual = D.astype(jnp.float32)[..., None] * pad_by_size(hidden_states, pad_size)

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
    y_diag = jnp.einsum(                                         # [b, nc, c, h, p]
        'bqihn,bhqij,bqjhn,bqjhp->bqihp',
        C.astype(jnp.bfloat16), L.astype(jnp.bfloat16),        # [opt 3] bf16 GEMMs,
        B.astype(jnp.bfloat16), hidden_states.astype(jnp.bfloat16),  # fp32 accumulation
        preferred_element_type=jnp.float32,
    )

    # 2. Per-chunk state (right term of the low-rank off-diagonal factorization)
    decay_states = jnp.exp(a_cumsum[:, :, :, -1:] - a_cumsum)   # [b, h, nc, c]
    b_decay = B * jnp.transpose(decay_states, (0, 2, 3, 1))[..., None]  # [b, nc, c, h, n]
    states = jnp.einsum(                                          # [b, nc, h, p, n]
        'bqjhn,bqjhp->bqhpn',
        b_decay.astype(jnp.bfloat16), hidden_states.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )

    # 3. Inter-chunk recurrence — [opt 1] associative scan: O(NC log NC) vs O(NC²)
    # Each chunk is an affine map h[t] = decay[t]*h[t-1] + state[t].
    # Two affine maps compose as: (d_b, s_b) ∘ (d_a, s_a) = (d_b*d_a, d_b*s_a + s_b).
    chunk_decay = jnp.exp(a_cumsum[:, :, :, -1])                # [b, h, nc]

    def compose(a, b):
        d_a, s_a = a
        d_b, s_b = b
        return d_b * d_a, d_b[..., None, None] * s_a + s_b

    decay_scan = jnp.transpose(chunk_decay, (2, 0, 1))           # [nc, b, h]
    states_scan = jnp.transpose(states, (1, 0, 2, 3, 4))         # [nc, b, h, p, n]
    _, cumstates = jax.lax.associative_scan(compose, (decay_scan, states_scan))
    # Shift right: chunk t needs the state accumulated *before* it (chunks 0..t-1)
    cumstates = jnp.concatenate([jnp.zeros_like(cumstates[:1]), cumstates[:-1]], axis=0)
    states = jnp.transpose(cumstates, (1, 0, 2, 3, 4))           # [b, nc, h, p, n]

    # 4. State -> output per chunk (left term of the factorization)
    state_decay_out = jnp.exp(a_cumsum)                          # [b, h, nc, c]
    y_off = jnp.einsum(                                          # [b, nc, c, h, p]
        'bqihn,bqhpn->bqihp',
        C.astype(jnp.bfloat16), states.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    ) * jnp.transpose(state_decay_out, (0, 2, 3, 1))[..., None]

    output = (y_diag + y_off).reshape(batch, -1, num_heads, head_dim)
    if d_residual is not None:
        output = output + d_residual
    if pad_size > 0:
        output = output[:, :seq_len]
    return output


def apply_mask_to_padding_states(
    hidden_states: jnp.ndarray, attention_mask: jnp.ndarray | None
) -> jnp.ndarray:
    """Zero out padded positions — see state-spaces/mamba#66."""
    if attention_mask is None:
        return hidden_states
    return hidden_states * attention_mask[:, :, None].astype(hidden_states.dtype)
