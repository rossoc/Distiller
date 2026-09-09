# Distiller

Normalized span extraction from low-resource (Danish) technical text, and an
experiment in doing it with a model ~40x smaller than the one that knows the
language.

## Problem

Standard Named Entity Recognition assumes the entity appears verbatim in the
source text. This task instead requires the entity to be **normalized**:

- a plural surface form must be returned in its singular canonical form,
- a quantity written in words ("treogtyve") must be returned as digits,
- abbreviations and morphological variants must collapse onto one label.

Source and target strings differ in surface form while sharing semantics, so
purely extractive tagging is insufficient. Distiller treats each (row, field)
pair as a conditional generation problem: given the source text and a field
marker, emit the normalized value.

## Two models

| | `model=default` | `model=mamba2` |
|---|---|---|
| Architecture | DFM-Mimir (HRM, attention) | Mamba2 (SSM, recurrent) |
| Sequence mixing | attention, `H_cycles x L_cycles` recurrent unroll | linear-time selective scan |
| Trainable params | ~1.05B (full fine-tune) | **21.4M** |
| Deployed params | ~1.05B | **288M** |
| Embeddings | learned | frozen DFM-Mimir table, learned 1536→512 projection |
| Output head | learned | frozen DFM-Mimir head, learned 1536→512 projection |

`model=default` is the baseline: full fine-tuning of
[DFM-Mimir](https://huggingface.co/danish-foundation-models/DFM-Mimir).
Everything below is about the second one.

## Approach: keep the embeddings, shrink everything else

DFM-Mimir spends most of itself on two tables, both `[262144, 1536]`:

```
embed_tokens.weight   402M params  ┐
lm_head.weight        402M params  ┘  805M of ~1.05B
```

That is where a model's knowledge of a low-resource language actually lives —
which morphemes exist, which surface forms are the same word, how numbers are
spelled. The 16 transformer blocks on top are general machinery that a much
smaller network can supply, especially for a task that only has to *find* a
few keywords rather than reason about them.

So: keep both tables verbatim, frozen, and learn only a low-rank view of them.

```
                    FROZEN (805M values, never a checkpoint tensor)
                 ┌──────────────────┐            ┌──────────────────┐
                 │  E_donor         │            │  W_donor         │
                 │  [262144, 1536]  │            │  [262144, 1536]  │
                 └────────┬─────────┘            └────────▲─────────┘
                          │ gather                        │
   input_ids ────────────▶│ rows[ids]                     │
                          ▼                               │
                 ┌──────────────────┐            ┌────────┴─────────┐
                 │  P [1536, 512]   │            │  Q [1536, 512]   │   TRAINED
                 │  786K trainable  │            │  786K trainable  │   (1.6M)
                 └────────┬─────────┘            └────────▲─────────┘
                          ▼                               │
                 ┌────────────────────────────────────────┴─────────┐
                 │      Mamba2 backbone — 12 SSM blocks, d=512      │   TRAINED
                 │      19.8M trainable, conventional backprop      │   (19.8M)
                 └──────────────────────────────────────────────────┘
```

Formally the small model's tables are

```
E_small = E_donor @ P      W_small = W_donor @ Q
```

Gradients flow *through* the frozen table values into `P` and `Q`; the tables
themselves receive no update and are not optimizer leaves. Because the
projection is re-applied on every forward pass, the effective embedding matrix
is always the product of the *current* `P` with the donor table — recomputing
the embeddings after each optimizer step is structural, not a step anyone has
to remember. `export_standalone()` collapses the product into concrete
`[262144, 512]` tables when you want a self-contained model.

Two identities keep this cheap (both exact, neither an approximation):

- **Embeddings gather first, project second.** `E_donor[ids] @ P` touches only
  the `B*T` rows in the batch. Rows outside the batch contribute nothing to
  `dP` either way, so the gradient is identical to materializing all 262144
  rows first — there is a test for exactly that.
- **The head projects up, not out.** `h @ (W_donor @ Q)ᵀ == (h @ Qᵀ) @ W_donorᵀ`.
  The right-hand side costs one `[B,T,512] → [B,T,1536]` matmul plus the
  donor's own head, and never allocates the `[262144, 512]` product or its
  gradient.

### Initialisation

`P` and `Q` start at each donor table's leading principal subspace (uncentered
PCA, so the map stays linear) — the best rank-512 linear compression of that
table under squared error. At 1536 → 512 that retains **60%** of the embedding
table's energy and **88.5%** of the head's, so training starts from the most
donor information 512 dimensions can hold and only has to re-shape the
subspace. The bases are cached to disk (3 MiB each), because the
eigendecomposition depends only on (donor, table, width) and is otherwise
recomputed on every fold of every trial.

Each projection also carries one learnable scalar gain, initialised so the
projected embeddings start at unit RMS and the initial logits at O(1). Without
it the donor head's own scale puts the starting cross-entropy in the hundreds.
The scalar folds into the projection on export, so the map stays exactly
linear.

### Why an RNN

The task is keyword extraction, not reasoning, so the sequence mixer only has
to carry a small amount of state forward. Mamba2's selective scan is
linear-time in sequence length and has no KV cache to grow, and at `d_model`
512 each block is ~9x cheaper than a 1536-wide one.

### The vocabulary is the memory bottleneck

Reusing the donor's head means a 262144-way softmax. One-shot float32 logits
for an 8x256 batch are 2 GiB before the backward pass needs their gradient,
which OOMs a 15 GiB card long before the 21M-parameter body does.

So the head is applied `loss_chunk_tokens` positions at a time, each chunk
wrapped in `torch.utils.checkpoint`: the forward pass keeps only hidden
states, and logits are recomputed and freed one chunk at a time during
backward. Peak logit memory becomes a function of the chunk size rather than
the batch size. It is the same mean-over-unmasked-tokens cross-entropy on the
same one-position shift — value and every gradient match HF's one-shot loss to
float32 rounding, which the test suite checks at three chunk sizes.

## Running it

```bash
# K-fold CV
python src/cv.py model=mamba2 training=mamba2

# Optuna search (K-fold CV per trial)
python src/optuna_search.py model=mamba2 training=mamba2 optuna.n_trials=50

# Override anything via Hydra
python src/cv.py model=mamba2 training=mamba2 model.d_model=768 training.batch_size=8

# The DFM-Mimir baseline is still the default
python src/cv.py
```

Evaluation on the held-out test split (never touched during CV):

```bash
python src/predict.py model=mamba2 training=mamba2 \
    predict.mode=evaluate \
    predict.checkpoint_dir=outputs/fold_0/best.ckpt
```

`save_pretrained` writes two things to one directory: a stock
`Mamba2ForCausalLM` with the projections already collapsed into concrete
tables (loadable with plain `transformers`, no donor needed), and
`donor_projections.pt` holding `P`/`Q` so training can be resumed.

## Performance notes

`mamba-ssm` and `causal-conv1d` are **not** required, but without them
transformers falls back to a pure-PyTorch selective scan that materializes
several `[batch, seq, chunk_size, heads, state]` float32 tensors per layer.
That fallback, not the model size, sets the batch size this trains at. The
defaults in `src/config/model/mamba2.yaml` are chosen for it; measured on a
15 GiB card at batch 16 x seq 256:

| `state_size` | `chunk_size` | grad ckpt | s/step | peak VRAM |
|---|---|---|---|---|
| 128 | 32 | on | 2.38 | 6.2 GiB |
| **64** | **32** | **on** | **1.56** | **4.9 GiB** |
| 64 | 32 | off | 1.31 | 8.4 GiB |
| 64 | 64 | on | 1.90 | 4.9 GiB |
| 128 | 128 | either | OOM | — |

`state_size` is the dominant knob; `chunk_size` is the one to lower first if
something OOMs. Installing the fused kernels (`pip install mamba-ssm
causal-conv1d`, needs `nvcc`) removes the whole table.

## Repository layout

```
src/
├── cv.py                      # K-fold CV entrypoint
├── optuna_search.py           # hyper-parameter search (CV per trial)
├── train.py                   # run_fold: one fold, model-agnostic
├── predict.py                 # held-out evaluation / generation
├── lit_datamodule.py          # splits, K-fold indices, tokenization, loaders
├── callback.py                # checkpointing, Optuna pruning, epoch heartbeat
├── utils.py                   # seeding, device, DataLoader runtime
├── model/
│   ├── factory.py             # cfg.model.kind -> LightningModule
│   ├── dfm_mimir.py           # baseline: full DFM-Mimir fine-tune
│   ├── mimir_mamba2.py        # Mamba2 + frozen-donor projections
│   ├── donor_projection.py    # the frozen tables and their learned views
│   └── hf_compat.py           # transformers version shims
├── data/loader.py             # XLSX -> samples, splits, K-fold indices
└── config/                    # Hydra config groups (untracked; see below)
```

## Setup

```bash
uv sync
```

`src/config/` and the root `config.yaml` are excluded from version control via
`.git/info/exclude`, so a fresh clone has no config tree. `src/config/` needs
these groups: `data`, `model`, `training`, `optuna`, `predict`, `wandb`,
`runtime`, plus the top-level `config.yaml` that wires them together and
instantiates the Lightning `Trainer`.

The donor model is downloaded from the Hub on first use (or point
`model.donor_model_id` at a local directory).
