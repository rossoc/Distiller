# Distiller

A latent-space sequence-to-sequence model for **normalized span extraction**
from low-resource text.

## Problem

Standard Named Entity Recognition assumes the entity appears verbatim in the
source text. Many real-world tasks instead require the entity to be
**normalized**:

- a plural surface form must be returned in its singular canonical form,
- a quantity written in words ("twenty-three") must be returned as digits,
- abbreviations and morphological variants must be collapsed onto a single
  reference label.

Because the source and target strings differ in surface form while sharing
semantics, purely extractive tagging is insufficient. Distiller treats the
problem as **sequence-to-sequence in embedding space**, then projects back to
text via nearest-neighbour retrieval over a closed label vocabulary.

## Approach 1 — Latent-Space Decoder (primary)

```
                   ┌──────────────────┐   ┌───────────────┐   ┌──────────────┐
  source text ───▶ │ frozen embedding │──▶│  Transformer  │──▶│  predicted   │
                   │     encoder      │   │    decoder    │   │  embeddings  │
                   └──────────────────┘   └───────────────┘   └──────┬───────┘
                                                                     │
                                                              ┌──────▼───────┐
                                target text  ◀─── FAISS NN ───│ mean-pooled  │
                                                              └──────────────┘
```

A frozen encoder (EmbeddingGemma 300M or Qwen3.5 0.8B, both quantized and run
on CPU via `llama-cpp`) produces token-level embeddings of the source. A
Transformer decoder is trained from scratch to map these onto the embeddings
of the canonical target. At inference time, the predicted vector is matched
against a FAISS index of all candidate labels.

### Components

| File | Role |
|------|------|
| `src/model/encoder.py` | Selects between Gemma / Qwen GGUF encoders; token-level pooling (`LLAMA_POOLING_TYPE_NONE`). |
| `src/model/decoder.py` | `nn.TransformerDecoder` (configurable depth / heads / FFN), Xavier init. |
| `src/model/diffusion_trainer.py` | Lightning module. Combined **MSE + cosine** loss (`DiffusionLoss`), OneCycleLR warmup, two-step iterative refinement inside `training_step`. |
| `src/model/faiss_retriever.py` | `IndexFlatIP` over L2-normalised, mean-pooled target embeddings (inner product ≡ cosine). |
| `src/data/datamodule.py`, `src/data/dataset.py` | Pre-computes encoder embeddings once, pads variable-length sequences via `embedding_collate_fn`. |
| `src/data/split_dataset.py` | Reproducible train / eval / test splits. |
| `src/optuna_search.py` | Optuna HPO (TPE sampler, Median pruner, Lightning integration). |

### Training

```bash
python src/train_latent_space.py \
    --encoder gemma \
    --num_layers 6 \
    --fwd_dim 2048 \
    --num_heads 8 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --loss_alpha 0.5
```

Each run writes weights, config, and W&B logs to a timestamped folder under
`outputs/`.

### Inference

The FAISS index is built once over the target vocabulary, then reused:

```bash
# 1. build the FAISS index from the test-set targets
python src/test_latent_space.py \
    --model_path outputs/<run>/best_model.pt \
    --build_index

# 2. run retrieval-based inference
python src/test_latent_space.py \
    --model_path outputs/<run>/best_model.pt \
    --k 3
```

## Approach 2 — LoRA Fine-Tuning (secondary experiment)

A direct text-to-text baseline was implemented for comparison: 4-bit QLoRA
fine-tuning of Qwen3.5-0.8B with Unsloth and TRL's `SFTTrainer`
(`src/model/qwen_finetuner_v2.py`, entrypoints `src/train_lora.py` and
`src/test_lora.py`).

This approach **underperformed** the latent-space decoder on the target
dataset. The most likely cause is that a sub-billion-parameter base model
carries weak coverage of the low-resource language used in the corpus: its
token embeddings do not encode enough lexical or morphological signal for the
LoRA adapters to learn the normalization mapping reliably with the available
data. The code is kept in-repo as a documented baseline.

## Repository layout

```
src/
├── train_latent_space.py     # primary training entrypoint
├── test_latent_space.py      # FAISS-based inference
├── train_lora.py             # LoRA baseline training
├── test_lora.py              # LoRA baseline inference
├── optuna_search.py          # hyper-parameter search
├── model/                    # decoder, encoder, trainer, retriever, LoRA wrapper
├── data/                     # data module, dataset, split, schema utilities
└── util/                     # logging, seeding, callbacks, result analysis

notebooks/
├── data-exploration.ipynb    # dataset statistics and sanity checks
├── finetuning-qwen.ipynb     # LoRA experiment walkthrough
└── model-usage.ipynb         # loading a checkpoint and running retrieval
```

## Setup

```bash
uv sync
```

The encoders are loaded from local GGUF files referenced in
`src/model/encoder.py`:

```
models/embeddinggemma-300M-Q8.gguf
models/Qwen3.5-0.8B-Q8_0.gguf
```

Place the quantised weights at those paths before running training or
inference.
