<h1 style="text-align:center;"> Distiller </h1>

## Embedding Decoder: Text-to-Embeddings-to-Text

This project includes an **Embedding Decoder** module that enables generating text from embeddings. The architecture uses EmbeddingGemma as an encoder and a Gemma decoder with a trainable projection layer.

### Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────┐     ┌──────────┐
│   Input     │────▶│ EmbeddingGemma   │────▶│  Projection     │────▶│  Gemma   │────▶│  Output  │
│    Text     │     │    (Encoder)     │     │     Layer       │     │ Decoder  │     │   Text   │
│             │     │                  │     │                 │     │          │     │          │
│             │     │ 768-dim embedding│     │768→2304 projection│   │          │     │          │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────┘     └──────────┘
```

### Installation

```bash
uv sync
```

### Training

Train the projection layer to map EmbeddingGemma embeddings to Gemma decoder input space:

```bash
python src/train_decoder.py \
    --data_path data/training_texts.txt \
    --output_dir outputs/decoder_checkpoint \
    --encoder_model "google/embeddinggemma-300m" \
    --decoder_model "google/gemma-2-2b" \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4
```

**Data format**: One text per line in your training file.

### Inference

Generate text from embeddings:

```bash
python scripts/embed_to_text.py \
    --checkpoint outputs/decoder_checkpoint/best_model.pt \
    --prompt "The future of artificial intelligence" \
    --max-new-tokens 100 \
    --temperature 0.8 \
    --num-samples 3
```

### Key Components

| File | Description |
|------|-------------|
| `src/model/embedding_decoder.py` | Core model architecture with `EmbeddingDecoderModel` and `EmbeddingEncoderWrapper` |
| `src/train_decoder.py` | Training pipeline with dataset, collator, and training loop |
| `scripts/embed_to_text.py` | Inference script for generating text from embeddings |

### How It Works

1. **Encoding**: Input text is encoded to a 768-dimensional embedding using EmbeddingGemma
2. **Projection**: The embedding is projected to the decoder's hidden space (2304-dim for Gemma-2-2B) via a trainable MLP
3. **Decoding**: The Gemma decoder autoregressively generates tokens from the projected embedding

### Training Details

- **Objective**: Causal language modeling - predict text continuation given prefix embeddings
- **Frozen components**: EmbeddingGemma encoder and Gemma decoder weights are frozen
- **Trainable components**: Only the projection layer is trained
- **Pooling**: Uses sentence-level embeddings (mean-pooled) from the encoder

### Limitations

- The decoder generates text conditioned on the **semantic meaning** of the input embedding, not a word-for-word reconstruction
- Quality depends on training data diversity and size
- Currently uses sentence-level embeddings; token-level embeddings would enable more precise conditioning

### Model Variants

| Decoder | Hidden Size | Parameters | Use Case |
|---------|-------------|------------|----------|
| Gemma-2-2B | 2304 | ~2.6B | Balanced quality/speed |
| Gemma-2-9B | 3584 | ~9.4B | Higher quality generation |
| Llama-3-8B | 4096 | ~8B | Alternative architecture |

To use a different decoder, change the `--decoder_model` argument during training.
