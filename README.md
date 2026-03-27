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

---

## LLM Fine-tuning (LoRA/QLoRA)

This project also supports fine-tuning large language models like **Qwen3.5-0.8B** using parameter-efficient methods (LoRA/QLoRA).

### Installation

The required dependencies are included in the main `pyproject.toml`. If installing manually:

```bash
uv sync
```

### Quick Start

**Fine-tune Qwen3.5-0.8B on your data:**

```bash
python src/train_qwen_finetune.py \
    --data_path data/instructions.json \
    --model_name unsloth/Qwen3.5-0.8B-Q8_0 \
    --lora_rank 8 \
    --learning_rate 2e-4 \
    --epochs 3
```

### Data Formats

The fine-tuning script supports multiple data formats:

**1. Instruction Format (JSON/JSONL):**
```json
[
  {
    "instruction": "Translate to French",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  },
  {
    "instruction": "Summarize the text",
    "input": "Long text here...",
    "output": "Brief summary..."
  }
]
```

**2. Chat Format:**
```json
[
  {
    "messages": [
      {"role": "user", "content": "What is AI?"},
      {"role": "assistant", "content": "Artificial Intelligence is..."}
    ]
  }
]
```

**3. Simple Text (TXT/JSONL):**
```
One text sample per line
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `unsloth/Qwen3.5-0.8B-Q8_0` | HuggingFace model to fine-tune |
| `--data_path` | (required) | Path to training data |
| `--format_type` | `instruction` | Data format: `instruction`, `chat`, `text` |
| `--lora_rank` | `8` | LoRA rank (higher = more parameters) |
| `--lora_alpha` | `16` | LoRA alpha scaling |
| `--learning_rate` | `2e-4` | Learning rate |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |
| `--max_length` | `512` | Maximum sequence length |
| `--use_quantization` | `True` | Use 4-bit quantization (QLoRA) |
| `--use_lora` | `True` | Use LoRA adapters |

### Memory Requirements

| Method | GPU Memory (approx.) |
|--------|---------------------|
| QLoRA (4-bit) | ~6 GB |
| LoRA (8-bit) | ~10 GB |
| Full fine-tuning | ~20+ GB |

### Inference

**Single prompt:**
```bash
python src/run_qwen_finetuned_inference.py \
    --model_dir outputs/finetuned_model \
    --prompt "What is machine learning?" \
    --max_new_tokens 256
```

**Interactive mode:**
```bash
python src/run_qwen_finetuned_inference.py \
    --model_dir outputs/finetuned_model \
    --interactive
```

**Batch inference:**
```bash
python src/run_qwen_finetuned_inference.py \
    --model_dir outputs/finetuned_model \
    --input_file prompts.txt \
    --output_file generations.json
```

### Example: Fine-tuning for Question Answering

1. **Prepare your data** (`data/qa_dataset.json`):
```json
[
  {
    "instruction": "Answer the question based on the context.",
    "input": "Context: Paris is the capital of France. Question: What is the capital of France?",
    "output": "The capital of France is Paris."
  }
]
```

2. **Train the model:**
```bash
python src/train_qwen_finetune.py \
    --data_path data/qa_dataset.json \
    --format_type instruction \
    --lora_rank 16 \
    --learning_rate 1e-4 \
    --epochs 5 \
    --batch_size 4 \
    --run_name qa_finetuned
```

3. **Run inference:**
```bash
python src/run_qwen_finetuned_inference.py \
    --model_dir outputs/finetuned_qa_finetuned \
    --prompt "Context: London is the capital of UK. Question: What is the capital of UK?" \
    --interactive
```

### Output Structure

After training, the output directory contains:
```
outputs/finetuned_model_YYYYMMDD_HHMMSS/
├── adapter_model/       # LoRA adapter weights
├── tokenizer/           # Tokenizer files
├── checkpoints/         # Training checkpoints
├── finetune_config.json # Training configuration
└── lightning_logs/      # Training logs
```

### Limitations

- QLoRA uses quantized models which may have slightly reduced quality compared to full fine-tuning
- The 0.8B parameter model has limited capacity for complex tasks
- For best results, use high-quality, task-specific training data
