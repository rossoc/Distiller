#!/usr/bin/env python3
"""
Embedding to Text Inference Script

This script generates text from embeddings using a trained EmbeddingDecoder model.
It takes text input, encodes it using EmbeddingGemma, and then decodes the
embedding back to text using a Gemma decoder.

Usage:
    python scripts/embed_to_text.py \
        --checkpoint outputs/decoder_checkpoint/best_model.pt \
        --prompt "The future of artificial intelligence" \
        --max-new-tokens 100 \
        --temperature 0.8
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate text from embeddings using EmbeddingDecoder"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained decoder checkpoint"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world!",
        help="Text to encode and then decode"
    )
    
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="google/embeddinggemma-300m",
        help="Encoder model name"
    )
    
    parser.add_argument(
        "--decoder-model",
        type=str,
        default="google/gemma-2-2b",
        help="Base decoder model name"
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    encoder_model_name: str,
    decoder_model_name: str,
    device: str
):
    """Load the trained decoder model and encoder."""
    from model.embedding_decoder import EmbeddingDecoderModel, EmbeddingDecoderConfig, EmbeddingEncoderWrapper
    
    print(f"Loading models...", file=sys.stderr)
    
    # Load config from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get('config', EmbeddingDecoderConfig(
        encoder_model_name=encoder_model_name,
        decoder_model_name=decoder_model_name,
    ))
    
    # Create and load decoder model
    decoder_model = EmbeddingDecoderModel(config)
    decoder_model.load_state_dict(checkpoint['model_state_dict'])
    decoder_model.to(device)
    decoder_model.eval()
    
    # Load encoder
    encoder = EmbeddingEncoderWrapper(config.encoder_model_name, device=device)
    
    # Get tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)
    
    print(f"Models loaded successfully!", file=sys.stderr)
    
    return decoder_model, encoder, tokenizer, config


def generate_text(
    decoder_model,
    encoder,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    num_samples: int = 1,
    device: str = "cpu",
) -> List[str]:
    """
    Generate text from a prompt.
    
    The process:
    1. Encode the prompt to an embedding using EmbeddingGemma
    2. Use the decoder to generate text from the embedding
    3. Decode generated tokens to text
    """
    # Step 1: Encode prompt to embedding
    with torch.no_grad():
        embedding = encoder.encode([prompt])[0]  # Shape: (embedding_dim,)
    
    # Step 2: Generate from embedding
    with torch.no_grad():
        # Repeat embedding for multiple samples
        input_embeddings = embedding.unsqueeze(0).repeat(num_samples, 1)
        input_embeddings = input_embeddings.to(device)
        
        # Get EOS token ID
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id or 0
        
        # Generate
        generated_ids = decoder_model.generate(
            input_embeddings=input_embeddings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    
    # Step 3: Decode generated tokens to text
    generated_texts = []
    for i in range(num_samples):
        sample_ids = generated_ids[i]
        # Remove padding and special tokens
        text = tokenizer.decode(
            sample_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        generated_texts.append(text)
    
    return generated_texts


def main():
    args = get_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load models
    decoder_model, encoder, tokenizer, config = load_model(
        checkpoint_path=str(checkpoint_path),
        encoder_model_name=args.encoder_model,
        decoder_model_name=args.decoder_model,
        device=args.device,
    )
    
    # Print configuration
    if args.verbose:
        print(f"\nConfiguration:", file=sys.stderr)
        print(f"  Encoder: {config.encoder_model_name}", file=sys.stderr)
        print(f"  Decoder: {config.decoder_model_name}", file=sys.stderr)
        print(f"  Device: {args.device}", file=sys.stderr)
        print(f"  Temperature: {args.temperature}", file=sys.stderr)
        print(f"  Top-p: {args.top_p}", file=sys.stderr)
        print(f"  Max new tokens: {args.max_new_tokens}", file=sys.stderr)
        print(file=sys.stderr)
    
    # Generate text
    print(f"Input: {args.prompt}", file=sys.stderr)
    print(f"\nGenerating {args.num_samples} sample(s)...", file=sys.stderr)
    
    generated_texts = generate_text(
        decoder_model=decoder_model,
        encoder=encoder,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_samples=args.num_samples,
        device=args.device,
    )
    
    # Output results
    print("\n" + "="*60)
    if args.num_samples == 1:
        print(f"Generated text:")
        print(generated_texts[0])
    else:
        for i, text in enumerate(generated_texts, 1):
            print(f"\nSample {i}:")
            print(text)
    print("="*60)


if __name__ == "__main__":
    main()
