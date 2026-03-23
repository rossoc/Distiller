#!/usr/bin/env python3
"""
Direct Embedding to Text Inference Script

This script generates text directly from pre-computed embeddings.
Unlike embed_to_text.py, this script accepts embedding vectors as input
(not text), making it suitable for using stored/embedded embeddings.

Usage:
    # From a numpy file containing embeddings
    python scripts/embedding_to_text.py \
        --checkpoint outputs/decoder_checkpoint/best_model.pt \
        --embeddings-path embeddings.npy \
        --max-new-tokens 100

    # From a single embedding vector
    python scripts/embedding_to_text.py \
        --checkpoint outputs/decoder_checkpoint/best_model.pt \
        --embedding-vector "[0.1, 0.2, 0.3, ...]" \
        --max-new-tokens 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate text directly from pre-computed embeddings"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained decoder checkpoint"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    input_group.add_argument(
        "--embeddings-path",
        type=str,
        help="Path to .npy or .pt file containing embeddings (shape: [num_embeddings, embedding_dim])"
    )
    
    input_group.add_argument(
        "--embedding-vector",
        type=str,
        help="Single embedding as a comma-separated list of floats, e.g., '0.1,0.2,0.3,...'"
    )
    
    input_group.add_argument(
        "--embedding-json",
        type=str,
        help="Path to JSON file containing embedding(s)"
    )
    
    # Generation parameters
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
        help="Number of samples to generate per embedding"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save generated texts (optional, prints to stdout if not provided)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    return parser.parse_args()


def load_embeddings(
    embeddings_path: Optional[str] = None,
    embedding_vector: Optional[str] = None,
    embedding_json: Optional[str] = None,
) -> torch.Tensor:
    """Load embeddings from various sources."""
    
    if embeddings_path is not None:
        path = Path(embeddings_path)
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        
        if path.suffix == ".npy":
            embeddings = np.load(path)
        elif path.suffix == ".pt":
            embeddings = torch.load(path, weights_only=True)
            if isinstance(embeddings, dict):
                embeddings = embeddings.get("embeddings", embeddings.get("embedding"))
            embeddings = embeddings.cpu().numpy()
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
    elif embedding_vector is not None:
        # Parse comma-separated floats
        values = [float(x.strip()) for x in embedding_vector.split(",")]
        embeddings = torch.tensor([values], dtype=torch.float32)
        
    elif embedding_json is not None:
        with open(embedding_json, "r") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                # Multiple embeddings
                embeddings = torch.tensor(data, dtype=torch.float32)
            else:
                # Single embedding
                embeddings = torch.tensor([data], dtype=torch.float32)
        elif isinstance(data, dict):
            embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
        else:
            raise ValueError("Invalid JSON format for embeddings")
    
    # Ensure 2D shape
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    
    return embeddings


def load_model(
    checkpoint_path: str,
    device: str
):
    """Load the trained decoder model and encoder."""
    from model.embedding_decoder import EmbeddingDecoderModel, EmbeddingDecoderConfig
    
    print(f"Loading decoder model...", file=sys.stderr)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get('config', EmbeddingDecoderConfig())
    
    # Create and load decoder model
    decoder_model = EmbeddingDecoderModel(config)
    decoder_model.load_state_dict(checkpoint['model_state_dict'])
    decoder_model.to(device)
    decoder_model.eval()
    
    # Get tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)
    
    print(f"Model loaded successfully!", file=sys.stderr)
    
    return decoder_model, tokenizer, config


def generate_from_embeddings(
    decoder_model,
    tokenizer,
    embeddings: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    num_samples: int = 1,
    device: str = "cpu",
) -> List[List[str]]:
    """
    Generate text from pre-computed embeddings.
    
    Args:
        decoder_model: The trained decoder model
        tokenizer: Tokenizer for decoding generated tokens
        embeddings: Tensor of shape (num_embeddings, embedding_dim)
        max_new_tokens: Maximum tokens to generate per sample
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty
        num_samples: Number of samples to generate per embedding
        device: Device to run on
    
    Returns:
        List of lists containing generated texts
    """
    embeddings = embeddings.to(device)
    num_embeddings = embeddings.shape[0]
    
    all_generated_texts = []
    
    with torch.no_grad():
        for i in range(num_embeddings):
            embedding = embeddings[i:i+1]  # Shape: (1, embedding_dim)
            
            # Repeat for multiple samples
            input_embeddings = embedding.repeat(num_samples, 1)
            
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
            
            # Decode to text
            generated_texts = []
            for j in range(num_samples):
                sample_ids = generated_ids[j]
                text = tokenizer.decode(
                    sample_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                generated_texts.append(text)
            
            all_generated_texts.append(generated_texts)
    
    return all_generated_texts


def main():
    args = get_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load embeddings
    print(f"Loading embeddings...", file=sys.stderr)
    try:
        embeddings = load_embeddings(
            embeddings_path=args.embeddings_path,
            embedding_vector=args.embedding_vector,
            embedding_json=args.embedding_json,
        )
        print(f"Loaded {embeddings.shape[0]} embedding(s) with dimension {embeddings.shape[1]}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading embeddings: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load model
    decoder_model, tokenizer, config = load_model(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
    )
    
    # Print configuration
    if args.verbose:
        print(f"\nConfiguration:", file=sys.stderr)
        print(f"  Decoder: {config.decoder_model_name}", file=sys.stderr)
        print(f"  Embedding dim: {config.embedding_dim}", file=sys.stderr)
        print(f"  Device: {args.device}", file=sys.stderr)
        print(f"  Temperature: {args.temperature}", file=sys.stderr)
        print(f"  Top-p: {args.top_p}", file=sys.stderr)
        print(f"  Max new tokens: {args.max_new_tokens}", file=sys.stderr)
        print(file=sys.stderr)
    
    # Generate text
    print(f"Generating text from {embeddings.shape[0]} embedding(s)...", file=sys.stderr)
    
    all_results = generate_from_embeddings(
        decoder_model=decoder_model,
        tokenizer=tokenizer,
        embeddings=embeddings,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_samples=args.num_samples,
        device=args.device,
    )
    
    # Output results
    output_lines = []
    
    for i, generated_texts in enumerate(all_results):
        if embeddings.shape[0] > 1:
            header = f"\n{'='*60}\nEmbedding {i+1}\n{'='*60}"
            output_lines.append(header)
        
        if args.num_samples == 1:
            output_lines.append(generated_texts[0])
        else:
            for j, text in enumerate(generated_texts, 1):
                output_lines.append(f"\nSample {j}:\n{text}")
    
    output_text = "\n".join(output_lines)
    
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(output_text)
        print(f"\nResults saved to {args.output_path}", file=sys.stderr)
    else:
        print("\n" + output_text)
    
    print(f"\n{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
