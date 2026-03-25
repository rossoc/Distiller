#!/usr/bin/env python3
"""
Inference script for the Decoder2 (Vocabulary-based Decoder).

This script:
1. Loads a trained Decoder2 model
2. Loads the test dataset
3. Runs predictions through the decoder
4. Converts predicted logits to token IDs (argmax)
5. Decodes tokens back to text
6. Saves results with predictions and ground truth

Usage:
    python src/run_decoder_2_inference.py --model_path outputs/decoder_2_XXX/best_model.pt

Output:
    outputs/inference_decoder_2/inference_results_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

from model.encoder import gemma_encoder
from model.decoder_2 import Decoder2
from model.gemma_tokenizer import GemmaTokenizer
from data.util import load_dataset
from util.randomness import set_seed
from data.decoder_2_dataset import Decoder2Dataset, decoder2_collate_fn
from torch.utils.data import DataLoader


def load_model(model_path: str, device: str = "cpu") -> Decoder2:
    """
    Load a trained Decoder2 model from checkpoint.

    Args:
        model_path: Path to the .pt model file
        device: Device to load the model on

    Returns:
        Loaded Decoder2 model
    """
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    config = checkpoint.get(
        "model_config",
        checkpoint.get(
            "config",
            {
                "vocab_size": 256000,
                "emb_dim": 768,
                "num_layers": 6,
                "fwd_dim": 2048,
                "num_heads": 8,
                "dropout": 0.1,
            },
        ),
    )

    model = Decoder2(
        vocab_size=config.get("vocab_size", 256000),
        emb_dim=config.get("emb_dim", 768),
        num_layers=config.get("num_layers", 6),
        fwd_dim=config.get("fwd_dim", 2048),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.1),
    )

    # Load state dict - handle both Lightning-wrapped and plain Decoder2 checkpoints
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # If keys are prefixed with "model.", strip the prefix (Lightning format)
    if any(key.startswith("model.") for key in state_dict.keys()):
        print("Detected Lightning checkpoint, stripping 'model.' prefix...")
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded: {config}")
    return model


def run_predictions(
    model: Decoder2,
    dataloader: DataLoader,
    tokenizer: GemmaTokenizer,
    device: str = "cpu",
) -> Tuple[List[List[int]], List[List[str]], List[str], List[str]]:
    """
    Run predictions on the test set.

    Args:
        model: Trained decoder model
        dataloader: DataLoader with test data
        tokenizer: Tokenizer for decoding
        device: Device to run on

    Returns:
        predicted_tokens: List of predicted token ID sequences
        predicted_texts: List of decoded predicted texts
        input_texts: List of input texts
        target_texts: List of ground truth target texts
    """
    print("Running predictions...")

    predicted_tokens = []
    predicted_texts = []
    input_texts = []
    target_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_embeddings = batch["input_embeddings"].to(device)
            memory_mask = batch["input_attention_mask"].to(device) == 0

            # Get batch size and sequence length
            batch_size = input_embeddings.shape[0]

            # For inference, we predict tokens autoregressively or in one pass
            # Here we use a simple approach: predict all tokens at once
            # using the input embeddings as memory
            
            # Create a dummy target sequence of appropriate length
            # In practice, you might want to use a more sophisticated approach
            max_input_len = input_embeddings.shape[1]
            
            # Create placeholder target (model will attend to memory)
            # Use zeros as placeholder - the model will predict based on memory
            dummy_tgt = torch.zeros(
                batch_size, max_input_len, dtype=torch.long, device=device
            )

            # Run prediction
            # Output shape: (batch, seq_len, vocab_size)
            logits = model(
                memory=input_embeddings,
                tgt=dummy_tgt,
                memory_mask=memory_mask,
            )

            # Get predicted tokens (argmax over vocabulary)
            pred_tokens = logits.argmax(dim=-1)  # (batch, seq_len)

            # Convert to texts
            for i in range(batch_size):
                # Get tokens for this sample, excluding padding
                sample_tokens = pred_tokens[i].cpu().numpy().tolist()
                
                # Decode to text
                text = tokenizer.decode(sample_tokens)
                
                predicted_tokens.append(sample_tokens)
                predicted_texts.append(text)

            input_texts.extend(batch["input_text"])
            target_texts.extend(batch["target_text"])

    return predicted_tokens, predicted_texts, input_texts, target_texts


def calculate_metrics(
    predicted_tokens: List[List[int]],
    target_texts: List[str],
    tokenizer: GemmaTokenizer,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        predicted_tokens: List of predicted token sequences
        target_texts: List of ground truth texts
        tokenizer: Tokenizer for encoding targets

    Returns:
        Dictionary of metrics
    """
    # Tokenize target texts for comparison
    target_tokens_list = [tokenizer.encode(text) for text in target_texts]

    # Calculate token-level accuracy
    total_correct = 0
    total_tokens = 0

    for pred_tokens, target_tokens in zip(predicted_tokens, target_tokens_list):
        # Compare up to the minimum length
        min_len = min(len(pred_tokens), len(target_tokens))
        if min_len > 0:
            correct = sum(
                p == t for p, t in zip(pred_tokens[:min_len], target_tokens[:min_len])
            )
            total_correct += correct
            total_tokens += min_len

    token_accuracy = total_correct / max(total_tokens, 1)

    return {
        "token_accuracy": token_accuracy,
        "total_samples": len(predicted_tokens),
    }


def save_results(
    output_path: str,
    input_texts: List[str],
    target_texts: List[str],
    predicted_texts: List[str],
    predicted_tokens: List[List[int]],
    model_path: str,
    metrics: Dict[str, float],
):
    """
    Save inference results to JSON.

    Args:
        output_path: Path to save results
        input_texts: List of input texts
        target_texts: List of ground truth texts
        predicted_texts: List of predicted texts
        predicted_tokens: List of predicted token sequences
        model_path: Path to the model used
        metrics: Dictionary of evaluation metrics
    """
    results = []
    for i in range(len(input_texts)):
        result = {
            "input": input_texts[i],
            "ground_truth": target_texts[i],
            "prediction": predicted_texts[i],
            "predicted_tokens": predicted_tokens[i][:20],  # First 20 tokens for reference
        }
        results.append(result)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "n_samples": len(results),
        "metrics": metrics,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with Decoder2 (Vocabulary-based Decoder)"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference_decoder_2",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gguf_model_path",
        type=str,
        default="models/embeddinggemma-300M-Q8.gguf",
        help="Path to GGUF model for tokenizer",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.4, help="Ratio of data for testing"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model_path, device)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GemmaTokenizer.from_gguf_model(args.gguf_model_path)
    print(f"Tokenizer loaded: vocab size = {tokenizer.vocab_size}")

    # Load test dataset
    print("\nLoading test dataset...")
    _, _, (X_test, y_test) = load_dataset(
        schema="simple_diffusion",
        split_ratio=(0.5, 0.1, 0.4),
    )

    X_test_list = X_test.tolist()
    y_test_list = y_test.tolist()

    print(f"Test samples: {len(X_test_list)}")

    # Create test dataset
    print("\nPreparing test data...")
    encoder = gemma_encoder()
    test_dataset = Decoder2Dataset(
        X_texts=X_test_list,
        y_texts=y_test_list,
        encoder=encoder,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=decoder2_collate_fn,
    )

    # Run predictions
    predicted_tokens, predicted_texts, input_texts, target_texts = run_predictions(
        model, test_loader, tokenizer, device
    )

    print(f"Generated {len(predicted_texts)} predictions")

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(predicted_tokens, target_texts, tokenizer)

    print(f"\nMetrics:")
    print(f"  Token Accuracy: {metrics['token_accuracy']:.4f}")
    print(f"  Total Samples: {metrics['total_samples']}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"inference_results_{timestamp}.json"

    # Save results
    save_results(
        output_path=str(output_path),
        input_texts=input_texts,
        target_texts=target_texts,
        predicted_texts=predicted_texts,
        predicted_tokens=predicted_tokens,
        model_path=args.model_path,
        metrics=metrics,
    )

    # Print sample results
    print("\n" + "=" * 60)
    print("Sample Results:")
    print("=" * 60)
    for i in range(min(3, len(input_texts))):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Input: {input_texts[i][:100]}...")
        print(f"Ground Truth: {target_texts[i][:100]}...")
        print(f"Prediction: {predicted_texts[i][:100]}...")

    print("\n" + "=" * 60)
    print(f"Inference complete! Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
