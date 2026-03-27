#!/usr/bin/env python3
"""
Inference script for the Diffusion-based Embedding Decoder with Token Extraction.

This script uses a token-by-token extraction approach instead of pooling:
1. Loads a trained DiffusionTrainer model checkpoint
2. Loads the test dataset using the same pipeline as training
3. Runs predictions through the decoder (without teacher forcing)
4. Uses cosine similarity to extract tokens from predicted embeddings
5. Decodes token sequences to text
6. Saves results with predictions and ground truth

Key differences from run_diffusion_inference.py:
- No FAISS pooling: processes each token position independently
- Token extraction via cosine similarity with embedding weight matrix
- Generates text token-by-token (more interpretable)
- Can generate novel sentences not in training set
- Provides confidence scores for each token prediction

Usage:
    python src/run_token_extraction_inference.py --model_path outputs/diffusion_XXX/best_model.pt
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
from dataclasses import asdict

import numpy as np
import torch
from tqdm import tqdm

from model.diffusion_trainer import DiffusionTrainer
from model.token_extractor import TokenExtractor, TokenExtractionResult
from data.datamodule import EmbeddingDecoderDataModule
from util.randomness import set_seed
from torch.utils.data import DataLoader


def run_predictions(
    model: DiffusionTrainer,
    dataloader: DataLoader,
    extractor: TokenExtractor,
    device: str = "cpu",
    max_output_length: int = 256,
    stop_at_eos: bool = False,
) -> Tuple[List[TokenExtractionResult], List[str], List[str], List[np.ndarray]]:
    """
    Run predictions on the test set using token-by-token extraction.

    For inference without teacher forcing, we:
    1. Use input embeddings as memory (context)
    2. Predict output embeddings in a single forward pass
    3. Extract tokens from each position using cosine similarity

    Args:
        model: Trained DiffusionTrainer model
        dataloader: DataLoader with test data
        extractor: TokenExtractor for converting embeddings to tokens
        device: Device to run on
        max_output_length: Maximum output sequence length

    Returns:
        extraction_results: List of token extraction results
        input_texts: List of input texts
        target_texts: List of ground truth target texts
        full_predictions: List of full sequence predictions for analysis
    """
    print("Running predictions with token extraction...")

    extraction_results = []
    input_texts = []
    target_texts = []
    full_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_embeddings = batch["input_embeddings"].to(device)
            target_embeddings = batch["target_embeddings"].to(device)
            memory_mask = batch["input_attention_mask"].to(device) == 0
            tgt_mask = batch["target_attention_mask"].to(device) == 0

            init_tgt = torch.zeros_like(target_embeddings)

            # Run forward pass
            output = model(
                memory=input_embeddings,
                tgt=init_tgt,
                memory_padding_mask=memory_mask,
            )

            # Output shape: (batch, seq_len, 768)
            # Process each sample in the batch
            for i in range(tgt_mask.shape[0]):
                # Get actual sequence length from mask
                seq_len = tgt_mask[i].sum().item()

                # Extract the predicted embeddings for this sample
                sample_embeddings = output[i, :seq_len, :].cpu().numpy()

                # Get attention mask (True = valid token)
                sample_mask = (~tgt_mask[i]).cpu().numpy()

                # Extract tokens using cosine similarity
                # For fixed-length generation, we don't stop at EOS
                result = extractor.extract_sequence(
                    sample_embeddings,
                    attention_mask=sample_mask,
                    stop_at_eos=stop_at_eos,
                    return_confidence=True,
                )

                extraction_results.append(result)
                full_predictions.append(sample_embeddings)

            input_texts.extend(batch["input_text"])
            target_texts.extend(batch["target_text"])

    print(f"Generated {len(extraction_results)} predictions")
    print(
        f"Average sequence length: {np.mean([r.sequence_length for r in extraction_results]):.1f}"
    )
    print(
        f"Average confidence: {np.mean([np.mean(r.confidence_scores) for r in extraction_results]):.3f}"
    )

    return extraction_results, input_texts, target_texts, full_predictions


def compute_metrics(
    extraction_results: List[TokenExtractionResult],
    target_texts: List[str],
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for the predictions.

    Args:
        extraction_results: List of token extraction results
        target_texts: List of ground truth texts

    Returns:
        Dictionary of evaluation metrics
    """
    # Confidence statistics
    all_confidences = []
    for result in extraction_results:
        all_confidences.extend(result.confidence_scores)

    confidence_stats = {
        "mean": float(np.mean(all_confidences)),
        "std": float(np.std(all_confidences)),
        "min": float(np.min(all_confidences)),
        "max": float(np.max(all_confidences)),
    }

    # Sequence length statistics
    seq_lengths = [r.sequence_length for r in extraction_results]
    length_stats = {
        "mean": float(np.mean(seq_lengths)),
        "std": float(np.std(seq_lengths)),
        "min": int(np.min(seq_lengths)),
        "max": int(np.max(seq_lengths)),
    }

    # Exact match rate (if predictions match ground truth exactly)
    exact_matches = sum(
        1
        for r, t in zip(extraction_results, target_texts)
        if r.text.strip() == t.strip()
    )
    exact_match_rate = exact_matches / len(target_texts)

    return {
        "confidence": confidence_stats,
        "sequence_length": length_stats,
        "exact_match_rate": exact_match_rate,
        "total_samples": len(extraction_results),
    }


def save_results(
    output_path: str,
    input_texts: List[str],
    target_texts: List[str],
    extraction_results: List[TokenExtractionResult],
    model_path: str,
    metrics: Dict[str, Any],
    full_predictions: List[np.ndarray] = None,
):
    """
    Save inference results to JSON.

    Args:
        output_path: Path to save results
        input_texts: List of input texts
        target_texts: List of ground truth texts
        extraction_results: List of token extraction results
        model_path: Path to the model used
        metrics: Evaluation metrics
        full_predictions: Optional full sequence predictions for analysis
    """
    results = []
    for i in range(len(input_texts)):
        result = {
            "input": input_texts[i],
            "ground_truth": target_texts[i],
            "prediction": extraction_results[i].text,
            "tokens": extraction_results[i].tokens,
            "token_ids": extraction_results[i].token_ids,
            "confidence_scores": extraction_results[i].confidence_scores,
            "average_confidence": float(
                np.mean(extraction_results[i].confidence_scores)
            )
            if extraction_results[i].confidence_scores
            else 0.0,
            "sequence_length": extraction_results[i].sequence_length,
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
        description="Inference with Diffusion-based Embedding Decoder using Token Extraction"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt or .pt file)",
    )
    parser.add_argument(
        "--gguf_model_path",
        type=str,
        default="models/embeddinggemma-300M-Q8.gguf",
        help="Path to GGUF model for token extraction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/token_extraction_inference",
        help="Directory to save inference results",
    )

    # Data arguments (should match training)
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for embeddings (should match training)",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=256,
        help="Maximum output sequence length for token extraction",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Analysis options
    parser.add_argument(
        "--save_full_predictions",
        action="store_true",
        help="Save full sequence predictions (not just extracted text) for analysis",
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
    model = DiffusionTrainer.from_checkpoint(args.model_path, device=device)

    # Load token extractor
    print("\nLoading token extractor...")
    extractor = TokenExtractor.from_gguf_model(args.gguf_model_path)
    print(
        f"Token extractor loaded: vocab size = {extractor.vocab_size}, embedding dim = {extractor.embedding_dim}"
    )

    # Setup data module (same as training)
    print("\nLoading test dataset...")
    datamodule = EmbeddingDecoderDataModule(
        train_ratio=0.5,
        eval_ratio=0.1,
        test_ratio=0.4,
        schema="simple_diffusion",
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 for inference to avoid multiprocessing issues
        seed=args.seed,
    )
    datamodule.setup()

    # Get test data
    X_test, y_test = datamodule.test_data
    print(f"Test samples: {len(X_test)}")

    # Get test dataloader
    test_loader = datamodule.test_dataloader()

    # Run predictions
    extraction_results, input_texts, target_texts, full_predictions = run_predictions(
        model, test_loader, extractor, device, args.max_output_length, stop_at_eos=False
    )

    # Compute metrics
    metrics = compute_metrics(extraction_results, target_texts)

    # Print metrics
    print("\n" + "=" * 60)
    print("Evaluation Metrics:")
    print("=" * 60)
    print(
        f"Confidence (mean): {metrics['confidence']['mean']:.4f} ± {metrics['confidence']['std']:.4f}"
    )
    print(
        f"Sequence length (mean): {metrics['sequence_length']['mean']:.1f} ± {metrics['sequence_length']['std']:.1f}"
    )
    print(
        f"Exact match rate: {metrics['exact_match_rate']:.4f} ({int(metrics['exact_match_rate'] * len(target_texts))}/{len(target_texts)})"
    )

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
        extraction_results=extraction_results,
        model_path=args.model_path,
        metrics=metrics,
    )

    # Optionally save full predictions for deeper analysis
    if args.save_full_predictions:
        full_pred_path = output_dir / f"full_predictions_{timestamp}.npy"
        # Save as object array since sequences have variable lengths
        np.save(full_pred_path, np.array(full_predictions, dtype=object))
        print(f"Full predictions saved to: {full_pred_path}")

    # Print sample results
    print("\n" + "=" * 60)
    print("Sample Results:")
    print("=" * 60)
    for i in range(min(3, len(input_texts))):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Input: {input_texts[i][:100]}...")
        print(f"Ground Truth: {target_texts[i][:100]}...")
        print(f"Prediction: {extraction_results[i].text[:100]}...")
        print(f"Avg Confidence: {np.mean(extraction_results[i].confidence_scores):.3f}")
        print(f"Tokens: {extraction_results[i].tokens[:20]}...")  # Show first 20 tokens

    print("\n" + "=" * 60)
    print(f"Inference complete! Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
