#!/usr/bin/env python3
"""
Inference script for the Diffusion-based Embedding Decoder with FAISS retrieval.

This script:
1. Loads a trained DiffusionTrainer model checkpoint
2. Loads the test dataset using the same pipeline as training
3. Runs predictions through the decoder (without teacher forcing)
4. Uses FAISS to convert predicted embeddings to text
5. Saves results with predictions and ground truth

Key differences from run_inference.py:
- Loads DiffusionTrainer Lightning module instead of plain Decoder
- Uses sequence-to-sequence prediction (not single vector)
- Matches training data pipeline (max_length=2048, same augmentations)
- Handles Lightning checkpoint format

Usage:
    # First, build the FAISS index (only needed once per dataset)
    python src/run_diffusion_inference.py --build_index --model_path outputs/diffusion_XXX/best_model.pt

    # Then run inference
    python src/run_diffusion_inference.py --model_path outputs/diffusion_XXX/best_model.pt
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

from model.diffusion_trainer import DiffusionTrainer
from model.faiss_retriever import FAISSRetriever
from data.datamodule import EmbeddingDecoderDataModule
from data.dataset import EmbeddingDecoderDataset, embedding_collate_fn
from util.randomness import set_seed
from torch.utils.data import DataLoader


def load_model(model_path: str, device: str = "cpu") -> DiffusionTrainer:
    """
    Load a trained DiffusionTrainer model from checkpoint.

    Wrapper around DiffusionTrainer.from_checkpoint for backward compatibility.

    Args:
        model_path: Path to the .ckpt or .pt model file
        device: Device to load the model on

    Returns:
        Loaded DiffusionTrainer model
    """
    return DiffusionTrainer.from_checkpoint(model_path, device=device)


def build_faiss_index(
    texts: List[str],
    embeddings: List[np.ndarray],
    save_path: str,
) -> FAISSRetriever:
    """
    Build a FAISS index from texts and their embeddings.

    For FAISS, we need a single vector per text. We use mean pooling
    over the sequence of token embeddings.

    Args:
        texts: List of texts
        embeddings: List of embedding arrays (seq_len, 768)
        save_path: Path to save the FAISS index

    Returns:
        FAISSRetriever with built index
    """
    print("Building FAISS index...")

    # Mean pool over sequence to get single vector per text
    pooled_embeddings = []
    for emb in embeddings:
        pooled = emb.mean(axis=0)  # (768,)
        pooled_embeddings.append(pooled)

    pooled_embeddings = np.array(pooled_embeddings, dtype=np.float32)

    # Build retriever
    retriever = FAISSRetriever(embedding_dim=768)
    retriever.build_index(texts, pooled_embeddings)
    retriever.save(save_path)

    print(f"FAISS index built with {len(texts)} entries")
    print(f"Saved to: {save_path}.index and {save_path}.data")

    return retriever


def run_predictions(
    model: DiffusionTrainer,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[str], List[str], List[np.ndarray]]:
    """
    Run predictions on the test set using sequence-to-sequence decoding.

    For inference without teacher forcing, we:
    1. Use input embeddings as memory (context)
    2. Predict output embeddings in a single forward pass
    3. Use mean pooling to get a single vector for FAISS retrieval

    Args:
        model: Trained DiffusionTrainer model
        dataloader: DataLoader with test data
        device: Device to run on

    Returns:
        pooled_predictions: Array of pooled predicted embeddings (n_samples, 768)
        input_texts: List of input texts
        target_texts: List of ground truth target texts
        full_predictions: List of full sequence predictions for analysis
    """
    print("Running predictions...")

    pooled_predictions = []
    full_predictions = []
    input_texts = []
    target_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_embeddings = batch["input_embeddings"].to(device)
            target_embeddings = batch["target_embeddings"].to(device)
            memory_mask = batch["input_attention_mask"].to(device) == 0
            tgt_mask = batch["target_attention_mask"].to(device) == 0

            batch_size = input_embeddings.shape[0]

            # For inference, we pass target embeddings through the model
            # The model learns to map input->target during training
            # During inference, we can use:
            # Option 1: Pass zeros/dummy target (simplest)
            # Option 2: Pass input embeddings as initial target guess
            # Option 3: Use iterative refinement (more complex)

            # We use Option 2: Initialize with input embeddings projected to target length
            # Get target sequence length
            max_tgt_len = target_embeddings.shape[1]

            # Create initial target from input (repeat/pad to match target length)
            # This gives the model a starting point based on the input
            if input_embeddings.shape[1] >= max_tgt_len:
                # Input is longer, just truncate
                init_tgt = input_embeddings[:, :max_tgt_len, :].clone()
            else:
                # Input is shorter, repeat and pad
                repeats = (max_tgt_len // input_embeddings.shape[1]) + 1
                init_tgt = input_embeddings.repeat(1, repeats, 1)[:, :max_tgt_len, :].clone()

            # Add some noise to avoid exact copies (optional, can help with diversity)
            # noise = torch.randn_like(init_tgt) * 0.1
            # init_tgt = init_tgt + noise

            # Run forward pass
            output = model(
                memory=input_embeddings,
                tgt=init_tgt,
                memory_padding_mask=memory_mask,
                tgt_padding_mask=tgt_mask,
            )

            # Output shape: (batch, seq_len, 768)
            # Mean pool to get single vector per sample for FAISS
            # Apply mask to ignore padding tokens
            output_mask = (~tgt_mask).unsqueeze(-1).float()
            pooled_output = (output * output_mask).sum(dim=1) / output_mask.sum(dim=1).clamp(min=1)

            # Move to CPU and store
            pooled_output = pooled_output.cpu().numpy()
            output_seq = output.cpu().numpy()

            pooled_predictions.append(pooled_output)
            full_predictions.append(output_seq)
            input_texts.extend(batch["input_text"])
            target_texts.extend(batch["target_text"])

    # Concatenate all batches
    pooled_predictions = np.vstack(pooled_predictions)
    full_predictions = np.vstack(full_predictions)

    print(f"Generated {len(pooled_predictions)} predictions")
    print(f"Pooled prediction shape: {pooled_predictions.shape}")
    print(f"Full prediction shape: {full_predictions.shape}")

    return pooled_predictions, input_texts, target_texts, full_predictions


def retrieve_with_faiss(
    predicted_embeddings: np.ndarray,
    retriever: FAISSRetriever,
    k: int = 1,
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Retrieve closest texts for predicted embeddings using FAISS.

    Args:
        predicted_embeddings: Array of predicted embeddings (n_samples, 768)
        retriever: FAISS retriever
        k: Number of nearest neighbors to return

    Returns:
        retrieved_texts: List of lists of retrieved texts
        scores: List of lists of similarity scores
    """
    print("Retrieving texts with FAISS...")
    return retriever.search(predicted_embeddings, k)


def save_results(
    output_path: str,
    input_texts: List[str],
    target_texts: List[str],
    retrieved_texts: List[List[str]],
    scores: List[List[float]],
    model_path: str,
    full_predictions: List[np.ndarray] = None,
):
    """
    Save inference results to JSON.

    Args:
        output_path: Path to save results
        input_texts: List of input texts
        target_texts: List of ground truth texts
        retrieved_texts: List of retrieved texts
        scores: List of similarity scores
        model_path: Path to the model used
        full_predictions: Optional full sequence predictions for analysis
    """
    results = []
    for i in range(len(input_texts)):
        result = {
            "input": input_texts[i],
            "ground_truth": target_texts[i],
            "predictions": [
                {
                    "text": retrieved_texts[i][j]
                    if j < len(retrieved_texts[i])
                    else None,
                    "score": float(scores[i][j]) if j < len(scores[i]) else None,
                }
                for j in range(len(retrieved_texts[0]) if retrieved_texts else 0)
            ],
        }
        results.append(result)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "n_samples": len(results),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with Diffusion-based Embedding Decoder"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt or .pt file)",
    )
    parser.add_argument(
        "--faiss_index_path",
        type=str,
        default="outputs/faiss_index_diffusion",
        help="Path to FAISS index (without extension)",
    )
    parser.add_argument(
        "--build_index",
        action="store_true",
        help="Build FAISS index instead of running inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/diffusion_inference",
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
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of nearest neighbors to retrieve"
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
        help="Save full sequence predictions (not just pooled) for analysis",
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

    # Get encoder from datamodule for FAISS index building
    encoder = datamodule.encoder

    if args.build_index:
        # Build FAISS index from test target texts
        print("\nBuilding FAISS index from test set targets...")

        # Create dataset to get pre-computed embeddings
        test_dataset = EmbeddingDecoderDataset(
            X_test.tolist(),
            y_test.tolist(),
            encoder,
            args.max_length,
        )

        # Build index from target embeddings
        build_faiss_index(
            texts=test_dataset.y_texts,
            embeddings=test_dataset.target_embeddings,
            save_path=args.faiss_index_path,
        )
        print("\nFAISS index built successfully!")
        return

    # Load existing FAISS index
    if not os.path.exists(f"{args.faiss_index_path}.index"):
        print(f"\nFAISS index not found at {args.faiss_index_path}")
        print("Run with --build_index first to create the index")
        return

    print(f"\nLoading FAISS index from: {args.faiss_index_path}")
    retriever = FAISSRetriever()
    retriever.load(args.faiss_index_path)

    # Get test dataloader
    test_loader = datamodule.test_dataloader()

    # Run predictions
    (
        pooled_predictions,
        input_texts,
        target_texts,
        full_predictions,
    ) = run_predictions(model, test_loader, device)

    # Retrieve texts with FAISS
    retrieved_texts, scores = retrieve_with_faiss(
        pooled_predictions, retriever, k=args.k
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
        retrieved_texts=retrieved_texts,
        scores=scores,
        model_path=args.model_path,
    )

    # Optionally save full predictions for deeper analysis
    if args.save_full_predictions:
        full_pred_path = output_dir / f"full_predictions_{timestamp}.npy"
        np.save(full_pred_path, full_predictions)
        print(f"Full predictions saved to: {full_pred_path}")

    # Print sample results
    print("\n" + "=" * 60)
    print("Sample Results:")
    print("=" * 60)
    for i in range(min(3, len(input_texts))):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Input: {input_texts[i][:100]}...")
        print(f"Ground Truth: {target_texts[i][:100]}...")
        for j, (text, score) in enumerate(zip(retrieved_texts[i], scores[i])):
            print(f"Prediction {j + 1} (score={score:.4f}): {text[:100]}...")

    print("\n" + "=" * 60)
    print(f"Inference complete! Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
