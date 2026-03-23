#!/usr/bin/env python3
"""
Inference script for the Embedding Decoder with FAISS retrieval.

This script:
1. Loads a trained decoder model
2. Loads the test dataset
3. Runs predictions through the decoder
4. Uses FAISS to convert predicted embeddings to text
5. Saves results with predictions and ground truth

Usage:
    # First, build the FAISS index (only needed once per dataset)
    python src/run_inference.py --build_index --model_path outputs/decoder_XXX/best_model.pt
    
    # Then run inference
    python src/run_inference.py --model_path outputs/decoder_XXX/best_model.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model.encoder import gemma_encoder
from model.decoder import Decoder
from model.faiss_retriever import FAISSRetriever
from data.util import load_dataset, Datasets_Variations, set_seed
from data.dataset import EmbeddingDecoderDataset, embedding_collate_fn
from torch.utils.data import DataLoader


def load_model(model_path: str, device: str = "cpu") -> Decoder:
    """
    Load a trained decoder model from checkpoint.
    
    Args:
        model_path: Path to the .pt model file
        device: Device to load the model on
        
    Returns:
        Loaded Decoder model
    """
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    config = checkpoint.get("model_config", checkpoint.get("config", {
        "output_dim": 768,
        "emb_dim": 768,
        "num_layers": 6,
        "fwd_dim": 2048,
        "num_heads": 8,
        "dropout": 0.1,
    }))
    
    model = Decoder(
        output_dim=config.get("output_dim", 768),
        emb_dim=config.get("emb_dim", 768),
        num_layers=config.get("num_layers", 6),
        fwd_dim=config.get("fwd_dim", 2048),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {config}")
    return model


def encode_texts(texts: List[str], encoder, max_length: int = 512) -> List[np.ndarray]:
    """
    Encode texts to sequences of embeddings.
    
    Args:
        texts: List of texts to encode
        encoder: Encoder model
        max_length: Maximum sequence length
        
    Returns:
        List of embedding arrays (seq_len, 768)
    """
    embeddings = []
    
    for i, text in enumerate(tqdm(texts, desc="Encoding texts")):
        try:
            emb_list = encoder.embed([text])
            if emb_list and len(emb_list) > 0:
                token_embs = np.array(emb_list[0], dtype=np.float32)
                if token_embs.ndim == 1:
                    token_embs = token_embs.reshape(1, -1)
                if len(token_embs) > max_length:
                    token_embs = token_embs[:max_length]
                embeddings.append(token_embs)
            else:
                embeddings.append(np.zeros((1, 768), dtype=np.float32))
        except Exception as e:
            embeddings.append(np.zeros((1, 768), dtype=np.float32))
    
    return embeddings


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
    model: Decoder,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Run predictions on the test set.
    
    Args:
        model: Trained decoder model
        dataloader: DataLoader with test data
        device: Device to run on
        
    Returns:
        predicted_embeddings: List of predicted embedding arrays
        input_texts: List of input texts
        target_texts: List of ground truth target texts
    """
    print("Running predictions...")
    
    predicted_embeddings = []
    input_texts = []
    target_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_embeddings = batch["input_embeddings"].to(device)
            memory_padding_mask = batch["input_attention_mask"].to(device) == 0
            
            # Run prediction
            # For inference, we predict one embedding that represents the full output
            # We use the mean of input embeddings as a query
            batch_size = input_embeddings.shape[0]
            
            # Mean pool input embeddings to get a single vector per sample
            input_mask = batch["input_attention_mask"].to(device).unsqueeze(-1)
            pooled_input = (input_embeddings * input_mask).sum(dim=1) / input_mask.sum(dim=1).clamp(min=1)
            
            # Pass through model to get predicted embedding
            # We need to reshape for the model
            pooled_input = pooled_input.unsqueeze(1)  # (batch, 1, 768)
            
            output = model(memory=pooled_input, memory_padding_mask=memory_padding_mask[:, :1])
            
            # Output shape: (batch, 1, 768) -> squeeze to (batch, 768)
            pred_emb = output.squeeze(1).cpu().numpy()
            
            predicted_embeddings.append(pred_emb)
            input_texts.extend(batch["input_text"])
            target_texts.extend(batch["target_text"])
    
    # Concatenate all batches
    predicted_embeddings = np.vstack(predicted_embeddings)
    
    return predicted_embeddings, input_texts, target_texts


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
    """
    results = []
    for i in range(len(input_texts)):
        result = {
            "input": input_texts[i],
            "ground_truth": target_texts[i],
            "predictions": [
                {
                    "text": retrieved_texts[i][j] if j < len(retrieved_texts[i]) else None,
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
    parser = argparse.ArgumentParser(description="Inference with Embedding Decoder")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--faiss_index_path",
        type=str,
        default="outputs/faiss_index",
        help="Path to FAISS index (without extension)"
    )
    parser.add_argument(
        "--build_index",
        action="store_true",
        help="Build FAISS index instead of running inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference",
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of nearest neighbors to retrieve"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
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
    
    # Load test dataset
    print("\nLoading test dataset...")
    _, _, (X_test, y_test) = load_dataset(
        dataset_variation=Datasets_Variations.SIMPLE_DIFFUSION,
        split_ratio=(0.5, 0.1, 0.4),
    )
    
    X_test_list = X_test.tolist()
    y_test_list = y_test.tolist()
    
    print(f"Test samples: {len(X_test_list)}")
    
    # Encode all target texts for FAISS index
    print("\nEncoding target texts for FAISS index...")
    encoder = gemma_encoder()
    target_embeddings = encode_texts(y_test_list, encoder, args.max_length)
    
    if args.build_index:
        # Build and save FAISS index
        build_faiss_index(
            texts=y_test_list,
            embeddings=target_embeddings,
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
    
    # Create test dataloader
    print("\nPreparing test data...")
    test_dataset = EmbeddingDecoderDataset(X_test_list, y_test_list, encoder, args.max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=embedding_collate_fn,
    )
    
    # Run predictions
    predicted_embeddings, input_texts, target_texts = run_predictions(
        model, test_loader, device
    )
    
    print(f"Generated {len(predicted_embeddings)} predictions")
    print(f"Prediction shape: {predicted_embeddings.shape}")
    
    # Retrieve texts with FAISS
    retrieved_texts, scores = retrieve_with_faiss(predicted_embeddings, retriever, k=args.k)
    
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
    
    # Print sample results
    print("\n" + "="*60)
    print("Sample Results:")
    print("="*60)
    for i in range(min(3, len(input_texts))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: {input_texts[i][:100]}...")
        print(f"Ground Truth: {target_texts[i][:100]}...")
        for j, (text, score) in enumerate(zip(retrieved_texts[i], scores[i])):
            print(f"Prediction {j+1} (score={score:.4f}): {text[:100]}...")
    
    print("\n" + "="*60)
    print(f"Inference complete! Results saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
