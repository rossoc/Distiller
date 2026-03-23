#!/usr/bin/env python3
"""
Inference script for the Embedding Decoder with FAISS retrieval.

This script demonstrates the inference flow:
1. Load input text
2. Encode input with gemma_encoder
3. Use decoder to predict target embedding
4. Use FAISS to find the closest matching text

Flow:
- Input Data -> Decoder -> Predicted Vector
- Predicted Vector -> FAISS Search -> Closest Sentence Found
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import faiss

from data.util import load_dataset, Datasets_Variations
from data.dataset import create_datasets
from model.encoder import gemma_encoder
from model.decoder import Decoder
from model.faiss_retriever import FAISSRetriever


def load_model(checkpoint_path: str, device: str = "cpu") -> Decoder:
    """Load a trained decoder model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    config = checkpoint.get("config", {
        "emb_dim": 768,
        "num_layers": 6,
        "fwd_dim": 2048,
        "num_heads": 8,
        "dropout": 0.1,
        "output_dim": 768,
    })
    
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
    
    return model


def load_faiss_retriever(index_path: str) -> FAISSRetriever:
    """Load a FAISS retriever from disk."""
    retriever = FAISSRetriever()
    retriever.load(index_path)
    return retriever


def encode_texts(texts: List[str], encoder) -> np.ndarray:
    """Encode texts to embeddings using the gemma_encoder."""
    embeddings = []
    batch_size = 4  # Small batch to avoid overflow
    expected_dim = 768  # EmbeddingGemma output dimension
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embs = encoder.embed(batch)
            for emb in batch_embs:
                emb_array = np.array(emb, dtype=np.float32)
                # Ensure correct dimension
                if len(emb_array) != expected_dim:
                    if len(emb_array) > expected_dim:
                        emb_array = emb_array[:expected_dim]
                    else:
                        padded = np.zeros(expected_dim, dtype=np.float32)
                        padded[:len(emb_array)] = emb_array
                        emb_array = padded
                embeddings.append(emb_array)
        except IndexError:
            # If batch fails, encode one by one
            for text in batch:
                try:
                    emb = encoder.embed([text])[0]
                    emb_array = np.array(emb, dtype=np.float32)
                    if len(emb_array) != expected_dim:
                        if len(emb_array) > expected_dim:
                            emb_array = emb_array[:expected_dim]
                        else:
                            padded = np.zeros(expected_dim, dtype=np.float32)
                            padded[:len(emb_array)] = emb_array
                            emb_array = padded
                    embeddings.append(emb_array)
                except Exception:
                    embeddings.append(np.zeros(expected_dim, dtype=np.float32))
    
    return np.array(embeddings, dtype=np.float32)


def predict_embeddings(
    input_texts: List[str],
    encoder,
    decoder: Decoder,
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict target embeddings from input texts.
    
    Args:
        input_texts: List of input text strings
        encoder: Encoder model
        decoder: Trained decoder model
        device: Device to run inference on
        
    Returns:
        predicted_embeddings: (batch_size, embedding_dim) array
    """
    # Encode inputs
    input_embeddings = encode_texts(input_texts, encoder)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(input_embeddings, dtype=torch.float32, device=device)
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(1)
    
    # Predict
    with torch.no_grad():
        predicted = decoder(memory=input_tensor)
        
        # Remove sequence dimension if present
        if predicted.dim() > 2:
            predicted = predicted[:, 0, :]
    
    return predicted.cpu().numpy()


def retrieve_texts(
    predicted_embeddings: np.ndarray,
    retriever: FAISSRetriever,
    k: int = 1,
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Retrieve closest texts for predicted embeddings.
    
    Args:
        predicted_embeddings: (batch_size, embedding_dim) array
        retriever: FAISS retriever
        k: Number of nearest neighbors to return
        
    Returns:
        texts: List of lists of retrieved texts
        scores: List of lists of similarity scores
    """
    return retriever.search(predicted_embeddings, k)


def run_inference(
    input_texts: List[str],
    decoder_path: str,
    faiss_index_path: str,
    device: str = "cpu",
    k: int = 1,
) -> List[dict]:
    """
    Run full inference pipeline.
    
    Args:
        input_texts: List of input texts
        decoder_path: Path to trained decoder checkpoint
        faiss_index_path: Path to FAISS index
        device: Device for inference
        k: Number of nearest neighbors to retrieve
        
    Returns:
        results: List of dictionaries with input, prediction, and retrieval
    """
    print("Loading models...")
    encoder = gemma_encoder()
    decoder = load_model(decoder_path, device)
    retriever = load_faiss_retriever(faiss_index_path)
    
    print(f"Running inference on {len(input_texts)} samples...")
    
    # Predict embeddings
    predicted_embeddings = predict_embeddings(
        input_texts, encoder, decoder, device
    )
    
    # Retrieve texts
    retrieved_texts, scores = retrieve_texts(predicted_embeddings, retriever, k)
    
    # Format results
    results = []
    for i, (input_text, texts, score) in enumerate(zip(input_texts, retrieved_texts, scores)):
        result = {
            "input": input_text,
            "retrieved": texts,
            "scores": score,
        }
        results.append(result)
        
        # Print result
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: {input_text}")
        print(f"Retrieved: {texts[0] if texts else 'N/A'}")
        print(f"Score: {score[0] if score else 'N/A'}")
    
    return results


def build_faiss_index_for_dataset(
    dataset_variation: Datasets_Variations = Datasets_Variations.SIMPLE_DIFFUSION,
    split_ratio: Tuple[float, float, float] = (0.5, 0.1, 0.4),
    save_path: str = "outputs/faiss_index",
):
    """
    Build a FAISS index for the entire training dataset.
    
    This should be run once after training to create the retrieval index
    for inference.
    
    Args:
        dataset_variation: Which dataset variation to use
        split_ratio: Train/eval/test split ratios
        save_path: Path to save the FAISS index
    """
    from src.data.dataset import create_datasets, EmbeddingDecoderDataset
    
    print("Loading datasets...")
    _, _, test_dataset = create_datasets(
        train_ratio=split_ratio[0],
        eval_ratio=split_ratio[1],
        test_ratio=split_ratio[2],
    )
    
    # Use all data for building the index (or just training data)
    # Here we use test dataset as example
    print("Building FAISS index from dataset...")
    
    # Get all target texts and embeddings
    texts = []
    embeddings = []
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        texts.append(sample["target_text"])
        embeddings.append(sample["target_embeddings"].numpy())
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Build retriever
    retriever = FAISSRetriever(embedding_dim=768)
    retriever.build_index(texts, embeddings)
    retriever.save(save_path)
    
    print(f"FAISS index saved to {save_path}")
    
    return retriever


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Embedding Decoder")
    
    parser.add_argument(
        "--decoder_path",
        type=str,
        default="outputs/decoder/best_model.pt",
        help="Path to trained decoder checkpoint"
    )
    parser.add_argument(
        "--faiss_index_path",
        type=str,
        default="outputs/faiss_index",
        help="Path to FAISS index"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of nearest neighbors to retrieve"
    )
    parser.add_argument(
        "--build_index",
        action="store_true",
        help="Build FAISS index instead of running inference"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=5,
        help="Number of test samples to run inference on"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.build_index:
        # Build FAISS index
        build_faiss_index_for_dataset(
            save_path=args.faiss_index_path,
        )
    else:
        # Run inference on test samples
        print("Loading test data...")
        (_, _), (_, _), (X_test, y_test) = load_dataset(
            dataset_variation=Datasets_Variations.SIMPLE_DIFFUSION,
            split_ratio=(0.5, 0.1, 0.4),
        )
        
        # Get a few test samples
        test_texts = X_test.tolist()[:args.test_samples]
        
        results = run_inference(
            input_texts=test_texts,
            decoder_path=args.decoder_path,
            faiss_index_path=args.faiss_index_path,
            device=args.device,
            k=args.k,
        )
        
        print("\n" + "="*50)
        print("Inference complete!")
        print("="*50)


if __name__ == "__main__":
    main()
