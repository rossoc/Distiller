import numpy as np
from model.faiss_retriever import FAISSRetriever


def build_faiss_index(
    texts: list[str],
    embeddings: list[np.ndarray],
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
