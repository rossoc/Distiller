from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE

# Mapping of encoder names to their embedding dimensions
ENCODER_DIMS = {
    "gemma": 768,
    "qwen": 1024,
}


def get_encoder_dim(name: str) -> int:
    """Get the embedding dimension for a given encoder name.

    Args:
        name: Encoder name ("gemma" or "qwen")

    Returns:
        Embedding dimension (768 for gemma, 1024 for qwen)

    Raises:
        ValueError: If encoder name is not recognized
    """
    if name not in ENCODER_DIMS:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(ENCODER_DIMS.keys())}")
    return ENCODER_DIMS[name]


def gemma_encoder():
    llm = LlamaEmbedding(
        model_path=str("models/embeddinggemma-300M-Q8.gguf"),
        n_ctx=2048,
        n_gpu_layers=0,  # CPU only
        verbose=False,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
    )

    return llm


def qwen_encoder():
    llm = LlamaEmbedding(
        model_path=str("models/Qwen3.5-0.8B-Q8_0.gguf"),
        n_ctx=2048,
        n_gpu_layers=0,  # CPU only
        verbose=False,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
    )

    return llm


def encoder(name: str):
    encoders = {"gemma": gemma_encoder, "qwen": qwen_encoder}

    if name not in encoders.keys():
        raise ValueError(f"Unknown Encoder {name}")

    return encoders[name]()
