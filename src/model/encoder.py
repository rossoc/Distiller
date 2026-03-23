from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE


def gemma_encoder():
    llm = LlamaEmbedding(
        model_path=str("models/embeddinggemma-300M-Q8.gguf"),
        n_ctx=2048,
        n_gpu_layers=0,  # CPU only
        verbose=False,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
    )

    return llm
