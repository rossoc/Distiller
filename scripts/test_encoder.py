from argparse import ArgumentParser
from pathlib import Path


def get_args():
    parser = ArgumentParser(
        description="A script to generate embeddings using embedding-gemma model."
    )

    parser.add_argument(
        "--prompt",
        default="Hello, world!",
        help="The text to generate embeddings for",
        type=str,
    )

    parser.add_argument(
        "--model-path",
        default="models/embeddinggemma-300M-Q8.gguf",
        help="Path to the model file (default: models/embeddinggemma-300M-Q8.gguf)",
        type=str,
    )

    parser.add_argument(
        "--n-cpu",
        default=4,
        help="Number of CPU threads to use",
        type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":
    from llama_cpp.llama_embedding import LlamaEmbedding
    import llama_cpp

    args = get_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please download the model first.")
        exit(1)

    # Load the model using LlamaEmbedding (optimized for embedding tasks)
    print(f"Loading model from {model_path}...")
    llm = LlamaEmbedding(
        model_path=str(model_path),
        n_ctx=2048,
        n_gpu_layers=0,  # CPU only
        verbose=False,
        pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE,
    )
    print("Model loaded successfully!")

    # Get embeddings for the provided text
    embedding = llm.embed(args.prompt)
    print(f"\nText: {args.prompt}")
    print(f"Embedding dimension: {len(embedding[0])}")
    print(embedding[0][:10])
