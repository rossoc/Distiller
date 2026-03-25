def print_verbose_setup_diffusion(args):
    print(f"\n{'=' * 60}")
    print("Training Configuration:")
    print(f"{'=' * 60}")
    print(
        f"  Model: {args.emb_dim}d emb, {args.num_layers} layers, {args.num_heads} heads"
    )
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

def print_verbose_training_complete(run_path):
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"Output folder: {run_path.absolute()}")
    print(f"{'=' * 60}")
