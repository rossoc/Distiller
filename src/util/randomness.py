import random
import numpy as np
import torch
import lightning as L

from .logger import create_output_folder


def set_seed(seed: int):
    """Set seed to make every experiment repeatable"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(seed)


def setup_run(args):
    set_seed(args.seed)

    run_path = create_output_folder(args.output_dir, args.run_name)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    print(f"\n{'=' * 60}")
    print(f"Training run: {run_path.name}")
    print(f"Output folder: {run_path.absolute()}")
    print(f"{'=' * 60}\n")

    return run_path
