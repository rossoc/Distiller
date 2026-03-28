"""
Dataset module for Unsloth-based Qwen fine-tuning (v2).
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
from datasets import Dataset, DatasetDict, load_dataset
from src.data.util import load_dataset as util_load_dataset

def formatting_prompts_func(examples):
    """
    Data formatting function for ChatML.
    """
    instructions = examples["source"]
    outputs      = examples["target"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return { "text" : texts }

def load_qwen_dataset_v2(
    data_path: Union[str, Path],
    validation_split: float = 0.1,
    seed: int = 42,
    schema: str = "simple_diffusion",
) -> DatasetDict:
    """
    Load dataset for unsloth-based fine-tuning.
    """
    data_path = Path(data_path)

    if data_path.suffix == ".xlsx":
        # Load from XLSX using util.load_dataset
        (X_all, y_all), _, _ = util_load_dataset(str(data_path), split_ratio=(1.0, 0.0, 0.0), schema=schema)

        # Convert to list of dicts for Hugging Face Dataset
        data = [{"source": X, "target": Y} for X, Y in zip(X_all, y_all)]
        dataset = Dataset.from_list(data)
    else:
        # Load from JSON (existing logic)
        dataset = load_dataset("json", data_files=str(data_path), split="train")

    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Split into train/test
    if validation_split > 0:
        dataset_dict = dataset.train_test_split(
            test_size=validation_split,
            seed=seed,
            shuffle=True,
        )
        return dataset_dict
    else:
        return DatasetDict({"train": dataset, "test": None})

            shuffle=True,
        )
        return dataset_dict
    else:
        return DatasetDict({"train": dataset, "test": None})
