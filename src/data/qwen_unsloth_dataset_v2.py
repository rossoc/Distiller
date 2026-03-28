"""
Dataset module for Unsloth-based Qwen fine-tuning (v2).
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
from datasets import Dataset, DatasetDict, load_dataset


def formatting_prompts_func(examples):
    """
    Data formatting function for ChatML.
    """
    instructions = examples["source"]
    outputs = examples["target"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        texts.append(text)
    return {"text": texts}


def load_qwen_dataset_v2(
    data_path: Union[str, Path],
    validation_split: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Load dataset for unsloth-based fine-tuning.
    """
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
