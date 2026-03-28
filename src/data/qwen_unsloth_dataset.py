"""
Dataset module for Unsloth-based Qwen fine-tuning.

Supports multiple data formats:
- Instruction tuning: {"instruction": "...", "input": "...", "output": "..."}
- Chat format: {"messages": [{"role": "...", "content": "..."}]}
- Simple text: {"text": "..."} or {"source": "...", "target": "..."}

Uses Qwen 3.5 ChatML format for optimal performance.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datasets import Dataset, DatasetDict
import pandas as pd


def load_unsloth_dataset(
    data_path: Union[str, Path],
    format_type: str = "instruction",
    validation_split: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Load dataset for unsloth-based fine-tuning.

    Args:
        data_path: Path to data file (JSON, JSONL, CSV, TXT)
        format_type: One of "instruction", "chat", "text", "source_target"
        validation_split: Fraction of data to use for validation
        seed: Random seed for shuffling

    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # Load data based on file extension
    if data_path.suffix in [".json", ".jsonl"]:
        data = _load_json_data(data_path)
    elif data_path.suffix == ".csv":
        data = _load_csv_data(data_path)
    elif data_path.suffix in [".txt", ".text"]:
        data = _load_text_data(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)

    # Split into train/test
    dataset_dict = dataset.train_test_split(
        test_size=validation_split,
        seed=seed,
        shuffle=True,
    )

    return dataset_dict


def _load_json_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSON/JSONL file."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        if data_path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                elif "examples" in data:
                    data = data["examples"]
                else:
                    data = [data]
    return data


def _load_csv_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load data from CSV file."""
    df = pd.read_csv(data_path)
    data = []
    for _, row in df.iterrows():
        sample = {}
        if "instruction" in df.columns:
            sample["instruction"] = row["instruction"]
        if "input" in df.columns:
            sample["input"] = row["input"]
        if "output" in df.columns:
            sample["output"] = row["output"]
        if "text" in df.columns:
            sample["text"] = row["text"]
        if "source" in df.columns:
            sample["source"] = row["source"]
        if "target" in df.columns:
            sample["target"] = row["target"]
        if "messages" in df.columns:
            sample["messages"] = json.loads(row["messages"])
        data.append(sample)
    return data


def _load_text_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load data from text file (one sample per line)."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append({"text": line})
    return data
