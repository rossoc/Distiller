"""
Lightning data module for fine-tuning LLMs (Qwen, Llama, etc.).

Supports multiple dataset formats:
- JSON/JSONL with instruction/input/output fields
- CSV/text files with configurable columns
- HuggingFace datasets
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from torch.utils.data import DataLoader, Dataset
import lightning as L
import pandas as pd

from transformers import AutoTokenizer, PreTrainedTokenizer
from util.randomness import set_seed


class FinetuneDataset(Dataset):
    """
    Dataset for LLM fine-tuning.

    Formats supported:
    - Instruction tuning: {"instruction": "...", "input": "...", "output": "..."}
    - Chat format: {"messages": [{"role": "...", "content": "..."}]}
    - Simple text: {"text": "..."} or just raw strings
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        format_type: str = "instruction",
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data: List of data samples (dicts or strings)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            format_type: One of "instruction", "chat", "text"
            prompt_template: Optional custom template for formatting
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
        self.prompt_template = prompt_template

        # Default instruction template
        if self.prompt_template is None and format_type == "instruction":
            self.prompt_template = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n{output}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        text = self._format_sample(sample)

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_sample(self, sample: Dict[str, Any]) -> str:
        """Format a sample according to the specified format type."""
        if self.format_type == "instruction":
            return self._format_instruction(sample)
        elif self.format_type == "chat":
            return self._format_chat(sample)
        elif self.format_type == "text":
            return self._format_text(sample)
        else:
            raise ValueError(f"Unknown format_type: {self.format_type}")

    def _format_instruction(self, sample: Dict[str, Any]) -> str:
        """Format instruction-style data."""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", sample.get("response", ""))

        if self.prompt_template:
            return self.prompt_template.format(
                instruction=instruction, input=input_text, output=output
            )
        else:
            # Simple format
            return f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"

    def _format_chat(self, sample: Dict[str, Any]) -> str:
        """Format chat-style data using tokenizer's chat template."""
        messages = sample.get("messages", [])
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Fallback: concatenate messages
            text_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text_parts.append(f"{role}: {content}")
            return "\n".join(text_parts)

    def _format_text(self, sample: Dict[str, Any]) -> str:
        """Format simple text data."""
        if isinstance(sample, str):
            return sample
        return sample.get("text", str(sample))


def load_finetune_data(
    data_path: Union[str, Path],
    split: str = "train",
    format_type: str = "instruction",
    validation_split: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Load fine-tuning data from various formats.

    Args:
        data_path: Path to data file or directory
        split: Data split ('train', 'val', 'test', or 'all')
        format_type: Format type for the data
        validation_split: Fraction of data to use for validation
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_data, val_data, test_data) - each is a list of samples
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # Load based on file extension
    if data_path.suffix == ".json" or data_path.suffix == ".jsonl":
        data = _load_json_data(data_path)
    elif data_path.suffix == ".csv":
        data = _load_csv_data(data_path)
    elif data_path.suffix in [".txt", ".text"]:
        data = _load_text_data(data_path)
    else:
        # Try to auto-detect or load as HuggingFace dataset
        try:
            from datasets import load_dataset
            dataset = load_dataset(str(data_path))
            data = list(dataset)
        except Exception as e:
            raise ValueError(f"Cannot load data format: {data_path.suffix}. Error: {e}")

    # Split data
    set_seed(seed)
    import random
    random.shuffle(data)

    n_samples = len(data)
    n_val = int(n_samples * validation_split)
    n_test = int(n_samples * validation_split)
    n_train = n_samples - n_val - n_test

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:n_train + n_val + n_test]

    if split == "train":
        return train_data
    elif split == "val":
        return val_data
    elif split == "test":
        return test_data
    else:
        return train_data, val_data, test_data


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
                # Handle different JSON structures
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
    # Convert to list of dicts
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


class FinetuneDataModule(L.LightningDataModule):
    """
    Lightning data module for LLM fine-tuning.

    Handles dataset loading, tokenization, and data loading for training.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        max_length: int = 512,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 42,
        format_type: str = "instruction",
        validation_split: float = 0.1,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the data module.

        Args:
            data_path: Path to training data
            tokenizer: Pre-loaded tokenizer (optional, will load from model_name if not provided)
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
            format_type: Format type for the data
            validation_split: Fraction of data to use for validation
            prompt_template: Optional custom template for formatting
        """
        super().__init__()

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.format_type = format_type
        self.validation_split = validation_split
        self.prompt_template = prompt_template

        self.train_data: Optional[List] = None
        self.val_data: Optional[List] = None
        self.test_data: Optional[List] = None

        self.train_dataset: Optional[FinetuneDataset] = None
        self.val_dataset: Optional[FinetuneDataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        Load tokenizer and create datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        set_seed(self.seed)

        # Load tokenizer if not provided
        if self.tokenizer is None:
            print(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data
        print(f"Loading data from {self.data_path}")
        self.train_data, self.val_data, self.test_data = load_finetune_data(
            data_path=self.data_path,
            split="all",
            format_type=self.format_type,
            validation_split=self.validation_split,
            seed=self.seed,
        )

        print(f"Loaded {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test samples")

        # Create datasets
        self.train_dataset = FinetuneDataset(
            data=self.train_data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            format_type=self.format_type,
            prompt_template=self.prompt_template,
        )

        self.val_dataset = FinetuneDataset(
            data=self.val_data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            format_type=self.format_type,
            prompt_template=self.prompt_template,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the datasets."""
        return {
            "data_path": self.data_path,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "format_type": self.format_type,
            "train_samples": len(self.train_data) if self.train_data else 0,
            "val_samples": len(self.val_data) if self.val_data else 0,
            "test_samples": len(self.test_data) if self.test_data else 0,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 0,
        }
