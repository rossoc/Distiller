"""
Fine-tuning module for LLMs (Qwen, Llama, etc.) using LoRA/QLoRA.

This module provides:
- Model loading with optional quantization (QLoRA)
- LoRA adapter configuration and application
- PyTorch Lightning training wrapper
- Model saving and loading utilities
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from peft.tuners.lora import LoraLayer
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha scaling
    lora_dropout: float = 0.05  # Dropout rate
    target_modules: Optional[List[str]] = None  # Modules to apply LoRA to
    bias: str = "none"  # Bias training: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantizationConfig:
    """Configuration for 4-bit/8-bit quantization (QLoRA)."""
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"  # Normal Float 4-bit
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True  # Nested quantization


@dataclass
class FineTuningConfig:
    """Complete fine-tuning configuration."""
    # Model
    model_name: str = "unsloth/Qwen3.5-0.8B-Q8_0"
    use_lora: bool = True
    use_quantization: bool = True
    
    # LoRA settings
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Quantization settings
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    
    # Optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "fp16", "bf16", or "no"
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    # Output
    output_dir: str = "outputs/finetuned_model"
    run_name: Optional[str] = None


class QwenFineTuner(L.LightningModule):
    """
    PyTorch Lightning module for fine-tuning Qwen and similar models.

    Supports:
    - Full fine-tuning
    - LoRA (Low-Rank Adaptation)
    - QLoRA (Quantized LoRA)

    Example:
        >>> config = FineTuningConfig(
        ...     model_name="Qwen/Qwen2.5-0.5B",
        ...     use_lora=True,
        ...     use_quantization=True,
        ... )
        >>> model = QwenFineTuner(config)
        >>> model.setup_model()
    """

    def __init__(
        self,
        config: FineTuningConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        Initialize the fine-tuner.

        Args:
            config: Fine-tuning configuration
            tokenizer: Optional pre-loaded tokenizer
        """
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.model: Optional[PreTrainedModel] = None

        # Save config for hyperparameters
        self.save_hyperparameters()

    def setup_model(self):
        """
        Load and configure the model.

        This should be called before training.
        """
        print(f"Loading model: {self.config.model_name}")

        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": self._get_torch_dtype(),
        }

        # Add quantization if enabled
        if self.config.use_quantization:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=self.config.quantization.load_in_4bit,
                load_in_8bit=self.config.quantization.load_in_8bit,
                bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
            )
            model_kwargs["quantization_config"] = quant_config
            print(f"Using quantization: 4bit={self.config.quantization.load_in_4bit}, 8bit={self.config.quantization.load_in_8bit}")
        else:
            # Remove device_map for non-quantized models (we'll handle device ourselves)
            model_kwargs.pop("device_map", None)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        # Load tokenizer if not provided
        if self.tokenizer is None:
            print(f"Loading tokenizer: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA if enabled
        if self.config.use_lora:
            print(f"Applying LoRA adapters (r={self.config.lora.r}, alpha={self.config.lora.lora_alpha})")
            self._apply_lora()

        # Print model info
        self._print_model_info()

    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from config string."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(
            self.config.quantization.bnb_4bit_compute_dtype,
            torch.float16
        )

    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        # Prepare model for k-bit training if using quantization
        if self.config.use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        # Determine target modules
        target_modules = self.config.lora.target_modules
        if target_modules is None:
            # Auto-detect based on model type
            target_modules = self._auto_detect_target_modules()
            print(f"Auto-detected target modules: {target_modules}")

        # Create LoRA config
        peft_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=target_modules,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)

    def _auto_detect_target_modules(self) -> List[str]:
        """Auto-detect target modules based on model architecture."""
        model_type = self.config.model_name.lower()

        # Qwen models (including Qwen3.5)
        if "qwen" in model_type:
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Llama models
        elif "llama" in model_type:
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Mistral models
        elif "mistral" in model_type:
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Gemma models
        elif "gemma" in model_type:
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Default: attention layers only
        else:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]

    def _print_model_info(self):
        """Print model information."""
        if self.model is None:
            return

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*50}")
        print(f"Model: {self.config.model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

        if self.config.use_lora:
            print(f"\nLoRA configuration:")
            print(f"  Rank (r): {self.config.lora.r}")
            print(f"  Alpha: {self.config.lora.lora_alpha}")
            print(f"  Dropout: {self.config.lora.lora_dropout}")

        print(f"{'='*50}\n")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a training step."""
        outputs = self.forward(batch)
        loss = outputs.loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Perform a validation step."""
        with torch.no_grad():
            outputs = self.forward(batch)
            loss = outputs.loss

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get parameters to optimize
        if self.config.use_lora:
            # Only optimize LoRA parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = self.model.parameters()

        # Optimizer
        optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def save_model(self, output_dir: str):
        """
        Save the fine-tuned model.

        Args:
            output_dir: Directory to save the model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving model to {output_dir}")

        if self.config.use_lora:
            # Save LoRA adapter only
            self.model.save_pretrained(output_dir)
        else:
            # Save full model
            self.model.save_pretrained(output_dir)

        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        # Save config
        import json
        config_dict = {
            "model_name": self.config.model_name,
            "use_lora": self.config.use_lora,
            "use_quantization": self.config.use_quantization,
            "lora_config": {
                "r": self.config.lora.r,
                "lora_alpha": self.config.lora.lora_alpha,
                "lora_dropout": self.config.lora.lora_dropout,
            },
        }
        with open(output_path / "finetune_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Model saved successfully to {output_dir}")

    @classmethod
    def load_model(
        cls,
        model_dir: str,
        config: Optional[FineTuningConfig] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a fine-tuned model.

        Args:
            model_dir: Directory containing saved model
            config: Optional fine-tuning config (will load from file if not provided)

        Returns:
            Tuple of (model, tokenizer)
        """
        import json

        model_path = Path(model_dir)

        # Load config if not provided
        if config is None:
            config_file = model_path / "finetune_config.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                config = FineTuningConfig(
                    model_name=config_dict.get("model_name", "unsloth/Qwen3.5-0.8B-Q8_0"),
                    use_lora=config_dict.get("use_lora", True),
                    use_quantization=config_dict.get("use_quantization", False),
                )
                if "lora_config" in config_dict:
                    config.lora.r = config_dict["lora_config"]["r"]
                    config.lora.lora_alpha = config_dict["lora_config"]["lora_alpha"]
                    config.lora.lora_dropout = config_dict["lora_config"]["lora_dropout"]
            else:
                # Default config
                config = FineTuningConfig()

        # Load base model
        print(f"Loading base model: {config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

        # Load LoRA adapter if applicable
        if config.use_lora:
            print(f"Loading LoRA adapter from {model_dir}")
            model = PeftModel.from_pretrained(model, model_dir)
            model = model.merge_and_unload()  # Optional: merge adapter weights

        # Load tokenizer
        print(f"Loading tokenizer from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        return model, tokenizer

    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from the model.

        Args:
            input_text: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return generated_text


def create_finetuning_config(
    model_name: str = "unsloth/Qwen3.5-0.8B-Q8_0",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    use_quantization: bool = True,
    **kwargs,
) -> FineTuningConfig:
    """
    Create a fine-tuning configuration with common presets.

    Args:
        model_name: HuggingFace model name
        lora_rank: LoRA rank (higher = more parameters)
        lora_alpha: LoRA alpha scaling
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        use_quantization: Whether to use 4-bit quantization (QLoRA)
        **kwargs: Additional config overrides

    Returns:
        FineTuningConfig instance
    """
    config = FineTuningConfig(
        model_name=model_name,
        lora=LoRAConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
        ),
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        use_quantization=use_quantization,
    )

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
