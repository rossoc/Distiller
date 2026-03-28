"""
Fine-tuning module for Qwen models using Unsloth and SFTTrainer (v2).
"""
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

class QwenFinetunerV2:
    def __init__(self, model_name, max_seq_length, load_in_4bit=True, **training_args):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.training_args = training_args
        self.model, self.tokenizer = self._init_model()
        self._setup_lora()

    def _init_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            dtype=None,  # Auto-detect
        )
        return model, tokenizer

    def _setup_lora(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    def train(self, train_dataset):
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            args=TrainingArguments(**self.training_args),
        )
        trainer_stats = trainer.train()
        return trainer_stats
