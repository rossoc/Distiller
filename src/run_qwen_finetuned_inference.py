#!/usr/bin/env python3
"""
Inference script for fine-tuned LLMs (Qwen, Llama, etc.).

This script loads a fine-tuned model and generates text based on user input.

Usage:
    # Interactive mode
    python src/run_qwen_finetuned_inference.py --model_dir outputs/finetuned_model

    # Single prompt
    python src/run_qwen_finetuned_inference.py \
        --model_dir outputs/finetuned_model \
        --prompt "What is machine learning?"

    # Batch inference from file
    python src/run_qwen_finetuned_inference.py \
        --model_dir outputs/finetuned_model \
        --input_file prompts.txt \
        --output_file generations.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from model.qwen_finetuner import QwenFineTuner, FineTuningConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned LLM"
    )

    # Model arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Base model name (optional, will load from config if not specified)",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Path to LoRA adapter (if separate from model_dir)",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for generation",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save generations",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling for generation",
    )
    parser.add_argument(
        "--no_sample",
        action="store_false",
        dest="do_sample",
        help="Use greedy decoding (no sampling)",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate per prompt",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on",
    )

    # Interactive mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Run in interactive mode (chat-like)",
    )

    return parser.parse_args()


def load_finetuned_model(
    model_dir: str,
    base_model_name: Optional[str] = None,
    adapter_dir: Optional[str] = None,
    device: str = "auto",
) -> tuple:
    """
    Load a fine-tuned model.

    Args:
        model_dir: Path to fine-tuned model directory
        base_model_name: Base model name (optional)
        adapter_dir: Path to LoRA adapter (optional)
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, config)
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load config
    config_file = model_path / "finetune_config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        base_model_name = base_model_name or config_dict.get("model_name", "unsloth/Qwen3.5-0.8B-Q8_0")
        use_lora = config_dict.get("use_lora", True)
        print(f"Loaded config from {config_file}")
        print(f"  Base model: {base_model_name}")
        print(f"  Uses LoRA: {use_lora}")
    else:
        print(f"No config file found, using defaults")
        use_lora = True

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # Load base model
    print(f"\nLoading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto" if device in ["cuda", "auto"] else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Load LoRA adapter if applicable
    adapter_path = adapter_dir if adapter_dir else model_dir
    if use_lora:
        adapter_config = Path(adapter_path) / "adapter_config.json"
        if adapter_config.exists():
            print(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            # Merge adapter weights for faster inference
            print("Merging adapter weights...")
            model = model.merge_and_unload()
        else:
            print("No adapter config found, using base model only")

    # Move to device if not already
    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    # Load tokenizer
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!\n")

    return model, tokenizer, config_dict


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    repetition_penalty: float = 1.1,
) -> List[str]:
    """
    Generate text from a prompt.

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        do_sample: Whether to use sampling
        num_return_sequences: Number of sequences to generate
        repetition_penalty: Repetition penalty

    Returns:
        List of generated texts
    """
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Determine if we need sampling parameters
    sampling_kwargs = {}
    if do_sample:
        sampling_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **sampling_kwargs,
        )

    # Decode outputs
    input_length = inputs["input_ids"].shape[1]
    generated_texts = []
    for output in outputs:
        generated_text = tokenizer.decode(
            output[input_length:],
            skip_special_tokens=True,
        )
        generated_texts.append(generated_text)

    return generated_texts


def interactive_mode(model, tokenizer, args):
    """Run interactive chat-like mode."""
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'clear' to clear conversation history")
    print("=" * 60 + "\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if user_input.lower() == "clear":
                history = []
                print("Conversation history cleared.\n")
                continue

            if not user_input:
                continue

            # Build prompt with history
            if history:
                prompt = "\n".join(history) + f"\nYou: {user_input}\nAssistant:"
            else:
                prompt = f"You: {user_input}\nAssistant:"

            # Generate response
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                num_return_sequences=1,
            )[0]

            print(f"Assistant: {response}\n")

            # Update history
            history.append(f"You: {user_input}")
            history.append(f"Assistant: {response}")

            # Limit history length
            if len(history) > 20:
                history = history[-20:]

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    args = parse_args()

    # Load model
    model, tokenizer, config = load_finetuned_model(
        model_dir=args.model_dir,
        base_model_name=args.base_model_name,
        adapter_dir=args.adapter_dir,
        device=args.device,
    )

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, args)
        return

    # Single prompt
    if args.prompt:
        print(f"\nPrompt: {args.prompt}\n")
        responses = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            repetition_penalty=args.repetition_penalty,
        )

        for i, response in enumerate(responses, 1):
            if args.num_return_sequences > 1:
                print(f"\n--- Response {i} ---")
            print(response)

        return

    # Batch inference from file
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            return

        with open(input_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(prompts)} prompts...")

        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] Processing: {prompt[:50]}...")
            responses = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.do_sample,
                num_return_sequences=args.num_return_sequences,
                repetition_penalty=args.repetition_penalty,
            )
            results.append({
                "prompt": prompt,
                "responses": responses,
            })

        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
        else:
            # Print results
            print("\n" + "=" * 60)
            print("Results:")
            print("=" * 60)
            for item in results:
                print(f"\nPrompt: {item['prompt']}")
                for i, response in enumerate(item['responses'], 1):
                    if len(item['responses']) > 1:
                        print(f"  Response {i}: {response}")
                    else:
                        print(f"  Response: {response}")

        return

    # No input specified - show usage
    print("\nNo input specified. Use one of:")
    print("  --prompt 'Your prompt here'  # Single prompt")
    print("  --input_file prompts.txt     # Batch from file")
    print("  --interactive                # Interactive chat mode")
    print("\nExample:")
    print(f"  python {args.__class__.__name__} --model_dir {args.model_dir} --prompt 'Hello, how are you?'")


if __name__ == "__main__":
    main()
