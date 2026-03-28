#!/usr/bin/env python3
"""
Batch inference script for fine-tuned Qwen v2 models using data from an Excel file.

This script loads a fine-tuned Unsloth model, reads prompts from an .xlsx file
by concatenating 'S_text' and 'L_text' columns, runs inference, and saves
the results to a new Excel file.

Usage:
    python src/run_qwen_finetuned_inference_v2.py
        --model_path "outputs"
        --input_file "data/data_district_heating.xlsx"
        --sheet_name "Sheet1"
        --output_file "outputs/inference_results.xlsx"
"""

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
import json
from datetime import datetime


def main():
    """
    Main function to run the batch inference script.
    """
    parser = argparse.ArgumentParser(
        description="Run batch inference with a Qwen v2 model using an Excel file."
    )
    # Model and data arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned adapter model directory.",
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input .xlsx file."
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        required=True,
        help="Name of the sheet to read from the Excel file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output file with results.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="xlsx",
        choices=["xlsx", "json"],
        help="Format to save the output results (xlsx or json).",
    )

    # Generation arguments
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for the model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference.",
    )
    args = parser.parse_args()

    # --- 1. Load Model ---
    print(f"Loading model from: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model.eval()
    FastLanguageModel.for_inference(model)

    # Set up tokenizer for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load Data ---
    print(f"Loading data from '{args.input_file}' sheet '{args.sheet_name}'...")
    try:
        df = pd.read_excel(args.input_file, sheet_name=args.sheet_name)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Validate columns
    if "S_text" not in df.columns or "L_text" not in df.columns:
        print("Error: The Excel sheet must contain 'S_text' and 'L_text' columns.")
        return

    # Fill any NaN values in the text columns to prevent errors
    df["S_text"] = df["S_text"].fillna("")
    df["L_text"] = df["L_text"].fillna("")

    # --- 3. Run Batch Inference ---
    inference_results = []
    prompt_template = """<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"""

    print(f"Running inference on {len(df)} rows with batch size {args.batch_size}...")
    
    # Process in batches
    for i in tqdm(range(0, len(df), args.batch_size), desc="Inference"):
        batch_df = df.iloc[i:i + args.batch_size]
        
        batch_prompts = []
        batch_original_texts = []
        
        for _, row in batch_df.iterrows():
            prompt_text = str(row["S_text"]) + " " + str(row["L_text"])
            batch_original_texts.append(prompt_text)
            batch_prompts.append(prompt_template.format(prompt_text))

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            padding=True, 
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        # Decode batch
        # We need to slice the output to only get the newly generated tokens
        input_lengths = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_lengths:]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for original_text, decoded_output in zip(batch_original_texts, decoded_outputs):
            assistant_response = decoded_output.strip()
            
            # Store results for both JSON and XLSX output
            inference_results.append({
                "original_prompt": original_text,
                "generated_response": assistant_response
            })

    # --- 4. Save Results ---
    print(f"\nSaving results to '{args.output_file}'...")
    if args.output_format == "json":
        json_output = {
            "timestamp": datetime.now().isoformat(),
            "model_path": args.model_path,
            "n_samples": len(df),
            "results": [
                {
                    "input": res["original_prompt"],
                    "predictions": [{"text": res["generated_response"]}]
                } for res in inference_results
            ]
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print("Inference complete. Results saved successfully in JSON format.")
    else: # Default to xlsx
        # Add generated responses back to the original DataFrame for XLSX output
        df["generated_response"] = [res["generated_response"] for res in inference_results]
        try:
            pd.DataFrame(df).to_excel(args.output_file, index=False)
            print("Inference complete. Results saved successfully in XLSX format.")
        except Exception as e:
            print(f"Error saving results to Excel file: {e}")


if __name__ == "__main__":
    main()
