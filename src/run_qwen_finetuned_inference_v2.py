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

def main():
    """
    Main function to run the batch inference script.
    """
    parser = argparse.ArgumentParser(
        description="Run batch inference with a Qwen v2 model using an Excel file."
    )
    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned adapter model directory.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .xlsx file.")
    parser.add_argument("--sheet_name", type=str, required=True, help="Name of the sheet to read from the Excel file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output .xlsx file with results.")

    # Generation arguments
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate per prompt.")
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
    if 'S_text' not in df.columns or 'L_text' not in df.columns:
        print("Error: The Excel sheet must contain 'S_text' and 'L_text' columns.")
        return
    
    # Fill any NaN values in the text columns to prevent errors
    df['S_text'] = df['S_text'].fillna('')
    df['L_text'] = df['L_text'].fillna('')

    # --- 3. Run Batch Inference ---
    results = []
    prompt_template = "<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"

    print(f"Running inference on {len(df)} rows...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Inference"):
        # Concatenate columns to form the prompt
        prompt_text = str(row['S_text']) + " " + str(row['L_text'])
        full_prompt = prompt_template.format(prompt_text)

        # Tokenize and generate
        inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and clean response
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        try:
            assistant_response = decoded_output.split("<|im_start|>assistant
")[1].strip()
        except IndexError:
            assistant_response = "ERROR: Model did not generate a response in the expected format."
        
        results.append(assistant_response)

    # --- 4. Save Results ---
    df['generated_response'] = results
    print(f"
Saving results to '{args.output_file}'...")
    try:
        # Create output directory if it doesn't exist
        pd.DataFrame(df).to_excel(args.output_file, index=False)
        print("Inference complete. Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to Excel file: {e}")


if __name__ == "__main__":
    main()
