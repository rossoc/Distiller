# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: energylabels
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Development


# %% Create Config
from dotenv import load_dotenv
from os import getenv
from src.data.util import by_group
from src.util.logger import write_data_config, read_data_config
from src.util.result_analysis import (
    compute_per_label_metrics,
    parse_inference,
    compute_match_statistics,
    print_match_statistics,
    print_per_label_metrics,
)

load_dotenv()

# %%

filename = getenv("dataset", "")
config = getenv("config", "")

schemas = read_data_config(config)

# print(schemas)
# print(schemas["all"])

feature = schemas["all"]["feature"]
target = [schemas["all"]["target"][i::6] for i in range(6)]
schemas["by_group"] = schemas["all"].copy()
schemas["by_group"]["target"] = target

# write_data_config("by_group", schemas["by_group"], config)
# print(schemas)

# import by_group
# print(by_group(filename)[1])


# %% [markdown]
# # Check inference statistics
# %%
json_path = "outputs/diffusion_inference/inference_results_20260325_213809.json"
print(f"Loading inference results from: {json_path}")
result = parse_inference(json_path)
print(f"Loaded {len(result)} samples")
# %%
stats = compute_match_statistics(json_path)
print_match_statistics(stats)
# %%
metrics = compute_per_label_metrics(result)
print_per_label_metrics(metrics)
# %%
json_path = "outputs/diffusion_inference/inference_results_20260326_070130.json"
print(f"Loading inference results from: {json_path}")
result = parse_inference(json_path)
print(f"Loaded {len(result)} samples")
# %%
stats = compute_match_statistics(json_path)
print_match_statistics(stats)
# %%
metrics = compute_per_label_metrics(result)
print_per_label_metrics(metrics)
# %% [markdown]
# Best small model:
#   - num_layers < 8
#   - ~13M parameters
#   - found with optuna (over 200 trials on wide selection of hyperparameters)
# ```sh
# python src/train_diffusion.py \
# --num_heads 4 \
# --num_layers 2 \
# --fwd_dim 1024 \
# --learning_rate 0.0009388399822421634 \
# --weight_decay 0.012536891876922682 \
# --dropout 0.3 \
# --loss_alpha 0.9 \
# --label_smoothing 0.01 \
# --epochs 200 \
# --batch_size 64
# ```
# %%
json_path = "outputs/diffusion_inference/inference_results_20260327_083425.json"
result = parse_inference(json_path)
stats = compute_match_statistics(json_path)
print_match_statistics(stats)
# %% [markdown]
# Changing the loss_alpha value changes the loss and I was curious to see how
# the model was affected:
# - loss_alpha = 0.5
# Apparently in prediction the change is substantial
# %%
json_path = "outputs/diffusion_inference/inference_results_20260327_083559.json"
result = parse_inference(json_path)
stats = compute_match_statistics(json_path)
print_match_statistics(stats)
# %% [markdown]
# loss_alpha = 0.1
# %%
json_path = "outputs/diffusion_inference/inference_results_20260327_084642.json"
result = parse_inference(json_path)
stats = compute_match_statistics(json_path)
print_match_statistics(stats)
# %% [markdown]
# loss_alpha = 0.1
# epochs = 1000
# %%
json_path = "outputs/diffusion_inference/inference_results_20260327_094648.json"
result = parse_inference(json_path)
stats = compute_match_statistics(json_path)
print_match_statistics(stats)

# metrics = compute_per_label_metrics(result)
# print_per_label_metrics(metrics)
# %%
