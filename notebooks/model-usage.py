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
from src.util.inference import (
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
json_path = "outputs/inference/inference_results_20260324_155925.json"
print(f"Loading inference results from: {json_path}")
result = parse_inference(json_path)
print(f"Loaded {len(result)} samples")
# %% [markdown]
# Compute and print overall statistics
# %%
stats = compute_match_statistics(json_path)
print_match_statistics(stats)
# %%
metrics = compute_per_label_metrics(result)
print_per_label_metrics(metrics)
# %%
