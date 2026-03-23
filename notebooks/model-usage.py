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

load_dotenv()


# %%

filename = getenv("dataset", "")
config = getenv("config", "")

schemas = read_data_config(config)
print(schemas)
print(schemas["all"])

feature = schemas["all"]["feature"]
target = [schemas["all"]["target"][i::6] for i in range(6)]
schemas["by_group"] = schemas["all"].copy()
schemas["by_group"]["target"] = target

# write_data_config("by_group", schemas["by_group"], config)
# print(schemas)

# import by_group
# print(by_group(filename)[1])

# %%

# %%
