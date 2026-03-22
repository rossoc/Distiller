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
# # Data Exploration
#
# ## Data Import

# %%
import numpy as np
import pandas as pd
import sys


# %%

data_folder = "data/"

# %%
d_hjorring = pd.read_excel(
    data_folder + "data_district_heating.xlsx", sheet_name="Hjorring"
)
d_favrskov = pd.read_excel(
    data_folder + "data_district_heating.xlsx", sheet_name="Favrskov"
)
d_aarhus = pd.read_excel(
    data_folder + "data_district_heating.xlsx", sheet_name="Aarhus"
)
d_gt = pd.read_excel(
    data_folder + "data_district_heating.xlsx", sheet_name="Ground Truth"
)
print(d_gt.head())

# %% [markdown]
# ## Data Size

# %%
print("Data Favrskov shape: ", d_favrskov.shape)
print("Data Hjorring shape: ", d_hjorring.shape)
print("Data Aarhus shape: ", d_aarhus.shape)
print("Data Ground Truth shape: ", d_gt.shape)

# %% [markdown]
# Let's merge the dataset on the SerialID to check whether the ground truth data are present in the rest of the samples.

# %%
d_f_gt = pd.merge(left=d_favrskov, right=d_gt, on="SerialID")
d_h_gt = pd.merge(left=d_hjorring, right=d_gt, on="SerialID")
d_a_gt = pd.merge(left=d_aarhus, right=d_gt, on="SerialID")
print(d_f_gt.shape)
print(d_h_gt.shape)
print(d_a_gt.shape)

# %% [markdown]
# From the previous result, we can say that the ground truth is not present in data.
# Thus we have 2272 labeled samples.

# %%
print(d_favrskov.shape[0] + d_hjorring.shape[0] + d_aarhus.shape[0])

# %% [markdown]
# On the other hand, we have 35791 entries we want to label.

# %% [markdown]
# ## Features of the Ground Truths

# %%
print(d_gt[d_gt["SerialID"] == 311795931].iloc[0, 0:31])

# %%
d_gt_labels = d_gt.iloc[:, 7:31]
print(d_gt_labels.columns)
print("Number of lables: ", d_gt_labels.shape[1])

# %%
print(100 - d_gt_labels.isnull().mean() * 100)

# %% [markdown]
# From the previous result, we know that every entry has informations about `Pieces1` and `Manufacturer1`.
# We can see that `NominalEffectEach4` is always empty, and thus we do not have information about this column.
# Finally, we can see that data are very sparse, indeed very few columns are filled.

# %%
null_counts = d_gt_labels.isnull().sum(axis=1)
missing_values = null_counts.value_counts()

missing_index = np.array(missing_values.index)
missing_index.sort()
missing_values = np.array(missing_values[missing_index])
missing_values_cum = np.cumsum(missing_values)
percentages = missing_values_cum / missing_values_cum[-1] * 100

for i in range(missing_index.shape[0]):
    print(
        "Number of samples: ",
        missing_values_cum[i],
        f"\tCorresponding percentage: {percentages[i]:.3f} No more than {missing_index[i]} entries missing",
    )

# %%
print(
    f"Average length of S_text: {d_gt['S_text'].str.len().mean():.2f}\t standard deviation: {d_gt['S_text'].str.len().std():.2f}"
)
print(
    f"Average length of L_text: {d_gt['L_text'].str.len().mean():.2f} standard deviation: {d_gt['L_text'].str.len().std():.2f}"
)

# %% [markdown]
# # Export Gold Label in csv

# %%
d_gt["text"] = (
    d_gt["S_text"].fillna("").astype(str) + ". " + d_gt["L_text"].fillna("").astype(str)
)
print(
    "Total number of sentences: ", (d_gt["text"].str.count(r"[.!?](?:\s|$)") + 1).sum()
)

# %%
print("Total number of lables: ", d_gt_labels.count().sum())

# %%
n_sentences = (d_gt["text"].str.count(r"[.!?](?:\s|$)") + 1).sum()
n_labels = d_gt_labels.count().sum()
print(
    "Labels on sentences ratio: ",
    n_labels / n_sentences,
    " Labels on sentences percentage: ",
    n_labels / n_sentences * 100,
)

# %%
for col in d_gt_labels.columns:
    print(f"{col} has the following values: {d_gt_labels[col].value_counts()}")

# %%
col = "HxType1"
print(f"{col} has the following values: {d_gt_labels[col].value_counts()}")

# %%
print(d_gt.loc[195, "text"])


# %%

for col in d_gt_labels.columns[:6]:
    print(col, "has", len(d_gt_labels[col].unique()), "unique fields")


# %%
d_gt_labels["Manufacturer1"].unique()


# %%


# %%
# %%
