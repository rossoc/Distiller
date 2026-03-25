from dotenv import load_dotenv
from os import getenv
import pandas as pd
import numpy as np
import yaml


from typing import Tuple


load_dotenv()


def simple_ner(filename):
    d_gt = pd.read_excel(filename, sheet_name="Ground Truth")

    with open("config.yaml", "r") as file:
        settings = yaml.safe_load(file)["simple_diffusion"]

    X = d_gt[settings["feature"]].fillna("").astype(str).agg(". ".join, axis=1)
    y = d_gt[settings["target"]]

    return X, y


def all(filename):
    d_gt = pd.read_excel(filename, sheet_name="Ground Truth")

    with open("config.yaml", "r") as file:
        settings = yaml.safe_load(file)["all"]

    X = d_gt[settings["feature"]].fillna("").astype(str).agg(". ".join, axis=1)
    y = d_gt.iloc[:, 7:31]

    return X, y


def simple_diffusion(filename):
    d_gt = pd.read_excel(filename, sheet_name="Ground Truth")

    with open("config.yaml", "r") as file:
        settings = yaml.safe_load(file)["one"]

    X = d_gt[settings["feature"]].fillna("").astype(str).agg(". ".join, axis=1)

    y = pd.Series("", index=d_gt.index)
    for col in settings["target"]:
        y += col + ": " + d_gt[col].fillna("").astype(str) + "\n"

    return X, y


def by_group(filename):
    d_gt = pd.read_excel(filename, sheet_name="Ground Truth")

    with open("config.yaml", "r") as file:
        settings = yaml.safe_load(file)["by_group"]

    X = d_gt[settings["feature"]].fillna("").astype(str).agg(". ".join, axis=1)
    y = [d_gt[col] for col in settings["target"]]
    return X, y


def split_dataset(X, y, train_ratio, eval_ratio):
    indices = np.random.permutation(len(X))
    train_size = int(train_ratio * len(X))
    eval_size = int(eval_ratio * len(X))

    train_idx = indices[:train_size]
    eval_idx = indices[train_size : train_size + eval_size]
    test_idx = indices[train_size + eval_size :]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_eval, y_eval = X.iloc[eval_idx], y.iloc[eval_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Reset indices to be sequential (0, 1, 2, ...) for proper integer indexing
    X_train = X_train.reset_index(drop=True)
    X_eval = X_eval.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_eval = y_eval.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return ((X_train, y_train), (X_eval, y_eval), (X_test, y_test))


def load_dataset(
    split_ratio: Tuple[float, float, float],
    schema: str = "simple_diffusion",
):
    datasets = {
        "simple_ner": simple_ner,
        "simple_diffusion": simple_diffusion,
        "field": simple_ner,  # todo!
        "all": all,
    }

    # schema = "simple_diffusion"

    # dataset_variation = Datasets_Variations.SIMPLE_DIFFUSION
    if schema not in datasets:
        raise ValueError(f"""Wrong value for dataset_name, expected one of
                         {", ".join([str(i) for i in Datasets_Variations])}, found:
                         {schema}""")

    filename = getenv("dataset", "")
    X, y = datasets[schema](filename)

    return split_dataset(X, y, split_ratio[0], split_ratio[1])
