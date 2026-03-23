from dotenv import load_dotenv
from os import getenv
import pandas as pd
import numpy as np


from typing import Tuple
from enum import Enum


class Datasets_Variations(Enum):
    SIMPLE_NER = 0  # todo!
    SIMPLE_DIFFUSION = 1
    FIELD = 2  # todo!
    ALL = 3  # todo!


load_dotenv()


def simple_ner(filename):
    d_gt = pd.read_excel(filename, sheet_name="Ground Truth")

    X = (
        d_gt["S_text"].fillna("").astype(str)
        + ". "
        + d_gt["L_text"].fillna("").astype(str)
    )

    y = d_gt[
        "Pieces1", "Manufacturer1", "SubType1", "HxType1", "NominelEffectEach1", "Year1"
    ]

    return X, y


def simple_diffusion(filename):
    d_gt = pd.read_excel(filename, sheet_name="Ground Truth")

    X = (
        d_gt["S_text"].fillna("").astype(str)
        + ". "
        + d_gt["L_text"].fillna("").astype(str)
    )

    y = (
        "Piece: "
        + d_gt["Pieces1"].fillna("").astype(str)
        + "\n"
        + "Manufacturer: "
        + d_gt["Manufacturer1"].fillna("").astype(str)
        + "\n"
        + "SubType: "
        + d_gt["SubType1"].fillna("").astype(str)
        + "\n"
        + "HxType: "
        + d_gt["HxType1"].fillna("").astype(str)
        + "\n"
        + "NominelEffectEach: "
        + d_gt["NominelEffectEach1"].fillna("").astype(str)
        + "\n"
        + "Year: "
        + d_gt["Year1"].fillna("").astype(str)
    )
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

    return ((X_train, y_train), (X_eval, y_eval), (X_test, y_test))


def load_dataset(
    dataset_variation: Datasets_Variations, split_ratio: Tuple[float, float, float]
):
    if dataset_variation not in Datasets_Variations:
        raise ValueError(f"""Wrong value for dataset_name, expected one of
                         {", ".join([str(i) for i in Datasets_Variations])}, found:
                         {dataset_variation}""")

    datasets = [
        simple_ner,
        simple_diffusion,
        simple_ner,
        simple_ner,
    ]

    file = getenv("dataset", "")
    X, y = datasets[dataset_variation.value](file)

    return split_dataset(X, y, split_ratio[0], split_ratio[1])
