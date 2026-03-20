from dotenv import load_dotenv
from os import getenv
import pandas as pd

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


def load_dataset(dataset_name, is_train):
    if dataset_name not in ["simple", "fields", "enumeration", "all"]:
        raise ValueError("""Wrong value for dataset_name, expected one of
                         "simple", "fields", "enumeration", "all", found:
                         {dataset_name}""")

    datasets = {
        "simple_ner": simple_ner,
        "simple_diffusion": simple_diffusion,
        "field": simple_ner,
        "all": simple_ner,
    }

    file = getenv("dataset", "")
    datasets[dataset_name](file)
