"""Shared settings, paths and data loading."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
PARAMS_FILE = PROJECT_DIR / "params.yaml"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MLFLOW_URI = "sqlite:///" + (PROJECT_DIR / "mlflow.db").as_posix()

# Name of the extra input that carries the prediction of the layer above
UPPER_COL = "SM layer above"


def load_params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def all_features(params):
    """Union of the inputs used by any depth, in a stable order."""
    out = []
    for feats in params["features"].values():
        out += [f for f in feats if f not in out]
    return out


def load_data(params):
    """Rows with measured soil moisture from 30 to 150 cm (180 cm is a subset).

    Soil moisture is stored in % and converted to cm³/cm³ (÷100) here.
    """
    df = pd.read_csv(PROJECT_DIR / params["data_file"])
    df = df.rename(columns={"Fields": "Field", "Points": "Point"})
    df = df.dropna(subset=["S150"]).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    for d in params["depths"]:
        df[d] = df[d] / 100
    keep = ["Date", "Year", "Field", "Point"] + all_features(params) + params["depths"]
    return df[keep]


def split_data(df, params):
    """One shared train/test split for every depth, so no test row is ever seen by any model in the chain."""
    s = params["split"]
    return train_test_split(df, test_size=s["test_size"], random_state=s["random_state"])
