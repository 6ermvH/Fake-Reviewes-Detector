import os
import pandas as pd
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi

def load_config(path: str = "config/config.yml") -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def download_dataset(config: dict) -> None:
    """Download & unzip dataset from Kaggle to raw_data_path."""
    dest = config["raw_data_path"]
    os.makedirs(dest, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(config["dataset"], path=dest, unzip=True)

def load_raw_data(config: dict) -> pd.DataFrame:
    """Read the first CSV in raw_data_path into a DataFrame."""
    files = [f for f in os.listdir(config["raw_data_path"]) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("Нет CSV в " + config["raw_data_path"])
    return pd.read_csv(os.path.join(config["raw_data_path"], files[0]))

def save_processed(df: pd.DataFrame, config: dict) -> str:
    """Save processed DataFrame to processed_data_path and return filepath."""
    os.makedirs(config["processed_data_path"], exist_ok=True)
    out = os.path.join(config["processed_data_path"], "processed.csv")
    df.to_csv(out, index=False)
    return out
