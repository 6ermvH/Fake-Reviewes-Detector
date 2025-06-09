import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(dataset: str, dest_path: str) -> None:
    """
    Download and unzip a Kaggle dataset.
    """
    os.makedirs(dest_path, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=dest_path, unzip=True)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV file into a DataFrame.
    """
    return pd.read_csv(file_path)
