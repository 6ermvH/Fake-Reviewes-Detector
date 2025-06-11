import os
import subprocess
from pathlib import Path
import pandas as pd

from fake_reviews_detector.utils import load_yaml_config


def download_kaggle_dataset(dataset_image: str, target: str, file_name: str = None) -> Path:
    """
    Download Kaggle dataset or a specific file within it.

    Args:
        dataset_image: Kaggle dataset identifier, e.g. "owner/dataset-name".
        target: path to a directory or full file path where data will be placed.
        file_name: exact name of the file inside the dataset to download. If None, download all files.

    Returns:
        Path to the downloaded file (if file_name provided) or the directory containing the dataset.
    """
    target_path = Path(target)
    # Determine directory and desired file path
    if target_path.suffix:
        target_dir = target_path.parent
        desired_file = target_path
    else:
        target_dir = target_path
        desired_file = None

    target_dir.mkdir(parents=True, exist_ok=True)

    # Build kaggle CLI command
    cmd = ["kaggle", "datasets", "download", "-d", dataset_image]
    if file_name:
        cmd.extend(["-f", file_name])
    cmd.extend(["--unzip", "-p", str(target_dir)])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # If downloading a single file, locate and rename it
    if file_name:
        # Search for the file in nested folders
        matches = list(target_dir.rglob(file_name))
        if not matches:
            raise FileNotFoundError(f"File '{file_name}' not found in '{target_dir}'")
        src = matches[0]
        if desired_file and src != desired_file:
            src.rename(desired_file)
            src = desired_file
        return src

    # Otherwise, return the dataset directory
    return target_dir


def load_raw_dataset(config_path: str) -> pd.DataFrame:
    """
    Load raw dataset into a pandas DataFrame.

    Uses download_kaggle_dataset under the hood, passing raw_data_file if specified.
    """
    cfg = load_yaml_config(config_path)["dataset"]
    raw_dir  = cfg["raw_data_dir"]
    raw_file = cfg.get("raw_data_file")

    path = download_kaggle_dataset(
        dataset_image=cfg["image"],
        target=os.path.join(raw_dir, raw_file) if raw_file else raw_dir,
        file_name=raw_file
    )

    df = pd.read_csv(path)
    print(f"Loaded raw data from '{path}' (shape: {df.shape})")
    return df


def save_processed_dataset(df: pd.DataFrame, config_path: str) -> None:
    """
    Save processed DataFrame to the specified location in config.
    """
    cfg       = load_yaml_config(config_path)["dataset"]
    proc_dir  = Path(cfg["processed_data_dir"])
    proc_file = cfg["processed_data_file"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = proc_dir / proc_file
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to '{out_path}'")

