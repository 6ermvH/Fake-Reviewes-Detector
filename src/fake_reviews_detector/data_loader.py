from fake_reviews_detector.utils import load_yaml_config
import os
import subprocess
from pathlib import Path
import pandas as pd
import yaml


def load_yaml_config(config_path="../../config/local_dev.yaml") -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Загрузка датасета с Kaggle
def download_kaggle_dataset(dataset_image: str, target: str, file_name: str = None) -> Path:
    target_path = Path(target)
    if target_path.suffix:
        target_dir = target_path.parent
        desired_file = target_path
    else:
        target_dir = target_path
        desired_file = None

    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_image]
    if file_name:
        cmd.extend(["-f", file_name])
    cmd.extend(["--unzip", "-p", str(target_dir)])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if file_name:
        matches = list(target_dir.rglob(file_name))
        if not matches:
            raise FileNotFoundError(f"File '{file_name}' not found in '{target_dir}'")
        src = matches[0]
        if desired_file and src != desired_file:
            src.rename(desired_file)
            src = desired_file
        return src

    return target_dir


# Загрузка датасета в DataFrame
def load_raw_dataset(config_path: str) -> pd.DataFrame:
    cfg = load_yaml_config(config_path)["dataset"]
    raw_dir = cfg["raw_data_dir"]
    raw_file = cfg.get("raw_data_file")

    path = download_kaggle_dataset(
        dataset_image=cfg["image"],
        target=os.path.join(raw_dir, raw_file) if raw_file else raw_dir,
        file_name=raw_file
    )

    df = pd.read_csv(path)
    print(f"Loaded raw data from '{path}' (shape: {df.shape})")
    return df


# Сохранение обработанного датасета в файл
def save_processed_dataset(df: pd.DataFrame, config_path: str) -> None:
    processed_df = preprocess_dataset(df, config_path)
    cfg = load_yaml_config(config_path)["dataset"]
    proc_dir = Path(cfg["processed_data_dir"])
    proc_file = cfg["processed_data_file"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = proc_dir / proc_file
    processed_df.to_csv(out_path, index=False)
    print(f"Saved processed data to '{out_path}'")
