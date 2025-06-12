from pathlib import Path
import kagglehub

def download_kaggle_dataset(image: str) -> Path:
    dataset_dir = Path(kagglehub.dataset_download(image))
    csv_files = list(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {dataset_dir}")
    if len(csv_files) > 1:
        print(f"Warning: multiple CSVs found, using {csv_files[0].name}")
    print(f"Downloaded dataset from {image} to {dataset_dir}")
    return csv_files[0]
