from pathlib import Path
import kagglehub

# Download kaggle dataset
def download_kaggle_dataset(image: str) -> Path:
    dataset_dir = Path(kagglehub.dataset_download(image))
    csv_files = list(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {dataset_dir}")
    if len(csv_files) > 1:
        print(f"Warning: multiple CSVs found, using {csv_files[0].name}")
    return csv_files[0]
