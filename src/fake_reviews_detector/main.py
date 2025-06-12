from fake_reviews_detector.data_loader import download_kaggle_dataset
from fake_reviews_detector.utils import load_yaml_config
import os

cfg = load_yaml_config("config/local_dev.yaml")["dataset"]
# cfg["raw_data_dir"] — папка, cfg["raw_data_file"] — имя файла
target_path = os.path.join(cfg["raw_data_dir"], cfg["raw_data_file"])

# при таком вызове файл внутри папки будет переименован в cfg["raw_data_file"]
download_kaggle_dataset(cfg["image"], target_path)

