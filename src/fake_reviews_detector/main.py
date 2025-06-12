from nltk import data
from numpy.lib import utils
from pandas._config.display import cf
from fake_reviews_detector.data_loader import download_kaggle_dataset
from fake_reviews_detector.preprocessing import *
from fake_reviews_detector.utils import load_yaml_config

cfg = load_yaml_config('config/local_dev.yaml')

dataset = download_kaggle_dataset('mexwell/fake-reviews-dataset')

create_processed_csv(dataset, cfg, cfg['dataset_path'])
