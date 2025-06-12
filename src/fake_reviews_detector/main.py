from fake_reviews_detector.data_loader import download_kaggle_dataset
from fake_reviews_detector.preprocessing import *
from fake_reviews_detector.preview import preview_single
from fake_reviews_detector.utils import load_yaml_config
from fake_reviews_detector.train import train_model
from fake_reviews_detector.preview import *

cfg = load_yaml_config('config/local_dev.yaml')

dataset = download_kaggle_dataset('mexwell/fake-reviews-dataset')

create_processed_csv(dataset, cfg, cfg['dataset_path'])

train_model(cfg)

print(preview(
    ["Hello world!!!", 
     "It`s so baaaad", 
     "What???",
     "Love this! Well made, sturdy, and very comfortable. I love it!Very pretty",
     "love it, a great upgrade from the original. I've had mine for a couple of years",
     "Super rough, not soft wash cloths, more like bar towels",
     "Like this little guy. Use it often. He is small."
     ],cfg))

print(preview_single("Hello world!!! my name is german", cfg))
