from fake_reviews_detector.utils import load_yaml_config
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Нужно будет один раз скачать
nltk.download('stopwords')

def clean_text(text: str, config: dict) -> str:
    if config["lowercase"]:
        text = text.lower()
    if config["remove_punctuation"]:
        text = re.sub(r'[^\w\s]', '', text)
    if config["remove_stopwords"]:
        stop_words = set(stopwords.words('english'))
        text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_dataset(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    config = load_yaml_config(config_path)
    
    review_col = config["dataset"]["columns"]["review"]
    label_col = config["dataset"]["columns"]["label"]
    value_map = {
        config["dataset"]["values"]["good"]: 1,
        config["dataset"]["values"]["bad"]: 0
    }
    preprocessing_cfg = config["preprocessing"]
    
    # Rename columns
    df = df.rename(columns={
        review_col: "text",
        label_col: "label"
    })
    
    # Map labels
    df["label"] = df["label"].map(value_map)
    
    # Clean text
    df["text"] = df["text"].apply(lambda x: clean_text(str(x), preprocessing_cfg))
    
    print(f"Preprocessed dataset shape: {df.shape}")
    
    # Optionally save processed data
    processed_path = config["dataset"]["processed_data_path"]
    df.to_csv(processed_path, index=False)
    print(f"Saved preprocessed dataset to {processed_path}")
    
    return df

