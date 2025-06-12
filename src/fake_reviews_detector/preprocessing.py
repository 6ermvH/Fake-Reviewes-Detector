from fake_reviews_detector.utils import load_yaml_config
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Загрузка конфига
config = load_yaml_config("../../config/local_dev.yaml")

# Загрузка шумовых слов
nltk.download('stopwords', quiet=True)


# Очистка и превращение текста в удобный вид
def clean_text(text: str) -> str:
    if config["lowercase"]:
        text = text.lower()
    if config["remove_punctuation"]:
        text = re.sub(r'[^\w\s]', '', text)
    if config["remove_stopwords"]:
        stop_words = set(stopwords.words('english'))
        text = " ".join([word for word in text.split() if word not in stop_words])
    return text


# Предобработка данных
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    review_col = config["dataset"]["columns"]["review"]
    label_col = config["dataset"]["columns"]["label"]
    value_map = {
        config["dataset"]["values"]["good"]: 1,
        config["dataset"]["values"]["bad"]: 0
    }
    preprocessing_cfg = config["preprocessing"]

    df = df.rename(columns={
        review_col: "text",
        label_col: "label"
    })

    df["label"] = df["label"].map(value_map)
    df["text"] = df["text"].apply(lambda x: clean_text(str(x), preprocessing_cfg))
    print(f"Preprocessed dataset shape: {df.shape}")
    processed_path = config["dataset"]["processed_data_path"]
    df.to_csv(processed_path, index=False)
    print(f"Saved preprocessed dataset to {processed_path}")
    return df
