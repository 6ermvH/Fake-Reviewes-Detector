import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data_handler import load_config
from .preprocess import clean_text


def train_model(config_path: str = "config/config.yml") -> None:
    """Train logistic-regression on processed.csv and dump to model_path."""
    cfg = load_config(config_path)
    df = pd.read_csv(os.path.join(cfg["processed_data_path"], "processed.csv"))
    df = df.dropna(subset=["clean_text", "label"])
    df = df[df["clean_text"].str.strip() != ""]
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42
    )
    vec = TfidfVectorizer(max_features=1000)
    Xv = vec.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xv, y_train)
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    joblib.dump({"vectorizer": vec, "model": model}, cfg["model_path"])


def predict_text(text: str, config_path: str = "config/config.yml") -> str:
    """Load model, preprocess one text, predict and return 'fake'|'real'."""
    cfg = load_config(config_path)
    data = joblib.load(cfg["model_path"])
    vec, model = data["vectorizer"], data["model"]
    clean = clean_text(text)
    pred = model.predict(vec.transform([clean]))[0]
    return "fake" if pred == 1 else "real"
