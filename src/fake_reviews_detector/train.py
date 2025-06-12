import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def split_data(df: pd.DataFrame, tr_cfg: dict):
    print("Splitting data into train")
    X = df["review"]
    y = df["label"]
    stratify = y if tr_cfg.get("stratify", False) else None
    return train_test_split(
        X, y,
        test_size=tr_cfg["test_size"],
        random_state=tr_cfg["random_state"],
        stratify=stratify
    )


def build_vectorizer(v_cfg: dict):
    print(f"Building vectorizer of type {v_cfg['type']} with max_features={v_cfg['max_features']} and ngram_range={v_cfg['ngram_range']}")
    if v_cfg["type"] == "tfidf":
        return TfidfVectorizer(
            max_features=v_cfg["max_features"],
            ngram_range=tuple(v_cfg["ngram_range"])
        )
    return CountVectorizer(
        max_features=v_cfg["max_features"],
        ngram_range=tuple(v_cfg["ngram_range"])
    )


def build_model(m_cfg: dict):
    print(f"Building model of type {m_cfg['type']} with hyperparameters {m_cfg['hyperparameters']}")
    t = m_cfg["type"]
    params = m_cfg["hyperparameters"]
    if t == "logistic_regression":
        return LogisticRegression(**params)
    if t == "svm":
        return SVC(**params)
    if t == "random_forest":
        return RandomForestClassifier(**params)
    if t == "naive_bayes":
        return MultinomialNB(**params)
    raise ValueError(f"Unknown model type: {t}")


def training(cfg) -> None:
    v_cfg  = cfg["vectorizer"]
    m_cfg  = cfg["model"]
    tr_cfg = cfg["training"]

    df = pd.read_csv(cfg["dataset_path"])
    print(df)
    df["review"] = df["review"].fillna("")
    df = df[df["review"].str.strip() != ""]

    vectorizer_path  = Path(cfg["vectorizer_path"])
    model_path       = Path(cfg["model_path"])

    X_train, X_test, y_train, y_test = split_data(df, tr_cfg)
    vec = build_vectorizer(v_cfg)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec  = vec.transform(X_test)

    model = build_model(m_cfg)
    model.fit(X_train_vec, y_train)

    print(f"Model info:")

    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    for path in (vectorizer_path.parent, model_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    joblib.dump(vec,  vectorizer_path)
    joblib.dump(model, model_path)
    print(f"Saved vectorizer to {vectorizer_path}")
    print(f"Saved model to {model_path}")

