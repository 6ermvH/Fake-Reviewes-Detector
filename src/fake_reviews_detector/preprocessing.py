import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def clean_text(text: str, p_cfg: dict) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower() if p_cfg.get("lowercase", False) else str(text)
    if p_cfg.get("remove_punctuation", False):
        t = re.sub(r"[^\w\s]", "", t)
    tokens = t.split()
    if p_cfg.get("remove_stopwords", False):
        sw = set(stopwords.words("english"))
        tokens = [w for w in tokens if w not in sw]
    if p_cfg.get("stemming", False):
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(w) for w in tokens]
    if p_cfg.get("lemmatization", False):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def rename_and_map(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.rename(
        columns={
            cfg["columns"]["review"]: "review",
            cfg["columns"]["label"]: "label"})
    df["label"] = df["label"].map(
        {cfg["values"]["good"]: 1, cfg["values"]["bad"]: 0})
    return df


def clean_reviews(df: pd.DataFrame, p_cfg: dict) -> pd.DataFrame:
    df["review"] = df["review"].fillna("")

    sw = set(stopwords.words("english"))
    stemmer = PorterStemmer() if p_cfg["stemming"] else None
    lemmatizer = WordNetLemmatizer() if p_cfg["lemmatization"] else None

    def _clean(text: str) -> str:
        if pd.isna(text):
            return ""
        t = str(text).lower() if p_cfg["lowercase"] else str(text)
        if p_cfg["remove_punctuation"]:
            t = re.sub(r"[^\w\s]", "", t)
        tokens = t.split()
        if p_cfg["remove_stopwords"]:
            tokens = [w for w in tokens if w not in sw]
        if stemmer:
            tokens = [stemmer.stem(w) for w in tokens]
        if lemmatizer:
            tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)

    df["review"] = df["review"].apply(_clean)
    return df


def create_processed_csv(
        raw_csv_path,
        config,
        output_csv_path) -> pd.DataFrame:
    file = Path(output_csv_path)
    if file.exists() and file.is_file():
        print(f"File `{file}` already exists, skipping preprocessing.")
        return

    ds_cfg = config["dataset"]
    pp_cfg = config["preprocessing"]

    df = pd.read_csv(raw_csv_path)
    df = df.dropna(subset=[ds_cfg["columns"]["review"]])
    df = rename_and_map(df, ds_cfg)
    df = clean_reviews(df, pp_cfg)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["review", "label"]].to_csv(out_path, index=False)

    print(f"Processed data saved to `{out_path}`")
    return df
