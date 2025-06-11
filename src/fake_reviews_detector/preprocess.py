import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from .data_handler import load_config, load_raw_data, save_processed

# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
TOKENIZER = RegexpTokenizer(r'\w+')


def clean_text(text: str) -> str:
    """
    Lowercase, remove URLs & non-letters, tokenize, remove stopwords, lemmatize.
    """
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = TOKENIZER.tokenize(text)
    tokens = [LEMMATIZER.lemmatize(tok) for tok in tokens if tok not in STOPWORDS]
    return " ".join(tokens)



def preprocess_data(config_path: str = "config/config.yml") -> str:
    """
    Load raw data, clean text_, map labels OR->1, CG->0, save processed CSV, return file path.
    """
    cfg = load_config(config_path)
    # Load raw data and drop missing text entries
    df = load_raw_data(cfg).dropna(subset=["text_"])

    # Clean text and drop empty results
    df["clean_text"] = df["text_"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    # Map string labels to numeric: OR -> 1, CG -> 0
    label_mapping = {"OR": 1, "CG": 0}
    df["label"] = df["label"].map(label_mapping)
    if df["label"].isnull().any():
        raise ValueError("Unexpected label values. Expected 'OR' or 'CG'.")

    # Prepare output and save
    out_df = df[["clean_text", "label"]]
    return save_processed(out_df, cfg)


