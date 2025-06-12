# src/fake_reviews_detector/preview.py

from pathlib import Path
import joblib
import pandas as pd

from fake_reviews_detector.preprocessing import clean_text


def load_artifacts(model_path: Path, vectorizer_path: Path):
    """
    Load trained model and vectorizer from disk.
    """
    vec = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vec, model


def preview(
    texts: list[str],
    cfg,
) -> pd.DataFrame:
    model_path = Path(cfg['model_path'])
    vectorizer_path = Path(cfg['vectorizer_path'])
    vec, model = load_artifacts(model_path, vectorizer_path)
    pp_cfg = cfg['preprocessing']

    cleaned = [clean_text(text, pp_cfg) for text in texts]

    # Vectorize
    X = vec.transform(cleaned)
    # Predict
    preds = model.predict(X)

    # Decode numeric back to original labels if needed
    inv_map = {1: 1, 0: 0}
    decoded = [inv_map.get(int(p), p) for p in preds]

    # Build output
    return pd.DataFrame({
        'raw': texts,
        'cleaned': cleaned,
        'pred_numeric': preds,
        'pred_label': decoded
    })

def preview_single(text: str, cfg) -> pd.DataFrame:
    model_path = Path(cfg['model_path'])
    vectorizer_path = Path(cfg['vectorizer_path'])
    vec, model = load_artifacts(model_path, vectorizer_path)
    pp_cfg = cfg['preprocessing']

    cleaned = [clean_text(text, pp_cfg)]

    X = vec.transform(cleaned)
    preds = model.predict(X)

    return preds[0]
