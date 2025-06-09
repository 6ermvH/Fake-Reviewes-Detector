import sys
import joblib

from fake_reviews_classifier.preprocess import clean_text


def predict_text(model_path: str, text: str) -> str:
    data = joblib.load(model_path)
    vec = data["vectorizer"]
    model = data["model"]
    clean = clean_text(text)
    pred = model.predict(vec.transform([clean]))[0]
    return "fake" if pred == 1 else "real"


@click.command()
@click.argument("model_path")
@click.argument("text")
def main(model_path: str, text: str) -> None:
    print(predict_text(model_path, text))


if __name__ == "__main__":
    main()

