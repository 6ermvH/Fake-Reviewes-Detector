import click
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from fake_reviews_classifier.data import load_data
from fake_reviews_classifier.preprocess import clean_text


def train_model(data_csv: str, model_path: str) -> None:
    df = load_data(data_csv)
    df["clean_review"] = df["review"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    joblib.dump({"vectorizer": vectorizer, "model": model}, model_path)


@click.command()
@click.argument("data_csv")
@click.argument("model_path")
def main(data_csv: str, model_path: str) -> None:
    train_model(data_csv, model_path)


if __name__ == "__main__":
    main()
