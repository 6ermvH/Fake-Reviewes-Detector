import click

from .data_handler import load_config, download_dataset
from .preprocess import preprocess_data
from .model import train_model, predict_text


@click.group()
@click.option(
    "--config",
    default="config/config.yml",
    help="Path to YAML config",
)
@click.pass_context
def cli(ctx, config):
    """Fake Reviews Detector CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.pass_context
def download(ctx):
    """Скачать CSV с Kaggle."""
    cfg = load_config(ctx.obj["config"])
    download_dataset(cfg)
    click.echo("Dataset downloaded to " + cfg["raw_data_path"])


@cli.command()
@click.pass_context
def preprocess(ctx):
    """Предобработать текст и сохранить processed.csv."""
    path = preprocess_data(ctx.obj["config"])
    click.echo("Data preprocessed → " + path)


@cli.command()
@click.pass_context
def train(ctx):
    """Обучить модель и сохранить её."""
    train_model(ctx.obj["config"])
    click.echo("Model trained → " + load_config(ctx.obj["config"])["model_path"])


@cli.command()
@click.argument("text")
@click.pass_context
def predict(ctx, text):
    """Сделать предсказание для одного отзыва."""
    result = predict_text(text, ctx.obj["config"])
    click.echo(result)


if __name__ == "__main__":
    cli()
