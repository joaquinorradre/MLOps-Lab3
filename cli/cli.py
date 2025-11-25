"""
cli.py
----------------
Command-line interface for image processing and prediction.
"""

import click
from mylib import logic


@click.group()
def cli():
    """CLI for image processing and prediction."""

@cli.command("predict")
@click.argument('image_path', type=click.Path(exists=True))
def predict(image_path):
    """Predict the class of an image.

    Example:
        uv run python -m cli.cli predict input.jpg
    """

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prediction = logic.predict(image_bytes)
    print(f"Prediction: {prediction}")

@cli.command("resize")
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--width', default=100, help='Width to resize to')
@click.option('--height', default=100, help='Height to resize to')
@click.argument('output_path', type=click.Path())
def resize(image_path, width, height, output_path):
    """Resize an image and save it.

    Example:
        uv run python -m cli.cli resize input.jpg --width 200 --height 200 output.png
    """

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    resized_bytes = logic.resize(image_bytes, width, height)

    with open(output_path, "wb") as f:
        f.write(resized_bytes)
    print(f"Image resized and saved to: {output_path}")

@cli.command("grayscale")
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def grayscale(image_path, output_path):
    """Convert an image to grayscale and save it.

    Example:
        uv run python -m cli.cli grayscale input.jpg output_gray.png
    """

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    grayscale_bytes = logic.convert_to_grayscale(image_bytes)

    with open(output_path, "wb") as f:
        f.write(grayscale_bytes)
    print(f"Grayscale image saved to: {output_path}")

if __name__ == "__main__": # pragma: no cover
    cli()
