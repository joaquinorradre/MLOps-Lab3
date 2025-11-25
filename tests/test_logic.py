# pylint: disable=pointless-string-statement
"""
Unit testing for logic module.
"""
import io
from PIL import Image
from mylib import logic


def test_predict():
    """
    Test that the predict function returns a string and that the string
    is one of the expected class names.
    """
    img = Image.new("RGB", (10, 10), color="black")
    byte_io = io.BytesIO()
    img.save(byte_io, format="PNG")
    image_bytes = byte_io.getvalue()

    prediction = logic.predict(image_bytes)
    assert isinstance(prediction, str), "Prediction is not a string."
    assert prediction in [
        "cat",
        "dog",
        "bird",
        "snake",
        "bear",
    ], "Prediction is not in expected classes."


def test_resize():
    """Test the resize function."""
    img = Image.new("RGB", (50, 50), color="black")
    byte_io = io.BytesIO()
    img.save(byte_io, format="PNG")
    image_bytes = byte_io.getvalue()

    resized_bytes = logic.resize(image_bytes, 20, 20)
    resized_img = Image.open(io.BytesIO(resized_bytes))

    assert resized_img.size == (
        20,
        20,
    ), "Image was not resized to the correct dimensions."
    assert resized_img.format == "PNG", "Resized image is not in PNG format."

def test_convert_to_grayscale():
    """Test the convert_to_grayscale function."""
    img_color = Image.new('RGB', (50, 50), color = 'red')
    byte_io = io.BytesIO()
    img_color.save(byte_io, format='PNG')
    image_bytes = byte_io.getvalue()

    gray_bytes = logic.convert_to_grayscale(image_bytes)

    gray_img = Image.open(io.BytesIO(gray_bytes))
    assert gray_img.mode == 'L', "Image was not converted to grayscale (mode 'L')."
    assert gray_img.size == (50, 50), "Grayscale image has incorrect dimensions."
