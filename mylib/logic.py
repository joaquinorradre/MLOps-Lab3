"""
logic.py
----------------
Logic for image processing and prediction.
"""

import random
import io
from PIL import Image, ImageOps


def predict(_image_bytes: bytes) -> str:
    """
    Predict the class of an image.
    In this lab, it just returns a random class. [cite: 10]
    """
    class_names = ["cat", "dog", "bird", "snake", "bear"]
    return random.choice(class_names)

def resize(image_bytes: bytes, width: int, height: int) -> bytes:
    """
    Resize an image to a specific size.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img_resized = img.resize((width, height))

    byte_io = io.BytesIO()
    img_resized.save(byte_io, format="PNG")
    byte_io.seek(0)

    return byte_io.getvalue()

def convert_to_grayscale(image_bytes: bytes) -> bytes:
    """
    Convert an image to grayscale.
    """
    img = Image.open(io.BytesIO(image_bytes))

    img_gray = ImageOps.grayscale(img)

    byte_io = io.BytesIO()
    img_gray.save(byte_io, format="PNG")
    byte_io.seek(0)

    return byte_io.getvalue()
