"""
logic.py
----------------
Logic for image processing and prediction.
"""

import io
from PIL import Image, ImageOps
from mylib.classifier import PetClassifier

try: #pragma: no cover
    _classifier = PetClassifier()
    print("Classifier initialized successfully in logic module")
except (FileNotFoundError, OSError, ValueError) as e:
    print(f"Warning: Could not load classifier: {e}")
    _classifier = None
except ImportError as e:  #pragma: no cover
    print(f"Warning: Missing dependencies: {e}")
    _classifier = None

def predict(image_bytes: bytes) -> str:
    """
    Predict the class of an image using the trained ONNX model.

    """
    if _classifier is None:
        return "Error: Model not loaded"
    
    return _classifier.predict(image_bytes)

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
