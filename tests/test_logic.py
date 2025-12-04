# pylint: disable=pointless-string-statement
"""
Unit testing for logic module.
"""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
from mylib import logic


def test_predict():
    """
    Test that the predict function returns a string.
    
    We mock the classifier to test the predict function behavior.
    """
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = "Abyssinian"
    
    original_classifier = logic._classifier
    logic._classifier = mock_classifier
    
    try:
        img = Image.new("RGB", (224, 224), color="red")
        byte_io = io.BytesIO()
        img.save(byte_io, format="PNG")
        image_bytes = byte_io.getvalue()
        
        prediction = logic.predict(image_bytes)
        
        assert isinstance(prediction, str), "Prediction is not a string."
        assert not prediction.startswith("Error:"), f"Prediction failed: {prediction}"
        assert prediction == "Abyssinian", f"Expected 'Abyssinian', got '{prediction}'"
        
        mock_classifier.predict.assert_called_once_with(image_bytes)
        
    finally:
        logic._classifier = original_classifier

def test_predict_handles_missing_model():
    """
    Test that predict handles gracefully when model is not loaded.
    
    We force the classifier to be None to ensure this path is tested
    regardless of whether the model loaded successfully or not.
    """
    original_classifier = logic._classifier
    
    logic._classifier = None
    
    try:
        img = Image.new("RGB", (224, 224), color="blue")
        byte_io = io.BytesIO()
        img.save(byte_io, format="PNG")
        image_bytes = byte_io.getvalue()
        
        prediction = logic.predict(image_bytes)
        
        assert prediction == "Error: Model not loaded", (
            "Should return error message when model is not loaded"
        )
        
    finally:
        logic._classifier = original_classifier

def test_predict_calls_classifier():
    """
    Test that predict calls the classifier's predict method when classifier is loaded.
    """
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = "Abyssinian"
    
    original_classifier = logic._classifier
    logic._classifier = mock_classifier
    
    try:
        img = Image.new("RGB", (224, 224), color="red")
        byte_io = io.BytesIO()
        img.save(byte_io, format="PNG")
        image_bytes = byte_io.getvalue()
        
        result = logic.predict(image_bytes)
        
        mock_classifier.predict. assert_called_once_with(image_bytes)
        
        assert result == "Abyssinian", (
            f"Expected 'Abyssinian', got '{result}'"
        )
    finally:
        logic._classifier = original_classifier


def test_resize():
    """Test the resize function."""
    img = Image.new("RGB", (50, 50), color="black")
    byte_io = io.BytesIO()
    img.save(byte_io, format="PNG")
    image_bytes = byte_io.getvalue()
    
    resized_bytes = logic.resize(image_bytes, 20, 20)
    resized_img = Image.open(io.BytesIO(resized_bytes))
    
    assert resized_img.size == (20, 20), (
        f"Image was not resized to the correct dimensions. "
        f"Expected (20, 20), got {resized_img.size}"
    )
    assert resized_img.format == "PNG", (
        f"Resized image is not in PNG format. Got {resized_img.format}"
    )


def test_convert_to_grayscale():
    """Test the convert_to_grayscale function."""
    img_color = Image.new('RGB', (50, 50), color='red')
    byte_io = io.BytesIO()
    img_color.save(byte_io, format='PNG')
    image_bytes = byte_io.getvalue()
    
    gray_bytes = logic.convert_to_grayscale(image_bytes)
    gray_img = Image.open(io.BytesIO(gray_bytes))
    
    assert gray_img.mode == 'L', (
        f"Image was not converted to grayscale (mode 'L'). Got mode '{gray_img.mode}'"
    )
    assert gray_img.size == (50, 50), (
        f"Grayscale image has incorrect dimensions. "
        f"Expected (50, 50), got {gray_img.size}"
    )


def test_predict_with_different_sizes():
    """
    Test that predict works with images of different sizes.
    
    The model should handle preprocessing internally.
    """
    # Test with small image
    img_small = Image.new("RGB", (100, 100), color="green")
    byte_io = io.BytesIO()
    img_small.save(byte_io, format="PNG")
    
    prediction_small = logic.predict(byte_io.getvalue())
    assert isinstance(prediction_small, str), "Prediction should be a string"
    
    # Test with large image
    img_large = Image.new("RGB", (500, 500), color="blue")
    byte_io = io.BytesIO()
    img_large.save(byte_io, format="PNG")
    
    prediction_large = logic.predict(byte_io.getvalue())
    assert isinstance(prediction_large, str), "Prediction should be a string"


def test_predict_consistency():
    """
    Test that the same image produces the same prediction.
    
    This verifies model determinism.
    """
    img = Image.new("RGB", (224, 224), color="yellow")
    byte_io = io.BytesIO()
    img.save(byte_io, format="PNG")
    image_bytes = byte_io.getvalue()
    
    prediction1 = logic.predict(image_bytes)
    prediction2 = logic.predict(image_bytes)
    
    assert prediction1 == prediction2, (
        "Same image should produce same prediction. "
        f"Got '{prediction1}' and '{prediction2}'"
    )