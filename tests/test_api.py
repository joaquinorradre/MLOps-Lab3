# pylint: disable=pointless-string-statement, redefined-outer-name
"""
Integration testing for the API (api/api.py)
"""
import io
import json
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api.api import app
from unittest.mock import patch


@pytest.fixture
def client():
    """Testing client from FastAPI."""
    return TestClient(app)


def create_fake_image_bytes():
    """
    Creates a dummy PNG image in memory to be used for testing.
    """
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


def test_home_endpoint(client):
    """Verify that the endpoint / returns the HTML home page."""
    response = client. get("/")
    assert response.status_code == 200
    assert response.headers['content-type'] == 'text/html; charset=utf-8'


def test_predict_endpoint(client):
    """
    Verify that the /predict endpoint correctly receives an image
    and returns a JSON prediction. 
    """
    with patch('api.api.logic.predict') as mock_predict:
        mock_predict. return_value = "Abyssinian"
        
        img_bytes = create_fake_image_bytes()

        response = client.post(
            "/predict",
            files={"file": ("test_image. png", img_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response. json()
        assert "filename" in data
        assert "prediction" in data
        
        assert isinstance(data["prediction"], str), "Prediction should be a string"
        assert not data["prediction"].startswith("Error:"), (
            f"Prediction failed: {data['prediction']}"
        )
        
        assert data["prediction"] == "Abyssinian", (
            f"Expected 'Abyssinian', got '{data['prediction']}'"
        )
        
        mock_predict.assert_called_once()


def test_resize_endpoint(client):
    """
    Verify that the /resize endpoint correctly resizes an image
    and returns the new image. 
    """
    img_bytes = create_fake_image_bytes()
    target_width = 50
    target_height = 50

    response = client.post(
        "/resize",
        files={"file": ("test_image.png", img_bytes, "image/png")},
        data={"width": str(target_width), "height": str(target_height)}
    )

    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'

    resized_img = Image.open(io.BytesIO(response. content))
    assert resized_img.size == (target_width, target_height)
    assert resized_img.format == 'PNG'


def test_predict_missing_file(client):
    """
    Verify that the /predict endpoint returns a 422 error
    if no file is provided.
    """
    response = client.post("/predict")

    assert response.status_code == 422


def test_resize_missing_data(client):
    """
    Verify that the /resize endpoint returns a 422 error
    if form data (width/height) is missing.
    """
    img_bytes = create_fake_image_bytes()

    response = client.post(
        "/resize",
        files={"file": ("test_image. png", img_bytes, "image/png")}
    )

    assert response.status_code == 422


def test_grayscale_endpoint(client):
    """
    Verify that the /grayscale endpoint correctly converts an image
    and returns the new image. 
    """
    img_bytes = create_fake_image_bytes()

    response = client.post(
        "/grayscale",
        files={"file": ("test_image.png", img_bytes, "image/png")}
    )

    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'

    gray_img = Image.open(io.BytesIO(response.content))
    assert gray_img.mode == 'L'


def test_grayscale_missing_file(client):
    """
    Verify that the /grayscale endpoint returns a 422 error
    if no file is provided.
    """
    response = client.post("/grayscale")
    assert response.status_code == 422


def test_predict_handles_exception(client, monkeypatch):
    """
    Verify that the /predict endpoint returns a 500 error
    if the logic. predict function fails.
    """
    def mock_predict(_image_bytes):
        raise RuntimeError("Simulated Logic Error")
    monkeypatch.setattr("api.api.logic.predict", mock_predict)

    img_bytes = create_fake_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test_image. png", img_bytes, "image/png")}
    )

    assert response.status_code == 500
    assert "Error processing image" in response.json()["detail"]


def test_resize_handles_value_error(client, monkeypatch):
    """
    Verify that the /resize endpoint returns a 400 error
    if the logic.resize function fails with a ValueError.
    """
    def mock_resize(_image_bytes, _width, _height):
        raise ValueError("Simulated Value Error")

    monkeypatch.setattr("api.api.logic.resize", mock_resize)

    img_bytes = create_fake_image_bytes()
    response = client.post(
        "/resize",
        files={"file": ("test_image.png", img_bytes, "image/png")},
        data={"width": "50", "height": "50"}
    )

    assert response.status_code == 400
    assert "Invalid input" in response.json()["detail"]


def test_resize_handles_general_exception(client, monkeypatch):
    """
    Verify that the /resize endpoint returns a 500 error
    if the logic. resize function fails with a general Exception. 
    """
    def mock_resize(_image_bytes, _width, _height):
        raise RuntimeError("Simulated General Error")

    monkeypatch.setattr("api.api.logic.resize", mock_resize)

    img_bytes = create_fake_image_bytes()
    response = client.post(
        "/resize",
        files={"file": ("test_image.png", img_bytes, "image/png")},
        data={"width": "50", "height": "50"}
    )

    assert response.status_code == 500
    assert "Error resizing image" in response.json()["detail"]


def test_grayscale_handles_exception(client, monkeypatch):
    """
    Verify that the /grayscale endpoint returns a 500 error
    if the logic.convert_to_grayscale function fails. 
    """
    def mock_grayscale(_image_bytes):
        raise RuntimeError("Simulated Grayscale Error")

    monkeypatch.setattr("api.api.logic.convert_to_grayscale", mock_grayscale)

    img_bytes = create_fake_image_bytes()
    response = client.post(
        "/grayscale",
        files={"file": ("test_image.png", img_bytes, "image/png")}
    )

    assert response. status_code == 500
    assert "Error converting image to grayscale" in response.json()["detail"]


def test_predict_with_different_sizes(client):
    """
    Test that the /predict endpoint handles images of different sizes.
    
    The model should handle preprocessing internally. 
    """
    img_small = Image.new("RGB", (100, 100), color="green")
    byte_io_small = io.BytesIO()
    img_small.save(byte_io_small, format="PNG")
    byte_io_small.seek(0)
    
    response_small = client.post(
        "/predict",
        files={"file": ("test_small.png", byte_io_small, "image/png")}
    )
    
    assert response_small.status_code == 200
    assert isinstance(response_small.json()["prediction"], str)
    
    img_large = Image.new("RGB", (500, 500), color="blue")
    byte_io_large = io.BytesIO()
    img_large.save(byte_io_large, format="PNG")
    byte_io_large.seek(0)
    
    response_large = client.post(
        "/predict",
        files={"file": ("test_large. png", byte_io_large, "image/png")}
    )
    
    assert response_large.status_code == 200
    assert isinstance(response_large.json()["prediction"], str)


def test_predict_consistency(client):
    """
    Test that the same image produces the same prediction.
    
    This verifies model determinism through the API.
    """
    img = Image.new("RGB", (224, 224), color="yellow")
    byte_io = io.BytesIO()
    img.save(byte_io, format="PNG")
    image_bytes = byte_io.getvalue()
    
    byte_io_1 = io.BytesIO(image_bytes)
    response1 = client.post(
        "/predict",
        files={"file": ("test_consistent.png", byte_io_1, "image/png")}
    )
    
    byte_io_2 = io.BytesIO(image_bytes)
    response2 = client.post(
        "/predict",
        files={"file": ("test_consistent. png", byte_io_2, "image/png")}
    )
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    prediction1 = response1.json()["prediction"]
    prediction2 = response2.json()["prediction"]
    
    assert prediction1 == prediction2, (
        "Same image should produce same prediction through API.  "
        f"Got '{prediction1}' and '{prediction2}'"
    )