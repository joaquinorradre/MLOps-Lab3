# pylint: disable=pointless-string-statement
"""
Unit testing for classifier module.
"""
import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
import numpy as np
from PIL import Image
from mylib.classifier import PetClassifier


@pytest.fixture
def mock_onnx_model():
    """Create a mock ONNX model file."""
    with tempfile. NamedTemporaryFile(mode='wb', suffix='.onnx', delete=False) as f:
        f.write(b'fake_onnx_model_data')
        model_path = f.name
    yield model_path
    os. unlink(model_path)


@pytest.fixture
def mock_labels_file():
    """Create a mock class labels JSON file."""
    labels = {
        "0": "Abyssinian",
        "1": "Bengal",
        "2": "Birman"
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='. json', delete=False) as f:
        json.dump(labels, f)
        labels_path = f.name
    yield labels_path
    os. unlink(labels_path)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    img = Image.new('RGB', (224, 224), color='red')
    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')
    return byte_io.getvalue()


def test_classifier_initialization_with_paths(mock_onnx_model, mock_labels_file):
    """
    Test that PetClassifier initializes correctly with provided paths.
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        assert classifier.session is not None, "Session should be initialized"
        assert classifier.input_name == 'input', "Input name should be 'input'"
        assert len(classifier.class_labels) == 3, "Should have 3 class labels"
        assert classifier.class_labels["0"] == "Abyssinian"


def test_classifier_initialization_default_paths():
    """
    Test that PetClassifier uses default paths when none provided.
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session, \
         patch('mylib.classifier.os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='{"0": "Cat", "1": "Dog"}')):
        
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier()
        
        assert classifier.session is not None
        assert classifier.input_name == 'input'


def test_classifier_initialization_model_not_found():
    """
    Test that PetClassifier raises FileNotFoundError when model file doesn't exist.
    """
    with pytest.raises(FileNotFoundError) as exc_info:
        PetClassifier(
            model_path="nonexistent_model.onnx",
            labels_path="dummy. json"
        )
    
    assert "Model file not found" in str(exc_info.value)


def test_classifier_initialization_labels_not_found(mock_onnx_model):
    """
    Test that PetClassifier raises FileNotFoundError when labels file doesn't exist. 
    """
    with pytest.raises(FileNotFoundError) as exc_info:
        PetClassifier(
            model_path=mock_onnx_model,
            labels_path="nonexistent_labels.json"
        )
    
    assert "Labels file not found" in str(exc_info.value)


def test_classifier_session_options(mock_onnx_model, mock_labels_file):
    """
    Test that PetClassifier configures ONNX session with correct options.
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session, \
         patch('mylib.classifier.ort.SessionOptions') as mock_options:
        
        mock_options_instance = MagicMock()
        mock_options.return_value = mock_options_instance
        
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        assert mock_options_instance.intra_op_num_threads == 4
        
        mock_session. assert_called_once()
        call_kwargs = mock_session.call_args[1]
        assert "CPUExecutionProvider" in call_kwargs['providers']


def test_preprocess_image_shape(mock_onnx_model, mock_labels_file, sample_image_bytes):
    """
    Test that preprocess returns correct shape (1, 3, 224, 224).
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        preprocessed = classifier.preprocess(sample_image_bytes)
        
        assert preprocessed. shape == (1, 3, 224, 224), (
            f"Expected shape (1, 3, 224, 224), got {preprocessed.shape}"
        )
        assert preprocessed.dtype == np.float32, (
            f"Expected dtype float32, got {preprocessed.dtype}"
        )


def test_preprocess_normalization(mock_onnx_model, mock_labels_file):
    """
    Test that preprocess applies correct normalization.
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        byte_io = io.BytesIO()
        img.save(byte_io, format='PNG')
        image_bytes = byte_io.getvalue()
        
        preprocessed = classifier.preprocess(image_bytes)
        assert preprocessed.min() > -5, "Normalized values too low"
        assert preprocessed.max() < 5, "Normalized values too high"


def test_preprocess_converts_to_rgb(mock_onnx_model, mock_labels_file):
    """
    Test that preprocess converts images to RGB. 
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance. get_inputs.return_value = [mock_input]
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        img = Image.new('L', (100, 100), color=128)
        byte_io = io.BytesIO()
        img.save(byte_io, format='PNG')
        image_bytes = byte_io.getvalue()
        
        preprocessed = classifier.preprocess(image_bytes)
        
        assert preprocessed.shape[1] == 3, (
            f"Expected 3 channels, got {preprocessed.shape[1]}"
        )


def test_preprocess_resizes_image(mock_onnx_model, mock_labels_file):
    """
    Test that preprocess resizes images to 224x224.
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs. return_value = [mock_input]
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        img = Image.new('RGB', (500, 300), color='blue')
        byte_io = io.BytesIO()
        img.save(byte_io, format='PNG')
        image_bytes = byte_io.getvalue()
        
        preprocessed = classifier.preprocess(image_bytes)
        
        assert preprocessed.shape == (1, 3, 224, 224)


def test_predict_returns_class_label(mock_onnx_model, mock_labels_file, sample_image_bytes):
    """
    Test that predict returns a valid class label.
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        
        mock_logits = np.array([[0.1, 0.8, 0.1]])
        mock_session_instance.run.return_value = [mock_logits]
        
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        prediction = classifier.predict(sample_image_bytes)
        
        assert prediction == "Bengal", f"Expected 'Bengal', got '{prediction}'"
        assert isinstance(prediction, str), "Prediction should be a string"


def test_predict_calls_preprocess(mock_onnx_model, mock_labels_file, sample_image_bytes):
    """
    Test that predict calls preprocess method.
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        
        mock_logits = np.array([[0.5, 0.3, 0.2]])
        mock_session_instance.run.return_value = [mock_logits]
        
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        with patch. object(classifier, 'preprocess', wraps=classifier.preprocess) as mock_preprocess:
            classifier.predict(sample_image_bytes)
            
            mock_preprocess.assert_called_once_with(sample_image_bytes)


def test_predict_different_classes(mock_onnx_model, mock_labels_file, sample_image_bytes):
    """
    Test that predict returns different class labels based on model output.
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        mock_logits_0 = np.array([[0.9, 0.05, 0.05]])
        mock_session_instance.run. return_value = [mock_logits_0]
        prediction_0 = classifier.predict(sample_image_bytes)
        assert prediction_0 == "Abyssinian"
        
        mock_logits_2 = np.array([[0.1, 0.1, 0.8]])
        mock_session_instance.run. return_value = [mock_logits_2]
        prediction_2 = classifier.predict(sample_image_bytes)
        assert prediction_2 == "Birman"


def test_predict_uses_correct_input_name(mock_onnx_model, mock_labels_file, sample_image_bytes):
    """
    Test that predict uses the correct input name from the ONNX model.
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'custom_input_name'
        mock_session_instance.get_inputs.return_value = [mock_input]
        
        mock_logits = np.array([[0.5, 0.3, 0.2]])
        mock_session_instance.run.return_value = [mock_logits]
        
        mock_session. return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        assert classifier.input_name == 'custom_input_name'
        
        classifier.predict(sample_image_bytes)
        
        mock_session_instance.run.assert_called_once()
        call_args = mock_session_instance.run.call_args[0]
        inputs_dict = call_args[1]
        assert 'custom_input_name' in inputs_dict


def test_predict_with_various_image_formats(mock_onnx_model, mock_labels_file):
    """
    Test that predict works with different image formats (PNG, JPEG, etc.).
    """
    with patch('mylib.classifier.ort.InferenceSession') as mock_session:
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs. return_value = [mock_input]
        
        mock_logits = np.array([[0.7, 0.2, 0.1]])
        mock_session_instance.run.return_value = [mock_logits]
        
        mock_session.return_value = mock_session_instance
        
        classifier = PetClassifier(
            model_path=mock_onnx_model,
            labels_path=mock_labels_file
        )
        
        img_png = Image.new('RGB', (100, 100), color='red')
        byte_io_png = io.BytesIO()
        img_png.save(byte_io_png, format='PNG')
        prediction_png = classifier.predict(byte_io_png.getvalue())
        assert prediction_png == "Abyssinian"
        
        img_jpeg = Image.new('RGB', (100, 100), color='green')
        byte_io_jpeg = io.BytesIO()
        img_jpeg.save(byte_io_jpeg, format='JPEG')
        prediction_jpeg = classifier.predict(byte_io_jpeg.getvalue())
        assert prediction_jpeg == "Abyssinian"


def test_classifier_loads_correct_number_of_labels(mock_onnx_model):
    """
    Test that classifier loads all class labels from JSON file.
    """
    labels = {str(i): f"Class_{i}" for i in range(37)}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(labels, f)
        labels_path = f.name
    
    try:
        with patch('mylib.classifier.ort.InferenceSession') as mock_session:
            mock_session_instance = MagicMock()
            mock_input = MagicMock()
            mock_input.name = 'input'
            mock_session_instance.get_inputs.return_value = [mock_input]
            mock_session.return_value = mock_session_instance
            
            classifier = PetClassifier(
                model_path=mock_onnx_model,
                labels_path=labels_path
            )
            
            assert len(classifier.class_labels) == 37, (
                f"Expected 37 class labels, got {len(classifier.class_labels)}"
            )
    finally:
        os.unlink(labels_path)
