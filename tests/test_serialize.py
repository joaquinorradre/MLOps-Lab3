# pylint: disable=pointless-string-statement
"""
Unit testing for model selection and serialization module.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
import torch
import torch.nn as nn
from mlflow.entities. model_registry import ModelVersion
from mlflow.entities import Run, RunData, Metric
from mylib import serialize


class DummyModel(nn.Module):
    """Simple dummy model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def mock_model_versions():
    """Create mock model versions with different metrics."""
    versions = []
    
    version1 = MagicMock(spec=ModelVersion)
    version1.version = "1"
    version1.run_id = "run_id_1_abc123"
    versions.append(version1)
    
    version2 = MagicMock(spec=ModelVersion)
    version2.version = "2"
    version2.run_id = "run_id_2_def456"
    versions.append(version2)
    
    version3 = MagicMock(spec=ModelVersion)
    version3.version = "3"
    version3.run_id = "run_id_3_ghi789"
    versions.append(version3)
    
    return versions


@pytest.fixture
def mock_runs():
    """Create mock runs with metrics and parameters."""
    runs = {}
    
    # Run 1
    run1 = MagicMock(spec=Run)
    run1.data = MagicMock(spec=RunData)
    run1.data.metrics = {"final_val_accuracy": 0.85}
    run1.data. params = {
        "model_name": "resnet18",
        "batch_size": "32",
        "learning_rate": "0.001"
    }
    runs["run_id_1_abc123"] = run1
    
    # Run 2 - Best
    run2 = MagicMock(spec=Run)
    run2.data = MagicMock(spec=RunData)
    run2.data. metrics = {"final_val_accuracy": 0.92}
    run2.data. params = {
        "model_name": "resnet50",
        "batch_size": "64",
        "learning_rate": "0.0001"
    }
    runs["run_id_2_def456"] = run2
    
    # Run 3
    run3 = MagicMock(spec=Run)
    run3.data = MagicMock(spec=RunData)
    run3.data. metrics = {"final_val_accuracy": 0.88}
    run3.data.params = {
        "model_name": "resnet34",
        "batch_size": "32",
        "learning_rate": "0.0005"
    }
    runs["run_id_3_ghi789"] = run3
    
    return runs


def test_get_best_model(mock_model_versions, mock_runs):
    """
    Test that get_best_model correctly identifies the version with highest accuracy.
    """
    with patch('mylib.serialize.MlflowClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_client.search_model_versions.return_value = mock_model_versions
        mock_client.get_run.side_effect = lambda run_id: mock_runs[run_id]
        
        best_version, best_metric = serialize.get_best_model(
            model_name="PetClassifier",
            metric_name="final_val_accuracy"
        )
        
        assert best_version.version == "2", (
            f"Expected version 2 to be selected, got version {best_version.version}"
        )
        assert best_metric == 0.92, (
            f"Expected best metric to be 0.92, got {best_metric}"
        )
        mock_client.search_model_versions.assert_called_once_with("name='PetClassifier'")


def test_get_best_model_no_versions():
    """
    Test that get_best_model raises ValueError when no model versions are found. 
    """
    with patch('mylib.serialize.MlflowClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_client.search_model_versions.return_value = []
        
        with pytest.raises(ValueError) as exc_info:
            serialize.get_best_model(model_name="NonExistentModel")
        
        assert "No versions found for model 'NonExistentModel'" in str(exc_info.value)


def test_get_best_model_missing_metric(mock_model_versions):
    """
    Test that get_best_model handles missing metrics gracefully.
    
    When metrics are missing, it should return the first version with metric -1.
    """
    with patch('mylib.serialize.MlflowClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_client.search_model_versions.return_value = [mock_model_versions[0]]
        
        run = MagicMock(spec=Run)
        run. data = MagicMock(spec=RunData)
        run.data.metrics = {}
        run.data.params = {"model_name": "resnet18", "batch_size": "32", "learning_rate": "0.001"}
        
        mock_client.get_run. return_value = run
        
        best_version, best_metric = serialize.get_best_model(
            model_name="PetClassifier",
            metric_name="final_val_accuracy"
        )
        
        assert best_version is not None, "Should return a version even with missing metrics"
        assert best_version.version == "1", (
            f"Expected version 1, got {best_version.version}"
        )
        assert best_metric == -1, (
            f"Expected metric to be -1 when missing, got {best_metric}"
        )


def test_load_and_prepare_model():
    """
    Test that load_and_prepare_model correctly loads and prepares a model.
    """
    dummy_model = DummyModel()
    
    mock_version = MagicMock(spec=ModelVersion)
    mock_version.run_id = "test_run_id"
    
    with patch('mylib.serialize.mlflow.pytorch.load_model') as mock_load:
        mock_load.return_value = dummy_model
        
        model = serialize.load_and_prepare_model(mock_version)
        
        assert not model.training, "Model should be in evaluation mode"
        mock_load.assert_called_once_with(f"runs:/test_run_id/model")


def test_export_to_onnx():
    """
    Test that export_to_onnx creates an ONNX file.
    """
    dummy_model = DummyModel()
    dummy_model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_model.onnx"
        
        with patch('mylib.serialize.torch.onnx.export') as mock_export:
            result_path = serialize. export_to_onnx(
                dummy_model,
                output_path=str(output_path),
                input_size=(1, 3, 224, 224)
            )
            
            assert result_path == str(output_path), (
                f"Expected path {output_path}, got {result_path}"
            )
            
            mock_export.assert_called_once()
            
            call_args = mock_export.call_args
            assert call_args[0][0] == dummy_model, "Model not passed correctly"
            assert call_args[1]['opset_version'] == 18, "Wrong opset version"
            assert call_args[1]['input_names'] == ['input'], "Wrong input names"
            assert call_args[1]['output_names'] == ['output'], "Wrong output names"


def test_export_to_onnx_custom_input_size():
    """
    Test that export_to_onnx handles custom input sizes.
    """
    dummy_model = DummyModel()
    dummy_model. eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_model.onnx"
        
        with patch('mylib.serialize.torch.onnx.export') as mock_export:
            serialize.export_to_onnx(
                dummy_model,
                output_path=str(output_path),
                input_size=(2, 3, 256, 256)
            )
            
            call_args = mock_export.call_args
            dummy_input = call_args[0][1]
            assert dummy_input.shape == (2, 3, 256, 256), (
                f"Expected shape (2, 3, 256, 256), got {dummy_input. shape}"
            )


def test_save_class_labels():
    """
    Test that save_class_labels correctly downloads and saves class labels.
    """
    mock_version = MagicMock(spec=ModelVersion)
    mock_version.run_id = "test_run_id"
    
    test_labels = {
        "0": "Abyssinian",
        "1": "Bengal",
        "2": "Birman"
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source_class_labels.json"
        with open(source_path, 'w', encoding='utf-8') as f:
            json.dump(test_labels, f)
        
        output_path = Path(tmpdir) / "output_class_labels.json"
        
        with patch('mylib.serialize.MlflowClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_client.download_artifacts.return_value = str(source_path)
            
            class_labels = serialize.save_class_labels(
                mock_version,
                output_path=str(output_path)
            )
            
            assert class_labels == test_labels, (
                f"Expected labels {test_labels}, got {class_labels}"
            )
            
            assert output_path.exists(), "Output file was not created"
            
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_labels = json.load(f)
            
            assert saved_labels == test_labels, (
                "Saved labels don't match original labels"
            )
            
            mock_client.download_artifacts.assert_called_once_with(
                run_id="test_run_id",
                path="class_labels.json"
            )


def test_save_class_labels_verifies_count():
    """
    Test that save_class_labels returns the correct number of classes.
    """
    mock_version = MagicMock(spec=ModelVersion)
    mock_version. run_id = "test_run_id"
    
    test_labels = {str(i): f"Class_{i}" for i in range(37)}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source_class_labels. json"
        with open(source_path, 'w', encoding='utf-8') as f:
            json.dump(test_labels, f)
        
        output_path = Path(tmpdir) / "output_class_labels.json"
        
        with patch('mylib.serialize.MlflowClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_client.download_artifacts.return_value = str(source_path)
            
            class_labels = serialize.save_class_labels(
                mock_version,
                output_path=str(output_path)
            )
            
            assert len(class_labels) == 37, (
                f"Expected 37 classes, got {len(class_labels)}"
            )


def test_main_integration(mock_model_versions, mock_runs):
    """
    Test the main function end-to-end workflow.
    """
    dummy_model = DummyModel()
    
    test_labels = {
        "0": "Abyssinian",
        "1": "Bengal",
        "2": "Birman"
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        source_labels_path = Path(tmpdir) / "source_class_labels. json"
        with open(source_labels_path, 'w', encoding='utf-8') as f:
            json.dump(test_labels, f)
        
        with patch('mylib.serialize.MlflowClient') as mock_client_class, \
             patch('mylib.serialize.mlflow.pytorch.load_model') as mock_load, \
             patch('mylib.serialize.torch.onnx.export') as mock_export, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('mylib.serialize.json.dump') as mock_json_dump, \
             patch('mylib.serialize.json.load') as mock_json_load:
            
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_client.search_model_versions.return_value = mock_model_versions
            mock_client.get_run. side_effect = lambda run_id: mock_runs[run_id]
            mock_client.download_artifacts.return_value = str(source_labels_path)
            
            mock_load.return_value = dummy_model
            
            mock_json_load.return_value = test_labels
            
            serialize.main()
            
            mock_client.search_model_versions.assert_called()
            mock_load.assert_called_once()
            mock_export.assert_called_once()
            mock_client.download_artifacts.assert_called_once()
            
            assert mock_json_dump.call_count >= 1, "model_info. json should be written"


def test_main_creates_all_output_files(mock_model_versions, mock_runs):
    """
    Test that main function creates all expected output files.
    """
    dummy_model = DummyModel()
    
    test_labels = {"0": "Class1", "1": "Class2"}
    
    with tempfile. TemporaryDirectory() as tmpdir:
        source_labels_path = Path(tmpdir) / "source_class_labels.json"
        with open(source_labels_path, 'w', encoding='utf-8') as f:
            json.dump(test_labels, f)
        
        with patch('mylib.serialize.MlflowClient') as mock_client_class, \
             patch('mylib.serialize.mlflow.pytorch.load_model') as mock_load, \
             patch('mylib.serialize.torch.onnx.export'), \
             patch('mylib.serialize.json.dump') as mock_json_dump, \
             patch('mylib.serialize.json.load', return_value=test_labels):
            
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_client.search_model_versions.return_value = mock_model_versions
            mock_client.get_run.side_effect = lambda run_id: mock_runs[run_id]
            mock_client.download_artifacts. return_value = str(source_labels_path)
            
            mock_load.return_value = dummy_model
            
            serialize.main()
            
            model_info_calls = [call for call in mock_json_dump.call_args_list 
                              if 'model_version' in str(call)]
            
            assert len(model_info_calls) > 0, "model_info. json should be created"
