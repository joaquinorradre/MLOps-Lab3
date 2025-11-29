# pylint: disable=pointless-string-statement
"""
Unit testing for training module.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from mylib import train


def test_set_seed():
    """
    Test that set_seed sets all random seeds correctly.
    """
    train.set_seed(42)
    
    import random
    val1 = random.random()
    train.set_seed(42)
    val2 = random.random()
    assert val1 == val2, "Python random seed not working correctly"
    
    train.set_seed(42)
    np_val1 = np.random. rand()
    train.set_seed(42)
    np_val2 = np.random. rand()
    assert np_val1 == np_val2, "NumPy random seed not working correctly"
    
    train.set_seed(42)
    torch_val1 = torch.rand(1). item()
    train.set_seed(42)
    torch_val2 = torch.rand(1).item()
    assert torch_val1 == torch_val2, "PyTorch random seed not working correctly"
    
    assert os.environ['PYTHONHASHSEED'] == '42', (
        "PYTHONHASHSEED environment variable not set correctly"
    )


def test_set_seed_different_values():
    """
    Test that different seeds produce different random values.
    """
    train.set_seed(42)
    val1 = torch.rand(1).item()
    
    train.set_seed(123)
    val2 = torch.rand(1).item()
    
    assert val1 != val2, "Different seeds should produce different random values"


@patch('mylib.train.datasets.OxfordIIITPet')
@patch('mylib.train.DataLoader')
def test_prepare_data(mock_dataloader, mock_dataset):
    """
    Test that prepare_data correctly creates train and validation loaders.
    """
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.class_to_idx = {
        'Abyssinian': 0,
        'Bengal': 1,
        'Birman': 2
    }
    mock_dataset_instance.__len__ = MagicMock(return_value=100)
    mock_dataset. return_value = mock_dataset_instance
    
    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()
    mock_dataloader. side_effect = [mock_train_loader, mock_val_loader]
    
    train_loader, val_loader, idx_to_class, num_classes = train.prepare_data(
        data_dir="test_data",
        batch_size=32,
        seed=42
    )
    
    assert train_loader == mock_train_loader, "Train loader not returned correctly"
    assert val_loader == mock_val_loader, "Val loader not returned correctly"
    assert num_classes == 3, f"Expected 3 classes, got {num_classes}"
    assert idx_to_class == {0: 'Abyssinian', 1: 'Bengal', 2: 'Birman'}, (
        "idx_to_class mapping incorrect"
    )
    
    mock_dataset.assert_called_once()
    call_kwargs = mock_dataset.call_args[1]
    assert call_kwargs['root'] == "test_data"
    assert call_kwargs['split'] == "trainval"
    assert call_kwargs['download'] is True


@patch('mylib.train.datasets.OxfordIIITPet')
@patch('mylib.train.DataLoader')
def test_prepare_data_split_ratio(mock_dataloader, mock_dataset):
    """
    Test that prepare_data uses 80/20 train/val split.
    """
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.class_to_idx = {'class1': 0}
    mock_dataset_instance.__len__ = MagicMock(return_value=100)
    mock_dataset.return_value = mock_dataset_instance
    
    mock_dataloader. side_effect = [MagicMock(), MagicMock()]
    
    with patch('mylib.train.random_split') as mock_split:
        mock_split.return_value = (MagicMock(), MagicMock())
        
        train. prepare_data(batch_size=32, seed=42)
        
        mock_split.assert_called_once()
        split_sizes = mock_split.call_args[0][1]
        assert split_sizes == [80, 20], f"Expected [80, 20] split, got {split_sizes}"


def test_build_model_mobilenet_v2():
    """
    Test that build_model correctly builds a MobileNetV2 model.
    """
    with patch('mylib.train.models.mobilenet_v2') as mock_mobilenet:
        mock_model = MagicMock()
        
        mock_param = MagicMock()
        mock_model.features.parameters = MagicMock(return_value=[mock_param])
        
        mock_classifier = MagicMock()
        mock_linear = MagicMock()
        mock_linear.in_features = 1280
        mock_classifier.__getitem__ = MagicMock(return_value=mock_linear)
        mock_classifier.__setitem__ = MagicMock()
        
        real_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_classifier.parameters = MagicMock(return_value=[real_param])
        
        mock_model.classifier = mock_classifier
        mock_mobilenet.return_value = mock_model
        
        model, criterion, optimizer = train.build_model(
            model_arch="mobilenet_v2",
            num_classes=37,
            learning_rate=0.001
        )
        
        mock_mobilenet.assert_called_once_with(weights="IMAGENET1K_V1")
        assert isinstance(criterion, nn.CrossEntropyLoss), (
            "Criterion should be CrossEntropyLoss"
        )
        assert isinstance(optimizer, torch.optim.Adam), (
            "Optimizer should be Adam"
        )


def test_build_model_efficientnet_b0():
    """
    Test that build_model correctly builds an EfficientNetB0 model. 
    """
    with patch('mylib.train.models.efficientnet_b0') as mock_efficientnet:
        mock_model = MagicMock()
        
        mock_param = MagicMock()
        mock_model.features.parameters = MagicMock(return_value=[mock_param])
        
        mock_classifier = MagicMock()
        mock_linear = MagicMock()
        mock_linear.in_features = 1280
        mock_classifier.__getitem__ = MagicMock(return_value=mock_linear)
        mock_classifier.__setitem__ = MagicMock()
        
        real_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_classifier.parameters = MagicMock(return_value=[real_param])
        
        mock_model.classifier = mock_classifier
        mock_efficientnet.return_value = mock_model
        
        model, criterion, optimizer = train. build_model(
            model_arch="efficientnet_b0",
            num_classes=37,
            learning_rate=0.001
        )
        
        mock_efficientnet.assert_called_once_with(weights="IMAGENET1K_V1")
        assert isinstance(criterion, nn.CrossEntropyLoss), (
            "Criterion should be CrossEntropyLoss"
        )
        assert isinstance(optimizer, torch.optim. Adam), (
            "Optimizer should be Adam"
        )


def test_build_model_resnet18():
    """
    Test that build_model correctly builds a ResNet18 model.
    """
    with patch('mylib.train.models.resnet18') as mock_resnet:
        mock_model = MagicMock()
        
        mock_param = MagicMock()
        mock_model.parameters = MagicMock(return_value=[mock_param])
        
        mock_fc = MagicMock()
        mock_fc.in_features = 512
        
        real_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_fc.parameters = MagicMock(return_value=[real_param])
        
        mock_model.fc = mock_fc
        mock_resnet.return_value = mock_model
        
        model, criterion, optimizer = train. build_model(
            model_arch="resnet18",
            num_classes=37,
            learning_rate=0.001
        )
        
        mock_resnet.assert_called_once_with(weights="IMAGENET1K_V1")
        assert isinstance(criterion, nn.CrossEntropyLoss), (
            "Criterion should be CrossEntropyLoss"
        )
        assert isinstance(optimizer, torch.optim.Adam), (
            "Optimizer should be Adam"
        )

def test_build_model_unsupported():
    """
    Test that build_model raises ValueError for unsupported models.
    """
    with pytest.raises(ValueError) as exc_info:
        train.build_model(
            model_arch="unsupported_model",
            num_classes=37,
            learning_rate=0.001
        )
    
    assert "not supported" in str(exc_info.value). lower(), (
        "Should raise ValueError for unsupported model"
    )


def test_train_model_epochs():
    """
    Test that train_model runs for the specified number of epochs. 
    """
    model = nn.Linear(10, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
    val_data = TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))
    train_loader = DataLoader(train_data, batch_size=4)
    val_loader = DataLoader(val_data, batch_size=4)
    
    device = torch.device("cpu")
    
    with patch('mylib.train.mlflow.log_metric'):
        trained_model, history = train.train_model(
            model, criterion, optimizer,
            train_loader, val_loader,
            device, epochs=3
        )
    
    assert len(history['train_loss']) == 3, (
        f"Expected 3 epochs of train loss, got {len(history['train_loss'])}"
    )
    assert len(history['val_loss']) == 3, (
        f"Expected 3 epochs of val loss, got {len(history['val_loss'])}"
    )
    assert len(history['train_acc']) == 3, (
        f"Expected 3 epochs of train acc, got {len(history['train_acc'])}"
    )
    assert len(history['val_acc']) == 3, (
        f"Expected 3 epochs of val acc, got {len(history['val_acc'])}"
    )


def test_train_model_logs_metrics():
    """
    Test that train_model logs metrics to MLflow.
    """
    model = nn.Linear(10, 2)
    criterion = nn. CrossEntropyLoss()
    optimizer = torch.optim. Adam(model.parameters(), lr=0.001)
    
    train_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
    val_data = TensorDataset(torch.randn(10, 10), torch. randint(0, 2, (10,)))
    train_loader = DataLoader(train_data, batch_size=4)
    val_loader = DataLoader(val_data, batch_size=4)
    
    device = torch.device("cpu")
    
    with patch('mylib.train.mlflow.log_metric') as mock_log_metric:
        train.train_model(
            model, criterion, optimizer,
            train_loader, val_loader,
            device, epochs=2
        )
        
        logged_metrics = [call_args[0][0] for call_args in mock_log_metric. call_args_list]
        
        assert "train_loss" in logged_metrics, "train_loss should be logged"
        assert "train_accuracy" in logged_metrics, "train_accuracy should be logged"
        assert "val_loss" in logged_metrics, "val_loss should be logged"
        assert "val_accuracy" in logged_metrics, "val_accuracy should be logged"
        assert "final_train_accuracy" in logged_metrics, "final_train_accuracy should be logged"
        assert "final_val_accuracy" in logged_metrics, "final_val_accuracy should be logged"


def test_train_model_returns_trained_model():
    """
    Test that train_model returns a trained model in eval mode.
    """
    model = nn.Linear(10, 2)
    criterion = nn. CrossEntropyLoss()
    optimizer = torch.optim. Adam(model.parameters(), lr=0.001)
    
    train_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
    val_data = TensorDataset(torch.randn(10, 10), torch. randint(0, 2, (10,)))
    train_loader = DataLoader(train_data, batch_size=4)
    val_loader = DataLoader(val_data, batch_size=4)
    
    device = torch.device("cpu")
    
    with patch('mylib.train.mlflow.log_metric'):
        trained_model, history = train.train_model(
            model, criterion, optimizer,
            train_loader, val_loader,
            device, epochs=1
        )
    
    assert not trained_model.training, "Model should be in eval mode after training"


def test_plot_metrics():
    """
    Test that plot_metrics creates a plot file.
    """
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.6, 0.5, 0.4],
        'train_acc': [0.7, 0.8, 0.85],
        'val_acc': [0.65, 0.75, 0.8]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "test_plot.png"
        
        train.plot_metrics(history, filename=str(filename))
        
        assert filename.exists(), "Plot file should be created"
        assert filename.stat().st_size > 0, "Plot file should not be empty"


def test_args_class():
    """
    Test that Args class correctly creates attributes from dictionary.
    """
    test_dict = {
        'model_name': 'resnet18',
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.0001
    }
    
    args = train.Args(test_dict)
    
    assert args.model_name == 'resnet18', "model_name not set correctly"
    assert args.epochs == 10, "epochs not set correctly"
    assert args.batch_size == 64, "batch_size not set correctly"
    assert args.learning_rate == 0.0001, "learning_rate not set correctly"
    assert args.run_name is None, "run_name should be initialized to None"


def test_args_class_dynamic_attributes():
    """
    Test that Args class allows setting additional attributes. 
    """
    test_dict = {'model_name': 'resnet18'}
    args = train.Args(test_dict)
    
    args.run_name = "custom_run"
    assert args.run_name == "custom_run", "Should be able to set run_name dynamically"


@patch('mylib.train.mlflow.start_run')
@patch('mylib.train.mlflow.log_params')
@patch('mylib.train.mlflow.log_metric')
@patch('mylib.train.mlflow.log_artifact')
@patch('mylib.train.mlflow.pytorch.log_model')
@patch('mylib.train.prepare_data')
@patch('mylib.train.build_model')
@patch('mylib.train.train_model')
@patch('mylib.train.plot_metrics')
@patch('mylib.train.torch.cuda.is_available', return_value=False)
def test_main_integration(mock_cuda, mock_plot, mock_train, mock_build,
                          mock_prepare, mock_log_model, mock_log_artifact,
                          mock_log_metric, mock_log_params, mock_start_run):
    """
    Test the main function end-to-end workflow.
    """
    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()
    idx_to_class = {0: 'class1', 1: 'class2'}
    mock_prepare.return_value = (mock_train_loader, mock_val_loader, idx_to_class, 2)
    
    mock_model = MagicMock()
    mock_criterion = MagicMock()
    mock_optimizer = MagicMock()
    mock_build.return_value = (mock_model, mock_criterion, mock_optimizer)
    
    history = {
        'train_loss': [0.5],
        'val_loss': [0.6],
        'train_acc': [0.7],
        'val_acc': [0.65]
    }
    mock_train.return_value = (mock_model, history)
    
    mock_start_run.return_value.__enter__ = MagicMock()
    mock_start_run. return_value.__exit__ = MagicMock(return_value=False)
    
    args = train.Args({
        'model_name': 'mobilenet_v2',
        'epochs': 1,
        'batch_size': 32,
        'learning_rate': 0.001,
        'seed': 42
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            train.main(args)
            
            mock_log_params.assert_called_once()
            mock_log_model.assert_called_once()
            
            assert Path("class_labels.json").exists(), "class_labels.json should be created"
            
        finally:
            os.chdir(original_dir)


@patch('mylib.train.mlflow.start_run')
@patch('mylib.train.mlflow.log_params')
@patch('mylib.train.mlflow.log_metric')
@patch('mylib.train.mlflow.log_artifact')
@patch('mylib.train.mlflow.pytorch.log_model')
@patch('mylib.train.prepare_data')
@patch('mylib.train.build_model')
@patch('mylib.train.train_model')
@patch('mylib.train.plot_metrics')
@patch('mylib.train.torch.cuda.is_available', return_value=True)
def test_main_uses_cuda_when_available(mock_cuda, mock_plot, mock_train, mock_build,
                                        mock_prepare, mock_log_model, mock_log_artifact,
                                        mock_log_metric, mock_log_params, mock_start_run):
    """
    Test that main uses CUDA device when available.
    """
    mock_prepare.return_value = (MagicMock(), MagicMock(), {0: 'class1'}, 1)
    mock_build.return_value = (MagicMock(), MagicMock(), MagicMock())
    history = {'train_loss': [0.5], 'val_loss': [0.6], 'train_acc': [0.7], 'val_acc': [0.65]}
    mock_train.return_value = (MagicMock(), history)
    
    mock_start_run.return_value.__enter__ = MagicMock()
    mock_start_run. return_value.__exit__ = MagicMock(return_value=False)
    
    args = train.Args({
        'model_name': 'mobilenet_v2',
        'epochs': 1,
        'batch_size': 32,
        'learning_rate': 0.001,
        'seed': 42
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            train.main(args)
            
            mock_train. assert_called_once()
            
        finally:
            os. chdir(original_dir)


@patch('mylib.train.mlflow.start_run')
@patch('mylib.train.mlflow.set_experiment')
@patch('mylib.train.mlflow.log_params')
@patch('mylib.train.mlflow.log_metric')
@patch('mylib.train.mlflow.log_artifact')
@patch('mylib.train.mlflow.pytorch.log_model')
@patch('mylib.train.prepare_data')
@patch('mylib.train.build_model')
@patch('mylib.train.train_model')
@patch('mylib.train.plot_metrics')
@patch('mylib.train.torch.cuda.is_available', return_value=False)
def test_main_sets_experiment(mock_cuda, mock_plot, mock_train, mock_build,
                               mock_prepare, mock_log_model, mock_log_artifact,
                               mock_log_metric, mock_log_params, mock_set_exp, mock_start_run):
    """
    Test that main sets the correct MLflow experiment.
    """
    mock_prepare.return_value = (MagicMock(), MagicMock(), {0: 'class1'}, 1)
    mock_build.return_value = (MagicMock(), MagicMock(), MagicMock())
    history = {'train_loss': [0.5], 'val_loss': [0.6], 'train_acc': [0.7], 'val_acc': [0.65]}
    mock_train. return_value = (MagicMock(), history)
    
    mock_start_run.return_value.__enter__ = MagicMock()
    mock_start_run. return_value.__exit__ = MagicMock(return_value=False)
    
    args = train.Args({
        'model_name': 'mobilenet_v2',
        'epochs': 1,
        'batch_size': 32,
        'learning_rate': 0.001,
        'seed': 42
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            train.main(args)
            
            mock_set_exp.assert_called_once_with("Lab3_DeepLearning")
            
        finally:
            os.chdir(original_dir)


def test_main_custom_run_name():
    """
    Test that main uses custom run_name when provided.
    """
    args = train.Args({
        'model_name': 'mobilenet_v2',
        'epochs': 1,
        'batch_size': 32,
        'learning_rate': 0.001,
        'seed': 42
    })
    args.run_name = "custom_experiment_name"
    
    with patch('mylib.train.mlflow.start_run') as mock_start_run, \
         patch('mylib.train.prepare_data'), \
         patch('mylib.train.build_model'), \
         patch('mylib.train.train_model'), \
         patch('mylib.train.plot_metrics'), \
         patch('mylib.train.mlflow.log_params'), \
         patch('mylib.train.mlflow.log_artifact'), \
         patch('mylib.train.mlflow.pytorch.log_model'):
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                with patch('mylib.train.prepare_data', return_value=(MagicMock(), MagicMock(), {0: 'c1'}, 1)):
                    with patch('mylib.train.build_model', return_value=(MagicMock(), MagicMock(), MagicMock())):
                        with patch('mylib.train.train_model', return_value=(MagicMock(), {
                            'train_loss': [0.5], 'val_loss': [0.6], 
                            'train_acc': [0.7], 'val_acc': [0.65]
                        })):
                            train.main(args)
                
                mock_start_run.assert_called_with(run_name="custom_experiment_name")
                
            finally:
                os.chdir(original_dir)