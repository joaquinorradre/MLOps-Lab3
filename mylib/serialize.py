import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import json
import os

def get_best_model(model_name="PetClassifier", metric_name="final_val_accuracy"):
    """
    Search all the registered model versions and select the best one based on the specified metric.
    """
    client = MlflowClient()
    
    print(f"Searching versions of model '{model_name}'...")
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    if not model_versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    
    print(f"Found {len(model_versions)} versions of the model.")
    
    best_version = None
    best_metric_value = -1
    
    print(f"\nComparing models based on '{metric_name}':")
    print("-" * 80)
    
    for version in model_versions:
        run_id = version.run_id
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        metric_value = metrics.get(metric_name, -1)
        
        params = run.data.params
        model_type = params.get("model_name", "unknown")
        batch_size = params.get("batch_size", "unknown")
        learning_rate = params.get("learning_rate", "unknown")
        
        print(f"Version {version.version} | Run ID: {run_id[:8]}... | "
              f"Model: {model_type} | BS: {batch_size} | LR: {learning_rate} | "
              f"{metric_name}: {metric_value:.4f}")
        
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_version = version
    
    if best_version is None:
        best_version = model_versions[0]
        print("\nWarning: No valid metrics found, selecting first available version")

    print("-" * 80)
    print("\nBest model selected:")
    print(f"  - Version: {best_version.version}")
    print(f"  - Run ID: {best_version.run_id}")
    print(f"  - {metric_name}: {best_metric_value:.4f}")
    
    return best_version, best_metric_value

def load_and_prepare_model(best_version):
    """
    Load the best model and prepare it for production (CPU + eval mode).
    """
    print("\nLoading model from MLflow...")
    model_uri = f"runs:/{best_version.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    print("Moving model to CPU...")
    model = model.to('cpu')
    
    print("Setting model to evaluation mode...")
    model.eval()
    
    return model

def export_to_onnx(model, output_path="models/pet_classifier_model.onnx", 
                   input_size=(1, 3, 224, 224)):
    """
    Export the model to ONNX format.
    """
    print("\nExporting model to ONNX format...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dummy_input = torch.randn(input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model exported to: {output_path}")
    return output_path

def save_class_labels(best_version, output_path="models/class_labels.json"):
    """
    Download and save the class labels of the best model.
    """
    print("\nDownloading class labels...")
    client = MlflowClient()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    artifact_path = client.download_artifacts(
        run_id=best_version.run_id,
        path="class_labels.json"
    )
    
    print(f"Artifact downloaded at: {artifact_path}")
    
    with open(artifact_path, 'r', encoding='utf-8') as f:
        class_labels = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(class_labels, f, indent=2)
    
    print(f"✓ Class labels saved to: {output_path}")
    print(f"  Total classes: {len(class_labels)}")
    
    return class_labels

def main():
    print("=" * 80)
    print("SELECTION AND SERIALIZATION OF THE BEST MODEL")
    print("=" * 80)
    
    os.makedirs("models", exist_ok=True)
    
    best_version, best_accuracy = get_best_model(
        model_name="PetClassifier",
        metric_name="final_val_accuracy"
    )
    
    model = load_and_prepare_model(best_version)
    
    onnx_path = export_to_onnx(
        model, 
        output_path="models/pet_classifier_model.onnx"
    )
    
    class_labels = save_class_labels(
        best_version, 
        output_path="models/class_labels.json"
    )
    
    model_info = {
        "model_version": best_version.version,
        "run_id": best_version.run_id,
        "validation_accuracy": best_accuracy,
        "onnx_model_path": onnx_path,
        "class_labels_path": "models/class_labels.json",
        "num_classes": len(class_labels)
    }
    
    with open("models/model_info.json", 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ PROCESS COMPLETED")
    print("=" * 80)
    print("Generated files:")
    print(f"  1. {onnx_path} - Model serialized in ONNX format")
    print("  2. models/class_labels.json - Class labels")
    print("  3. models/model_info.json - Selected model information")
    print("=" * 80)

if __name__ == "__main__":  # pragma: no cover
    main()
