import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import mlflow
import mlflow.pytorch
import json

def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(data_dir="data", batch_size=32):
    """
    Download and prepare the Oxford-IIIT Pet dataset.
    Apply transformations: Resize 224x224, ToTensor, Normalize.
    Split into 80% train and 20% validation.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Downloading/Loading dataset...")
    full_dataset = datasets.OxfordIIITPet(root=data_dir, split="trainval", target_types="category", download=True, transform=transform)
    
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, idx_to_class, len(class_to_idx)

def build_model(num_classes, learning_rate=0.001):
    """
    Load pre-trained MobileNet_v2, freeze weights, and modify the classifier.
    """
    print("Loading MobileNet_v2...")
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train_model(model, criterion, optimizer, train_loader, val_loader, device, epochs=5):
    """
    Train the model and evaluate each epoch.
    """
    model.to(device)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_acc.item(), step=epoch)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_corrects.double() / val_total

        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_epoch_acc.item(), step=epoch)

    return model

def main(args):
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow.set_experiment("Lab3_DeepLearning")
    
    run_name = f"MobileNetV2_BS{args.batch_size}_LR{args.learning_rate}_E{args.epochs}"

    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "model_name": "mobilenet_v2",
            "weights": "IMAGENET1K_V1",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "dataset": "OxfordIIITPet"
        }
        mlflow.log_params(params)

        train_loader, val_loader, idx_to_class, num_classes = prepare_data(
            batch_size=args.batch_size
        )
        
        print("Saving class labels...")
        with open("class_labels.json", "w") as f:
            json.dump(idx_to_class, f)
        mlflow.log_artifact("class_labels.json")

        model, criterion, optimizer = build_model(
            num_classes=num_classes, 
            learning_rate=args.learning_rate
        )

        print("Starting training...")
        model = train_model(
            model, criterion, optimizer, 
            train_loader, val_loader, 
            device, epochs=args.epochs
        )

        print("Logging model in MLflow...")
        mlflow.pytorch.log_model(model, "model")
        
        print(f"Training completed. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 with MLflow")
    
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--run_experiments", action="store_true", help="Run multiple experiments with different configurations")

    args = parser.parse_args()

    if args.run_experiments:
        print("🚀 Running multi-experiment mode")
        experiments = [
            {"epochs": 3, "batch_size": 16, "learning_rate": 0.01, "seed": 42},
            {"epochs": 3, "batch_size": 32, "learning_rate": 0.001, "seed": 42},
            {"epochs": 3, "batch_size": 64, "learning_rate": 0.0005, "seed": 42},
        ]
        
        for i, exp_config in enumerate(experiments):
            print(f"\n--- Starting Experiment {i+1}/{len(experiments)} ---")
            print(f"Config: {exp_config}")
            
            class Args:
                def __init__(self, dictionary):
                    for k, v in dictionary.items():
                        setattr(self, k, v)
            
            current_args = Args(exp_config)
            main(current_args)
            
        print("\n✅ All experiments have completed.")
    else:
        main(args)
