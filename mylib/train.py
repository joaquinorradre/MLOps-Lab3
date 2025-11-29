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
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(data_dir="data", batch_size=32, seed=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.OxfordIIITPet(
        root=data_dir, split="trainval", target_types="category",
        download=True, transform=transform
    )

    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    g = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, idx_to_class, len(class_to_idx)

def build_model(model_name, num_classes, learning_rate=0.001):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        params_to_update = model.classifier.parameters()

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        params_to_update = model.classifier.parameters()

    elif model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        params_to_update = model.fc.parameters()
    else:
        raise ValueError(f"Model {model_name} not supported.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    return model, criterion, optimizer

def train_model(model, criterion, optimizer, train_loader, val_loader, device, epochs=5):
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
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
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_acc.item(), step=epoch)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

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
        mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_epoch_acc.item(), step=epoch)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
    
    mlflow.log_metric("final_train_accuracy", history['train_acc'][-1])
    mlflow.log_metric("final_train_loss", history['train_loss'][-1])
    mlflow.log_metric("final_val_accuracy", history['val_acc'][-1])
    mlflow.log_metric("final_val_loss", history['val_loss'][-1])

    return model, history

def plot_metrics(history, filename="loss_curve.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main(args):
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mlflow.set_experiment("Lab3_DeepLearning")
    run_name = args.run_name if hasattr(args, "run_name") else f"{args.model_name}_BS{args.batch_size}_LR{args.learning_rate}"

    with mlflow.start_run(run_name=run_name):
        params = {
            "model_name": args.model_name,
            "weights": "IMAGENET1K_V1",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "dataset": "OxfordIIITPet"
        }
        mlflow.log_params(params)

        train_loader, val_loader, idx_to_class, num_classes = prepare_data(
            batch_size=args.batch_size, seed=args.seed
        )

        with open("class_labels.json", "w") as f:
            json.dump(idx_to_class, f)
        mlflow.log_artifact("class_labels.json")

        model, criterion, optimizer = build_model(
            model_name=args.model_name,
            num_classes=num_classes,
            learning_rate=args.learning_rate
        )

        model, history = train_model(
            model, criterion, optimizer,
            train_loader, val_loader,
            device, epochs=args.epochs
        )

        plot_filename = "loss_curves.png"
        plot_metrics(history, plot_filename)
        mlflow.log_artifact(plot_filename)

        mlflow.pytorch.log_model(model, "model", registered_model_name="PetClassifier")

class Args:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mobilenet_v2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_experiments", action="store_true")
    args = parser.parse_args()

    if args.run_experiments:
        models_to_test = ["mobilenet_v2", "efficientnet_b0", "resnet18"]
        configs = [
            {"batch_size": 16, "learning_rate": 0.001, "seed": 42},
            {"batch_size": 32, "learning_rate": 0.001, "seed": 42},
            {"batch_size": 32, "learning_rate": 0.0001, "seed": 42},
            {"batch_size": 64, "learning_rate": 0.0001, "seed": 42},
        ]
        
        experiment_counter = 0
        for model_name in models_to_test:
            for config in configs:
                exp_config = {
                    "model_name": model_name,
                    "epochs": 5,
                    "batch_size": config["batch_size"],
                    "learning_rate": config["learning_rate"],
                    "seed": config["seed"]
                }
                current_args = Args(exp_config)
                current_args.run_name = f"{model_name}_BS{config['batch_size']}_LR{config['learning_rate']}_exp{experiment_counter}"
                print(f"\n{'='*60}")
                print(f"Running experiment {experiment_counter + 1}/9: {current_args.run_name}")
                print(f"{'='*60}\n")
                main(current_args)
                experiment_counter += 1
    else:
        main(args)

