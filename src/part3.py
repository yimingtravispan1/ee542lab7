
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results" / "part3"


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.fc2.bias, 0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    epochs = 10
    learning_rate = 0.001
    train_batch_size = 64
    test_block_size = 512

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Using device:", device, flush=True)

    # 1. 加载已有的训练集和测试集
    transform = transforms.Compose([
        transforms.Resize(
            (64, 64),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train", transform=transform
    )
    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test", transform=transform
    )

    assert (
        train_dataset.class_to_idx
        == test_dataset.class_to_idx
        == {"cat": 0, "dog": 1}
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_block_size,
        shuffle=False,
        num_workers=0,
    )

    model = CatDogCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    history = []
    synchronize(device)
    training_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        synchronize(device)
        epoch_start = time.perf_counter()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            count = images.size(0)
            total_loss += loss.item() * count
            correct += (
                (outputs.detach() > 0.5) == (labels > 0.5)
            ).sum().item()
            total += count

        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        history.append({
            "epoch": epoch + 1,
            "train_mse": epoch_loss,
            "train_accuracy_during_updates": epoch_accuracy,
            "seconds": epoch_seconds,
        })

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"MSE: {epoch_loss:.4f} | "
            f"Train accuracy: {epoch_accuracy:.2%} | "
            f"Time: {epoch_seconds:.2f}s",
            flush=True,
        )

    synchronize(device)
    training_seconds = time.perf_counter() - training_start

  
    model.eval()
    with torch.no_grad():
        warmup_images = torch.zeros(
            test_block_size, 3, 64, 64, device=device
        )
        for _ in range(3):
            model(warmup_images)
        synchronize(device)
        del warmup_images

    blocks = []
    test_correct = 0
    test_total = 0
    test_loss_sum = 0.0

    synchronize(device)
    test_start = time.perf_counter()
    test_iterator = iter(test_loader)

    with torch.no_grad():
        for block_number in range(1, len(test_loader) + 1):
            synchronize(device)
            block_start = time.perf_counter()

            images, labels = next(test_iterator)
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            count = images.size(0)
            block_correct = (
                (outputs > 0.5) == (labels > 0.5)
            ).sum().item()

            test_loss_sum += loss.item() * count
            test_correct += block_correct
            test_total += count

            synchronize(device)
            block_seconds = time.perf_counter() - block_start

            blocks.append({
                "block": block_number,
                "images": count,
                "accuracy": block_correct / count,
                "seconds": block_seconds,
                "images_per_second": count / block_seconds,
            })

    synchronize(device)
    test_seconds = time.perf_counter() - test_start

    metrics = {
        "torch_version": str(torch.__version__),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda" else None
        ),
        "model_seed": 42,
        "class_mapping": train_dataset.class_to_idx,
        "train_images": len(train_dataset),
        "test_images": test_total,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "test_block_size": test_block_size,
        "num_workers": 0,
        "resize": "Pillow RGB BICUBIC",
        "dtype": "float32",
        "loss": "MSE",
        "optimizer": "Adam",
        "output_activation": "ReLU",
        "prediction_threshold": 0.5,
        "training_seconds": training_seconds,
        "test_seconds": test_seconds,
        "test_mse": test_loss_sum / test_total,
        "test_accuracy": test_correct / test_total,
        "test_images_per_second": test_total / test_seconds,
        "timing_scope": "loading, preprocessing, transfer, computation",
    }
    with (OUTPUT_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    for filename, rows in [
        ("losses.csv", history),
        ("test_blocks.csv", blocks),
    ]:
        with (OUTPUT_DIR / filename).open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")

    print("\nTraining and evaluation completed!")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Test MSE: {metrics['test_mse']:.4f}")
    print(f"Training time: {training_seconds:.2f}s")
    print(f"Test time: {test_seconds:.2f}s")
    print(
        f"Throughput: "
        f"{metrics['test_images_per_second']:.2f} images/s"
    )
    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()