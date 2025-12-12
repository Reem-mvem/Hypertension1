

import os
import sys
from typing import Optional

import cv2
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report

# ==== USER PATHS (update these) ====
RETFOUND_ROOT = r"C:\Users\reem2\Downloads\train_eye\RETFound"  # folder containing models_vit.py, util/, etc.
WEIGHTS_PATH = r"C:\Users\reem2\Downloads\train_eye\RETFound_cfp_weights.pth"
DATA_DIR = r"C:\Users\reem2\Downloads\train_eye\data"  # must contain class subfolders Normal/Hypertension
HYPERTENSIVE_IMAGES=r"C:\Users\reem2\Downloads\train_eye\data\Hypertension"  # e.g. r"C:\path\to\data_root\Hypertension"
NORMAL_IMAGES=r"C:\Users\reem2\Downloads\train_eye\data\Normal" # e.g. r"C:\path\to\data_root\Normal"

# Make RETFound imports available
if RETFOUND_ROOT not in sys.path:
    sys.path.append(RETFOUND_ROOT)

from util.pos_embed import interpolate_pos_embed  # noqa: E402
import models_vit  # noqa: E402


def show_samples(folder: Optional[str], limit: int = 5) -> None:
    """Optionally preview a few images using cv2.imshow (requires GUI)."""
    if not folder or not os.path.isdir(folder):
        return
    for image_name in list(os.listdir(folder))[:limit]:
        img_path = os.path.join(folder, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        cv2.imshow("sample", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def build_dataloaders(data_dir: str, batch_size: int = 16, val_ratio: float = 0.2):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomRotation(10),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    print("Classes:", full_dataset.classes)

    n_total = len(full_dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(num_classes: int, weights_path: str, device: torch.device):
    model = models_vit.RETFound_mae(
        num_classes=num_classes,
        drop_path_rate=0.2,
        global_pool=True,
    )

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"]

    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("Loaded checkpoint:", msg)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    return model


def train(model, train_loader, val_loader, device, epochs: int = 5, base_lr: float = 1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=0.05,
    )

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / max(train_total, 1)
        epoch_train_acc = train_correct / max(train_total, 1)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / max(val_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- train_loss: {epoch_train_loss:.4f}, train_acc: {epoch_train_acc:.4f} "
            f"- val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_acc:.4f}"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Optional preview (remove if not needed)
    show_samples(HYPERTENSIVE_IMAGES)
    show_samples(NORMAL_IMAGES)

    train_loader, val_loader = build_dataloaders(DATA_DIR, batch_size=16, val_ratio=0.2)
    model = build_model(num_classes=2, weights_path=WEIGHTS_PATH, device=device)
    train(model, train_loader, val_loader, device, epochs=5, base_lr=1e-4)

    torch.save(model.state_dict(), "fundus_balanced_retefound.pth")
    print("Model saved to fundus_balanced_retefound.pth")


if __name__ == "__main__":
    main()

