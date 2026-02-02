"""
02_cnn_model_training.py
========================

Description:
    This script is responsible for training the Deep Learning backbone (FontCNN)
    of the hybrid pipeline. It operates in "Phase 1" of the project, where the goal
    is to learn robust visual feature representations from the pixel data.

Key Operations:
    1.  **Data Loading**: Uses `src.data_loader` to load tensors from `data/processed_fonts.pt`.
    2.  **Model Initialization**: Instantiates the `FontCNN` (with Spatial Attention).
    3.  **Training Loop**:
        - Optimizes weights using Adam (lr=0.001) and Cross-Entropy Loss.
        - Monitors Validation Accuracy after every epoch.
        - Saves the *best* model (state_dict) to `cnn_model_output/best_cnn.pth`.
    4.  **visualization**: Plots Loss and Accuracy curves at the end of training.

Usage:
    python 02_cnn_model_training.py
"""

import torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

# Import your modules
from src.data_loader import get_dataloaders
from src.cnn_architecture import FontCNN

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "cnn_model_output/best_cnn.pth"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    # Progress bar for the batch loop
    loop = tqdm(loader, leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward Pass (Ignore the 'embedding' output for now)
        logits, _ = model(images)

        # Loss & Backward
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar
        loop.set_description(f"Loss: {loss.item():.4f}")

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits, _ = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()

            # Calculate Accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), correct / total


def main():
    print(f" Starting Phase 1 Training on {DEVICE}...")

    # 1. Load Data
    # Ensure you have run preprocess.py or have the cache ready
    train_loader, val_loader, _, num_classes, img_dim = get_dataloaders(
        data_path="data/processed_fonts.pt",
        batch_size=BATCH_SIZE,
        include_other=False,
    )

    if isinstance(img_dim, tuple) or isinstance(img_dim, list):
        input_size_int = img_dim[-1]  # Take the last value (width)
    else:
        input_size_int = img_dim

    print(f" Data Loaded. Classes: {num_classes}, Input Size: {input_size_int}")

    # 2. Initialize Model (Pass the integer, not the tuple)
    model = FontCNN(num_classes=num_classes, input_dim=input_size_int).to(DEVICE)

    # 3. Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create folder for saving
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Lists for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_acc = 0.0

    # Start Timer
    start_time = time.time()

    # 4. Training Loop
    print("\n Training Start:")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc

            # Calculate elapsed time so far
            current_duration = time.time() - start_time

            # SAVE EVERYTHING (Weights + Metadata)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "training_time": current_duration,
                    "best_acc": best_acc,
                    "epochs": epoch + 1,
                },
                SAVE_PATH,
            )
            print(
                f"   🏆 New Best Model Saved! ({val_acc:.2%}  in Time: {current_duration:.2f}s)"
            )

    print("\n Training Complete.")
    print(f"Best Validation Accuracy: {best_acc:.2%}")
    print(f"Model saved to: {SAVE_PATH}")

    # 5. Plot Results
    plt.figure(figsize=(10, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, color="green", label="Val Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
