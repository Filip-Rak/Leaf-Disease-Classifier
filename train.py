import copy
import os
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import seaborn as sns

# ------------------------------
# Attributes
CONFIG = {
    "DATA_DIR": "dataset/",             # Root folder for datasets.
    "TRAIN_DIR": "dataset/train/",      # Training images grouped into subfolders (each for a class).
    "VAL_DIR": "dataset/val/",          # Validation directory for measuring model performance.
    "OUTPUT_DIR": "output/",            # Directory for saved models.
    "IMAGE_SIZE": (224, 224),           # Fixed image size.
    "NUM_CLASSES": 7,                   # The number of categories the model classifies.
    "BATCH_SIZE": 16,                   # Number of images processed in one forward and backward pass.
    "NUM_EPOCHS": 50,                   # How many times the full dataset will pass through during training.
    "LEARNING_RATE": 0.00025,           # Step size for model updates.
    "GPU_ID": 0,                        # Specifies which CUDA GPU to use.
    "NUM_WORKERS": 1,                   # Number of CPU threads used for data loading.
}

# Global variable to store logs
LOG_OUTPUT = ""

# ------------------------------
# Functions
class DiseaseClassifier(nn.Module):
    """EfficientNet-based classifier for plant disease classification."""

    def __init__(self, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Modify classifier to add Dropout before final layer
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 30% Dropout
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

    # Define the prediction flow
    def forward(self, x):
        return self.model(x)

def init_device(gpu_id: int):
    """Initialize and return the device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        torch.backends.cudnn.benchmark = True  # Allow PyTorch picking the fastest convolution algorithm on GPU
        log_print(f"Using device: {device} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        device = torch.device("cpu")
        log_print("CUDA not available, using CPU.")
    return device

def get_data_loaders(train_dir, val_dir, batch_size, image_size, num_workers):
    """Prepare and return the training and validation data loaders."""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        # transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize between [-1, 1]
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize between [-1, 1]
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # pin_memory = Moves data to pinned (non-pageable) memory, making CPU to GPU transfers faster
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def train_model(model, train_loader, device, optimizer, criterion, num_epochs, allowed_loss_increases: int, scheduler=None):
    """Train the model and print progress."""
    # Switch the model into training mode
    model.train()

    total_time = 0.0  # Sum total time
    lowest_loss = sys.float_info.max
    loss_increases_in_a_row = 0
    best_model_state = copy.deepcopy(model.state_dict())  # Store the initial model state
    scaler = torch.amp.GradScaler('cuda')    # Automatic Mixed Precision Scaler

    # Go through entire dataset num_epochs times
    for epoch in range(num_epochs):
        start_time = time.time()    # Start the timer and keep track of total loss
        running_loss = 0.0

        # Go through every training batch
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data onto GPU if enabled
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()                   # Zero gradient from previous batch

            with torch.amp.autocast('cuda'):         # Enable mixed precision
                outputs = model(inputs)             # Run a prediction
                loss = criterion(outputs, labels)   # Compute loss

            scaler.scale(loss).backward()           # Scaled backpropagation
            scaler.step(optimizer)                  # Step optimizer
            scaler.update()                         # Update scaling factor

            running_loss += loss.item()             # Add up entire loss of this epoch

            # Print progress every 10 batches
            if batch_idx % 40 == 0:
                log_print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        # Update the scheduler
        if scheduler:
            scheduler.step(avg_loss)

        # If validation loss is worse, count it
        if avg_loss > lowest_loss:
            loss_increases_in_a_row += 1
            log_print(f"NOTICE: Validation loss increased in this epoch ({loss_increases_in_a_row}/{allowed_loss_increases + 1}).")

            # If the loss increased too many times, revert the model
            if loss_increases_in_a_row > allowed_loss_increases:
                # Debug average loss and training time
                epoch_time = time.time() - start_time
                total_time += epoch_time
                log_print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

                # End training
                log_print(f"NOTICE: Reverting to previous best model state (loss: {lowest_loss:.4f}). Ending training")
                model.load_state_dict(best_model_state)  # Restore best model
                break
        else:
            lowest_loss = avg_loss
            loss_increases_in_a_row = 0  # Reset counter if loss improves
            best_model_state = copy.deepcopy(model.state_dict())  # Save the best model

        # Debug average loss and training time
        epoch_time = time.time() - start_time
        total_time += epoch_time
        log_print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    # Print total time spent in training
    log_print(f"Total time spent in training: {total_time:.2f}s")

    # Return the best model
    return model

def validate_model(model, val_loader, device, criterion):
    """Evaluate the model on the validation dataset."""
    # Set model to validation mode
    model.eval()

    # Track results
    correct = 0
    total = 0
    val_loss = 0.0

    # For confusion matrix
    all_predictions = []
    all_labels = []

    # Disable gradient calculation (faster validation)
    with torch.no_grad():
        # Go through batches of images
        for inputs, labels in val_loader:
            # Move data to selected device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)                 # Make a prediction
            loss = criterion(outputs, labels)       # Compute loss
            val_loss += loss.item()                 # Convert to numerical

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Track correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions & labels for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())  # Convert to list
            all_labels.extend(labels.cpu().numpy())  # Convert to list

    # Debug accuracy and loss
    accuracy = correct / total * 100
    avg_loss = val_loss / len(val_loader)
    log_print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    return cm

def save_output(model, cm, save_path, val_loader):
    """Save the trained model, confusion matrix, and logs to the same directory."""
    os.makedirs(save_path, exist_ok=True)

    # Save model
    model_path = os.path.join(save_path, f"model-{os.path.basename(save_path)}.pth")
    torch.save(model.state_dict(), model_path)
    log_print(f"Model saved to {model_path}")

    # Save confusion matrix
    cm_path = os.path.join(save_path, f"cm-{os.path.basename(save_path)}.png")
    plt.figure(figsize=(10, 8))
    class_names = val_loader.dataset.classes
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Adjust layout to prevent labels from being cut off
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(cm_path)
    log_print(f"Confusion matrix saved to {cm_path}")
    plt.close()

    # Save logs
    log_path = os.path.join(save_path, f"log-{os.path.basename(save_path)}.txt")
    with open(log_path, "w") as log_file:
        log_file.write(LOG_OUTPUT)
    log_print(f"Logs saved to {log_path}")

def get_output_dir_name(save_dir: str):
    """Ask the user for a directory name and handle overwrite checks."""
    os.makedirs(save_dir, exist_ok=True)

    while True:
        dir_name = log_input("Enter output directory name: ").strip()

        if dir_name == "":
            dir_name = "unnamed"

        save_path = os.path.join(save_dir, dir_name)

        if os.path.exists(save_path):
            overwrite = log_input(f"'{dir_name}' already exists. Overwrite? (Y/N): ").strip().lower()
            if overwrite == 'y':
                log_print("Overwriting existing output...")
                return save_path  # Save with the chosen name
            else:
                log_print("Enter a different name.")
        else:
            return save_path  # Save with the new name

def log_print(*args, **kwargs):
    """Custom print function to store logs while also displaying them."""
    global LOG_OUTPUT
    message = " ".join(map(str, args))  # Convert all print args to a single string
    LOG_OUTPUT += message + "\n"        # Append to global log string
    print(message, **kwargs)            # Still print normally to console

def log_input(prompt):
    """Custom input function that logs user input along with the prompt."""
    global LOG_OUTPUT
    user_response = input(prompt)           # Get user input
    log_entry = f"{prompt}{user_response}"  # Combine prompt + input
    LOG_OUTPUT += log_entry + "\n"          # Log input
    return user_response                    # Return input as normal

# ------------------------------
# Main Function
def main():
    """Main training loop."""
    # Query user on the process
    log_print("/* Querying */")
    save_path = get_output_dir_name(CONFIG["OUTPUT_DIR"])

    # Set properties used in training
    log_print("/* Initializing */")
    device = init_device(CONFIG["GPU_ID"])
    train_loader, val_loader = get_data_loaders(CONFIG["TRAIN_DIR"], CONFIG["VAL_DIR"], CONFIG["BATCH_SIZE"], CONFIG["IMAGE_SIZE"], CONFIG["NUM_WORKERS"])
    model = DiseaseClassifier(CONFIG["NUM_CLASSES"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    allowed_loss_increases = CONFIG["NUM_EPOCHS"]

    # Train the model
    log_print("/* Training */")
    model = train_model(model, train_loader, device, optimizer, criterion, CONFIG["NUM_EPOCHS"], allowed_loss_increases, scheduler)

    # Validate the model
    log_print("/* Validation */")
    cm = validate_model(model, val_loader, device, criterion)

    # Save the model
    log_print("/* Saving */")
    save_output(model, cm, save_path, val_loader)

# ------------------------------
# Entry Point
if __name__ == "__main__":
    main()
