import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import os

# ------------------------------
# Configuration
CONFIG = {
    "DATA_DIR": "dataset/",             # Root folder for datasets.
    "TRAIN_DIR": "dataset/train/",      # Training images grouped into subfolders (each for a class).
    "VAL_DIR": "dataset/val/",          # Validation directory for measuring model performance.
    "MODELS_DIR": "models/",            # Directory for saved models.
    "IMAGE_SIZE": (224, 224),           # Fixed image size.
    "NUM_CLASSES": 7,                   # The number of categories the model classifies.
    "BATCH_SIZE": 64,                   # Number of images processed in one forward and backward pass.
    "NUM_EPOCHS": 1,                    # How many times the full dataset will pass through during training.
    "LEARNING_RATE": 0.001,             # Step size for model updates.
    "GPU_ID": 0,                        # Specifies which CUDA GPU to use.
    "NUM_WORKERS": 8,                   # Number of CPU threads used for data loading.
}


# ------------------------------
# Device Initialization
def init_device(gpu_id: int):
    """Initialize and return the device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        torch.backends.cudnn.benchmark = True  # Allow PyTorch picking the fastest convolution algorithm on GPU
        print(f"Using device: {device} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")
    return device


# ------------------------------
# Data Preparation
def get_data_loaders(train_dir, val_dir, batch_size, image_size, num_workers):
    """Prepare and return the training and validation data loaders."""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ------------------------------
# Model Definition
class DiseaseClassifier(nn.Module):
    """EfficientNet-based classifier for plant disease classification."""

    def __init__(self, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    # Define the prediction flow
    def forward(self, x):
        return self.model(x)


# ------------------------------
# Training Function
def train_model(model, train_loader, device, optimizer, criterion, num_epochs):
    """Train the model and print progress."""
    # Switch the model into training mode
    model.train()

    # Go through entire dataset num_epochs times
    for epoch in range(num_epochs):
        # Start the timer and keep track of total loss
        start_time = time.time()
        running_loss = 0.0

        # Go through every training batch
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data onto GPU if enabled
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()               # Zero gradient from previous batch
            outputs = model(inputs)             # Run a prediction
            loss = criterion(outputs, labels)   # Compute loss
            loss.backward()                     # Compute the gradient of the loss
            optimizer.step()                    # Update model weights

            running_loss += loss.item()         # Add up entire loss of this epoch

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Debug average loss and training time
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")


# ------------------------------
# Validation Function
def validate_model(model, val_loader, device, criterion):
    """Evaluate the model on the validation dataset."""
    # Set model to validation mode
    model.eval()

    # Track results
    correct = 0
    total = 0
    val_loss = 0.0

    # Disable gradient calculation
    with torch.no_grad():
        # Go through batches of images
        for inputs, labels in val_loader:
            # Transfer data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)                 # Make a prediction
            loss = criterion(outputs, labels)       # Compute loss
            val_loss += loss.item()                 # Convert to numerical

            _, predicted = torch.max(outputs, 1)    # Sum correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Debug accuracy and loss
    accuracy = correct / total * 100
    avg_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return accuracy, avg_loss


# ------------------------------
# Model Saving Function
def save_model(model, save_dir, filename="model.pth"):
    """Save the trained model to disk."""
    os.makedirs(save_dir, exist_ok=True)            # Create save dir if it doesn't exist
    save_path = os.path.join(save_dir, filename)    # Create the full path
    torch.save(model.state_dict(), save_path)       # Save the model
    print(f"Model saved to {save_path}")


# ------------------------------
# Main Function
def main():
    """Main training loop."""
    device = init_device(CONFIG["GPU_ID"])
    train_loader, val_loader = get_data_loaders(CONFIG["TRAIN_DIR"], CONFIG["VAL_DIR"], CONFIG["BATCH_SIZE"],
                                                CONFIG["IMAGE_SIZE"], CONFIG["NUM_WORKERS"])

    model = DiseaseClassifier(CONFIG["NUM_CLASSES"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, device, optimizer, criterion, CONFIG["NUM_EPOCHS"])
    validate_model(model, val_loader, device, criterion)

    save_model(model, CONFIG["MODELS_DIR"])


# ------------------------------
# Entry Point
if __name__ == "__main__":
    main()
