import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import kornia.augmentation as K
import torchvision.models as models
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import seaborn as sns

from efficient_net import CustomEfficientNetB0

# Attributes
# ------------------------------
CONFIG = {
    "DATA_DIR": "dataset/",             # Root directory for datasets.
    "TRAIN_DIR": "dataset/train/",      # Training images grouped into subfolders (each for a class).
    "VAL_DIR": "dataset/val/",          # Validation directory for measuring model performance.
    "TEST_DIR": "dataset/test/",        # Directory with images for the final model test.
    "OUTPUT_DIR": "output/V2/",         # Directory for saved models.
    "IMAGE_SIZE": (224, 224),           # Fixed image size.
    "NUM_CLASSES": 7,                   # The number of categories the model classifies.
    "BATCH_SIZE": 16,                   # Number of images processed in one forward and backward pass.
    "NUM_EPOCHS": 90,                   # How many times the full dataset will pass through during training.
    "LEARNING_RATE": 0.00025,           # Step size for model updates.
    "LABEL_SMOOTHING": 0.05,            # Prevents overconfidence by slightly adjusting target labels.
    "SCHEDULER_MODE": "min",            # Tack 'min' (validation loss decreasing) or 'max' (accuracy increasing).
    "SCHEDULER_FACTOR": 0.5,            # Factor by which the learning rate is reduced when the scheduler is triggered.
    "SCHEDULER_PATIENCE": 2,            # Number of epochs with no improvement before the learning rate is reduced.
    "DROPOUT_FACTOR": 0.3,              # The percentage of neurons randomly disabled during training to prevent overfitting.
    "GPU_ID": 0,                        # Specifies which CUDA GPU to use.
    "NUM_WORKERS": 1,                   # Number of CPU threads used for data loading.
}

# Global variable to store logs
LOG_OUTPUT = ""

class DiseaseClassifier(nn.Module):
    """EfficientNet-based classifier for plant disease classification."""

    def __init__(self, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # self.model = CustomEfficientNetB0()

        # Modify classifier to add Dropout before final layer
        self.model.classifier = nn.Sequential(
            nn.Dropout(CONFIG["DROPOUT_FACTOR"]),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

    # Define the prediction flow
    def forward(self, x):
        return self.model(x)

gpu_augmentations = torch.nn.Sequential(
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),
    K.RandomRotation(15.0),
    # K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    # K.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    # K.RandomGaussianBlur((3, 3), (0.1, 2.0)),
    # K.RandomMotionBlur(kernel_size=(3, 5), angle=10.0, direction=0.1),
).cuda()

# Functions
# ------------------------------

# Initialization
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

def get_data_loaders(train_dir, val_dir, test_dir, batch_size, image_size, num_workers):
    """Prepare and return the training and validation data loaders."""
    generic_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize between [-1, 1]
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=generic_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=generic_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=generic_transform)

    # pin_memory = Moves data to pinned (non-pageable) memory, making CPU to GPU transfers faster
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# Training
def train_one_epoch(model, device, train_loader, optimizer, scaler, criterion, epoch):
    total_epoch_loss = 0.0
    model.train()   # Set the model to training mode

    # Loop through all the batches
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move data onto device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Apply GPU augmentations
        inputs = gpu_augmentations(inputs)

        optimizer.zero_grad()                   # Make sure gradients are zeroed

        with torch.amp.autocast('cuda'):        # Use mixed precision for performance boost
            outputs = model(inputs)             # Run a prediction
            loss = criterion(outputs, labels)   # Compute loss

        scaler.scale(loss).backward()           # Scaled backpropagation
        scaler.step(optimizer)                  # Step optimizer
        scaler.update()                         # Update scaling factor

        total_epoch_loss += loss.item()         # Add up entire loss of this epoch

        # Occasionally log and print the progress
        # if batch_idx % 40 == 0:
        # log_print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Return the average loss of this epoch
    return total_epoch_loss / len(train_loader)

def validate_model(model, val_loader, device, criterion):
    model.eval()    # Set the model to validation mode
    total_loss = 0.0

    # Disable gradient calculation for faster validation
    with torch.no_grad():
        # Go through all images
        for inputs, labels in val_loader:
            # Move data onto device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)                 # Make a prediction
            loss = criterion(outputs, labels)       # Compute loss
            total_loss += loss.item()               # Convert to numerical

    # Return the average loss of validation
    return total_loss / len(val_loader)

def main_training_loop(model, train_loader, val_loader, device, optimizer, criterion, num_epochs, allowed_loss_increases, scheduler):
    """Train and validate the model while printing progress"""
    total_training_time = 0.0
    loss_increases_in_a_row = 0
    lowest_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict)
    scaler = torch.amp.GradScaler('cuda')

    # Go through entire dataset num_epochs times
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Train the model
        training_loss = train_one_epoch(model, device, train_loader, optimizer, scaler, criterion, epoch)

        # Validate the model
        validation_loss = validate_model(model, val_loader, device, criterion)

        # Analyse the loss
        scheduler.step(validation_loss)

        if validation_loss > lowest_val_loss:
            loss_increases_in_a_row += 1
            log_print(f"NOTICE: Validation loss is worse than best: {lowest_val_loss:.4f}. In a row: {loss_increases_in_a_row}")

            # If the loss increased too many times, revert the model
            if loss_increases_in_a_row > allowed_loss_increases:
                # Debug average loss and training time
                epoch_time = time.time() - epoch_start_time
                total_training_time += epoch_time

                # Log and print data about the loss and time
                log_print(f"Epoch {epoch + 1}/{num_epochs}, Training loss: {training_loss:.4f}, Validation loss: {validation_loss:.4f}, Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']}")

                # End training
                log_print(f"NOTICE: Reverting to previous best model state (loss: {lowest_val_loss:.4f}). Ending training")
                break
        else:
            lowest_val_loss = validation_loss
            loss_increases_in_a_row = 0  # Reset counter if loss improves
            best_model_state = copy.deepcopy(model.state_dict())  # Save the best model

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time

        # Log and print data about the loss and time
        log_print(f"Epoch {epoch + 1}/{num_epochs}, Training loss: {training_loss:.4f}, Validation loss: {validation_loss:.4f}, Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']}")

    # Print total time spent in training
    log_print(f"Total time spent in training: {total_training_time:.2f}s")

    # Return the best model
    log_print(f"Loading the best model with validation loss: {lowest_val_loss:.4f}")
    model.load_state_dict(best_model_state)
    return model

# Testing
def test_model(model, test_loader, device, criterion):
    """Evaluate the model on the test dataset."""
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
        for inputs, labels in test_loader:
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
    avg_loss = val_loss / len(test_loader)
    log_print(f"Test loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    return cm

# Miscellaneous
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

# Main Function
# ------------------------------
def main():
    """Main execution function."""
    # Query user on the process
    log_print("/* Querying */")
    save_path = get_output_dir_name(CONFIG["OUTPUT_DIR"])

    # Set properties used in training
    log_print("/* Initializing */")
    device = init_device(CONFIG["GPU_ID"])
    train_loader, val_loader, test_loader = get_data_loaders(CONFIG["TRAIN_DIR"], CONFIG["VAL_DIR"], CONFIG["TEST_DIR"], CONFIG["BATCH_SIZE"], CONFIG["IMAGE_SIZE"], CONFIG["NUM_WORKERS"])
    model = DiseaseClassifier(CONFIG["NUM_CLASSES"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["LABEL_SMOOTHING"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=CONFIG["SCHEDULER_MODE"], factor=CONFIG["SCHEDULER_FACTOR"], patience=CONFIG["SCHEDULER_PATIENCE"])
    allowed_loss_increases = CONFIG["NUM_EPOCHS"]

    # Train the model
    log_print("/* Training */")
    model = main_training_loop(model, train_loader, val_loader, device, optimizer, criterion, CONFIG["NUM_EPOCHS"], allowed_loss_increases, scheduler)

    # Test the model
    log_print("/* Final Testing */")
    cm = test_model(model, test_loader, device, criterion)

    # Save the model
    log_print("/* Saving */")
    save_output(model, cm, save_path, val_loader)

# Entry Point
# ------------------------------
if __name__ == "__main__":
    main()
