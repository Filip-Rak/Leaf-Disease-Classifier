from torchvision.models import EfficientNet_B0_Weights
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch

# ------------------------------
# Settings

# Paths
DATA_DIR = "dataset/"
TRAIN_DIR = DATA_DIR + "train/"
TEST_DIR = DATA_DIR + "test/"

# Image
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64

# Model
BASE_MODEL = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
NUMBER_OF_FEATURES = 7
CRITERION = nn.CrossEntropyLoss()
NUM_EPOCHS = 20
GPU_ID = 0
LOAD_PROC = 6


# ------------------------------
# Functions
def arrange_data(train_dir: str, test_dir: str, batch_size: int, image_size: tuple, load_proc: int) -> tuple:
    # Transform training dataset with augmentations
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Transform testing dataset without augmentations
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load data
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # Load data in batches using DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=load_proc,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=load_proc, pin_memory=True)

    return train_loader, test_loader


def init_device(gpu_id: int):
    # Check if CUDA is available and select GPU manually
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu_id))  # Force GPU
        torch.cuda.set_device(gpu_id)  # Explicitly set the device
        torch.backends.cudnn.benchmark = True  # Optimizes training speed for your hardware
        torch.backends.cudnn.enabled = True  # Ensures cuDNN is being used
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def init_model(base_model, features: int, device):
    # Use provided model
    model = base_model

    # Set number of features
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, features)

    # Move the model to the device
    model = model.to(device)

    return model


def train_model(model, train_loader, device, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Before moving to device -> Inputs: {inputs.device}, Labels: {labels.device}")

            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            print(f"Batch {batch_idx}: After moving to device -> Inputs: {inputs.device}, Labels: {labels.device}")

            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Predict
            loss = criterion(outputs, labels)  # Estimate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")


# ------------------------------
# Main Function
def main():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA-compatible GPU found.")

    train_loader, test_loader = arrange_data(TRAIN_DIR, TEST_DIR, BATCH_SIZE, IMAGE_SIZE, LOAD_PROC)
    device = init_device(GPU_ID)
    model = init_model(BASE_MODEL, NUMBER_OF_FEATURES, device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, device, optimizer, CRITERION, NUM_EPOCHS)


# ------------------------------
# Entry Point - Runs if executed directly
if __name__ == "__main__":
    main()
