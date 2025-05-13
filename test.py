import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from efficient_net import CustomEfficientNetB0
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialization
num_classes = 7 
model = CustomEfficientNetB0(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
pth_path = "output/V2/CustomEffNet_90ep-BS16_LR25e-4-LS5e-2-MA-DR50/model-CustomEffNet_90ep-BS16_LR25e-4-LS5e-2-MA.pth"
checkpoint = torch.load(pth_path, map_location=device)

# Check if it's a full checkpoint or just the state_dict
if isinstance(checkpoint, dict):
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Remove prefixes if present
new_state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}

# Load the weights
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# Prepare test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder("dataset/test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Make predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1) 
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Evaluate
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
accuracy = accuracy_score(all_labels, all_preds)

# Print evaluation results
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Define paths to save reports
report_dir = os.path.dirname(pth_path)
text_report_path = os.path.join(report_dir, "evaluation_report.txt")
image_report_path = os.path.join(report_dir, "confusion_matrix.png")


# Save classification report to text file
with open(text_report_path, "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm, separator=', '))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write(f"\n\nAccuracy: {accuracy:.4f}")

# Save confusion matrix as an image
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Adjust layout to prevent label clipping
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Save and close plot
plt.savefig(image_report_path)
plt.close()

print(f"\nText report saved to: {text_report_path}")
print(f"Confusion matrix image saved to: {image_report_path}")
