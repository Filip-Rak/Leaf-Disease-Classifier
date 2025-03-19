import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """Swish Activation Function x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation Block: Enhances important channels"""
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Reduce each channel to a single value
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)  # Reduce channel size
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)  # Expand channel size back

    def forward(self, x):
        scale = self.pool(x)                    # Compute channel-wise statistics
        scale = F.silu(self.fc1(scale))         # Apply Swish activation
        scale = torch.sigmoid(self.fc2(scale))  # Generate per channel scaling factors
        return x * scale                        # Rescale the input feature map

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv) Block"""
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1):
        super(MBConv, self).__init__()
        hidden_dim = in_channels * expansion_factor # Expand channel dimensions
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels) # Skip connection if input matches output

        # Expansion phase (1x1 convolution to increase channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # Depth wise Convolution (3x3 spatial filtering)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # Squeeze-and-Excitation layer (feature reweighing)
        self.se = SqueezeExcitation(hidden_dim)

        # Projection phase (1x1 convolution to reduce channels)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))   # Apply expansion
        out = F.silu(self.bn2(self.conv2(out))) # Apply depth wise convolution
        out = self.se(out)                      # Apply squeeze and excitation
        out = self.bn3(self.conv3(out))         # Apply projection

        if self.use_residual:
            out += x    # Add residual connection (skip connection)

        return out

class CustomEfficientNetB0(nn.Module):
    """Custom Implementation of EfficientNet B0"""
    def __init__(self, num_classes=7):
        super(CustomEfficientNetB0, self).__init__()

        # Initial layer (Convolution) to extract low-level features
        self.stem = nn.Sequential(
            # Convert 3 RGB channels to 32 channel feature map
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), # Normalize data
            Swish() # Apply activation function
        )

        # Extracts features from the image (edges, textures, shapes, objects)
        self.blocks = nn.Sequential(
            MBConv(32, 16, expansion_factor=1, stride=1),   # Initial lightweight block
            MBConv(16, 24, expansion_factor=6, stride=2),   # Downsample to lear mid-level features
            MBConv(24, 40, expansion_factor=6, stride=2),   # Downsample further
            MBConv(40, 80, expansion_factor=6, stride=2),   # Learn shapes and structures
            MBConv(80, 112, expansion_factor=6, stride=1),  # Focus on important object parts
            MBConv(112, 192, expansion_factor=6, stride=2), # Further downsample. Capture complex patterns
            MBConv(192, 320, expansion_factor=6, stride=1)  # Fully extracted deep features
        )

        # Final processing before classification
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),    # Expand feature map
            nn.BatchNorm2d(1280),       # Normalize features for stable training
            Swish(),                    # Apply activation for better gradient flow
            nn.AdaptiveAvgPool2d(1)     # Convert the entire feature map to 1x1 representation
        )

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Randomly disable 30% of neurons to prevent overfitting
            nn.Linear(1280, num_classes)  # Map feature vector to class possibilities
        )

    def forward(self, x):
        """Image is passed through several transformations and classification is returned """
        x = self.stem(x)            # Extract initial low-level features.
        x = self.blocks(x)          # Extract deep hierarchical features with MBConv.
        x = self.head(x)            # Summarize all extracted features into a single compact feature vector.
        x = torch.flatten(x, 1)     # Convert multidimensional feature map into a 1D vector of 1280 values.
        x = self.classifier(x)      # Transform extracted features into a class prediction.
        return x
