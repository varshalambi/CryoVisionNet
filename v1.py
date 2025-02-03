import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        # ... add more encoder blocks
        
        # Decoder (Upsampling)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        # ... add more decoder blocks
        
        # Output heads
        self.detection_head = nn.Conv3d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Implement U-Net skip connections
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        # ...
        y = self.dec1(x2)
        # ...
        return self.detection_head(y)

# Custom loss (F-beta weighted)
class ParticleLoss(nn.Module):
    def __init__(self, beta=4, class_weights=[1,1,1,2,2]):
        super().__init__()
        self.beta = beta
        self.class_weights = torch.tensor(class_weights)
        
    def forward(self, pred, target):
        # pred: (B, C, D, H, W), target: (B, C, D, H, W)
        tp = (pred * target).sum(dim=(2,3,4))
        fp = (pred * (1 - target)).sum(dim=(2,3,4))
        fn = ((1 - pred) * target).sum(dim=(2,3,4))
        
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        
        f_beta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall)
        weighted_f = (f_beta * self.class_weights.to(pred.device)).mean()
        return 1 - weighted_f



import zarr
import numpy as np

# Load a tomogram
tomogram_path = "train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr"
store = zarr.DirectoryStore(tomogram_path)
root = zarr.group(store=store)
data = root['0'][:]  # Highest resolution scale
print(f"Tomogram shape: {data.shape}")

from copick.models import CopickPicks

picks = CopickPicks.from_file("train/overlay/ExperimentRuns/TS_5_4/Picks/ribosome.json")
coordinates = [(p.x, p.y, p.z) for p in picks.points]

def create_heatmap(shape, coordinates, radius):
    heatmap = np.zeros(shape)
    for (x, y, z) in coordinates:
        # Convert coordinates to voxel indices
        xx, yy, zz = int(x), int(y), int(z)
        # Create Gaussian blob
        # (Implement 3D Gaussian kernel centered at (xx,yy,zz))
    return heatmap

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=[1,1,1,2,2], gamma=2):
        super().__init__()
        self.alpha = torch.tensor(alpha)  # Weights for [easy, easy, easy, hard, hard]
        self.gamma = gamma

    def forward(self, preds, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        weighted_loss = focal_loss * self.alpha.to(preds.device)[None,:,None,None,None]
        return weighted_loss.mean()
    

from scipy.ndimage import maximum_filter

def radius_nms(heatmap, radius=5):
    max_filtered = maximum_filter(heatmap, size=2*radius+1)
    peaks = (heatmap == max_filtered) & (heatmap > threshold)
    return np.argwhere(peaks)


scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()