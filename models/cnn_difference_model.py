import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage import median_filter
from skimage.metrics import structural_similarity as ssim

class SimpleDiffCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv(x)

class CNNDifferenceModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = SimpleDiffCNN().to(self.device)
        self.model.eval()  # No training, just inference
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def predict(self, img1: Image.Image, img2: Image.Image, method='absdiff') -> np.ndarray:
        t1 = self.transform(img1)
        t2 = self.transform(img2)
        if method == 'cnn':
            diff = torch.abs(t1.unsqueeze(0) - t2.unsqueeze(0)).to(self.device)
            with torch.no_grad():
                mask = self.model(diff)
            mask = mask.squeeze().cpu().numpy()
            return mask
        elif method == 'absdiff':
            diff = torch.abs(t1 - t2)
            diff = diff.mean(dim=0).numpy()  # Average over channels
            return diff
        elif method == 'absdiff_median':
            diff = torch.abs(t1 - t2)
            diff = diff.mean(dim=0).numpy()
            mask = median_filter(diff, size=5)
            return mask
        elif method == 'ssim':
            arr1 = np.array(img1.resize((256, 256))).astype(np.float32) / 255.0
            arr2 = np.array(img2.resize((256, 256))).astype(np.float32) / 255.0
            # Compute SSIM for each channel, then average
            ssim_map = np.zeros(arr1.shape[:2])
            for c in range(3):
                ssim_map_c, _ = ssim(arr1[:,:,c], arr2[:,:,c], full=True)
                ssim_map += ssim_map_c
            ssim_map /= 3.0
            mask = 1.0 - ssim_map  # Invert: high value = more change
            return mask
        else:
            raise ValueError(f"Unknown method: {method}") 