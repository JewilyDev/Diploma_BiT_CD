from PIL import Image
import torch
from torchvision import transforms

# Preprocess a pair of images for change detection
# Concatenates two images along the channel dimension

def preprocess_pair(img1: Image.Image, img2: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    t1 = transform(img1)
    t2 = transform(img2)
    # Stack or concatenate along channel dimension: result has 6 channels
    return torch.cat([t1, t2], dim=0) 