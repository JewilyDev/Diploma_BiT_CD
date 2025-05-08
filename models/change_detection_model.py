import torch
from argparse import Namespace
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
import torchvision.transforms.functional as TF

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from BIT_CD.models.basic_model import CDEvaluator
from BIT_CD.utils import get_device

def split_into_patches(img: Image.Image, patch_size=256, overlap=64):
    w, h = img.size
    patches = []
    coords = []
    step = patch_size - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            box = (x, y, min(x+patch_size, w), min(y+patch_size, h))
            patch = img.crop(box)
            if patch.size != (patch_size, patch_size):
                new_patch = Image.new("RGB", (patch_size, patch_size))
                new_patch.paste(patch, (0, 0))
                patch = new_patch
            patches.append(patch)
            coords.append((x, y))
    return patches, coords, (w, h)

def stitch_patches(patches, coords, full_size, patch_size=256, overlap=64):
    w, h = full_size
    mask = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    
    # Weight matrix (Gaussian window)
    weight = np.ones((patch_size, patch_size), dtype=np.float32)
    for i in range(overlap):
        weight[:, i] *= (i + 1) / overlap
        weight[:, -i-1] *= (i + 1) / overlap
        weight[i, :] *= (i + 1) / overlap
        weight[-i-1, :] *= (i + 1) / overlap
    
    for patch, (x, y) in zip(patches, coords):
        arr = np.array(patch).astype(np.float32) / 255.0
        arr = arr[:patch_size, :patch_size]
        
        x_end = min(x + patch_size, w)
        y_end = min(y + patch_size, h)
        pw = x_end - x
        ph = y_end - y
        
        mask[y:y_end, x:x_end] += arr[:ph, :pw] * weight[:ph, :pw]
        weights[y:y_end, x:x_end] += weight[:ph, :pw]
    
    mask = np.divide(mask, weights, where=weights != 0)
    return (mask > 0.5).astype(np.uint8)

class ChangeDetectionModel:
    def __init__(self, model_path: str,
                 net_G: str = 'base_transformer_pos_s4_dd8_dedim8',
                 device: str = 'cpu'):
        self.device = device
        
        # Prepare BIT-CD args similar to demo.py
        args = Namespace(
            project_name='BIT_LEVIR',
            gpu_ids=[0] if device == 'cuda' else [],  # Keep as list for basic_model.py
            checkpoint_root='checkpoints',
            output_folder='samples/predict',
            num_workers=0,
            dataset='CDDataset',
            data_name='quick_start',
            batch_size=1,
            split="demo",
            img_size=256,
            n_class=2,
            net_G=net_G,
            checkpoint_name='best_ckpt.pt'
        )
        
        # Set up the checkpoint directory
        args.checkpoint_dir = "C:\\Users\\monch\\OneDrive\\Документы\\diploma_bit_cd\\Diploma_BiT_CD\\BIT_CD\\checkpoints\\BIT_LEVIR"
        # Initialize model directly without get_device
        self.model = CDEvaluator(args)
        
        # Load checkpoint
        self.model.load_checkpoint(args.checkpoint_name)
        self.model.eval()
        
        # Preprocessing for bitemporal inputs
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, img1: Image.Image, img2: Image.Image) -> np.ndarray:
        # Apply transforms
        t1 = self.transform(img1.convert('RGB')).unsqueeze(0)  # Add batch dimension
        t2 = self.transform(img2.convert('RGB')).unsqueeze(0)  # Add batch dimension

        # Create batch dictionary similar to demo.py
        batch = {
            'A': t1,
            'B': t2,
            'name': 'prediction'
        }
        
        # Forward pass
        with torch.no_grad():
            score_map = self.model._forward_pass(batch)
            pred = torch.argmax(score_map, dim=1)
        
        return pred.squeeze().cpu().numpy()

    def patchwise_predict(self, img1: Image.Image, img2: Image.Image, patch_size=256, overlap=64) -> np.ndarray:
        patches1, coords, full_size = split_into_patches(img1, patch_size, overlap)
        patches2, _, _ = split_into_patches(img2, patch_size, overlap)
        
        mask_patches = []
        for p1, p2 in zip(patches1, patches2):
            mask = self.predict(p1, p2)
            mask_patches.append(Image.fromarray((mask * 255).astype(np.uint8)))
        
        mask_full = stitch_patches(mask_patches, coords, full_size, patch_size, overlap)
        return mask_full