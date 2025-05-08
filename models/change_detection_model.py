import torch
from argparse import Namespace
from torchvision import transforms
from PIL import Image
import numpy as np
from BIT_CD.models.networks import define_G

def split_into_patches(img: Image.Image, patch_size=256, overlap=0):
    w, h = img.size
    patches = []
    coords = []
    step = patch_size - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            box = (x, y, min(x+patch_size, w), min(y+patch_size, h))
            patch = img.crop(box)
            # Pad if needed
            if patch.size != (patch_size, patch_size):
                new_patch = Image.new("RGB", (patch_size, patch_size))
                new_patch.paste(patch, (0, 0))
                patch = new_patch
            patches.append(patch)
            coords.append((x, y))
    return patches, coords, (w, h)

def stitch_patches(patches, coords, full_size, patch_size=256, overlap=0):
    w, h = full_size
    mask = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    step = patch_size - overlap
    for patch, (x, y) in zip(patches, coords):
        arr = np.array(patch)
        arr = arr[:patch_size, :patch_size]
        arr = arr.squeeze()
        x_end = min(x+patch_size, w)
        y_end = min(y+patch_size, h)
        mask[y:y_end, x:x_end] += arr[:y_end-y, :x_end-x]
        count[y:y_end, x:x_end] += 1
    mask = mask / np.maximum(count, 1)
    return mask

class ChangeDetectionModel:
    def __init__(self, model_path: str,
                 net_G: str = 'base_transformer_pos_s4_dd8_dedim8',
                 device: str = 'cpu'):
        self.device = device
        # Prepare BIT-CD args
        args = Namespace(net_G=net_G, gpu_ids=[] if device == 'cpu' else [0])
        # Instantiate the BIT-CD network
        self.model = define_G(args, init_type='normal', init_gain=0.02, gpu_ids=args.gpu_ids)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_G_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        # Preprocessing for bitemporal inputs
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, img1: Image.Image, img2: Image.Image) -> np.ndarray:
        # Apply transforms
        t1 = self.transform(img1)
        t2 = self.transform(img2)
        x1 = t1.unsqueeze(0).to(self.device)
        x2 = t2.unsqueeze(0).to(self.device)
        # Forward pass
        with torch.no_grad():
            out = self.model(x1, x2)  # shape [1, 2, H, W]
            pred = torch.argmax(out, dim=1)  # shape [1, H, W]
        return pred.squeeze().cpu().numpy()

    def patchwise_predict(self, img1: Image.Image, img2: Image.Image, patch_size=256, overlap=0) -> np.ndarray:
        # Split both images into patches
        patches1, coords, full_size = split_into_patches(img1, patch_size, overlap)
        patches2, _, _ = split_into_patches(img2, patch_size, overlap)
        mask_patches = []
        for p1, p2 in zip(patches1, patches2):
            mask = self.predict(p1, p2)
            mask_patches.append(Image.fromarray((mask*255).astype(np.uint8)))
        # Stitch back
        mask_full = stitch_patches(mask_patches, coords, full_size, patch_size, overlap)
        mask_full = mask_full / 255.0  # Normalize to [0,1]
        return mask_full 