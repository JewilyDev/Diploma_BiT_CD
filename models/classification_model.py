import torch
import os
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torchvision.models import ResNet18_Weights

class ClassificationModel:
    def __init__(self, model_path: str, class_names: list, device: str = "cpu"):
        self.class_names = class_names
        num_classes = len(class_names)
        self.device = device
        
        # Use new weights parameter format
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
        
        # Load local weights if available
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Successfully loaded classifier weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load classifier weights: {e}")
                print("Using ImageNet pretrained weights with custom classification layer")
        else:
            print(f"Warning: classifier weights not found at {model_path}")
            print("Using ImageNet pretrained weights with custom classification layer")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image) -> dict:
        """
        Predict the class of the input image.
        Returns a dict with 'predicted' (str) and 'probabilities' (dict of class->float).
        Handles all-black or empty images gracefully.
        """
        # Check if the image is all black or empty
        np_img = np.array(image)
        if np_img.sum() == 0:
            return {
                "predicted": "No significant change",
                "probabilities": {name: 0.0 for name in self.class_names}
            }
            
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_name = self.class_names[pred_idx]
            
        # Map probabilities to class names
        prob_dict = {name: float(probs[idx]) for idx, name in enumerate(self.class_names)}
        return {
            "predicted": pred_name,
            "probabilities": prob_dict
        } 