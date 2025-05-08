from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image
import io
import torch
import os
import numpy as np
import base64
from skimage.transform import resize as sk_resize
import cv2
from scipy.ndimage import label, find_objects

from models.classification_model import ClassificationModel
from models.change_detection_model import ChangeDetectionModel
from models.cnn_difference_model import CNNDifferenceModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize models (update paths, num_classes, and devices as needed)
classification_model = ClassificationModel(
    model_path="models/classifier.pth",
    class_names=[
        "Urban Area", "Forest", "Water Body", "Agricultural Land", 
        "Barren Land", "Grassland", "Mountain", "Desert", "Coastal Area",
        "Snow/Ice", "Wetland", "Industrial Area"
    ],
    device="cuda" if torch.cuda.is_available() else "cpu"
)
change_model = ChangeDetectionModel(
    model_path="BIT_CD/checkpoints/BIT_LEVIR/best_ckpt.pt",
    net_G="base_transformer_pos_s4_dd8_dedim8",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
cnn_diff_model = CNNDifferenceModel(device="cuda" if torch.cuda.is_available() else "cpu")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("classify.html", {"request": request})

@app.get("/classify")
async def classify_page(request: Request):
    return templates.TemplateResponse("classify.html", {"request": request})

@app.get("/change-detection")
async def change_detection_page(request: Request):
    return templates.TemplateResponse("change_detection.html", {"request": request})

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    result = classification_model.predict(image)
    return JSONResponse(content=result)

@app.post("/detect-change")
async def detect_change(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    method: str = Query("bitcd", description="Change detection method: 'bitcd', 'cnn', 'absdiff', 'absdiff_median', or 'ssim'")
):
    # Read and convert images
    before_content = await before.read()
    after_content = await after.read()
    before_img = Image.open(io.BytesIO(before_content)).convert("RGB")
    after_img = Image.open(io.BytesIO(after_content)).convert("RGB")
    
    # Store original size for later use
    original_size = before_img.size
    
    # For BiT-CD, use patch-wise inference; for others, keep original size
    if method == "bitcd":
        # Use patch-wise inference for BIT-CD
        change_mask = change_model.patchwise_predict(before_img, after_img, patch_size=256, overlap=0)
    elif method in ["cnn", "absdiff", "absdiff_median", "ssim"]:
        change_mask = cnn_diff_model.predict(before_img, after_img, method=method)
        # Normalize mask to [0, 1] for non-cnn methods
        if method != "cnn":
            change_mask = (change_mask - np.min(change_mask)) / (np.max(change_mask) - np.min(change_mask) + 1e-8)
    else:
        return JSONResponse(status_code=400, content={"detail": f"Unknown method: {method}"})
    
    # If mask is not the same size as the original, upsample for visualization/statistics
    if change_mask.shape != (original_size[1], original_size[0]):
        change_mask = sk_resize(change_mask, (original_size[1], original_size[0]), order=1, mode='reflect', anti_aliasing=True)
    
    # Try different thresholds for better change detection
    thresholds = [0.3, 0.5, 0.7]
    threshold_results = {}
    
    for threshold in thresholds:
        binary_mask = (change_mask > threshold).astype(np.uint8)
        total_pixels = binary_mask.size
        changed_pixels = np.sum(binary_mask)
        change_percentage = (changed_pixels / total_pixels) * 100
        threshold_results[f"threshold_{threshold}"] = float(change_percentage)
    
    # Use adaptive thresholding for better results
    mean_change = np.mean(change_mask)
    std_change = np.std(change_mask)
    adaptive_threshold = mean_change + std_change
    binary_mask = (change_mask > adaptive_threshold).astype(np.uint8)
    
    # Create masked images for classification
    before_array = np.array(before_img)
    after_array = np.array(after_img)
    
    # Resize binary mask to match image dimensions
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    binary_mask = np.repeat(binary_mask, 3, axis=-1)
    
    # Get changed regions from both images
    changed_regions_before = before_array * binary_mask
    changed_regions_after = after_array * binary_mask
    
    # Convert back to PIL Images for classification
    changed_before = Image.fromarray(changed_regions_before)
    changed_after = Image.fromarray(changed_regions_after)
    
    # Classify changed regions
    before_classification = classification_model.predict(changed_before)
    after_classification = classification_model.predict(changed_after)
    
    # Calculate change statistics
    total_pixels = binary_mask.size // 3  # Divide by 3 for RGB channels
    changed_pixels = np.sum(binary_mask[:,:,0])  # Use one channel for counting
    change_percentage = (changed_pixels / total_pixels) * 100
    
    # Create enhanced visualization
    # Convert change mask to RGB for better visualization
    mask_rgb = np.zeros_like(before_array)
    mask_rgb[binary_mask[:,:,0] == 1] = [255, 0, 0]  # Red for changed areas
    
    # Create overlay
    overlay = before_array.copy()
    overlay = overlay * 0.7 + mask_rgb * 0.3  # Blend with original image
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image and resize back to original size
    mask_img = Image.fromarray(overlay)
    # mask_img is already at original size
    
    # Save visualization
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "change_mask": mask_base64,
        "change_percentage": float(change_percentage),
        "classification": {
            "before": before_classification,
            "after": after_classification
        },
        "threshold_results": threshold_results,
        "change_stats": {
            "total_pixels": int(total_pixels),
            "changed_pixels": int(changed_pixels),
            "mean_change": float(mean_change),
            "std_change": float(std_change)
        },
        "method": method
    } 