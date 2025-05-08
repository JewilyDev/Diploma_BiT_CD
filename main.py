from fastapi import FastAPI, File, UploadFile, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
from typing import List
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
    model_path="C:\\Users\\monch\\OneDrive\\Документы\\diploma_bit_cd\\Diploma_BiT_CD\\BIT_CD\\checkpoints\\BIT_LEVIR\\best_ckpt.pt",
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
        change_mask = change_model.predict(before_img, after_img, patch_size=256, overlap=0)
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

@app.post("/detect-timeline-changes")
async def detect_timeline_changes(files: list[UploadFile] = File(..., description="Images to process")):
    try:
        # Print debug information
        print("Processing timeline change detection request")
        print(f"Received {len(files)} files")
        
        # Validate minimum number of files
        if len(files) < 2:
            return JSONResponse(
                status_code=400,
                content={"detail": f"At least two images are required for timeline analysis. Found {len(files)} files."}
            )
        
        # List all files
        for i, file in enumerate(files):
            print(f"File {i+1}: {file.filename}")
        
        # Read all images
        images = []
        for i, file in enumerate(files):
            try:
                content = await file.read()
                image = Image.open(io.BytesIO(content)).convert("RGB")
                images.append(image)
                print(f"Successfully processed image {i+1}: {file.filename}, size: {image.size}")
            except Exception as e:
                print(f"Error processing image {file.filename}: {str(e)}")
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Error processing image {file.filename}: {str(e)}"}
                )
        
        # Store original size for later use
        original_size = images[0].size
        print(f"Using original size: {original_size}")
        
        # Process each consecutive pair of images
        comparisons = []
        summary_mask = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
        
        for i in range(len(images) - 1):
            print(f"Processing image pair {i+1}/{len(images)-1}")
            before_img = images[i]
            after_img = images[i + 1]
            
            # Use patch-wise inference with smaller patch size and more overlap for better results
            print(f"Running patchwise prediction with patch_size=128, overlap=32")
            change_mask = change_model.patchwise_predict(before_img, after_img, patch_size=128, overlap=32)
            
            # If mask is not the same size as the original, upsample
            if change_mask.shape != (original_size[1], original_size[0]):
                print(f"Resizing mask from {change_mask.shape} to {(original_size[1], original_size[0])}")
                change_mask = sk_resize(change_mask, (original_size[1], original_size[0]), order=1, mode='reflect', anti_aliasing=True)
            
            # Apply some enhancement to the mask to make changes more visible
            # Contrast enhancement
            p_low, p_high = np.percentile(change_mask, (1, 99))
            change_mask_enhanced = np.clip((change_mask - p_low) / (p_high - p_low + 1e-6), 0, 1)
            
            # Apply adaptive thresholding
            mean_val = np.mean(change_mask_enhanced)
            std_val = np.std(change_mask_enhanced)
            # Lower the threshold to detect more changes
            adaptive_threshold = max(0.3, mean_val - 0.5 * std_val)
            print(f"Using adaptive threshold: {adaptive_threshold}")
            
            # Create binary mask with adaptive threshold
            binary_mask = (change_mask_enhanced > adaptive_threshold).astype(np.uint8)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Calculate change statistics
            total_pixels = binary_mask.size
            changed_pixels = np.sum(binary_mask)
            change_percentage = (changed_pixels / total_pixels) * 100
            print(f"Change percentage: {change_percentage:.2f}%")
            
            # Create visualization with improved mask
            before_array = np.array(before_img)
            mask_rgb = np.zeros_like(before_array)
            # Use more vibrant color for changes
            mask_rgb[binary_mask == 1] = [255, 50, 50]  # Brighter red for changed areas
            
            # Create overlay with more emphasis on changes
            overlay = before_array.copy()
            # Increase change visibility by using a higher weight for the mask
            overlay = overlay * 0.6 + mask_rgb * 0.4
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            # Add contours to highlight changes
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            
            # Convert to PIL Image
            mask_img = Image.fromarray(overlay)
            
            # Save visualization
            buffered = io.BytesIO()
            mask_img.save(buffered, format="PNG")
            mask_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Add to comparisons list
            comparisons.append({
                "change_mask": mask_base64,
                "change_percentage": float(change_percentage),
                "change_stats": {
                    "total_pixels": int(total_pixels),
                    "changed_pixels": int(changed_pixels)
                }
            })
            
            # Add to summary mask (using the enhanced binary mask)
            if i == 0:
                summary_mask = binary_mask.astype(np.float32)
            else:
                summary_mask = np.maximum(summary_mask, binary_mask.astype(np.float32))
        
        print("Creating summary visualization")
        # Create summary visualization
        first_img_array = np.array(images[0])
        summary_rgb = np.zeros_like(first_img_array)
        summary_rgb[summary_mask > 0.5] = [255, 50, 50]  # Brighter red for changed areas
        
        # Create overlay with more emphasis on changes
        summary_overlay = first_img_array.copy()
        summary_overlay = summary_overlay * 0.6 + summary_rgb * 0.4
        summary_overlay = np.clip(summary_overlay, 0, 255).astype(np.uint8)
        
        # Add contours to highlight all changes
        summary_binary = summary_mask.astype(np.uint8)
        contours, _ = cv2.findContours(summary_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(summary_overlay, contours, -1, (0, 255, 0), 2)
        
        # Convert to PIL Image
        summary_img = Image.fromarray(summary_overlay)
        
        # Save visualization
        buffered = io.BytesIO()
        summary_img.save(buffered, format="PNG")
        summary_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        print("Returning results")
        return {
            "total_images": len(images),
            "total_comparisons": len(comparisons),
            "comparisons": comparisons,
            "summary_mask": summary_base64
        }
        
    except Exception as e:
        print(f"Error in detect-timeline-changes: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing images: {str(e)}"}
        ) 