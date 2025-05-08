# Satellite Image Change Detection and Classification

## Overview
This application provides a web interface and API for:
- **Image Classification**: Classify single satellite images into 12 common land‐use categories using a ResNet-18 based model.
- **Change Detection**: Detect and quantify changes between two satellite images using multiple methods (BiT-CD, CNN difference, absolute difference, median absolute difference, SSIM).
- **Integrated Analysis**: Automatically classify regions of change before and after to understand the nature of detected changes.

## Features
- FastAPI-powered backend with interactive web UI (Jinja2 templates).
- Multiple change detection methods with adaptive thresholding and multiple threshold analysis.
- Base64-encoded visualization of detected changes overlayed on original images.
- JSON-based API for easy integration in other services or scripts.

## Tech Stack
- Python 3.8+
- FastAPI, Uvicorn
- PyTorch (torch, torchvision)
- BiT-CD (Leveraging BIT_CD checkpoints)
- Jinja2, Python-Multipart, Pillow, OpenCV, NumPy, SciPy

## Requirements
All dependencies are listed in `requirements.txt`. Key versions:
```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
jinja2==3.1.2
pillow==10.1.0
torch==2.1.1
torchvision==0.16.1
einops>=0.3.0
numpy
opencv-python
```

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# (Optional) Create a virtual environment
python3 -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration
- **Model files**:
  - Classification model: `models/classifier.pth`
  - Change detection checkpoint: `BIT_CD/checkpoints/BIT_LEVIR/best_ckpt.pt`
- Ensure these paths exist or update `main.py` initialization arguments.
- GPU usage is automatic if CUDA is available; otherwise CPU is used.

## Running the Application
```bash
# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- Web UI: http://localhost:8000/
  - Classification page: `/classify`
  - Change detection page: `/change-detection`

## API Endpoints

### 1) Image Classification
- **Endpoint**: `POST /classify`
- **Parameters**: Multipart form field named `file` (image upload)
- **Response**: JSON with predicted class probabilities and labels.

**Example**:
```bash
curl -X POST "http://localhost:8000/classify" \
  -F file=@/path/to/image.jpg
```

### 2) Change Detection
- **Endpoint**: `POST /detect-change`
- **Parameters**:
  - `before`: first image file
  - `after`: second image file
  - Optional query `method` (default `bitcd`): one of `bitcd`, `cnn`, `absdiff`, `absdiff_median`, `ssim`
- **Response**: JSON containing:
  - `change_mask`: base64 PNG overlay of detected changes
  - `change_percentage`: overall percent of changed pixels
  - `classification`: object with `before` and `after` classification results
  - `threshold_results`: percentages at thresholds [0.3, 0.5, 0.7]
  - `change_stats`: raw stats (total_pixels, changed_pixels, mean_change, std_change)
  - `method`: method used

**Example**:
```bash
curl -X POST "http://localhost:8000/detect-change?method=ssim" \
  -F before=@/path/to/old.jpg \
  -F after=@/path/to/new.jpg
```

## Directory Structure
```
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── BIT_CD/                 # BiT-CD code & checkpoints
├── models/                 # Classification & change detection model files
├── templates/              # Jinja2 HTML templates
│   ├── index.html          # Home template (redirects to classify)
│   ├── classify.html       # Image classification UI
│   └── change_detection.html # Change detection UI
├── test.html               # Static test page for endpoint testing
└── utils/                  # Helper modules
```

## Testing
1. Open a browser and navigate to the UI pages.
2. Use Postman or `curl` examples above to test API calls.

## Contributing
Contributions, issues, and feature requests are welcome!

## License
This project is licensed under the MIT License. 