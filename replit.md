# Seasonal Colour Analysis App

## Overview

A personal colour analysis web app implementing the Italian **armocromia** 12-class seasonal classification system. Converted from a SwiftUI iOS app and integrating two GitHub ML repositories:

1. **[SeasonalColourClassification](https://github.com/JuliusPinsker/SeasonalColourClassification)** — timm-based PyTorch 12-class training framework (optional deep ML path)
2. **[raden-dosel/seasonal-color-classification](https://github.com/raden-dosel/seasonal-color-classification)** — deterministic SIVC colour-theory algorithm (default analysis engine)

Upload a face photo to get a real colour season result. The app automatically selects the best available analysis engine:
- **SIVC Colour Theory** (default, no model weights needed): extracts skin/hair/eye/lips colours via OpenCV, computes Subtone/Intensity/Value/Contrast metrics, matches against season reference vectors using Hamming distance
- **Deep ML Model** (optional): timm checkpoint loaded from `checkpoints/best_model.pth`

## Architecture

- **Backend**: Python 3.11 + Flask (runs on port 5000, host 0.0.0.0)
- **Frontend**: Vanilla HTML/CSS/JS served by Flask templates
- **Image Processing**: OpenCV (face detection/cropping) + Pillow + NumPy
- **ML (optional)**: PyTorch + timm — loaded at startup if checkpoint exists
- **Training**: `SeasonalColourClassification/` module (timm-based, Neptune logging, Optuna HPO)

## 12-Class System

| Season | Subtypes |
|--------|----------|
| Primavera (Spring) | Bright, Light, Warm |
| Estate (Summer) | Cool, Light, Soft |
| Autunno (Autumn) | Deep, Soft, Warm |
| Inverno (Winter) | Bright, Cool, Deep |

## Third-party dataset — Deep Armocromia (ECCV 2024)

Real labeled face images for all 12 armocromia sub-types, hosted on Google Drive:
- **Source**: https://github.com/lorenzo-stacchio/Deep-Armocromia
- **Drive folder**: `1QuFJqNxhbLVME5UyUjDGYHQ1ntAqgniT`
- **Access**: Fill the [request form](https://forms.gle/icac2opCYqF79RyE9) to receive the ZIP password
- **4,920 images** across 12 classes (train + test splits, `annotations.csv` included)
- ZIPs are encrypted with traditional PKZIP password encryption

### Dataset Download Flow

1. Google Drive is connected via Replit integration (OAuth already authorised)
2. `/dataset` web page provides a 3-step UI: get password → download subset → retrain
3. `download_dataset.py` uses HTTP Range requests + ZIP central-directory parsing to download
   only the required subset without pulling the full 1.6 GB ZIP
4. Images saved to `dataset/images/{partition}/{season}/{subtype}/` — nested structure required by `SeasonalColorDataset`
5. PKZIP decryption is implemented natively in Python (`_PkDecrypter` class)
6. Range requests include retry logic (4 attempts with exponential back-off + token refresh)
7. After download, clicking "Retrain Model" re-runs `train.py` and reloads the checkpoint

```bash
# Command-line usage:
python3 download_dataset.py --password YOUR_ZIP_PASSWORD --train 150 --test 50
```

### Current trained model state (real data)

- **1,800 training images** (150 per class × 12 classes) from Deep Armocromia ECCV 2024
- **600 test images** (50 per class × 12 classes)
- Pipeline: EfficientNet_b0 backbone (frozen) → StandardScaler → PCA 1280→256 → LogisticRegression
- Best C found via 5-fold cross-validation: C=0.01
- **Train accuracy: 62.4%** | **Test accuracy: 22.2%** (vs 8.3% random chance on 12 classes)
- PCA explained variance: 83.6%; LR and timm model verified at exact same accuracy
- Scaler + PCA + LR weights are baked into the model's single Linear(1280→12) classifier layer

## Project Structure

```
app.py                          Flask backend, inference + dataset management routes
color_analysis.py               SIVC colour-theory engine (OpenCV-based, no weights)
download_dataset.py             Drive→ZIP→image downloader (Range requests + PKZIP decrypt)
annotations.csv                 Deep Armocromia label file (4,920 images, 12 classes)
model_config.json               Deep ML model name + checkpoint path config
requirements.txt                Python dependencies
templates/index.html            Upload + results UI
templates/dataset.html          Dataset Manager page (download + retrain UI)
static/css/style.css            Styling
static/js/app.js                Frontend logic (shows SIVC metrics + dominant colours)
reference_palettes/             Season reference data from raden-dosel repo
  spring.csv / summer.csv / autumn.csv / winter.csv
SeasonalColourClassification/   Full ML training pipeline (from JuliusPinsker GitHub)
checkpoints/                    Drop best_model.pth here for deep ML inference
dataset/images/                 Training data (train/ and test/ subdirs per class)
ColorSeasonApp/                 Original SwiftUI iOS source (reference)
ColorSeasonApp.spec             PyInstaller spec for Windows .exe build
installer/setup.iss             Inno Setup script for Windows installer
BUILD_WINDOWS.md                Windows build instructions
```

## Running in Replit

The workflow `Start application` runs `python3 app.py` on port 5000.

## Enabling Real ML Inference

1. Train a model: `python -m SeasonalColourClassification.main --model tf_efficientnetv2_l.in21k_ft_in1k --epochs 250`
2. Drop the generated `best_model.pth` into `checkpoints/`
3. Update `model_config.json` with the correct `model_name` and `img_size`
4. Restart the app — the badge will change to "ML Model Active"

## Building the Windows x64 .exe Installer

See `BUILD_WINDOWS.md`. Requires Windows x64 + Python 3.11 + Inno Setup 6.

## SIVC Algorithm (colour_analysis.py)

The real analysis engine (no model download needed):

1. **Face detection** — OpenCV Haar cascade locates the face bounding box
2. **Region extraction** — estimates skin (HSV mask in cheek area), hair (above face), eyes (Haar + geometry), lips (lower face strip)
3. **KMeans dominant colours** — k=3 per region
4. **SIVC metrics**:
   - **S (Subtone)**: lips CIELab distance to peach vs purple → warm or cold
   - **I (Intensity)**: skin HSV saturation (threshold 0.422)
   - **V (Value)**: mean brightness of skin + hair + eyes (threshold 0.390)
   - **C (Contrast)**: |hair_V − eye_V| (threshold 0.200)
5. **Season matching**: soft product score against reference vectors (`spring=1111, summer=0010, autumn=1000, winter=0101`)
6. **Subtype scoring**: distributes confidence within each season to 3 subtypes using intensity + value + contrast

## Key Dependencies

| Package | Purpose |
|---------|---------|
| flask | Web server |
| opencv-python-headless | Face detection + region extraction |
| Pillow | Image I/O |
| scikit-learn | KMeans dominant colour extraction |
| scikit-image | CIELab colour distance (rgb2lab) |
| gunicorn | Production WSGI server |
| torch + timm | Deep ML inference (optional, install separately) |
