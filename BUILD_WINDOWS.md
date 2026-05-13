# Building the Windows x64 .exe Installer

## Prerequisites (on a Windows x64 machine)

1. **Python 3.11 x64** — https://www.python.org/downloads/
2. **Inno Setup 6** — https://jrsoftware.org/isdl.php

## Step 1: Install Dependencies

```bat
pip install flask Pillow opencv-python numpy gunicorn pyinstaller
```

To also enable the ML model in the .exe, add:

```bat
pip install torch torchvision timm
```

## Step 2: Build the .exe with PyInstaller

```bat
pyinstaller ColorSeasonApp.spec
```

This creates `dist\ColorSeasonApp\` with the standalone app.

## Step 3: Test Before Packaging

```bat
dist\ColorSeasonApp\ColorSeasonApp.exe
```

The app opens your browser at http://localhost:5000 automatically.

## Step 4: Build the Installer with Inno Setup

1. Open **Inno Setup Compiler**
2. Open `installer\setup.iss`
3. Click **Build → Compile**

Output: `installer_output\ColorSeasonApp_Setup_x64.exe`

---

## Training the ML Model (Before Building)

The app ships in Demo Mode. To activate real ML inference:

1. Install training dependencies:
   ```bat
   pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm neptune-client optuna plotly kaleido
   ```

2. Prepare the [Deep Armocromia Dataset](https://github.com/lorenzo-stacchio/Deep-Armocromia-Dataset) in `dataset/images/train` and `dataset/images/test`

3. Set your Neptune API token:
   ```bat
   set NEPTUNE_API_TOKEN=your_token_here
   ```

4. Run training:
   ```bat
   python -m SeasonalColourClassification.main --model tf_efficientnetv2_l.in21k_ft_in1k --epochs 250
   ```

5. Copy the checkpoint into `checkpoints/best_model.pth` (or update `model_config.json` with the actual path)

6. Now rebuild the .exe — it will bundle the checkpoint and run real inference.
