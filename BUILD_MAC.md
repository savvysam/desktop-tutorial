# Building the macOS .app Bundle

Produces `dist/ColorSeasonApp.app` — a self-contained double-clickable app that
runs on both **Intel (x86_64)** and **Apple Silicon (arm64)** Macs via a
universal2 binary.  No Python installation required on the end-user's machine.

---

## Prerequisites (on a macOS machine)

| Tool | Version | Where |
|------|---------|-------|
| Python | 3.11 (recommended) | https://www.python.org/downloads/ |
| Xcode Command Line Tools | any recent | `xcode-select --install` |
| PyInstaller | ≥ 6.0 | installed below |

> **Apple Silicon note** — build on the machine architecture you want to target
> first.  The `universal2` target in the spec merges both slices, but some
> native extensions (OpenCV, NumPy) must be available as universal2 wheels.
> If a wheel is Intel-only, the build still works on Intel; run `arch -x86_64`
> before the build command if needed.

---

## Step 1 — Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## Step 2 — Install runtime dependencies

```bash
pip install --upgrade pip
pip install \
    flask \
    Pillow \
    opencv-python-headless \
    numpy \
    scikit-learn \
    scikit-image \
    pyinstaller
```

### Optional — bundle the ML model (SVM + EfficientNet backbone)

Only needed if you want the `.app` to ship with real ML inference (34 % accuracy
vs 8 % random baseline).  This adds ~400 MB to the bundle.

```bash
pip install torch torchvision timm joblib
```

Then open `ColorSeasonApp_mac.spec` and un-comment this line in `hiddenimports`:

```python
# 'torch', 'timm', 'torchvision',
```

---

## Step 3 — Copy assets from Replit

Download (or `git clone`) the project from Replit.  Make sure these files are
present in the project root before building:

```
app.py
launcher.py
color_analysis.py
model_config.json
templates/
static/
reference_palettes/
checkpoints/best_model.pth   ← optional, enables ML inference
checkpoints/svm_head.pkl     ← optional, enables ML inference
```

---

## Step 4 — Build the .app with PyInstaller

```bash
pyinstaller ColorSeasonApp_mac.spec
```

The first run takes 2–5 minutes.  Output:

```
dist/
└── ColorSeasonApp.app      ← double-click to run
build/                      ← intermediate files (safe to delete)
```

---

## Step 5 — Test it

```bash
open dist/ColorSeasonApp.app
```

The app starts a local Flask server on port 5000 and opens your default browser
at `http://localhost:5000` automatically.  No internet connection required for
colour-theory (SIVC) mode.

---

## Step 6 — Distribute the app

### Option A — ZIP for direct sharing (no notarisation)

```bash
cd dist
zip -r ColorSeasonApp_mac.zip ColorSeasonApp.app
```

Recipients must right-click → Open the first time to bypass Gatekeeper (macOS
will warn that it is from an unidentified developer).

### Option B — Notarised DMG (recommended for wider distribution)

Requires a paid **Apple Developer Program** account ($99/year).

1. **Sign the app**

   ```bash
   codesign --deep --force --verify --verbose \
       --sign "Developer ID Application: Your Name (TEAMID)" \
       --entitlements entitlements.plist \
       dist/ColorSeasonApp.app
   ```

2. **Create a DMG**

   ```bash
   hdiutil create -volname "ColorSeasonApp" -srcfolder dist/ColorSeasonApp.app \
       -ov -format UDZO dist/ColorSeasonApp.dmg
   ```

3. **Notarise with Apple**

   ```bash
   xcrun notarytool submit dist/ColorSeasonApp.dmg \
       --apple-id your@apple.id \
       --team-id TEAMID \
       --password app-specific-password \
       --wait
   ```

4. **Staple the notarisation ticket**

   ```bash
   xcrun stapler staple dist/ColorSeasonApp.dmg
   ```

After stapling, the DMG opens without any Gatekeeper warning on any Mac.

---

## Adding an app icon

1. Create a 1024×1024 PNG of your icon.
2. Use **Image2icon** (free, App Store) or the command-line tool below to
   convert it to `.icns`:

   ```bash
   mkdir icon.iconset
   sips -z 16 16     icon.png --out icon.iconset/icon_16x16.png
   sips -z 32 32     icon.png --out icon.iconset/icon_16x16@2x.png
   sips -z 32 32     icon.png --out icon.iconset/icon_32x32.png
   sips -z 64 64     icon.png --out icon.iconset/icon_32x32@2x.png
   sips -z 128 128   icon.png --out icon.iconset/icon_128x128.png
   sips -z 256 256   icon.png --out icon.iconset/icon_128x128@2x.png
   sips -z 256 256   icon.png --out icon.iconset/icon_256x256.png
   sips -z 512 512   icon.png --out icon.iconset/icon_256x256@2x.png
   sips -z 512 512   icon.png --out icon.iconset/icon_512x512.png
   sips -z 1024 1024 icon.png --out icon.iconset/icon_512x512@2x.png
   iconutil -c icns icon.iconset -o installer/icon.icns
   ```

3. In `ColorSeasonApp_mac.spec`, replace both `icon=None` lines with:

   ```python
   icon='installer/icon.icns',
   ```

4. Rebuild: `pyinstaller ColorSeasonApp_mac.spec`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` at launch | Add the missing module to `hiddenimports` in the spec and rebuild |
| Blank browser page | Check that `templates/` and `static/` appear inside `ColorSeasonApp.app/Contents/Resources` |
| OpenCV cascade not found | Add `('cv2/data', 'cv2/data')` to `datas` in the spec |
| App killed immediately on Apple Silicon | Remove `target_arch='universal2'` and build native arm64 instead |
| Port 5000 already in use | Another process holds port 5000; quit it or change the port in `launcher.py` |
| Gatekeeper blocks the app | Right-click → Open, or notarise the build (Option B above) |
