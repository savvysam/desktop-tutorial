# PyInstaller spec file — run on macOS (Intel or Apple Silicon) to produce a .app bundle
# Usage: pyinstaller ColorSeasonApp_mac.spec
#
# Produces:  dist/ColorSeasonApp.app
# Supports:  universal2 (runs natively on both Intel x86_64 and Apple Silicon arm64)
#
# To notarise for distribution outside the App Store, see BUILD_MAC.md.

import sys
from pathlib import Path

block_cipher = None

# ---------------------------------------------------------------------------
# Optional: bundle the ML checkpoints so the app ships with real inference.
# If the files don't exist the build still succeeds — the app falls back to
# colour-theory (SIVC) mode at runtime.
# ---------------------------------------------------------------------------
_checkpoint_datas = []
for _ckpt in ('checkpoints/best_model.pth', 'checkpoints/svm_head.pkl'):
    if Path(_ckpt).exists():
        _checkpoint_datas.append((_ckpt, 'checkpoints'))

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('templates',          'templates'),
        ('static',             'static'),
        ('reference_palettes', 'reference_palettes'),
        ('model_config.json',  '.'),
    ] + _checkpoint_datas,
    hiddenimports=[
        # Flask / web stack
        'flask',
        'jinja2',
        'jinja2.ext',
        'werkzeug',
        'werkzeug.serving',
        'werkzeug.routing',
        'itsdangerous',
        'click',
        # Image processing
        'cv2',
        'PIL',
        'PIL.Image',
        'numpy',
        # Machine-learning (SIVC path)
        'sklearn',
        'sklearn.cluster',
        'sklearn.cluster._kmeans',
        'sklearn.neighbors._partition_nodes',
        'sklearn.utils._cython_blas',
        'sklearn.utils._weight_vector',
        'skimage',
        'skimage.color',
        'skimage.color.colorconv',
        # Machine-learning (SVM / deep path — optional)
        'joblib',
        'scipy',
        'scipy.special._ufuncs_cxx',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        # timm / torch are large; include only if you bundle the ML model
        # 'torch', 'timm', 'torchvision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy training-only packages from the app bundle
        'neptune',
        'optuna',
        'matplotlib',
        'IPython',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ColorSeasonApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX is unreliable on macOS arm64 — leave off
    console=False,      # No terminal window; app opens browser silently
    argv_emulation=True,  # Needed for proper macOS .app behaviour
    target_arch='universal2',  # Runs on both Intel and Apple Silicon
    codesign_identity=None,    # Set to 'Developer ID Application: …' for notarisation
    entitlements_file=None,
    icon=None,          # Replace with 'installer/icon.icns' if you create one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ColorSeasonApp',
)

app = BUNDLE(
    coll,
    name='ColorSeasonApp.app',
    icon=None,          # Replace with 'installer/icon.icns' if you create one
    bundle_identifier='com.colorseasonapp.app',
    info_plist={
        'CFBundleName':               'ColorSeasonApp',
        'CFBundleDisplayName':        'Seasonal Colour Analysis',
        'CFBundleVersion':            '1.0.0',
        'CFBundleShortVersionString': '1.0',
        'NSHighResolutionCapable':    True,
        'NSCameraUsageDescription':   'Used to analyse facial colours for season classification.',
        'LSMinimumSystemVersion':     '12.0',  # macOS Monterey+
        'LSUIElement':                False,   # Show in Dock
        'CFBundleDocumentTypes':      [],
    },
)
