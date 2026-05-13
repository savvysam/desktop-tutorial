#!/usr/bin/env python3
"""
Standalone training script for the 12-class armocromia seasonal colour classifier.

Strategy (CPU-friendly, completes in ~3 minutes):
  1. Generate synthetic face-colour dataset (no real photos needed)
  2. Extract features from pretrained backbone via ModelFactory (model_name from
     model_config.json); training images are passed N_PASSES=8 times through
     ColorAugmentation transforms (RandomErasing + Gaussian noise) to produce
     richer, augmented feature vectors; test images use a single clean pass.
  3. Fit an SVM head (sklearn SVC, RBF kernel) on the extracted features
  4. Save checkpoints/best_model.pth (backbone state dict, num_classes=0) and
     checkpoints/svm_head.pkl (StandardScaler + PCA + SVC pipeline)
  5. Update model_config.json

Notes:
  - Bypasses Neptune/Optuna (not needed here; those are for the research training pipeline)
  - SVM cannot be baked into a single Linear layer, so inference uses a two-step path:
    backbone → feature vector → sklearn SVM pipeline → class probabilities
  - For production use with real face photos, retrain with a real labeled dataset
"""

import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

# ---------------------------------------------------------------------------
# 12-class colour palette definitions
# Each entry: (skin, hair, eye, lip)  — RGB tuples
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    # ---- Primavera (Spring) — warm, clear, light ----
    "primavera_bright": {"skin": (230,188,158), "hair": ( 80, 52, 28), "eye": ( 90, 68, 38), "lip": (215,100, 88)},
    "primavera_light":  {"skin": (245,220,200), "hair": (200,170,118), "eye": (115, 95, 62), "lip": (225,168,150)},
    "primavera_warm":   {"skin": (210,168,128), "hair": (148, 90, 40), "eye": (108, 78, 38), "lip": (212,128,100)},

    # ---- Estate (Summer) — cool, muted, soft ----
    "estate_cool":  {"skin": (222,200,198), "hair": (148,145,162), "eye": (108,118,145), "lip": (198,145,158)},
    "estate_light": {"skin": (248,235,232), "hair": (200,195,208), "eye": (148,158,178), "lip": (218,182,185)},
    "estate_soft":  {"skin": (208,190,185), "hair": (162,150,148), "eye": (122,122,132), "lip": (185,155,158)},

    # ---- Autunno (Autumn) — warm, muted, rich earth ----
    "autunno_deep": {"skin": (168,122, 88), "hair": ( 48, 32, 18), "eye": ( 62, 42, 22), "lip": (158, 72, 60)},
    "autunno_soft": {"skin": (192,158,128), "hair": (118, 88, 55), "eye": ( 98, 72, 48), "lip": (188,122,100)},
    "autunno_warm": {"skin": (205,162,115), "hair": (145, 88, 38), "eye": (108, 75, 32), "lip": (202,112, 80)},

    # ---- Inverno (Winter) — cool, clear, high contrast ----
    "inverno_bright": {"skin": (248,232,232), "hair": ( 22, 18, 24), "eye": ( 48,100,188), "lip": (202, 78,102)},
    "inverno_cool":   {"skin": (228,212,218), "hair": ( 48, 42, 65), "eye": ( 68, 90,122), "lip": (178, 78,112)},
    "inverno_deep":   {"skin": (138, 98, 82), "hair": ( 18, 14, 22), "eye": ( 42, 32, 55), "lip": (138, 52, 78)},
}

CLASS_ORDER = [
    "primavera_bright", "primavera_light", "primavera_warm",
    "estate_cool",      "estate_light",    "estate_soft",
    "autunno_deep",     "autunno_soft",    "autunno_warm",
    "inverno_bright",   "inverno_cool",    "inverno_deep",
]

N_PASSES_DEFAULT = 8


def rand_color(base_rgb, noise=22):
    r = int(np.clip(base_rgb[0] + random.randint(-noise, noise), 0, 255))
    g = int(np.clip(base_rgb[1] + random.randint(-noise, noise), 0, 255))
    b = int(np.clip(base_rgb[2] + random.randint(-noise, noise), 0, 255))
    return (r, g, b)


def make_face_image(class_id, size=224):
    """Generate a synthetic face image with season-appropriate colours."""
    cols = CLASS_COLORS[class_id]
    brightness = random.uniform(0.88, 1.08)

    def bc(key, noise=20):
        c = rand_color(cols[key], noise)
        return tuple(int(np.clip(v * brightness, 0, 255)) for v in c)

    skin = bc("skin", 18)
    hair = bc("hair", 15)
    eye  = bc("eye",  15)
    lip  = bc("lip",  18)

    bg_v = random.randint(175, 215)
    img  = Image.new("RGB", (size, size), (bg_v, bg_v, bg_v))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, int(size * 0.52)

    # Hair mass
    draw.ellipse([cx - int(size*0.38), cy - int(size*0.46),
                  cx + int(size*0.38), cy + int(size*0.46)], fill=hair)

    # Face oval
    draw.ellipse([cx - int(size*0.30), cy - int(size*0.34),
                  cx + int(size*0.30), cy + int(size*0.42)], fill=skin)

    # Eyes
    eye_r = int(size * 0.055)
    eye_offset_x = int(size * 0.11)
    eye_y = cy - int(size * 0.07)
    for ex in [cx - eye_offset_x, cx + eye_offset_x]:
        draw.ellipse([ex - eye_r, eye_y - int(eye_r*0.6),
                      ex + eye_r, eye_y + int(eye_r*0.6)], fill=(255, 255, 255))
        iris_r = int(eye_r * 0.72)
        draw.ellipse([ex - iris_r, eye_y - iris_r,
                      ex + iris_r, eye_y + iris_r], fill=eye)
        pup_r = int(iris_r * 0.45)
        draw.ellipse([ex - pup_r, eye_y - pup_r,
                      ex + pup_r, eye_y + pup_r], fill=(12, 10, 12))

    # Lips
    lip_y = cy + int(size * 0.17)
    lip_rx, lip_ry = int(size * 0.12), int(size * 0.05)
    draw.ellipse([cx - lip_rx, lip_y - lip_ry,
                  cx + lip_rx, lip_y + lip_ry], fill=lip)
    draw.polygon([(cx - int(size*0.05), lip_y - int(size*0.01)),
                  (cx, lip_y - lip_ry - int(size*0.015)),
                  (cx + int(size*0.05), lip_y - int(size*0.01))], fill=skin)

    # Nose hint
    nose_y = cy + int(size * 0.04)
    nose_r = int(size * 0.028)
    draw.ellipse([cx - nose_r, nose_y - nose_r,
                  cx + nose_r, nose_y + nose_r],
                 fill=tuple(max(0, v - 18) for v in skin))

    # Texture noise
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr + np.random.normal(0, 5, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    return img


def generate_dataset(root_dir, n_train=80, n_test=20):
    season_subtype_map = {
        "primavera": ["bright", "light", "warm"],
        "estate":    ["cool",   "light", "soft"],
        "autunno":   ["deep",   "soft",  "warm"],
        "inverno":   ["bright", "cool",  "deep"],
    }
    total = 0
    for split, n in [("train", n_train), ("test", n_test)]:
        split_dir = os.path.join(root_dir, split)
        for season, subtypes in season_subtype_map.items():
            for subtype in subtypes:
                class_id = f"{season}_{subtype}"
                out_dir = os.path.join(split_dir, season, subtype)
                os.makedirs(out_dir, exist_ok=True)
                for i in range(n):
                    img = make_face_image(class_id)
                    img.save(os.path.join(out_dir, f"{i:04d}.jpg"), quality=92)
                total += n
    print(f"Generated {total} images ({n_train} train + {n_test} test per class, 12 classes)")


# ---------------------------------------------------------------------------
# Training — feature extraction + SVM head
# ---------------------------------------------------------------------------

def train():
    import torch
    import timm
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import joblib

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from SeasonalColourClassification.data.data_loader import SeasonalColorDataset
    from SeasonalColourClassification.data.data_augmentation import ColorAugmentation
    from SeasonalColourClassification.models.base_model import ModelFactory

    # Read settings from model_config.json; fall back to safe defaults.
    ARCH    = "efficientnet_b0"
    N_PASSES = N_PASSES_DEFAULT
    if os.path.exists("model_config.json"):
        try:
            with open("model_config.json") as f:
                _cfg = json.load(f)
            ARCH     = _cfg.get("model_name", ARCH)
            raw_passes = int(_cfg.get("n_passes", N_PASSES))
            if raw_passes < 1:
                print(f"Warning: n_passes={raw_passes} is invalid; clamping to 1.")
                raw_passes = 1
            N_PASSES = raw_passes
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            print(f"Warning: could not read model_config.json ({exc}); using defaults.")

    IMG_SIZE      = 224
    CKPT_DIR      = "checkpoints"
    CKPT_PATH     = os.path.join(CKPT_DIR, "best_model.pth")
    SVM_HEAD_PATH = os.path.join(CKPT_DIR, "svm_head.pkl")
    DATA_ROOT     = "dataset/images"
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"Backbone: {ARCH}  |  Augmented passes (train): {N_PASSES}  |  Test passes: 1")

    # 1. Generate dataset if needed
    train_dir = os.path.join(DATA_ROOT, "train")
    test_dir  = os.path.join(DATA_ROOT, "test")
    n_train_files = sum(len(f) for _, _, f in os.walk(train_dir)) if os.path.exists(train_dir) else 0
    if n_train_files < 200:
        print("Generating synthetic face-colour dataset (80 train + 20 test per class)...")
        generate_dataset(DATA_ROOT, n_train=80, n_test=20)
    else:
        print(f"Dataset already present ({n_train_files} train images)")

    # 2. Build transforms using ColorAugmentation (includes RandomErasing + Gaussian noise)
    import cv2

    class FaceCropTransform:
        """PIL → PIL: crops to the largest detected face or falls back to a square centre-crop.

        Padding is asymmetric to preserve hair-colour signal (critical for armocromia):
          - top:    100% of face height (captures full head / hair above the face)
          - bottom:  20% of face height (chin area)
          - sides:   25% of face width each
        Cascade: equalizeHist pre-processing, scaleFactor=1.05, minNeighbors=3.
        """

        def __init__(self, label=""):
            self.label      = label
            self.face_count = 0
            self.crop_count = 0
            self._cascade   = None

        def _cascade_xml(self):
            if self._cascade is None:
                self._cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
            return self._cascade

        def __call__(self, pil_img):
            img_rgb = np.array(pil_img.convert("RGB"))
            gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gray    = cv2.equalizeHist(gray)
            faces   = self._cascade_xml().detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )
            w_img, h_img = pil_img.size
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                # Asymmetric padding: generous top to capture hair above the face
                pad_top  = int(h * 1.00)   # 100% above — includes full head + hair
                pad_bot  = int(h * 0.20)   # 20% below chin
                pad_side = int(w * 0.25)   # 25% each side
                x1 = max(0, x - pad_side)
                y1 = max(0, y - pad_top)
                x2 = min(w_img, x + w + pad_side)
                y2 = min(h_img, y + h + pad_bot)
                self.face_count += 1
                cropped = pil_img.crop((x1, y1, x2, y2))
                return cropped
            else:
                side = min(w_img, h_img)
                left = (w_img - side) // 2
                top  = (h_img - side) // 2
                self.crop_count += 1
                return pil_img.crop((left, top, left + side, top + side))

        def report(self):
            total = self.face_count + self.crop_count
            if total == 0:
                return
            pct = 100.0 * self.face_count / total
            tag = f"[{self.label}] " if self.label else ""
            print(f"  {tag}Face detected: {self.face_count}/{total} ({pct:.1f}%)  "
                  f"Centre-crop fallback: {self.crop_count}/{total} ({100-pct:.1f}%)")

    face_crop_train = FaceCropTransform(label="train")
    face_crop_test  = FaceCropTransform(label="test")

    color_aug_train = ColorAugmentation.get_train_transforms(img_size=IMG_SIZE)
    color_aug_test  = ColorAugmentation.get_test_transforms(img_size=IMG_SIZE)

    train_tf = transforms.Compose([face_crop_train, color_aug_train])
    test_tf  = transforms.Compose([face_crop_test,  color_aug_test])

    train_ds = SeasonalColorDataset(train_dir, transform=train_tf)
    test_ds  = SeasonalColorDataset(test_dir,  transform=test_tf)
    print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)

    # 3. Load pretrained backbone via ModelFactory (num_classes=0 → feature extractor)
    CACHE = os.path.join(CKPT_DIR, "_feat_cache.npz")

    def extract_augmented(backbone, loader_tr, loader_te, device, n_passes=N_PASSES):
        """Run N augmented passes over training set; single clean pass for test set."""
        n_train_imgs = len(loader_tr.dataset)
        print(f"Extracting features: {n_passes} augmented passes × {n_train_imgs} "
              f"training images = {n_passes * n_train_imgs} effective train vectors…")

        all_train_feats, all_train_lbls = [], []
        for pass_idx in range(n_passes):
            torch.manual_seed(pass_idx * 137 + 42)
            random.seed(pass_idx * 137 + 42)
            np.random.seed(pass_idx * 137 + 42)
            pass_feats, pass_lbls = [], []
            with torch.no_grad():
                for imgs, labels in loader_tr:
                    f = backbone(imgs.to(device))
                    pass_feats.append(f.cpu().numpy())
                    pass_lbls.append(labels.numpy())
            all_train_feats.append(np.vstack(pass_feats))
            all_train_lbls.append(np.hstack(pass_lbls))
            print(f"  Pass {pass_idx + 1}/{n_passes} done")

        X_train = np.vstack(all_train_feats)
        y_train = np.hstack(all_train_lbls)
        print(f"{len(y_train)} augmented train features from {n_train_imgs} images "
              f"({n_passes} passes)")

        print("Extracting clean test features (1 pass)…")
        test_feats, test_lbls = [], []
        with torch.no_grad():
            for imgs, labels in loader_te:
                f = backbone(imgs.to(device))
                test_feats.append(f.cpu().numpy())
                test_lbls.append(labels.numpy())
        X_test = np.vstack(test_feats)
        y_test = np.hstack(test_lbls)
        return X_train, y_train, X_test, y_test

    if os.path.exists(CACHE):
        os.remove(CACHE)
        print(f"Deleted stale feature cache ({CACHE}) — will re-extract.")

    print(f"\nLoading pretrained {ARCH} backbone via ModelFactory…")
    device = torch.device("cpu")

    # Feature backbone: num_classes=0 → timm returns raw feature vector
    backbone_model = ModelFactory.get_model(ARCH, num_classes=0)
    backbone_model = backbone_model.to(device).eval()

    print(f"Augmentation pipeline: RandomErasing (p=0.1) + Gaussian noise (p=0.2) "
          f"+ ColorJitter + RandomRotation + RandomHorizontalFlip")
    X_train, y_train, X_test, y_test = extract_augmented(
        backbone_model, train_loader, test_loader, device
    )
    np.savez(CACHE, Xtr=X_train, ytr=y_train, Xte=X_test, yte=y_test)

    print("Face-crop detection statistics:")
    face_crop_train.report()
    face_crop_test.report()

    print(f"Feature shape: {X_train.shape}  (train: {len(y_train)}, test: {len(y_test)})")

    # 4. Save backbone state dict (num_classes=0) as best_model.pth
    backbone_state = backbone_model.model.state_dict()
    del backbone_model

    # 5. Fit SVM head with PCA dimensionality reduction
    print("\nFitting SVM (RBF kernel) head on extracted features…")
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # PCA: reduce to min(256, n_samples-1) to avoid underdetermined regime
    n_components = min(256, X_train_s.shape[0] - 1, X_train_s.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p  = pca.transform(X_test_s)
    print(f"PCA: {X_train_s.shape[1]} → {n_components} dims  "
          f"(explained var: {pca.explained_variance_ratio_.sum():.3f})")

    # Grid-search best C and gamma using 5-fold cross-val on training set
    best_C, best_gamma, best_cv = 1.0, "scale", -1
    param_grid = [
        (C, gamma)
        for C in [0.1, 1.0, 10.0, 100.0]
        for gamma in ["scale", "auto", 0.001, 0.01]
    ]
    print(f"Grid-searching C × gamma ({len(param_grid)} combos, 5-fold CV)…")
    for C, gamma in param_grid:
        cv = cross_val_score(
            SVC(kernel="rbf", C=C, gamma=gamma, probability=True),
            X_train_p, y_train, cv=5, scoring="accuracy"
        ).mean()
        if cv > best_cv:
            best_cv, best_C, best_gamma = cv, C, gamma
        print(f"  C={C:<6} gamma={str(gamma):<8}  cv_acc={cv:.3f}")
    print(f"Best C={best_C}  gamma={best_gamma}  (5-fold CV acc: {best_cv:.3f})")

    svm = SVC(kernel="rbf", C=best_C, gamma=best_gamma, probability=True)
    svm.fit(X_train_p, y_train)

    train_acc = svm.score(X_train_p, y_train)
    test_acc  = svm.score(X_test_p,  y_test)
    print(f"SVM (RBF)  →  train acc: {train_acc:.3f}  |  test acc: {test_acc:.3f}")

    # 6. Save the SVM pipeline (scaler + PCA + SVC) as svm_head.pkl
    svm_pipeline = {
        "scaler": scaler,
        "pca":    pca,
        "svm":    svm,
        "classes": train_ds.classes,
    }
    joblib.dump(svm_pipeline, SVM_HEAD_PATH)
    print(f"SVM pipeline saved → {SVM_HEAD_PATH}")

    # 7. Save backbone state dict as best_model.pth
    checkpoint = {
        "model_state_dict": backbone_state,
        "best_val_loss": 1.0 - test_acc,
        "epoch": 0,
        "arch": ARCH,
        "img_size": IMG_SIZE,
        "num_classes": 0,   # feature extractor — classifier head is in svm_head.pkl
        "classes": train_ds.classes,
    }
    torch.save(checkpoint, CKPT_PATH)
    print(f"Backbone checkpoint saved → {CKPT_PATH}")
    print(f"\nDone. Test accuracy: {test_acc:.1%} (SVM RBF)")

    # 8. Update model_config.json
    cfg = {
        "model_name":      ARCH,
        "img_size":        IMG_SIZE,
        "num_classes":     12,
        "n_passes":        N_PASSES,
        "checkpoint_path": CKPT_PATH,
        "svm_head_path":   SVM_HEAD_PATH,
        "note": (
            f"Trained on 12-class armocromia dataset via multi-pass augmented feature "
            f"extraction ({ARCH} backbone via ModelFactory, {N_PASSES} augmented passes, "
            f"ColorAugmentation with RandomErasing+GaussianNoise + SVM RBF head). "
            "Inference: backbone (num_classes=0) → feature vector → svm_head.pkl pipeline. "
            "For best real-photo accuracy, retrain with real labeled face photos."
        ),
    }
    with open("model_config.json", "w") as f:
        json.dump(cfg, f, indent=4)
    print("Updated model_config.json")


# ---------------------------------------------------------------------------
# Training — end-to-end fine-tuning via Trainer
# ---------------------------------------------------------------------------

def train_finetune(cfg, force_resume=False):
    """Two-stage end-to-end fine-tuning using Trainer from SeasonalColourClassification.

    Resume behaviour
    ----------------
    When ``force_resume=True`` **or** ``checkpoints/finetune_resume.pth`` exists on disk,
    training automatically continues from the last completed epoch rather than
    starting over.  The resume checkpoint is deleted once training finishes
    successfully so that the next run starts fresh.

    Parameters
    ----------
    cfg:
        Dict read from ``model_config.json``.
    force_resume:
        If ``True``, use the resume checkpoint even if the caller did not detect
        it automatically (useful when invoked from the CLI with ``--resume``).
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import cv2

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Patch config BEFORE importing Trainer so its module-level imports pick up
    # the correct values for FINE_TUNE and INITIAL_EPOCHS.
    import SeasonalColourClassification.config as _sc_cfg
    _sc_cfg.FINE_TUNE      = True
    _sc_cfg.INITIAL_EPOCHS = int(cfg.get("initial_epochs", 10))

    from SeasonalColourClassification.data.data_loader import SeasonalColorDataset
    from SeasonalColourClassification.data.data_augmentation import ColorAugmentation
    from SeasonalColourClassification.models.base_model import ModelFactory
    from SeasonalColourClassification.training.trainer import Trainer
    from SeasonalColourClassification.training.losses import LabelSmoothingLoss, FocalLoss

    ARCH           = cfg.get("model_name", "efficientnet_b0")
    IMG_SIZE       = int(cfg.get("img_size", 224))
    TOTAL_EPOCHS   = int(cfg.get("finetune_epochs", 30))
    LOSS_NAME      = cfg.get("loss", "label_smoothing")
    CKPT_DIR       = "checkpoints"
    CKPT_PATH      = os.path.join(CKPT_DIR, "best_model.pth")
    RESUME_PATH    = os.path.join(CKPT_DIR, "finetune_resume.pth")
    DATA_ROOT      = "dataset/images"
    BATCH_SIZE     = 16
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Per-epoch resume checkpoints are always written to RESUME_PATH so that any
    # interruption can be recovered.  The trainer only *restores* from the file
    # when it already exists (or --resume / force_resume was requested).
    resume_path = RESUME_PATH
    if force_resume and not os.path.exists(RESUME_PATH):
        print(f"[finetune] --resume requested but no resume checkpoint found at "
              f"{RESUME_PATH} — starting fresh.", flush=True)
    elif os.path.exists(RESUME_PATH):
        print(f"[finetune] Resume checkpoint found: {RESUME_PATH} — will resume.", flush=True)
    else:
        print(f"[finetune] Per-epoch checkpoints will be saved to: {RESUME_PATH}", flush=True)

    print(f"[finetune] Backbone: {ARCH}  |  Total epochs: {TOTAL_EPOCHS}  "
          f"|  Initial (head-only) epochs: {_sc_cfg.INITIAL_EPOCHS}  "
          f"|  Loss: {LOSS_NAME}", flush=True)

    # 1. Generate dataset if needed
    train_dir = os.path.join(DATA_ROOT, "train")
    test_dir  = os.path.join(DATA_ROOT, "test")
    n_train_files = sum(len(f) for _, _, f in os.walk(train_dir)) if os.path.exists(train_dir) else 0
    if n_train_files < 200:
        print("[finetune] Generating synthetic face-colour dataset "
              "(80 train + 20 test per class)…", flush=True)
        generate_dataset(DATA_ROOT, n_train=80, n_test=20)
    else:
        print(f"[finetune] Dataset present ({n_train_files} train images)", flush=True)

    # 2. Build transforms (reuse FaceCropTransform + ColorAugmentation)
    class FaceCropTransform:
        def __init__(self):
            self._cascade = None
        def _get_cascade(self):
            if self._cascade is None:
                self._cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            return self._cascade
        def __call__(self, pil_img):
            img_rgb = np.array(pil_img.convert("RGB"))
            gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gray    = cv2.equalizeHist(gray)
            faces   = self._get_cascade().detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
            w_img, h_img = pil_img.size
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad_top  = int(h * 1.00)
                pad_bot  = int(h * 0.20)
                pad_side = int(w * 0.25)
                x1 = max(0, x - pad_side)
                y1 = max(0, y - pad_top)
                x2 = min(w_img, x + w + pad_side)
                y2 = min(h_img, y + h + pad_bot)
                return pil_img.crop((x1, y1, x2, y2))
            side = min(w_img, h_img)
            left = (w_img - side) // 2
            top  = (h_img - side) // 2
            return pil_img.crop((left, top, left + side, top + side))

    face_crop = FaceCropTransform()
    train_tf = transforms.Compose([face_crop, ColorAugmentation.get_train_transforms(img_size=IMG_SIZE)])
    test_tf  = transforms.Compose([face_crop, ColorAugmentation.get_test_transforms(img_size=IMG_SIZE)])

    train_ds = SeasonalColorDataset(train_dir, transform=train_tf)
    val_ds   = SeasonalColorDataset(test_dir,  transform=test_tf)
    print(f"[finetune] Classes ({len(train_ds.classes)}): {train_ds.classes}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Build full end-to-end model (num_classes=12)
    print(f"[finetune] Building {ARCH} model via ModelFactory (num_classes=12)…", flush=True)
    model = ModelFactory.get_model(ARCH, num_classes=12)

    # 4. Loss function
    if LOSS_NAME == "focal":
        criterion = FocalLoss(gamma=2.0)
        print("[finetune] Using FocalLoss (gamma=2.0)", flush=True)
    elif LOSS_NAME == "cross_entropy":
        import torch.nn as _nn
        criterion = _nn.CrossEntropyLoss()
        print("[finetune] Using CrossEntropyLoss", flush=True)
    else:
        criterion = LabelSmoothingLoss(smoothing=0.1)
        print("[finetune] Using LabelSmoothingLoss (smoothing=0.1)", flush=True)

    # 5. Train via Trainer (two-stage: head-only → full fine-tune)
    #    Pass resume_path so the Trainer saves an epoch checkpoint after every
    #    epoch and auto-restores state on interrupted runs.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        save_dir=CKPT_DIR,
    )
    print(f"[finetune] Starting two-stage training ({TOTAL_EPOCHS} total epochs)…", flush=True)
    history = trainer.train(num_epochs=TOTAL_EPOCHS, resume_path=resume_path)

    # 6. Evaluate on val set using best checkpoint
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    best_val_loss = trainer.best_val_loss
    print(f"[finetune] Best val acc: {best_val_acc:.3f}  |  Best val loss: {best_val_loss:.4f}",
          flush=True)

    # 7. Enrich the saved checkpoint with metadata for inference
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    checkpoint.update({
        "arch":        ARCH,
        "img_size":    IMG_SIZE,
        "num_classes": 12,
        "classes":     train_ds.classes,
        "training_mode": "finetune",
    })
    torch.save(checkpoint, CKPT_PATH)
    print(f"[finetune] Checkpoint saved → {CKPT_PATH}", flush=True)

    # 8. Update model_config.json
    updated_cfg = {
        "model_name":      ARCH,
        "img_size":        IMG_SIZE,
        "num_classes":     12,
        "checkpoint_path": CKPT_PATH,
        "training_mode":   "finetune",
        "loss":            LOSS_NAME,
        "finetune_epochs": TOTAL_EPOCHS,
        "initial_epochs":  int(cfg.get("initial_epochs", 10)),
        "n_passes":        int(cfg.get("n_passes", 1)),
        "note": (
            f"Fine-tuned end-to-end: {ARCH}, {TOTAL_EPOCHS} epochs "
            f"({_sc_cfg.INITIAL_EPOCHS} head-only + remainder full fine-tune), "
            f"loss={LOSS_NAME}, LabelSmoothing/FocalLoss, ColorAugmentation. "
            "Inference: full model forward pass + softmax (no svm_head.pkl needed)."
        ),
    }
    with open("model_config.json", "w") as f:
        json.dump(updated_cfg, f, indent=4)
    print("[finetune] Updated model_config.json", flush=True)

    # 9. Delete resume checkpoint — training completed successfully
    if os.path.exists(RESUME_PATH):
        os.remove(RESUME_PATH)
        print(f"[finetune] Resume checkpoint removed ({RESUME_PATH})", flush=True)

    print(f"\n[finetune] Done. Best val accuracy: {best_val_acc:.1%}", flush=True)


if __name__ == "__main__":
    import argparse

    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Train the armocromia colour classifier.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume fine-tuning from checkpoints/finetune_resume.pth. "
            "Has no effect in feature_extraction mode. "
            "Auto-detection also kicks in when the file already exists."
        ),
    )
    args = parser.parse_args()

    # Read training_mode from model_config.json (default: feature_extraction)
    _training_mode = "feature_extraction"
    _cfg_for_dispatch = {}
    if os.path.exists("model_config.json"):
        try:
            with open("model_config.json") as _f:
                _cfg_for_dispatch = json.load(_f)
            _training_mode = _cfg_for_dispatch.get("training_mode", "feature_extraction")
        except (json.JSONDecodeError, OSError):
            pass

    if _training_mode == "finetune":
        print(f"training_mode=finetune — running end-to-end fine-tuning via Trainer", flush=True)
        train_finetune(_cfg_for_dispatch, force_resume=args.resume)
    else:
        print(f"training_mode=feature_extraction — running SVM head pipeline", flush=True)
        train()
