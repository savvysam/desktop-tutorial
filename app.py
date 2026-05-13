import os
import json
import random
import base64
import io
import logging
import threading
import subprocess
import shutil
import time

from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ---------------------------------------------------------------------------
# 12-class Armocromia palette definitions (matches SeasonalColourClassification)
# ---------------------------------------------------------------------------
SEASON_PALETTES = [
    {
        "name": "Primavera Bright", "class_id": "primavera_bright", "season": "spring",
        "overlay": "#FF9F7D",
        "swatches": ["#FF9F7D", "#FF6F61", "#FFD166", "#FFB347"],
        "description": "Vivid, warm, and saturated tones for a clear, warm complexion with high contrast."
    },
    {
        "name": "Primavera Light", "class_id": "primavera_light", "season": "spring",
        "overlay": "#F7C8A2",
        "swatches": ["#F7C8A2", "#FAD7B6", "#FDE6C8", "#FFE4D4"],
        "description": "Delicate, peachy, warm tones that complement a light, soft, warm complexion."
    },
    {
        "name": "Primavera Warm", "class_id": "primavera_warm", "season": "spring",
        "overlay": "#F4B183",
        "swatches": ["#F4B183", "#F7A45A", "#F9C784", "#FFE0A3"],
        "description": "Golden, sunny tones that brighten a distinctly warm, ivory-to-medium complexion."
    },
    {
        "name": "Estate Cool", "class_id": "estate_cool", "season": "summer",
        "overlay": "#A9C0E8",
        "swatches": ["#A9C0E8", "#C2D3F2", "#B0C4DE", "#9FB7D9"],
        "description": "Muted, blue-toned shades for a cool, rosy, medium complexion."
    },
    {
        "name": "Estate Light", "class_id": "estate_light", "season": "summer",
        "overlay": "#C8D8F0",
        "swatches": ["#C8D8F0", "#D9E2F2", "#E6EEF7", "#B8C4E6"],
        "description": "Soft, cool, airy pastels for a light, cool complexion."
    },
    {
        "name": "Estate Soft", "class_id": "estate_soft", "season": "summer",
        "overlay": "#B6BFC8",
        "swatches": ["#B6BFC8", "#C6CCD2", "#AEB4BD", "#D6DAE0"],
        "description": "Dusty, muted neutrals with a cool undertone for a soft, blended complexion."
    },
    {
        "name": "Autunno Deep", "class_id": "autunno_deep", "season": "autumn",
        "overlay": "#8C5E3C",
        "swatches": ["#8C5E3C", "#A4663F", "#6D432C", "#B57446"],
        "description": "Dark, rich, warm tones for a deep, warm complexion with strong contrast."
    },
    {
        "name": "Autunno Soft", "class_id": "autunno_soft", "season": "autumn",
        "overlay": "#C2A48C",
        "swatches": ["#C2A48C", "#D1B59C", "#B99575", "#E2C2A1"],
        "description": "Muted, earthy tones for a warm yet soft, blended complexion."
    },
    {
        "name": "Autunno Warm", "class_id": "autunno_warm", "season": "autumn",
        "overlay": "#C9834C",
        "swatches": ["#C9834C", "#D9935C", "#B8723B", "#E0A06A"],
        "description": "Rich, spicy, golden-brown colors for a distinctly warm, medium complexion."
    },
    {
        "name": "Inverno Bright", "class_id": "inverno_bright", "season": "winter",
        "overlay": "#5E6AD2",
        "swatches": ["#5E6AD2", "#2D35A7", "#9B5DE5", "#4EA8DE"],
        "description": "Bold, icy, jewel-toned colors for a high-contrast, vivid cool complexion."
    },
    {
        "name": "Inverno Cool", "class_id": "inverno_cool", "season": "winter",
        "overlay": "#4B6CB7",
        "swatches": ["#4B6CB7", "#3A5BA0", "#6B8DD6", "#2F4B7C"],
        "description": "True, cool-toned classics — navy, rose, icy tones — for a cool, clear complexion."
    },
    {
        "name": "Inverno Deep", "class_id": "inverno_deep", "season": "winter",
        "overlay": "#2B2D42",
        "swatches": ["#2B2D42", "#3D405B", "#1B1D2E", "#4A4E69"],
        "description": "Dark, dramatic, cool shades for a deep, cool complexion with high contrast."
    },
]

PALETTE_BY_CLASS_ID = {p["class_id"]: p for p in SEASON_PALETTES}

# ---------------------------------------------------------------------------
# Optional deep ML model — supports two inference modes:
#   "feature_extraction" : timm backbone (num_classes=0) + sklearn SVM head
#   "finetune"           : full timm model (num_classes=12) + softmax
# ---------------------------------------------------------------------------
_ml_backbone     = None     # feature-extraction mode: timm backbone (num_classes=0)
_ml_svm_head     = None     # feature-extraction mode: dict with scaler/pca/svm/classes
_ml_finetune_model = None   # finetune mode: full timm model (num_classes=12)
_ml_training_mode  = "feature_extraction"
_ml_img_size     = 224
_ml_model_loaded = False

def _try_load_ml_model():
    global _ml_backbone, _ml_svm_head, _ml_finetune_model
    global _ml_training_mode, _ml_img_size, _ml_model_loaded
    _ml_backbone       = None
    _ml_svm_head       = None
    _ml_finetune_model = None
    _ml_model_loaded   = False
    try:
        if not os.path.exists("model_config.json"):
            return
        with open("model_config.json") as f:
            cfg = json.load(f)

        training_mode = cfg.get("training_mode", "feature_extraction")
        ckpt_path     = cfg.get("checkpoint_path", "checkpoints/best_model.pth")
        model_name    = cfg.get("model_name", "efficientnet_b0")
        img_size      = int(cfg.get("img_size", 224))

        import torch
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from SeasonalColourClassification.models.base_model import ModelFactory

        if training_mode == "finetune":
            # Full end-to-end fine-tuned model: num_classes=12, softmax inference.
            # If the checkpoint doesn't yet exist or was saved without a 12-class
            # classifier head (i.e. it's still the feature-extraction backbone from
            # a previous run), fall through to the feature-extraction branch so the
            # app stays functional while fine-tuning is in progress.
            if not os.path.exists(ckpt_path):
                logger.info("Fine-tune mode configured but no checkpoint yet — falling back to SVM head")
                training_mode = "feature_extraction"   # re-route to SVM branch below
            else:
                _ft_ok = False
                try:
                    model = ModelFactory.get_model(model_name, num_classes=12)
                    checkpoint = torch.load(ckpt_path, map_location="cpu")
                    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                    model.eval()
                    _ml_finetune_model = model
                    _ml_training_mode  = "finetune"
                    _ml_img_size       = img_size
                    _ml_model_loaded   = True
                    _ft_ok = True
                    logger.info("Fine-tuned ML model loaded: %s (%s, num_classes=12)", ckpt_path, model_name)
                except Exception as ft_err:
                    logger.info("Fine-tuned checkpoint load failed (%s) — falling back to SVM head", ft_err)
                    training_mode = "feature_extraction"   # re-route to SVM branch below

        if training_mode != "finetune":
            # Feature-extraction mode: backbone (num_classes=0) + sklearn SVM head.
            # Also reached when finetune mode fell back due to missing/incompatible ckpt.
            svm_head_path = cfg.get("svm_head_path", "checkpoints/svm_head.pkl")
            if not os.path.exists(ckpt_path) or not os.path.exists(svm_head_path):
                return
            import joblib
            backbone = ModelFactory.get_model(model_name, num_classes=0)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            raw_state = checkpoint.get("model_state_dict", checkpoint)
            # Detect which level the keys are at and load accordingly.
            # Checkpoints saved via backbone_model.model.state_dict() have raw timm
            # keys (e.g. "conv_stem.weight"); ones saved via TimmModel.state_dict()
            # have a "model." prefix.  Handle both formats.
            _sample = next(iter(raw_state))
            if _sample.startswith("model."):
                # TimmModel-level keys → load into the TimmModel wrapper directly
                backbone.load_state_dict(raw_state, strict=False)
            else:
                # Raw timm model keys → load into backbone.model (the inner timm model)
                backbone.model.load_state_dict(raw_state, strict=False)
            backbone.eval()
            svm_head = joblib.load(svm_head_path)
            _ml_backbone      = backbone
            _ml_svm_head      = svm_head
            _ml_training_mode = "feature_extraction"
            _ml_img_size      = img_size
            _ml_model_loaded  = True
            logger.info("SVM ML model loaded: backbone=%s, svm_head=%s", ckpt_path, svm_head_path)
    except Exception as e:
        logger.info("No deep ML model loaded (%s)", e)

_try_load_ml_model()

# ---------------------------------------------------------------------------
# Import real colour-theory analyser
# ---------------------------------------------------------------------------
try:
    import color_analysis as _ca
    _COLOR_THEORY_AVAILABLE = True
    logger.info("Colour-theory SIVC analyser loaded")
except Exception as e:
    _COLOR_THEORY_AVAILABLE = False
    logger.warning("Colour-theory analyser unavailable: %s", e)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


def detect_and_crop_face(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) == 0:
        return img_bgr
    x, y, w, h = faces[0]
    pad = int(0.2 * max(w, h))
    return img_bgr[max(0, y-pad):min(img_bgr.shape[0], y+h+pad),
                   max(0, x-pad):min(img_bgr.shape[1], x+w+pad)]


def image_to_data_url(img_bgr, max_size=400):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def rgb_to_hex(rgb):
    if rgb is None:
        return None
    return "#{:02X}{:02X}{:02X}".format(*rgb)

# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------
ALL_CLASS_IDS = [p["class_id"] for p in SEASON_PALETTES]


def run_ml_inference(face_bgr):
    """ML inference — dispatches to fine-tune (softmax) or feature-extraction (SVM) path."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    tf = transforms.Compose([
        transforms.Resize((_ml_img_size, _ml_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    tensor = tf(pil).unsqueeze(0)

    if _ml_training_mode == "finetune":
        # Full model forward pass + softmax
        with torch.no_grad():
            logits = _ml_finetune_model(tensor)          # shape (1, 12)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
        scores = {cid: round(float(p), 4) for cid, p in zip(ALL_CLASS_IDS, probs)}
        return scores, "ml_model_finetune"
    else:
        # Step 1: extract feature vector from backbone (num_classes=0)
        with torch.no_grad():
            feat = _ml_backbone(tensor).cpu().numpy()    # shape (1, D)

        # Step 2: apply sklearn pipeline (StandardScaler → PCA → SVC)
        scaler  = _ml_svm_head["scaler"]
        pca     = _ml_svm_head["pca"]
        svm     = _ml_svm_head["svm"]
        classes = _ml_svm_head["classes"]

        feat_s = scaler.transform(feat)
        feat_p = pca.transform(feat_s)
        probs  = svm.predict_proba(feat_p)[0]            # shape (12,)

        # Map SVM class order → canonical ALL_CLASS_IDS order
        class_to_prob = {cls: float(prob) for cls, prob in zip(classes, probs)}
        scores = {cid: round(class_to_prob.get(cid, 0.0), 4) for cid in ALL_CLASS_IDS}
        return scores, "ml_model"


def run_color_theory(img_bgr):
    """Real SIVC colour-theory analysis (no weights needed)."""
    result = _ca.analyse(img_bgr)
    return result["class_scores"], "color_theory", result["metrics"], result["dominant_colors"]


def run_mock():
    """Random mock predictions as last resort."""
    base = random.uniform(0.55, 0.95)
    shuffled = list(range(len(SEASON_PALETTES)))
    random.shuffle(shuffled)
    scores = {}
    for rank, idx in enumerate(shuffled):
        scores[SEASON_PALETTES[idx]["class_id"]] = max(0.01, base - rank * 0.07)
    total = sum(scores.values())
    return {k: round(v/total, 4) for k, v in scores.items()}, "mock"


def build_predictions(class_scores):
    """Convert class_scores dict → sorted list of prediction dicts."""
    preds = [
        {"palette": PALETTE_BY_CLASS_ID[cid], "confidence": conf}
        for cid, conf in class_scores.items()
    ]
    preds.sort(key=lambda x: x["confidence"], reverse=True)
    return preds


def build_explanation(primary, secondary=None):
    if secondary:
        return (
            f"Your features fall between {primary['palette']['name']} and "
            f"{secondary['palette']['name']}. Both palettes suit you — lean toward "
            f"{primary['palette']['name']} while exploring "
            f"{secondary['palette']['name']} shades too."
        )
    return (
        f"Your coloring aligns with {primary['palette']['name']}. "
        f"This palette best matches your undertone, contrast, and overall depth."
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if _ml_model_loaded:
        # A model is loaded — show what's actually running so the badge is
        # accurate and consistent with what /analyze will use.
        mode = "ml_model_finetune" if _ml_training_mode == "finetune" else "ml_model"
    elif _COLOR_THEORY_AVAILABLE:
        mode = "color_theory"
    else:
        mode = "mock"

    # Also expose the configured training mode (from model_config.json) so the
    # template can show an "upcoming mode" hint when it differs from active mode.
    _configured_mode = _ml_training_mode
    try:
        if os.path.exists("model_config.json"):
            with open("model_config.json") as _f:
                _configured_mode = json.load(_f).get("training_mode", _ml_training_mode)
    except Exception:
        pass

    return render_template("index.html", analysis_mode=mode,
                           configured_training_mode=_configured_mode)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400
    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        face_img = detect_and_crop_face(img)

        # --- Choose analysis method ---
        metrics_data   = None
        dominant_colors = None
        analysis_mode  = "mock"

        if _ml_model_loaded:
            class_scores, analysis_mode = run_ml_inference(face_img)
        elif _COLOR_THEORY_AVAILABLE:
            class_scores, analysis_mode, metrics_data, dominant_colors = run_color_theory(face_img)
        else:
            class_scores, analysis_mode = run_mock()

        predictions = build_predictions(class_scores)
        primary   = predictions[0]
        gap       = predictions[0]["confidence"] - predictions[1]["confidence"]
        secondary = predictions[1] if gap < 0.08 else None
        explanation = build_explanation(primary, secondary)

        response = {
            "predictions":     predictions,
            "primary":         primary,
            "secondary":       secondary,
            "explanation":     explanation,
            "image_data_url":  image_to_data_url(face_img),
            "analysis_mode":   analysis_mode,
        }
        if metrics_data:
            response["metrics"] = metrics_data
        if dominant_colors:
            response["dominant_colors"] = {
                k: rgb_to_hex(v) for k, v in dominant_colors.items()
            }

        return jsonify(response)

    except Exception as e:
        logger.exception("Analysis error")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Dataset download + retrain management
# ---------------------------------------------------------------------------
_task_state = {
    "status": "idle",   # idle | downloading | retraining | cancelled | download_cancelled | done | error
    "progress": 0,
    "total": 0,
    "message": "",
    "log": [],
    "cancelled_at_epoch": None,
    "training_mode": None,      # "finetune" | "feature_extraction" | None
    "current_epoch": None,      # most-recently completed epoch (1-based)
    "total_epochs": None,       # total epochs for the run
    "phase": None,              # "initial" | "finetune"
    "train_acc":  [],           # per-epoch train accuracy (floats)
    "val_acc":    [],           # per-epoch val accuracy (floats)
    "train_loss": [],           # per-epoch train loss (floats)
    "val_loss":   [],           # per-epoch val loss (floats)
    "retrain_start_time": None, # wall-clock time when retraining began
    "avg_epoch_seconds": None,  # rolling average seconds per epoch
    "eta_seconds": None,        # estimated seconds remaining
    "last_test_accuracy": None, # final test accuracy from the last completed run (%)
}
_task_lock = threading.Lock()
_train_proc = None          # handle to the running train.py subprocess
_download_proc = None       # handle to the running download_dataset.py subprocess

_TASK_STATE_SIDECAR = os.path.join("checkpoints", "task_state.json")


def _persist_cancelled_state():
    """Write the current cancelled state to disk so it survives server restarts."""
    with _task_lock:
        snap = {
            "status":             _task_state["status"],
            "message":            _task_state["message"],
            "cancelled_at_epoch": _task_state.get("cancelled_at_epoch"),
            "training_mode":      _task_state.get("training_mode"),
            "current_epoch":      _task_state.get("current_epoch"),
            "total_epochs":       _task_state.get("total_epochs"),
            "phase":              _task_state.get("phase"),
        }
    try:
        os.makedirs("checkpoints", exist_ok=True)
        tmp = _TASK_STATE_SIDECAR + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f)
        os.replace(tmp, _TASK_STATE_SIDECAR)
        logger.info("Persisted cancelled task state to %s", _TASK_STATE_SIDECAR)
    except Exception as e:
        logger.warning("Could not persist task state: %s", e)


def _clear_task_state_sidecar():
    """Remove the persisted cancelled state (called when training (re)starts or completes)."""
    try:
        if os.path.exists(_TASK_STATE_SIDECAR):
            os.remove(_TASK_STATE_SIDECAR)
    except Exception as e:
        logger.warning("Could not remove task state sidecar: %s", e)


def _load_persisted_task_state():
    """
    On startup, restore a previously-cancelled training state so the UI
    doesn't show 'idle' when a resume checkpoint is waiting.
    Reads checkpoints/task_state.json if present; falls back to detecting
    checkpoints/finetune_resume.pth with no sidecar.
    """
    if os.path.exists(_TASK_STATE_SIDECAR):
        try:
            with open(_TASK_STATE_SIDECAR) as f:
                snap = json.load(f)
            if snap.get("status") == "cancelled":
                _task_state.update({
                    "status":             "cancelled",
                    "message":            snap.get("message",
                                                   "Training paused — click Retrain to resume."),
                    "cancelled_at_epoch": snap.get("cancelled_at_epoch"),
                    "training_mode":      snap.get("training_mode"),
                    "current_epoch":      snap.get("current_epoch"),
                    "total_epochs":       snap.get("total_epochs"),
                    "phase":              snap.get("phase"),
                })
                logger.info("Restored cancelled training state from %s", _TASK_STATE_SIDECAR)
                return
        except Exception as e:
            logger.warning("Could not read task state sidecar: %s", e)

    # Fallback: no sidecar but a resume checkpoint exists — show paused state
    resume_ckpt = os.path.join("checkpoints", "finetune_resume.pth")
    if os.path.exists(resume_ckpt):
        _task_state.update({
            "status":  "cancelled",
            "message": "Training paused — click Retrain to resume.",
        })
        logger.info("Detected finetune_resume.pth — setting status to cancelled")


_load_persisted_task_state()


def _append_log(msg):
    with _task_lock:
        _task_state["log"].append(msg)
        if len(_task_state["log"]) > 500:
            _task_state["log"] = _task_state["log"][-500:]
    logger.info("[task] %s", msg)


def _run_download(password, train_n, test_n):
    """Run download_dataset.py in a subprocess and stream its output."""
    global _download_proc
    with _task_lock:
        _task_state.update({"status": "downloading", "progress": 0, "total": 0,
                            "message": "Starting download…", "log": []})
    # Belt-and-suspenders: track the last class-partition directory we saw
    # being written so we can clean it up if the download is cancelled
    # (the subprocess also cleans up via its own SIGTERM handler, but we
    # cover the case where the handler output was not flushed in time).
    _last_partial_dir = None
    try:
        cmd = [
            "python3", "-u", "download_dataset.py",
            "--password", password,
            "--train",   str(train_n),
            "--test",    str(test_n),
        ]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        with _task_lock:
            _download_proc = proc

        for line in proc.stdout:
            line = line.rstrip()
            _append_log(line)
            # Authoritative active-directory tracking: emitted at every
            # class/partition boundary by download_dataset.py.
            if line.startswith("[ACTIVE_DIR]"):
                dir_str = line[len("[ACTIVE_DIR]"):].strip()
                if dir_str:
                    _last_partial_dir = dir_str
            # Parse progress from "[N/TOTAL] …" lines
            elif "[" in line and "/" in line and "]" in line:
                try:
                    inside = line[line.index("[")+1:line.index("]")]
                    cur, tot = inside.split("/")
                    with _task_lock:
                        _task_state["progress"] = int(cur.strip())
                        _task_state["total"]    = int(tot.strip())
                        _task_state["message"]  = line.strip()
                except Exception:
                    pass
            elif "Downloading" in line and "images" in line:
                try:
                    n = int(line.split()[1])
                    with _task_lock:
                        _task_state["total"] = n
                        _task_state["message"] = line.strip()
                except Exception:
                    pass
            else:
                with _task_lock:
                    _task_state["message"] = line.strip() or _task_state["message"]

        proc.wait()
        with _task_lock:
            _download_proc = None

        if proc.returncode == 0:
            with _task_lock:
                _task_state["status"] = "done"
                _task_state["message"] = "Download complete! You can now retrain the model."
            _append_log("✓ Download finished successfully.")
        else:
            with _task_lock:
                current_status = _task_state["status"]
            if current_status == "download_cancelled":
                _append_log("⚠ Download cancelled by user.")
                # Clean up any partial class directory left by the subprocess.
                # The subprocess SIGTERM handler already attempts this, but we
                # repeat it here in case the directory still exists.
                if _last_partial_dir and os.path.isdir(_last_partial_dir):
                    try:
                        shutil.rmtree(_last_partial_dir)
                        logger.info("Cleaned up partial download directory: %s",
                                    _last_partial_dir)
                        _append_log(f"✓ Cleaned up partial directory: {_last_partial_dir}")
                    except Exception as _ce:
                        logger.warning("Could not clean up %s: %s", _last_partial_dir, _ce)
                elif _last_partial_dir:
                    logger.info("Partial directory already removed by subprocess: %s",
                                _last_partial_dir)
                    _append_log(f"✓ Partial directory already cleaned up: {_last_partial_dir}")
            else:
                with _task_lock:
                    _task_state["status"] = "error"
                    _task_state["message"] = "Download failed — check the log below."
    except Exception as e:
        with _task_lock:
            _download_proc = None
            if _task_state["status"] != "download_cancelled":
                _task_state["status"] = "error"
                _task_state["message"] = str(e)
        _append_log(f"✗ Exception: {e}")


def _run_retrain():
    """Run train.py in a subprocess."""
    global _train_proc
    # Read training_mode from model_config.json so the status endpoint can
    # expose it and the frontend can decide whether to show epoch progress.
    _cfg_training_mode = "feature_extraction"
    try:
        if os.path.exists("model_config.json"):
            with open("model_config.json") as _f:
                _cfg_training_mode = json.load(_f).get("training_mode", "feature_extraction")
    except Exception:
        pass

    with _task_lock:
        _task_state.update({
            "status": "retraining", "progress": 0, "total": 0,
            "message": "Starting training…", "log": [],
            "cancelled_at_epoch": None,
            "training_mode": _cfg_training_mode,
            "current_epoch": None,
            "total_epochs": None,
            "phase": None,
            "train_acc":  [],
            "val_acc":    [],
            "train_loss": [],
            "val_loss":   [],
            "retrain_start_time": time.time(),
            "avg_epoch_seconds": None,
            "eta_seconds": None,
        })
    # Clear any persisted cancelled state — a fresh run is now in progress.
    _clear_task_state_sidecar()
    # Always remove stale feature cache so train.py re-extracts from current dataset
    cache_path = os.path.join("checkpoints", "_feat_cache.npz")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    try:
        import re as _re
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            ["python3", "-u", "train.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        with _task_lock:
            _train_proc = proc

        # Patterns for epoch progress — matches trainer.py print format:
        #   Initial phase: "Epoch {cur}/{total} - Train Loss: ..."
        #   Fine-tune phase: "Fine-tune Epoch {ft}/{ft_total} (Total Epoch {cur}/{total}) - ..."
        _finetune_total_pat = _re.compile(
            r'Total Epoch\s+(\d+)/(\d+)', _re.IGNORECASE)
        _finetune_ft_pat    = _re.compile(
            r'Fine-tune Epoch\s+(\d+)/(\d+)', _re.IGNORECASE)
        _initial_pat        = _re.compile(
            r'^Epoch\s+(\d+)/(\d+)\s+-', _re.IGNORECASE)
        _epoch_cancel_pat   = _re.compile(
            r'epoch\s*[:\-]?\s*(\d+)', _re.IGNORECASE)
        _pass_pattern       = _re.compile(r'pass\s+(\d+)/\d+', _re.IGNORECASE)
        # Pattern to extract accuracy/loss metrics from each epoch line
        _metrics_pat        = _re.compile(
            r'Train Loss:\s*([\d.]+),\s*Train Acc:\s*([\d.]+),\s*'
            r'Val Loss:\s*([\d.]+),\s*Val Acc:\s*([\d.]+)',
            _re.IGNORECASE)

        for line in proc.stdout:
            stripped = line.rstrip()
            _append_log(stripped)
            with _task_lock:
                _task_state["message"] = stripped or _task_state["message"]

                # --- Accuracy / loss metrics ---
                m_metrics = _metrics_pat.search(stripped)
                if m_metrics:
                    try:
                        _task_state["train_loss"].append(float(m_metrics.group(1)))
                        _task_state["train_acc"].append(float(m_metrics.group(2)))
                        _task_state["val_loss"].append(float(m_metrics.group(3)))
                        _task_state["val_acc"].append(float(m_metrics.group(4)))
                    except (ValueError, IndexError):
                        pass

                # --- Epoch progress tracking ---
                m_ft_total = _finetune_total_pat.search(stripped)
                m_ft       = _finetune_ft_pat.search(stripped)
                m_init     = _initial_pat.search(stripped)

                _epoch_updated = False
                if m_ft_total:
                    # Fine-tune phase total-epoch line
                    _task_state["current_epoch"] = int(m_ft_total.group(1))
                    _task_state["total_epochs"]  = int(m_ft_total.group(2))
                    _task_state["phase"]         = "finetune"
                    _task_state["cancelled_at_epoch"] = _task_state["current_epoch"]
                    _epoch_updated = True
                elif m_ft and not m_ft_total:
                    # Standalone fine-tune line without total-epoch info
                    _task_state["phase"] = "finetune"
                    _task_state["cancelled_at_epoch"] = int(m_ft.group(1))
                elif m_init:
                    _task_state["current_epoch"] = int(m_init.group(1))
                    _task_state["total_epochs"]  = int(m_init.group(2))
                    _task_state["phase"]         = "initial"
                    _task_state["cancelled_at_epoch"] = _task_state["current_epoch"]
                    _epoch_updated = True
                else:
                    # Fallback: any "epoch N" mention used only for cancel message
                    m = _epoch_cancel_pat.search(stripped) or _pass_pattern.search(stripped)
                    if m:
                        _task_state["cancelled_at_epoch"] = int(m.group(1))

                # --- ETA computation ---
                if _epoch_updated:
                    _cur = _task_state["current_epoch"]
                    _tot = _task_state["total_epochs"]
                    _t0  = _task_state["retrain_start_time"]
                    if _cur and _tot and _t0:
                        _elapsed = time.time() - _t0
                        _avg = _elapsed / _cur
                        _eta = _avg * (_tot - _cur)
                        _task_state["avg_epoch_seconds"] = round(_avg, 1)
                        _task_state["eta_seconds"] = round(_eta)

                # Write lightweight sidecar JSON so /dataset/status can read epoch
                # progress even if the server restarts mid-training.
                if _epoch_updated and _cfg_training_mode == "finetune":
                    _progress_snap = {
                        "current_epoch":    _task_state["current_epoch"],
                        "total_epochs":     _task_state["total_epochs"],
                        "phase":            _task_state["phase"],
                        "training_mode":    _cfg_training_mode,
                        "avg_epoch_seconds": _task_state.get("avg_epoch_seconds"),
                        "retrain_start_time": _task_state.get("retrain_start_time"),
                    }
                    try:
                        os.makedirs("checkpoints", exist_ok=True)
                        _sidecar = os.path.join("checkpoints", "finetune_progress.json")
                        with open(_sidecar + ".tmp", "w") as _sf:
                            json.dump(_progress_snap, _sf)
                        os.replace(_sidecar + ".tmp", _sidecar)
                    except Exception:
                        pass

        proc.wait()
        with _task_lock:
            _train_proc = None

        if proc.returncode == 0:
            # Parse final test accuracy from the training log
            _final_acc = None
            import re as _re2
            _acc_pat = _re2.compile(r'test\s+acc(?:uracy)?[:\s]+([0-9]+\.?[0-9]*)\s*%?', _re2.IGNORECASE)
            with _task_lock:
                for _log_line in reversed(_task_state.get("log", [])):
                    _m = _acc_pat.search(_log_line)
                    if _m:
                        try:
                            _val = float(_m.group(1))
                            # Normalise: values > 1 are already percentages
                            _final_acc = round(_val if _val > 1 else _val * 100, 1)
                        except ValueError:
                            pass
                        break

            # Persist accuracy + training timestamp to model_config.json
            try:
                _cfg_path = "model_config.json"
                _cfg = {}
                if os.path.exists(_cfg_path):
                    with open(_cfg_path) as _cf:
                        _cfg = json.load(_cf)
                if _final_acc is not None:
                    _cfg["last_test_accuracy"] = _final_acc
                import datetime as _dt
                _cfg["last_trained_at"] = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                with open(_cfg_path, "w") as _cf:
                    json.dump(_cfg, _cf, indent=4)
            except Exception as _ae:
                logger.warning("Could not save accuracy to model_config.json: %s", _ae)

            # Reload model
            _try_load_ml_model()
            _clear_task_state_sidecar()
            _done_msg = "Retraining complete! Model updated."
            if _final_acc is not None:
                _done_msg = f"Retraining complete! Test accuracy: {_final_acc}%"
            with _task_lock:
                _task_state["status"] = "done"
                _task_state["message"] = _done_msg
                _task_state["cancelled_at_epoch"] = None
                _task_state["last_test_accuracy"] = _final_acc
        elif proc.returncode == -15 or proc.returncode == 1:
            # SIGTERM (−15) or exit(1) after graceful stop — treat as cancelled
            # only if status was already set to cancelled by /dataset/cancel
            with _task_lock:
                if _task_state["status"] != "cancelled":
                    _task_state["status"] = "error"
                    _task_state["message"] = "Training stopped unexpectedly — check the log."
        else:
            with _task_lock:
                if _task_state["status"] != "cancelled":
                    _task_state["status"] = "error"
                    _task_state["message"] = "Training failed — check the log."
    except Exception as e:
        with _task_lock:
            _train_proc = None
            if _task_state["status"] != "cancelled":
                _task_state["status"] = "error"
                _task_state["message"] = str(e)


@app.route("/dataset/set-mode", methods=["POST"])
def dataset_set_mode():
    body = request.get_json(force=True, silent=True) or {}
    new_mode = body.get("training_mode")
    if new_mode not in ("feature_extraction", "finetune"):
        return jsonify({"error": "Invalid training_mode value"}), 400

    cfg_path = "model_config.json"
    try:
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path) as _f:
                cfg = json.load(_f)
        cfg["training_mode"] = new_mode
        with open(cfg_path, "w") as _f:
            json.dump(cfg, _f, indent=4)
    except Exception as e:
        logger.warning("Could not update model_config.json in set-mode: %s", e)
        return jsonify({"error": "Could not save training mode"}), 500

    return jsonify({"ok": True, "training_mode": new_mode})


@app.route("/dataset")
def dataset_page():
    annotations_exists = os.path.exists("annotations.csv")
    dataset_counts = {}
    dataset_root = "dataset/images"
    if os.path.isdir(dataset_root):
        for split in ("train", "test"):
            split_dir = os.path.join(dataset_root, split)
            if not os.path.isdir(split_dir):
                continue
            for season in sorted(os.listdir(split_dir)):
                season_dir = os.path.join(split_dir, season)
                if not os.path.isdir(season_dir):
                    continue
                for subtype in sorted(os.listdir(season_dir)):
                    subtype_dir = os.path.join(season_dir, subtype)
                    if not os.path.isdir(subtype_dir):
                        continue
                    count = len([f for f in os.listdir(subtype_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    key = f"{season}_{subtype}"
                    dataset_counts.setdefault(key, {})
                    dataset_counts[key][split] = count
    # Read training config from model_config.json for display on the dataset page
    cfg_training_mode   = None
    cfg_finetune_epochs = 30
    cfg_initial_epochs  = 10
    cfg_loss            = "label_smoothing"
    try:
        if os.path.exists("model_config.json"):
            with open("model_config.json") as _f:
                _mcfg = json.load(_f)
            cfg_training_mode   = _mcfg.get("training_mode")
            cfg_finetune_epochs = int(_mcfg.get("finetune_epochs", 30))
            cfg_initial_epochs  = int(_mcfg.get("initial_epochs", 10))
            cfg_loss            = _mcfg.get("loss", "label_smoothing")
    except Exception:
        pass

    last_test_accuracy = None
    last_trained_at    = None
    try:
        if os.path.exists("model_config.json"):
            with open("model_config.json") as _f2:
                _mcfg2 = json.load(_f2)
            last_test_accuracy = _mcfg2.get("last_test_accuracy")
            last_trained_at    = _mcfg2.get("last_trained_at")
    except Exception:
        pass

    return render_template("dataset.html",
                           annotations_exists=annotations_exists,
                           dataset_counts=dataset_counts,
                           model_loaded=_ml_model_loaded,
                           ml_training_mode=_ml_training_mode,
                           cfg_training_mode=cfg_training_mode,
                           cfg_finetune_epochs=cfg_finetune_epochs,
                           cfg_initial_epochs=cfg_initial_epochs,
                           cfg_loss=cfg_loss,
                           last_test_accuracy=last_test_accuracy,
                           last_trained_at=last_trained_at)


@app.route("/dataset/start", methods=["POST"])
def dataset_start():
    with _task_lock:
        if _task_state["status"] in ("downloading", "retraining"):
            return jsonify({"error": "A task is already running"}), 409
    data = request.get_json(force=True)
    password = (data.get("password") or "").strip()
    if not password:
        return jsonify({"error": "Password is required"}), 400
    train_n = max(5, int(data.get("train", 9999)))
    test_n  = max(2, int(data.get("test",  9999)))
    t = threading.Thread(target=_run_download, args=(password, train_n, test_n), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/dataset/retrain", methods=["POST"])
def dataset_retrain():
    with _task_lock:
        if _task_state["status"] in ("downloading", "retraining"):
            return jsonify({"error": "A task is already running"}), 409

    # Optionally accept training config overrides in the request body
    body = request.get_json(force=True, silent=True) or {}
    new_mode = body.get("training_mode")
    if new_mode in ("feature_extraction", "finetune"):
        # Validate numeric finetune params before touching disk
        if new_mode == "finetune":
            try:
                ft_ep = int(body["finetune_epochs"]) if "finetune_epochs" in body else None
                in_ep = int(body["initial_epochs"])  if "initial_epochs"  in body else None
            except (ValueError, TypeError) as _ve:
                return jsonify({"error": f"Invalid parameter value: {_ve}"}), 400
            if ft_ep is not None and ft_ep < 1:
                return jsonify({"error": "finetune_epochs must be at least 1"}), 400
            if in_ep is not None and in_ep < 1:
                return jsonify({"error": "initial_epochs must be at least 1"}), 400
            loss_val = body.get("loss")
            if loss_val is not None and loss_val not in ("label_smoothing", "focal", "cross_entropy"):
                return jsonify({"error": f"Unknown loss function: {loss_val}"}), 400

        # Persist the new mode (and optional finetune params) to model_config.json
        cfg_path = "model_config.json"
        try:
            cfg = {}
            if os.path.exists(cfg_path):
                with open(cfg_path) as _f:
                    cfg = json.load(_f)
            cfg["training_mode"] = new_mode
            if new_mode == "finetune":
                if ft_ep is not None:
                    cfg["finetune_epochs"] = ft_ep
                if in_ep is not None:
                    cfg["initial_epochs"] = in_ep
                if loss_val is not None:
                    cfg["loss"] = loss_val
            with open(cfg_path, "w") as _f:
                json.dump(cfg, _f, indent=4)
        except Exception as e:
            logger.warning("Could not update model_config.json: %s", e)
            return jsonify({"error": "Could not save training configuration"}), 500

    t = threading.Thread(target=_run_retrain, daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/dataset/cancel", methods=["POST"])
def dataset_cancel():
    import signal as _signal
    with _task_lock:
        current_status = _task_state["status"]

        if current_status == "downloading":
            proc = _download_proc
            if proc is None or proc.poll() is not None:
                return jsonify({"error": "Download process not available; please wait and retry"}), 409
            _task_state["status"] = "download_cancelled"
            _task_state["message"] = "Download cancelled — any partial class directory has been cleaned up automatically. Start a new download to try again."

        elif current_status == "retraining":
            proc = _train_proc
            if proc is None or proc.poll() is not None:
                return jsonify({"error": "Training process not available; please wait and retry"}), 409
            epoch = _task_state.get("cancelled_at_epoch")
            _task_state["status"] = "cancelled"
            _task_state["message"] = (
                f"Training paused — click Retrain to resume from epoch {epoch}."
                if epoch is not None
                else "Training paused — click Retrain to resume."
            )

        else:
            return jsonify({"error": "No cancellable task is running"}), 409

    # Lock is released here.
    # Persist cancelled training state outside the lock to avoid re-entrancy
    # deadlock (_persist_cancelled_state acquires _task_lock internally).
    if current_status == "retraining":
        _persist_cancelled_state()

    # SIGTERM is sent after releasing the lock so the worker thread's loop can
    # drain stdout and then proc.wait() without deadlocking on the lock.
    try:
        proc.send_signal(_signal.SIGTERM)
    except Exception as e:
        logger.warning("Could not send SIGTERM to subprocess: %s", e)

    if current_status == "retraining":
        _append_log("⚠ Training cancelled by user.")
    # Download cancel is logged by _run_download() once the process exits.
    return jsonify({"ok": True})


@app.route("/dataset/status")
def dataset_status():
    with _task_lock:
        state = dict(_task_state)

    # When retraining in finetune mode and in-memory epoch info is absent
    # (e.g., server restarted mid-training), read the lightweight sidecar JSON
    # written alongside each epoch by _run_retrain().
    if (state.get("status") == "retraining"
            and state.get("training_mode") == "finetune"
            and state.get("current_epoch") is None):
        _sidecar = os.path.join("checkpoints", "finetune_progress.json")
        try:
            with open(_sidecar) as _sf:
                _prog = json.load(_sf)
            state["current_epoch"] = _prog.get("current_epoch")
            state["total_epochs"]  = _prog.get("total_epochs")
            state["phase"]         = _prog.get("phase")
            # Restore ETA timing so it survives a server restart mid-training
            _avg = _prog.get("avg_epoch_seconds")
            if _avg is not None and state.get("avg_epoch_seconds") is None:
                state["avg_epoch_seconds"] = _avg
                _cur = state["current_epoch"]
                _tot = state["total_epochs"]
                if _cur is not None and _tot is not None:
                    state["eta_seconds"] = round(_avg * (_tot - _cur))
            # Also restore retrain_start_time if present (informational)
            if state.get("retrain_start_time") is None:
                state["retrain_start_time"] = _prog.get("retrain_start_time")
        except Exception:
            pass

    return jsonify(state)


def _drive_token_status() -> dict:
    """
    Return a dict describing the current Google Drive token state.
    Fields: valid (bool), expired (bool), expires_at (ISO str or None),
            seconds_remaining (int or None), message (str).
    """
    _base = os.path.dirname(os.path.abspath(__file__))
    token_file  = os.path.join(_base, ".gdrive_token")
    expiry_file = os.path.join(_base, ".gdrive_token_expiry")

    has_token = os.path.exists(token_file) and bool(open(token_file).read().strip())
    if not has_token:
        return {"valid": False, "expired": False, "expires_at": None,
                "seconds_remaining": None,
                "message": "No token — ask the assistant to reconnect Google Drive."}

    expires_at_str = None
    seconds_remaining = None
    expired = False
    if os.path.exists(expiry_file):
        try:
            from datetime import datetime, timezone
            expires_at_str = open(expiry_file).read().strip()
            dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
            seconds_remaining = int(dt.timestamp() - time.time())
            expired = seconds_remaining < 60
        except Exception:
            pass

    if expired:
        return {"valid": False, "expired": True, "expires_at": expires_at_str,
                "seconds_remaining": seconds_remaining,
                "message": "Token expired — ask the assistant to refresh the Google Drive token."}

    msg = "Connected"
    if seconds_remaining is not None:
        h, rem = divmod(max(0, seconds_remaining), 3600)
        m = rem // 60
        msg = f"Valid for {h}h {m:02d}m" if h else f"Valid for {m}m"

    return {"valid": True, "expired": False, "expires_at": expires_at_str,
            "seconds_remaining": seconds_remaining, "message": msg}


@app.route("/admin/drive-token-status")
def admin_drive_token_status():
    """Return the current Drive OAuth token validity as JSON."""
    return jsonify(_drive_token_status())


@app.route("/admin/refresh-token", methods=["POST"])
def admin_refresh_token():
    """
    Accept a fresh Google Drive access token + optional expiry from an external
    caller (e.g. the Replit agent's code_execution sandbox which can call
    listConnections) and write them to .gdrive_token / .gdrive_token_expiry.
    """
    data = request.get_json(force=True) or {}
    token = (data.get("token") or "").strip()
    if not token:
        return jsonify({"error": "token field required"}), 400
    _base = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(_base, ".gdrive_token"), "w") as f:
            f.write(token)
        expiry = (data.get("expires_at") or "").strip()
        if expiry:
            with open(os.path.join(_base, ".gdrive_token_expiry"), "w") as f:
                f.write(expiry)
        logger.info("Drive token refreshed via /admin/refresh-token (%d chars)", len(token))
        return jsonify({"ok": True, "length": len(token), "expires_at": expiry or None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
