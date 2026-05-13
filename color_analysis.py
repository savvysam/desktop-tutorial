"""
Real SIVC-based seasonal colour analysis.

Inspired by raden-dosel/seasonal-color-classification — uses colour theory to
determine season from four facial-region metrics:
  S = Subtone   (1=warm, 0=cold)   — lips colour vs peach/purple
  I = Intensity (1=high, 0=low)    — skin HSV saturation
  V = Value     (1=light, 0=dark)  — mean brightness of skin + hair + eyes
  C = Contrast  (1=high, 0=low)    — |hair_brightness − eye_brightness|

Segmentation is done with OpenCV (no heavy model weights needed).

Additional lip-season voting signal from Colorinsight (PSY222/Colorinsight):
  Per-season lip RGB reference vectors (Spring/Summer/Autumn/Winter) derived
  from Korean celebrity image research. L2 distance voting across sampled lip
  pixels produces a soft season prior that is blended 30% with the SIVC score.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import color as skcolor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Reference palette metrics vectors (S, I, V, C)
# Source: raden-dosel/seasonal-color-classification palettes/
# ---------------------------------------------------------------------------
REFERENCE_VECTORS = {
    "spring": [1, 1, 1, 1],   # warm, saturated, light, high-contrast
    "summer": [0, 0, 1, 0],   # cold,  muted,     light, low-contrast
    "autumn": [1, 0, 0, 0],   # warm,  muted,     dark,  low-contrast
    "winter": [0, 1, 0, 1],   # cold,  saturated, dark,  high-contrast
}

# ---------------------------------------------------------------------------
# Lip-season reference RGB vectors (from PSY222/Colorinsight — functions.py)
# Research-validated lip color archetypes per season derived from Korean
# celebrity image dataset. Used for L2-distance voting across lip pixels.
# ---------------------------------------------------------------------------
LIP_SEASON_REFS = {
    "spring": np.array([[253, 183, 169], [247, 98,  77],  [186, 33,  33]],  dtype=np.float32),
    "summer": np.array([[243, 184, 202], [211, 118, 155], [147, 70,  105]], dtype=np.float32),
    "autumn": np.array([[210, 124, 110], [155, 70,  60],  [97,  16,  28]],  dtype=np.float32),
    "winter": np.array([[237, 223, 227], [177, 47,  57],  [98,  14,  37]],  dtype=np.float32),
}

# Blend weight for the lip-season vote vs the SIVC score
LIP_VOTE_WEIGHT = 0.30

# Binarisation thresholds (from UserPaletteClassificationFilter)
THRESH_CONTRAST  = 0.200
THRESH_INTENSITY = 0.422
THRESH_VALUE     = 0.390

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _rgb_to_hsv_single(r, g, b):
    """Return (H°, S 0-1, V 0-1) for a single RGB pixel (0-255)."""
    arr = np.array([[[r, g, b]]], dtype=np.float32) / 255.0
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    return float(hsv[0, 0, 0]), float(hsv[0, 0, 1]), float(hsv[0, 0, 2])


def _get_lip_pixels(lips_roi_bgr, max_samples=60):
    """
    Extract filtered RGB pixel samples from the lip ROI.

    Applies the Colorinsight filter: keep pixels where red≥97 and blue≤227
    (removes skin/shadow noise that bleeds into the lip bounding box).
    Returns an (N, 3) float32 RGB array, or None if too few pixels survive.
    """
    if lips_roi_bgr is None or lips_roi_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(lips_roi_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    # Colorinsight filter: r>=97, b<=227
    mask = (rgb[:, 0] >= 97) & (rgb[:, 2] <= 227)
    filtered = rgb[mask]
    if len(filtered) < 10:
        return None
    # Random subsample for speed
    idx = np.random.choice(len(filtered), min(max_samples, len(filtered)), replace=False)
    return filtered[idx]


def lip_season_vote(lip_pixels):
    """
    Compute a soft season prior from lip pixel colours using L2 distance voting.

    For each sampled pixel, find the closest season reference colour (minimum L2
    over that season's 3 archetype vectors) and cast a vote. Votes are tallied
    and normalised to a probability distribution over the 4 base seasons.

    Credit: PSY222/Colorinsight — calc_dis() in functions.py
    Returns dict {season: 0-1} summing to 1, or uniform if no pixels.
    """
    seasons = list(LIP_SEASON_REFS.keys())
    if lip_pixels is None or len(lip_pixels) == 0:
        return {s: 0.25 for s in seasons}

    votes = {s: 0 for s in seasons}
    for pixel in lip_pixels:
        best_season = None
        best_dist = np.inf
        for season, refs in LIP_SEASON_REFS.items():
            # Minimum L2 distance to any of this season's 3 reference colours
            dists = np.linalg.norm(refs - pixel, axis=1)
            d = float(dists.min())
            if d < best_dist:
                best_dist = d
                best_season = season
        votes[best_season] += 1

    total = sum(votes.values()) or 1
    return {s: votes[s] / total for s in seasons}


def _dominant_color(region_bgr, k=3, min_pixels=20):
    """
    KMeans dominant colour for a BGR region.
    Returns (r, g, b) tuple or None if too few pixels.
    """
    if region_bgr is None or region_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) < min_pixels:
        return None
    k = min(k, len(pixels))
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    km.fit(pixels)
    labels, counts = np.unique(km.labels_, return_counts=True)
    dominant_idx = labels[np.argmax(counts)]
    c = km.cluster_centers_[dominant_idx]
    return (int(c[0]), int(c[1]), int(c[2]))


def _skin_mask(region_bgr):
    """HSV-based skin pixel mask for a BGR image."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 60], dtype=np.uint8)
    upper = np.array([25, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def _dominant_skin_color(region_bgr):
    """Dominant colour within the skin-coloured pixels of a region."""
    if region_bgr is None or region_bgr.size == 0:
        return None
    mask = _skin_mask(region_bgr)
    pixels = region_bgr[mask > 0]
    if len(pixels) < 30:
        # Fall back to full region
        return _dominant_color(region_bgr)
    rgb = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    k = min(3, len(rgb))
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    km.fit(rgb)
    labels, counts = np.unique(km.labels_, return_counts=True)
    c = km.cluster_centers_[labels[np.argmax(counts)]]
    return (int(c[0]), int(c[1]), int(c[2]))

# ---------------------------------------------------------------------------
# Metric computation (from raden-dosel palette.py)
# ---------------------------------------------------------------------------

def _cielab_distance(rgb1, rgb2):
    """Perceptual colour distance (CIELab Euclidean) between two RGB triples."""
    a = np.array([[[rgb1[0]/255, rgb1[1]/255, rgb1[2]/255]]], dtype=np.float64)
    b = np.array([[[rgb2[0]/255, rgb2[1]/255, rgb2[2]/255]]], dtype=np.float64)
    return float(np.linalg.norm(skcolor.rgb2lab(a) - skcolor.rgb2lab(b)))


def compute_subtone(lips_rgb):
    """S: 'warm' (1) if lips closer to peach, 'cold' (0) if closer to purple."""
    peach  = (255, 230, 182)
    purple = (210, 120, 180)
    if _cielab_distance(lips_rgb, peach) < _cielab_distance(lips_rgb, purple):
        return 1, _cielab_distance(lips_rgb, peach)  # warm
    return 0, _cielab_distance(lips_rgb, purple)     # cold


def compute_intensity(skin_rgb):
    """I: skin HSV saturation (raw 0-1, binarised at THRESH_INTENSITY)."""
    _, s, _ = _rgb_to_hsv_single(*skin_rgb)
    return s


def compute_value(skin_rgb, hair_rgb, eye_rgb):
    """V: mean HSV brightness of skin + hair (if available) + eyes."""
    values = []
    for rgb in [skin_rgb, eye_rgb]:
        if rgb:
            _, _, v = _rgb_to_hsv_single(*rgb)
            values.append(v)
    if hair_rgb:
        _, _, v = _rgb_to_hsv_single(*hair_rgb)
        values.append(v)
    return float(np.mean(values)) if values else 0.5


def compute_contrast(hair_rgb, eye_rgb):
    """C: |hair_V − eye_V| (None if no hair colour)."""
    if hair_rgb is None:
        return None
    _, _, hair_v = _rgb_to_hsv_single(*hair_rgb)
    _, _, eye_v  = _rgb_to_hsv_single(*eye_rgb)
    return abs(hair_v - eye_v)

# ---------------------------------------------------------------------------
# Season scoring (continuous, not just binary Hamming)
# ---------------------------------------------------------------------------

def season_scores(s_raw, i_raw, v_raw, c_raw):
    """
    Soft Hamming distance from each reference season.
    Returns dict {season: confidence} summing to 1.
    """
    # Continuous "agreement" with each reference bit
    bits = {
        "S": s_raw if isinstance(s_raw, float) else float(s_raw),
        "I": i_raw,
        "V": v_raw,
        "C": c_raw if c_raw is not None else 0.5,
    }

    def agree(value, target):
        """How much does `value` (0-1) agree with bit `target` (0 or 1)?"""
        return value if target == 1 else (1 - value)

    scores = {}
    for season, vec in REFERENCE_VECTORS.items():
        s_bit, i_bit, v_bit, c_bit = vec
        score = (agree(bits["S"], s_bit) *
                 agree(bits["I"], i_bit) *
                 agree(bits["V"], v_bit) *
                 agree(bits["C"], c_bit))
        scores[season] = score

    total = sum(scores.values())
    if total == 0:
        return {s: 0.25 for s in REFERENCE_VECTORS}
    return {s: v / total for s, v in scores.items()}

# ---------------------------------------------------------------------------
# Subtype classification (12-class armocromia)
# ---------------------------------------------------------------------------

SUBTYPE_MAP = {
    "spring": [
        ("primavera_bright", lambda i, v, c: i * (c if c else 0.5)),
        ("primavera_light",  lambda i, v, c: (1 - i) * v),
        ("primavera_warm",   lambda i, v, c: i * (1 - (c if c else 0.5))),
    ],
    "summer": [
        ("estate_cool",  lambda i, v, c: (1 - i) * (1 - (c if c else 0.5))),
        ("estate_light", lambda i, v, c: (1 - i) * v),
        ("estate_soft",  lambda i, v, c: (1 - i) * (1 - v)),
    ],
    "autumn": [
        ("autunno_deep", lambda i, v, c: (1 - v)),
        ("autunno_soft", lambda i, v, c: (1 - i) * (1 - v + 0.3)),
        ("autunno_warm", lambda i, v, c: i * (1 - v + 0.5)),
    ],
    "winter": [
        ("inverno_bright", lambda i, v, c: i * (c if c else 0.5)),
        ("inverno_cool",   lambda i, v, c: (1 - i) * (1 - v + 0.5)),
        ("inverno_deep",   lambda i, v, c: (1 - v)),
    ],
}


def full_12class_scores(season_conf, i_raw, v_raw, c_raw):
    """
    Combine season confidence with subtype scoring to produce 12 confidence values.
    """
    results = {}
    for season, season_weight in season_conf.items():
        subtypes = SUBTYPE_MAP[season]
        raw = [(cid, fn(i_raw, v_raw, c_raw)) for cid, fn in subtypes]
        total = sum(r[1] for r in raw) or 1
        for cid, r in raw:
            results[cid] = season_weight * (r / total)
    return results

# ---------------------------------------------------------------------------
# Face region extraction (OpenCV-based, no model weights needed)
# ---------------------------------------------------------------------------

def _detect_face(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return faces[0] if len(faces) > 0 else None


def _detect_eyes(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
    return eyes


def extract_facial_colors(img_bgr):
    """
    Returns (skin_rgb, hair_rgb, eye_rgb, lips_rgb) from the image.
    Any value can be None if the region can't be found.
    """
    face = _detect_face(img_bgr)
    if face is None:
        # No face detected — use full image as a rough fallback
        h, w = img_bgr.shape[:2]
        skin_rgb = _dominant_skin_color(img_bgr[h//4:3*h//4, w//4:3*w//4])
        return skin_rgb, None, None, None, None

    fx, fy, fw, fh = face
    face_roi = img_bgr[fy:fy+fh, fx:fx+fw]
    h, w = face_roi.shape[:2]

    # Skin — middle 60% of face (cheek area), using skin mask
    skin_roi = face_roi[int(h*0.3):int(h*0.8), int(w*0.15):int(w*0.85)]
    skin_rgb = _dominant_skin_color(skin_roi) or _dominant_color(skin_roi)

    # Hair — region above face (or forehead top if no area above)
    hair_y1 = max(0, fy - int(fh * 0.5))
    hair_y2 = max(0, fy - int(fh * 0.05))
    hair_roi = img_bgr[hair_y1:hair_y2, fx:fx+fw] if hair_y2 > hair_y1 else None
    if hair_roi is None or hair_roi.shape[0] < 5:
        # Fall back to top 12% of face
        hair_roi = face_roi[:int(h * 0.12), :]
    hair_rgb = _dominant_color(hair_roi)

    # Eyes — use eye cascade, else estimate from face geometry
    eye_rgb = None
    eyes = _detect_eyes(face_roi)
    if len(eyes) > 0:
        ex, ey, ew, eh = sorted(eyes, key=lambda e: e[2]*e[3])[-1]
        eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
        eye_rgb = _dominant_color(eye_roi)
    if eye_rgb is None:
        eye_roi = face_roi[int(h*0.25):int(h*0.45), int(w*0.2):int(w*0.8)]
        eye_rgb = _dominant_color(eye_roi)

    # Lips — bottom 20-30% of face, center strip
    lips_roi = face_roi[int(h*0.72):int(h*0.92), int(w*0.25):int(w*0.75)]
    lips_rgb = _dominant_color(lips_roi)
    # Colorinsight: also extract filtered pixel samples for L2 season voting
    lip_pixels = _get_lip_pixels(lips_roi)

    return skin_rgb, hair_rgb, eye_rgb, lips_rgb, lip_pixels

# ---------------------------------------------------------------------------
# Main public interface
# ---------------------------------------------------------------------------

def analyse(img_bgr):
    """
    Run the full SIVC analysis on a BGR image.

    Returns:
        dict with keys:
          'season_scores'   — {spring/summer/autumn/winter: 0-1}
          'class_scores'    — {class_id: 0-1} for all 12 subtypes
          'metrics'         — raw SIVC values
          'dominant_colors' — {skin/hair/eye/lips: (r,g,b) or None}
          'is_real'         — always True (real analysis, not mock)
    """
    skin, hair, eye, lips, lip_pixels = extract_facial_colors(img_bgr)

    # Fallback defaults when regions unavailable
    skin  = skin  or (220, 185, 160)
    eye   = eye   or (90, 70, 50)
    lips  = lips  or (200, 130, 120)

    s_raw, _ = compute_subtone(lips)
    i_raw    = compute_intensity(skin)
    v_raw    = compute_value(skin, hair, eye)
    c_raw    = compute_contrast(hair, eye)

    s_cont = float(s_raw)                  # 0 or 1 (discrete for subtone)
    c_cont = c_raw if c_raw is not None else 0.5

    # --- SIVC season scores (70% weight) ---
    sivc_seasons = season_scores(s_cont, i_raw, v_raw, c_cont)

    # --- Colorinsight lip L2 vote (30% weight) ---
    lip_vote = lip_season_vote(lip_pixels)
    lip_signal_used = lip_pixels is not None and len(lip_pixels) >= 10

    # Blend: 70% SIVC + 30% Colorinsight lip vote
    w_sivc = 1.0 - LIP_VOTE_WEIGHT if lip_signal_used else 1.0
    w_lip  = LIP_VOTE_WEIGHT        if lip_signal_used else 0.0
    seasons = {
        s: w_sivc * sivc_seasons[s] + w_lip * lip_vote[s]
        for s in sivc_seasons
    }
    # Re-normalise after blend
    total_s = sum(seasons.values()) or 1
    seasons = {s: v / total_s for s, v in seasons.items()}

    classes  = full_12class_scores(seasons, i_raw, v_raw, c_cont)

    # Normalise class scores to sum to 1
    total = sum(classes.values()) or 1
    classes = {k: round(v / total, 4) for k, v in classes.items()}

    return {
        "season_scores": seasons,
        "class_scores": classes,
        "metrics": {
            "subtone": "warm" if s_raw == 1 else "cold",
            "intensity": round(i_raw, 3),
            "value": round(v_raw, 3),
            "contrast": round(c_cont, 3),
            "lip_vote": {s: round(lip_vote[s], 3) for s in lip_vote},
        },
        "dominant_colors": {
            "skin": skin,
            "hair": hair,
            "eye": eye,
            "lips": lips,
        },
        "is_real": True,
    }
