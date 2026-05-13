"""
Download a subset of the Deep Armocromia dataset from Google Drive.

Uses HTTP Range requests + a seekable file-like wrapper to let Python's
built-in zipfile module read and decrypt individual entries on-demand,
without downloading the full 1.6 GB ZIP.

Usage:
    python3 download_dataset.py --password YOUR_ZIP_PASSWORD
    python3 download_dataset.py --password YOUR_ZIP_PASSWORD --train 50 --test 15
"""

import os, struct, zlib, argparse, zipfile, io, time, urllib.request, json, csv, signal, shutil
from pathlib import Path

# ── Graceful SIGTERM / cancel cleanup ─────────────────────────────────────
# Tracks the class-partition directory currently being populated.
# Set to a Path object just before we start writing files for each
# (cls, sub, partition) combination.  The SIGTERM handler deletes it so
# that no partial / incomplete class folder is left on disk after a cancel.
_current_class_dir: "Path | None" = None


def _sigterm_handler(signum, frame):
    global _current_class_dir
    if _current_class_dir is not None and _current_class_dir.exists():
        try:
            shutil.rmtree(_current_class_dir)
            print(f"\n⚠ Cancelled — removed partial directory: {_current_class_dir}",
                  flush=True)
        except Exception as exc:
            print(f"\n⚠ Cancelled — could not clean up {_current_class_dir}: {exc}",
                  flush=True)
    else:
        print("\n⚠ Cancelled — no partial directory to clean up", flush=True)
    raise SystemExit(1)


signal.signal(signal.SIGTERM, _sigterm_handler)

# ── Replit connector auth ──────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_TOKEN_FILE  = os.path.join(_BASE, ".gdrive_token")
_EXPIRY_FILE = os.path.join(_BASE, ".gdrive_token_expiry")

def _token_expires_at() -> "float | None":
    """Return the token expiry as a UTC unix timestamp, or None if unknown."""
    if not os.path.exists(_EXPIRY_FILE):
        return None
    try:
        raw = open(_EXPIRY_FILE).read().strip()
        # ISO-8601: 2026-05-11T09:35:40.499Z
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None

def get_access_token():
    """Read Google Drive access token; fail fast with a helpful message if expired."""
    tok = ""
    if os.path.exists(_TOKEN_FILE):
        tok = open(_TOKEN_FILE).read().strip()
    if not tok:
        tok = os.environ.get("GDRIVE_ACCESS_TOKEN", "")
    if not tok:
        raise RuntimeError(
            "No Google Drive access token found. "
            "Ask the assistant to refresh the Drive authorisation."
        )
    # Check expiry so we get a clear error instead of a cryptic 401 mid-download.
    expiry = _token_expires_at()
    if expiry is not None and time.time() > expiry - 60:
        raise RuntimeError(
            "Google Drive access token has expired. "
            "Ask the assistant to refresh the Drive authorisation "
            "(type: 'refresh the Google Drive token')."
        )
    return tok

def verify_token_live(token: str) -> None:
    """
    Make a cheap Drive API call to confirm the token is accepted by Google
    right now.  Raises RuntimeError with a clear user-facing message on 401.
    This is called once at the very start of main() so any auth problem is
    surfaced before any bytes are downloaded.
    """
    import urllib.error as _uerr
    url = "https://www.googleapis.com/drive/v3/about?fields=kind"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()   # discard; just need a 200
    except _uerr.HTTPError as e:
        if e.code == 401:
            raise RuntimeError(
                "Google Drive token rejected (401). "
                "Ask the assistant to refresh the Drive authorisation "
                "(type: 'refresh the Google Drive token')."
            ) from e
        raise


# ── Drive helpers ──────────────────────────────────────────────────────────
def _drive_get(url, token, extra_headers=None):
    h = {"Authorization": f"Bearer {token}"}
    if extra_headers:
        h.update(extra_headers)
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=90)

def get_file_size(file_id, token):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=size"
    with _drive_get(url, token) as r:
        return int(json.loads(r.read())["size"])

def range_download(file_id, token, start, end, retries=4, delay=2.0):
    """Download byte range [start, end] inclusive from a Drive file, with retries."""
    import urllib.error as _uerr
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    last_err = None
    for attempt in range(retries):
        try:
            with _drive_get(url, token, {"Range": f"bytes={start}-{end}"}) as r:
                return r.read()
        except _uerr.HTTPError as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                if e.code == 401:
                    # Force-expire the cache so get_fresh_token re-reads the file
                    _token_cache["read_at"] = 0.0
                token = get_fresh_token()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                token = get_fresh_token()
    raise RuntimeError(f"range_download failed after {retries} attempts: {last_err}")

# ── Seekable Drive file wrapper ────────────────────────────────────────────
class DriveRangeFile:
    """
    Seekable read-only file-like object backed by HTTP Range requests.
    Caches chunks of CHUNK_SIZE bytes to reduce the number of requests.
    Python's zipfile.ZipFile will use this to seek around and read the ZIP
    central directory + individual entries without downloading the whole file.
    """
    CHUNK_SIZE = 256 * 1024   # 256 KB per chunk

    def __init__(self, file_id, token_fn, total_size):
        self.file_id   = file_id
        self._token_fn = token_fn   # callable returning current token
        self.total_size = total_size
        self._pos = 0
        self._cache: dict[int, bytes] = {}   # chunk_index → bytes

    # ── internal ──
    def _get_chunk(self, idx: int) -> bytes:
        if idx in self._cache:
            return self._cache[idx]
        start = idx * self.CHUNK_SIZE
        end   = min(start + self.CHUNK_SIZE, self.total_size) - 1
        data  = range_download(self.file_id, self._token_fn(), start, end)
        # Evict old chunks if cache grows large (>= 50 chunks = 12.5 MB)
        if len(self._cache) >= 50:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[idx] = data
        return data

    # ── file interface ──
    def read(self, n=-1) -> bytes:
        if n == -1 or n is None:
            n = self.total_size - self._pos
        if n <= 0 or self._pos >= self.total_size:
            return b""

        result = bytearray()
        remaining = min(n, self.total_size - self._pos)
        pos = self._pos
        while remaining > 0:
            idx    = pos // self.CHUNK_SIZE
            offset = pos  % self.CHUNK_SIZE
            chunk  = self._get_chunk(idx)
            piece  = chunk[offset : offset + remaining]
            result.extend(piece)
            pos       += len(piece)
            remaining -= len(piece)
        self._pos = pos
        return bytes(result)

    def readline(self) -> bytes:
        # Required by zipfile on some paths
        data = bytearray()
        while True:
            b = self.read(1)
            if not b:
                break
            data.extend(b)
            if b == b"\n":
                break
        return bytes(data)

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self.total_size + pos
        self._pos = max(0, min(self._pos, self.total_size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False


# ── Dataset configuration ──────────────────────────────────────────────────
RGB_ZIP_ID  = "1N2GI5Q28YUXO8JRn4LI-M7i44578aieH"   # RGB.zip (1595 MB)
ANNOTATIONS = Path("annotations.csv")
DATASET_DIR = Path("dataset/images")

# ── Token refresh ──────────────────────────────────────────────────────────
_token_cache: dict = {"value": None, "read_at": 0.0}

def get_fresh_token() -> str:
    now = time.time()
    if now - _token_cache["read_at"] > 300 or not _token_cache["value"]:
        _token_cache["value"] = get_access_token()
        _token_cache["read_at"] = now
    return _token_cache["value"]

# ── Annotations ────────────────────────────────────────────────────────────
def load_annotations():
    """Parse annotations.csv using proper CSV reader to handle commas in filenames."""
    rows = []
    with open(ANNOTATIONS, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls       = row.get("class", "").strip()
            sub       = row.get("sub_class", "").strip()
            partition = row.get("partition", "").strip()
            path_rgb  = row.get("path_rgb_original", "").strip()
            if cls and sub and partition and path_rgb:
                rows.append((cls, sub, partition, path_rgb))
    return rows


def select_subset(rows, train_n, test_n):
    """Return {(cls, sub): {partition: [zip_path, ...]}} picking at most train_n/test_n per class."""
    from collections import defaultdict
    by_class = defaultdict(lambda: {"train": [], "test": []})
    for cls, sub, partition, path in rows:
        if partition in ("train", "test"):
            by_class[(cls, sub)][partition].append(path)

    selected = {}
    for (cls, sub) in sorted(by_class):
        splits = by_class[(cls, sub)]
        selected[(cls, sub)] = {
            "train": splits["train"][:train_n],
            "test":  splits["test"][:test_n],
        }
        t = len(selected[(cls, sub)]["train"])
        v = len(selected[(cls, sub)]["test"])
        print(f"  {cls}/{sub:<12} train={t:<4} test={v}")
    return selected


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", required=True,
                        help="ZIP password from the Deep Armocromia request form")
    parser.add_argument("--train", type=int, default=9999)
    parser.add_argument("--test",  type=int, default=9999)
    args = parser.parse_args()
    pwd = args.password.encode()

    print("=== Deep Armocromia Dataset Downloader ===\n")
    token = get_fresh_token()
    verify_token_live(token)
    print("✓ Google Drive auth OK")

    print("Getting ZIP file size…")
    total_size = get_file_size(RGB_ZIP_ID, token)
    print(f"  RGB.zip = {total_size/1024/1024:.1f} MB")

    print("Opening ZIP via Range-request file wrapper…")
    drive_file = DriveRangeFile(RGB_ZIP_ID, get_fresh_token, total_size)
    zf = zipfile.ZipFile(drive_file, "r")   # reads EOCD + central directory lazily
    names = set(zf.namelist())
    print(f"  ✓ Opened ZIP: {len(names)} entries")

    print("\nLoading annotations…")
    rows = load_annotations()
    print(f"  {len(rows)} labeled images")

    print(f"\nSelecting subset ({args.train} train + {args.test} test per class):")
    selected = select_subset(rows, args.train, args.test)

    total_needed = sum(len(v["train"]) + len(v["test"]) for v in selected.values())
    print(f"\nDownloading {total_needed} images…")

    # Create output dirs: dataset/images/{partition}/{season}/{subtype}/
    for (cls, sub) in selected:
        (DATASET_DIR / "train" / cls / sub).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "test"  / cls / sub).mkdir(parents=True, exist_ok=True)

    downloaded = skipped = errors = 0

    global _current_class_dir
    for (cls, sub), splits in selected.items():
        for partition, paths in splits.items():
            # Track current directory so the SIGTERM handler can clean it up.
            # Also emit a dedicated line so the parent process (app.py) can
            # reliably follow which directory is active without relying on
            # sparse per-file progress lines.
            _current_class_dir = DATASET_DIR / partition / cls / sub
            print(f"[ACTIVE_DIR] {_current_class_dir}", flush=True)
            for zip_path in paths:
                # zip_path = "RGB/train/autunno/deep/10306.jpg"
                filename = zip_path.split("/")[-1]
                out_path = DATASET_DIR / partition / cls / sub / filename

                if out_path.exists():
                    skipped += 1
                    downloaded += 1
                    continue

                if zip_path not in names:
                    print(f"  ✗ Not in ZIP: {zip_path}")
                    errors += 1
                    continue

                try:
                    img_bytes = zf.read(zip_path, pwd=pwd)
                    out_path.write_bytes(img_bytes)
                    downloaded += 1
                    if downloaded % 25 == 0 or downloaded <= 3:
                        print(f"  [{downloaded}/{total_needed}] {cls}/{sub}/{partition}  "
                              f"{len(img_bytes)//1024} KB  ({filename})")
                except Exception as e:
                    print(f"  ✗ {zip_path}: {e}")
                    errors += 1
            # Partition fully downloaded — clear the tracker so a late SIGTERM
            # does not delete a completed directory.
            _current_class_dir = None

    zf.close()
    print(f"\n{'='*50}")
    print(f"✓ {downloaded} images  ({skipped} already existed)  {errors} errors")
    print(f"Saved to {DATASET_DIR}/")
    if errors == 0:
        print("\nRun  python3 train.py  to retrain with real data.")


if __name__ == "__main__":
    main()
