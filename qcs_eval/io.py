"""Image listing, stem matching, per-folder feature cache with file fingerprints."""
import os, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder, on_duplicate="raise"):
    """folder: a path, or a list/tuple of paths whose images are combined (e.g. Input/Val + Input/Test).
    on_duplicate="raise": error if the same (lower-cased) stem appears twice (default; scoring code).
    on_duplicate="keep" : return every file (audit code reports the duplicates itself)."""
    dirs = folder if isinstance(folder, (list, tuple)) else [folder]
    files = []
    for d in dirs:
        files += [p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files = sorted(files, key=lambda p: p.name.lower())
    if on_duplicate == "keep":
        return files
    seen = {}
    for p in files:
        s = p.stem.lower()
        if s in seen:
            raise RuntimeError(f"duplicate stem '{s}' in {seen[s].parent} and {p.parent}")
        seen[s] = p
    return files


def stem(p):
    return Path(p).stem.lower()


def file_sha(path, nbytes=None):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read() if nbytes is None else fh.read(nbytes))
    return h.hexdigest()[:16]


def read_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def _score_one(args):
    path, feature_set, extras = args
    from .features import compute_features
    try:
        img = read_bgr(path)
    except IOError as e:
        return dict(Filename=Path(path).name, stem=stem(path), error=str(e))
    d = compute_features(img, feature_set, extras)
    d.update(Filename=Path(path).name, stem=stem(path), sha=file_sha(path))
    return d


def score_folder(folder, label, cache_dir, feature_set="legacy_r2", extras=True, workers=None, stems=None):
    """Score every image (or only `stems`) in folder; cache to cache_dir/<label>__<feature_set>.csv.
    Cache rows are reused only if the file sha matches (detects changed images)."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(cache_dir, exist_ok=True)
    cpath = Path(cache_dir) / f"{label}__{feature_set}.csv"
    files = list_images(folder)
    if stems is not None:
        stems = set(stems); files = [f for f in files if stem(f) in stems]
    cached = pd.read_csv(cpath) if cpath.exists() else pd.DataFrame(columns=["stem", "sha"])
    cmap = dict(zip(cached["stem"], cached["sha"]))
    todo = [f for f in files if cmap.get(stem(f)) != file_sha(f)]
    if todo:
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(workers) as ex:
            new = list(ex.map(_score_one, [(str(f), feature_set, extras) for f in todo], chunksize=16))
        new = pd.DataFrame(new)
        keep = cached[~cached["stem"].isin(new["stem"])]
        cached = pd.concat([keep, new], ignore_index=True).sort_values("stem")
        cached.to_csv(cpath, index=False)
    out = cached[cached["stem"].isin({stem(f) for f in files})].copy()
    out["Configuration"] = label
    return out
