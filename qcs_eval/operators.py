"""
Stress operators = the pipeline's own implementations, copied verbatim from
uw-clarity/src/postprocess.py (_dehaze_fast, _usm) and uw-clarity/src/preprocess.py (_clahe_lab).
Do not edit the three functions below; they must stay identical to the pipeline.

Units are the pipeline's parameter units:
  usm    : sharpen_usm amount (pipeline bound 0-0.6; blur sigma = 1 + 2*amount)
  dehaze : dehaze strength    (pipeline bound 0-0.6; transmission floor 0.6)
  clahe  : pre clahe_clip     (pipeline bound 0.005-0.03; cv2 clipLimit = clip*100), tile 8
Grids extend well beyond the pipeline bounds on purpose (stress test). Index 0 = identity.

Previous version (v0.3) used re-implementations from run_confound_control_experiment.py, which differ:
dehaze used a DCP airlight model, USM used a fixed sigma=1, CLAHE used clip*255 (2.55x stronger).
"""
import numpy as np
import cv2 as cv


# ---- verbatim from postprocess.py ----
def _dehaze_fast(bgr, strength=0.25):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    dark = cv.erode(gray, np.ones((15, 15), np.uint8))
    t = 1 - (gray.astype(np.float32) - dark.astype(np.float32)) / 255.0
    t = np.clip(1 - strength * (1 - t), 0.6, 1.0)
    t3 = cv.merge([t, t, t])
    return (bgr.astype(np.float32) / np.maximum(t3, 1e-3)).clip(0, 255).astype(np.uint8)


def _usm(bgr, amount=0.35):
    if amount <= 1e-6: return bgr
    blur = cv.GaussianBlur(bgr, (0, 0), sigmaX=1.0 + 2.0 * amount)
    return cv.addWeighted(bgr, 1 + amount, blur, -amount, 0)


# ---- verbatim from preprocess.py ----
def _clahe_lab(bgr, clip=0.01, tile=8):
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB); L, A, B = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=float(clip * 100.0), tileGridSize=(tile, tile))
    L2 = clahe.apply(L)
    return cv.cvtColor(cv.merge([L2, A, B]), cv.COLOR_LAB2BGR)


# ---- wrappers: strength 0 must return the input unchanged ----
def usm(img, s):
    return _usm(img, s)


def dehaze(img, s):
    return img if s <= 0 else _dehaze_fast(img, s)


def clahe(img, s):
    return img if s <= 0 else _clahe_lab(img, s, 8)


OPERATORS = {
    "usm":    (usm,    [0, 0.18, 0.35, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0]),   # LLaVA ~0.18, default 0.35, bound 0.6
    "dehaze": (dehaze, [0, 0.15, 0.25, 0.4, 0.6, 1.0, 2.0]),             # LLaVA ~0.15, default 0.25, bound 0.6
    "clahe":  (clahe,  [0, 0.006, 0.012, 0.018, 0.03, 0.06, 0.12]),      # LLaVA range 0.006-0.018, bound 0.03
}
