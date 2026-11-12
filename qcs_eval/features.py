"""
Feature implementations.

LEGACY_R2 = exact port of the functions used for the WACV Round-2 Table 1
(run_corrected_qcs_full_reanalysis.py: uiqm/uciqe/ag/ten). Do NOT edit these:
they exist so the Round-2 numbers can be reproduced bit-for-bit.

Known deviations from the published definitions (reviewer-visible, see README):
  * uiqm_legacy: UISM uses EME of an unsharp-masked channel (not Sobel edge map x channel),
    UIConM uses plain EME (not PLIP log-AMEE)  -> values ~19-32 instead of typical ~2-5.
  * uciqe_r2: CIE-Lab float scale, contrast = top/bottom 1% mean range, saturation = C/L
    -> values ~21-31 instead of typical ~0.4-0.7; C/L is unstable for very dark pixels.
  * uciqe_r1_bug: the Round-1 uint8-Lab implementation (kept only for audit).
"""
import numpy as np
import cv2

EPS = 1e-8
FEATURES = ["UIQM", "UCIQE", "AG", "Tenengrad"]


# ---------------- LEGACY_R2 (frozen) ----------------
def _eme(img, block=8):
    H, W = img.shape[:2]
    h, w = (H // block) * block, (W // block) * block
    if h == 0 or w == 0:
        return 0.0
    x = img[:h, :w].reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(-1, block, block)
    mn = x.min((1, 2)).astype(float)
    mx = x.max((1, 2)).astype(float)
    return float(np.mean(20 * np.log10(np.clip((mx + 1e-6) / (mn + 1e-6 + EPS), EPS, None))))


def _unsharp(g):
    b = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.addWeighted(g, 2.0, b, -1.0, 0)


def uiqm_legacy(img):
    def tm(x, a=.1):
        x = np.sort(x.ravel()); n = len(x); c = int(np.floor(a * n))
        return float(np.mean(x[c:n - c])) if 2 * c < n else float(np.mean(x))
    R = img[:, :, 2].astype(float); G = img[:, :, 1].astype(float); B = img[:, :, 0].astype(float)
    RG = R - G; YB = .5 * (R + G) - B
    mu = np.sqrt(tm(RG) ** 2 + tm(YB) ** 2); sig = np.sqrt(np.std(RG) ** 2 + np.std(YB) ** 2)
    uicm = .1586 * sig - .0268 * mu
    uism = .299 * _eme(_unsharp(R)) + .587 * _eme(_unsharp(G)) + .114 * _eme(_unsharp(B))
    uiconm = _eme(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return float(.0282 * uicm + .2953 * uism + 3.5753 * uiconm)


def uciqe_r2(img):
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).astype(float)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    C = np.sqrt(a * a + b * b)
    v = np.sort(L.ravel()); k = max(1, int(round(.01 * len(v))))
    con = float(np.mean(v[-k:]) - np.mean(v[:k]))
    s = np.zeros_like(C); m = L > 1e-12; s[m] = C[m] / L[m]
    return float(.468 * np.std(C) + .2745 * con + .2576 * np.mean(s))


def ag(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    dx = np.diff(g, 1, 1); dy = np.diff(g, 1, 0)
    return float(np.mean(np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)))


def tenengrad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0); sy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
    return float(np.mean(sx * sx + sy * sy))


# ---------------- audit only ----------------
def uciqe_r1_bug(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    L = lab[:, :, 0] / 255.0
    con_l = float(np.percentile(L, 95) - np.percentile(L, 5))
    return 0.4680 * np.std(chroma) + 0.2745 * con_l + 0.2576 * np.mean(chroma)


def dark_pixel_fraction(img, thr=5.0):
    """Fraction of pixels with Lab L < thr (L in 0..100). Diagnostic for C/L blow-up in uciqe_r2."""
    L = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)[:, :, 0]
    return float(np.mean(L < thr))


FEATURE_SETS = {
    "legacy_r2": {"UIQM": uiqm_legacy, "UCIQE": uciqe_r2, "AG": ag, "Tenengrad": tenengrad},
}


def compute_features(img, feature_set="legacy_r2", extras=False):
    fs = FEATURE_SETS[feature_set]
    out = {k: f(img) for k, f in fs.items()}
    if extras:
        out["UCIQE_r1_bug"] = float(uciqe_r1_bug(img))
        out["dark_frac_L5"] = dark_pixel_fraction(img)
        out["H"], out["W"] = img.shape[:2]
    return out
