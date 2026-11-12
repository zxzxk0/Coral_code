"""
QCS-v2 candidates A-D (ablation chain; each step adds one element).

Common to all candidates
  - resolution: image resized so the short side is 384 px (INTER_AREA down / INTER_CUBIC up)
  - features use published definitions (UIQM: Panetta et al. 2016 incl. Sobel-EME and PLIP logAMEE;
    UCIQE: Yang & Sowmya 2015 on CIELab with bounded saturation C/sqrt(C^2+L^2))
  - each feature -> robust z = (x - median) / (1.4826 * MAD) using a FROZEN calibration JSON
  - subscores = mean of their robust z's; final score = unweighted mean of subscores
  - continuous output, no clipping
  - NOTE on the ablation: subscores are equally weighted, so adding a subscore (B->C->D) also
    re-normalises the weights of the existing ones (1/2 -> 1/3 -> 1/4). A step therefore measures
    "add X with equal subscore weighting", not X in isolation.

  A : color(UIQM, UCIQE) + detail(AG, Tenengrad)
  B : color(UIQM, UCIQE) + detail(AG)                              -> remove sharpness redundancy
  C : B + artifact(-noise_flat, -clip_frac)                         -> penalise over-processing
  D : C + physics(-residual_backscatter)                            -> formation-model-inspired proxy
      residual_backscatter = mean dark channel / mean intensity. Under I = J t + B(1-t),
      pixels with J ~ 0 give I ~ B(1-t), so under the dark-channel prior the dark channel is a
      proxy for the backscatter left in the output. It is an observable proxy, NOT an estimate of
      a physical parameter (no depth, no B or t is recovered).
  D2: as D, but the proxy is computed on a sigma=3 low-pass image (A2a showed the full-band
      proxy rewards USM undershoot); this reduces, not removes, that sensitivity.
"""
import json, hashlib
import numpy as np
import cv2

SHORT_SIDE = 384
V2_FEATURES = ["UIQM_std", "UCIQE_std", "AG_r", "Ten_r", "noise_flat", "clip_frac", "resid_bs", "resid_bs_lp"]
# variance-stabilising transforms applied before robust z (gradient energies are heavy-tailed)
TRANSFORM = {"UIQM_std": "id", "UCIQE_std": "id", "AG_r": "log", "Ten_r": "log", "noise_flat": "log1p",
             "clip_frac": "sqrt", "resid_bs": "id", "resid_bs_lp": "id"}
Z_BOUND = 3.0   # soft bound: z -> B*tanh(z/B); continuous and strictly monotone (no ties), limits single-feature dominance
SIGN = {"UIQM_std": 1, "UCIQE_std": 1, "AG_r": 1, "Ten_r": 1, "noise_flat": -1, "clip_frac": -1, "resid_bs": -1, "resid_bs_lp": -1}
CANDIDATES = {
    "v2_A": {"color": ["UIQM_std", "UCIQE_std"], "detail": ["AG_r", "Ten_r"]},
    "v2_B": {"color": ["UIQM_std", "UCIQE_std"], "detail": ["AG_r"]},
    "v2_C": {"color": ["UIQM_std", "UCIQE_std"], "detail": ["AG_r"], "artifact": ["noise_flat", "clip_frac"]},
    "v2_D": {"color": ["UIQM_std", "UCIQE_std"], "detail": ["AG_r"], "artifact": ["noise_flat", "clip_frac"],
             "physics": ["resid_bs"]},
    # D2: physics term measured on a low-pass (sigma=3) image; reduces (does not eliminate) sensitivity to
    #     high-frequency sharpening undershoot
    "v2_D2": {"color": ["UIQM_std", "UCIQE_std"], "detail": ["AG_r"], "artifact": ["noise_flat", "clip_frac"],
              "physics": ["resid_bs_lp"]},
}


def resize_short(img, s=SHORT_SIDE):
    h, w = img.shape[:2]
    f = s / min(h, w)
    if abs(f - 1) < 1e-9:
        return img
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                      interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


# ---------------- UIQM (Panetta et al. 2016) ----------------
def _trimmed(x, a=0.1):
    x = np.sort(x.ravel()); n = len(x); k = int(np.ceil(a * n)); j = int(np.floor(a * n))
    return float(x[k:n - j].mean())


def _uicm(rgb):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    RG = R - G; YB = (R + G) / 2 - B
    mrg, myb = _trimmed(RG), _trimmed(YB)
    srg = float(((RG - mrg) ** 2).mean()); syb = float(((YB - myb) ** 2).mean())
    return -0.0268 * np.sqrt(mrg ** 2 + myb ** 2) + 0.1586 * np.sqrt(srg + syb)


def _blocks(x, ws):
    h, w = (x.shape[0] // ws) * ws, (x.shape[1] // ws) * ws
    x = x[:h, :w]
    if x.ndim == 2:
        return x.reshape(h // ws, ws, w // ws, ws).transpose(0, 2, 1, 3).reshape(-1, ws * ws)
    return x.reshape(h // ws, ws, w // ws, ws, x.shape[2]).transpose(0, 2, 1, 3, 4).reshape(-1, ws * ws * x.shape[2])


def _eme(x, ws=10):
    b = _blocks(x, ws); mx, mn = b.max(1), b.min(1)
    ok = (mn > 0) & (mx > 0)
    return float(2.0 / len(b) * np.log(mx[ok] / mn[ok]).sum()) if len(b) else 0.0


def _sobel_mag(ch):
    gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
    m = np.hypot(gx, gy); mx = m.max()
    return m * (255.0 / mx) if mx > 0 else m


def _uism(rgb):
    return sum(l * _eme(_sobel_mag(rgb[..., c]) * rgb[..., c]) for c, l in zip(range(3), (0.299, 0.587, 0.114)))


def _uiconm(rgb, ws=10):
    b = _blocks(rgb, ws); mx, mn = b.max(1), b.min(1)
    top, bot = mx - mn, mx + mn
    ok = (top > 0) & (bot > 0)
    r = top[ok] / bot[ok]
    return float(1.0 / len(b) * (r * np.log(r)).sum()) if len(b) else 0.0   # logAMEE, alpha=1


def uiqm_std(bgr):
    rgb = bgr[..., ::-1].astype(np.float64)
    return float(0.0282 * _uicm(rgb) + 0.2953 * _uism(rgb) + 3.5753 * _uiconm(rgb))


# ---------------- UCIQE (Yang & Sowmya 2015), bounded saturation ----------------
def uciqe_std(bgr):
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[..., 0] / 100.0
    C = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2) / 100.0
    v = np.sort(L.ravel()); k = max(1, int(round(0.01 * len(v))))
    con = v[-k:].mean() - v[:k].mean()
    sat = C / np.sqrt(C ** 2 + L ** 2 + 1e-12)
    return float(0.4680 * C.std() + 0.2745 * con + 0.2576 * sat.mean())


# ---------------- detail / artifact / physics ----------------
def ag(gray):
    dx = np.diff(gray, 1, 1); dy = np.diff(gray, 1, 0)
    return float(np.mean(np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)))


def tenengrad(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return float(np.mean(gx * gx + gy * gy))


def noise_flat(gray, struct_sigma=3.0, hf_sigma=1.0, frac=0.30):
    """Noise/ringing level in structurally flat regions.
    The flat mask comes from the gradient of a STRONGLY blurred luminance (sigma=3 px at 384 px short side),
    so high-frequency energy added by sharpening/noise cannot move pixels out of the mask.
    Measured quantity: robust std (1.4826*MAD) of the high-frequency residual gray - G_1(gray) inside the mask."""
    sm = cv2.GaussianBlur(gray, (0, 0), struct_sigma)
    gx = cv2.Sobel(sm, cv2.CV_64F, 1, 0); gy = cv2.Sobel(sm, cv2.CV_64F, 0, 1)
    g = np.hypot(gx, gy)
    m = g <= np.percentile(g, frac * 100)
    hf = gray - cv2.GaussianBlur(gray, (0, 0), hf_sigma)
    x = hf[m]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 0.0


def clip_frac(bgr):
    return float(((bgr <= 1) | (bgr >= 254)).any(axis=2).mean())


def resid_bs(bgr):
    dark = cv2.erode(bgr.min(axis=2), np.ones((15, 15), np.uint8)).astype(np.float64)
    return float(dark.mean() / (bgr.astype(np.float64).mean() + 1e-6))


def resid_bs_lp(bgr, sigma=3.0):
    """residual-backscatter proxy on a low-pass (sigma=3) image. Attenuates high-frequency overshoot/undershoot
    from sharpening before the dark channel is taken (A2a: the full-band proxy rewards USM). Reduces, does not
    eliminate, that sensitivity."""
    lp = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return resid_bs(lp)


def compute_v2_features(bgr):
    im = resize_short(bgr)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return {"UIQM_std": uiqm_std(im), "UCIQE_std": uciqe_std(im), "AG_r": ag(gray), "Ten_r": tenengrad(gray),
            "noise_flat": noise_flat(gray), "clip_frac": clip_frac(im), "resid_bs": resid_bs(im),
            "resid_bs_lp": resid_bs_lp(im)}


def _tf(f, x):
    t = TRANSFORM[f]
    x = np.asarray(x, float)
    if t == "log":   return np.log(np.maximum(x, 1e-9))
    if t == "log1p": return np.log1p(np.maximum(x, 0))
    if t == "sqrt":  return np.sqrt(np.maximum(x, 0))
    return x


# ---------------- frozen robust calibration ----------------
def fit_calibration(df, desc, provisional=False):
    c = {"desc": desc, "provisional": bool(provisional), "n": int(len(df)), "short_side": SHORT_SIDE, "transform": TRANSFORM, "z_bound": Z_BOUND, "features": {}}
    for f in V2_FEATURES:
        x = _tf(f, df[f].to_numpy(float)); x = x[np.isfinite(x)]
        med = float(np.median(x)); mad = float(1.4826 * np.median(np.abs(x - med)))
        if mad <= 1e-12:                       # e.g. clip_frac mostly 0 -> fall back to IQR/1.349, then std
            q = np.percentile(x, [25, 75]); mad = float((q[1] - q[0]) / 1.349) or float(x.std()) or 1.0
        c["features"][f] = {"median": med, "scale": mad}
    c["sha256"] = _hash(c)
    return c


def _hash(c):
    return hashlib.sha256(json.dumps({"f": c["features"], "s": c["short_side"], "t": c.get("transform"), "b": c.get("z_bound"),
                                      "p": c.get("provisional", False)},
                                     sort_keys=True).encode()).hexdigest()[:16]


def save_calibration(c, path):
    with open(path, "w") as fh:
        json.dump(c, fh, indent=1)


def load_calibration(path):
    with open(path) as fh:
        c = json.load(fh)
    if _hash(c) != c["sha256"]:
        raise RuntimeError("v2 calibration hash mismatch (file edited after fit)")
    return c


def score(feats, calib, candidate, return_subscores=False):
    B = calib.get("z_bound", Z_BOUND)
    tfm = calib.get("transform", TRANSFORM)
    if tfm != TRANSFORM:
        raise RuntimeError("calibration was fitted with different feature transforms")
    z = {}
    for f in V2_FEATURES:
        r = (float(_tf(f, feats[f])) - calib["features"][f]["median"]) / calib["features"][f]["scale"]
        z[f] = SIGN[f] * B * np.tanh(r / B)
    subs = {k: float(np.mean([z[f] for f in fs])) for k, fs in CANDIDATES[candidate].items()}
    s = float(np.mean(list(subs.values())))
    return (s, subs) if return_subscores else s
