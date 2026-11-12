"""
QCS transform: fit once, freeze to JSON (with content hash), apply unchanged.

cholesky : Round-2 definition  z = (x-mu) L^{-T},  zbar = mean(z),  QCS = clip((zbar-p5)/(p95-p5),0,1)
           (order-dependent; kept for reproduction)
zca      : symmetric whitening W = Sigma^{-1/2} (order-invariant alternative, for Phase B)
"""
import json, hashlib
import numpy as np

EPS = 1e-8


def fit(X, order, method="cholesky", ridge=1e-6, pct=(5, 95), calib_desc=""):
    X = np.asarray(X, float)
    mu = X.mean(0)
    cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * ridge
    if method == "cholesky":
        L = np.linalg.cholesky(cov)
        W = np.linalg.inv(L)
    elif method == "zca":
        ev, V = np.linalg.eigh(cov)
        W = V @ np.diag(ev ** -0.5) @ V.T
    else:
        raise ValueError(method)
    zb = ((X - mu) @ W.T).mean(1)
    p_lo, p_hi = np.percentile(zb, pct)
    t = dict(method=method, order=list(order), mu=mu.tolist(), cov=cov.tolist(), W=W.tolist(),
             p_lo=float(p_lo), p_hi=float(p_hi), pct=list(pct), ridge=ridge,
             n_calib=int(len(X)), calib_desc=calib_desc)
    t["sha256"] = transform_hash(t)
    return t


def transform_hash(t):
    core = {k: t[k] for k in ("method", "order", "mu", "W", "p_lo", "p_hi")}
    s = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def zbar(X, t):
    X = np.asarray(X, float)
    return ((X - np.array(t["mu"])) @ np.array(t["W"]).T).mean(1)


def apply(X, t, return_raw=False):
    z = zbar(X, t)
    q = np.clip((z - t["p_lo"]) / max(t["p_hi"] - t["p_lo"], EPS), 0, 1)
    return (q, z) if return_raw else q


def effective_weights(t):
    """QCS_raw = w . x + const ; w = mean over rows of W (per original feature unit)
    and standardized weight = w * sd(feature) (comparable across features)."""
    W = np.array(t["W"]); w = W.mean(0)
    sd = np.sqrt(np.diag(np.array(t["cov"])))
    std_w = w * sd
    return dict(zip(t["order"], w.tolist())), dict(zip(t["order"], (std_w / np.abs(std_w).sum()).tolist()))


def save(t, path):
    assert t["sha256"] == transform_hash(t), "transform modified after fit"
    with open(path, "w") as fh:
        json.dump(t, fh, indent=1)


def load(path):
    with open(path) as fh:
        t = json.load(fh)
    h = transform_hash(t)
    if h != t["sha256"]:
        raise RuntimeError(f"Frozen transform hash mismatch: file={t['sha256']} recomputed={h}")
    return t
