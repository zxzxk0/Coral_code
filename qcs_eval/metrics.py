"""
Metric plug-in registry.

A metric = name + higher_is_better + score(feats, img) -> (score, latent, clip_lo, clip_hi)
  feats : dict of shared hand-crafted features already computed for this image
  score : the value the metric reports (QCS after clipping, raw value for single features)
  latent: pre-clipping value (== score for unclipped metrics)

To add a v2 candidate: write a function returning the same tuple and call register().
Learned metrics (MUSIQ, NIQE via pyiqa) are optional; they load only if pyiqa is installed.
"""
import numpy as np
from . import transform as T

REGISTRY = {}
FAILURES = {}   # metric -> number of images it could not score (returned NaN)
UPSCALED = {}   # metric -> images upscaled by the pre-specified input rule
UPSCALE_224 = {"liqe"}   # metrics whose input short side is raised to 224 px when smaller
FAIL_MSG = {}   # metric -> first failure message (exception or non-finite output)


def register(name, fn, higher_is_better=True, needs_image=False):
    REGISTRY[name] = dict(fn=fn, hib=higher_is_better, needs_image=needs_image)


def _single(feat):
    return lambda f, img: (f[feat], f[feat], False, False)


for _f in ["Tenengrad", "AG", "UIQM", "UCIQE"]:
    register(_f.lower(), _single(_f))


def make_qcs_v1(transform_path):
    t = T.load(transform_path)
    order = t["order"]

    def fn(f, img):
        x = np.array([[f[k] for k in order]])
        q, z = T.apply(x, t, return_raw=True)
        return float(q[0]), float(z[0]), bool(q[0] <= 0), bool(q[0] >= 1)
    register("qcs_v1", fn)
    # pre-clipping version, so curves stay visible where the clipped score saturates at 0 or 1
    register("qcs_v1_latent", lambda f, img: (lambda q: (q[1], q[1], False, False))(fn(f, img)))
    return t


def make_v2(calib_path, candidates=None):
    """Register QCS-v2 candidates; needs the v2 features (qcs_eval.v2.compute_v2_features) in `f`."""
    from . import v2
    c = v2.load_calibration(calib_path)
    for name in (candidates or list(v2.CANDIDATES)):
        register(name, lambda f, img, name=name: (lambda s: (s, s, False, False))(v2.score(f, c, name)))
    return c


def enable_pyiqa(names=("musiq", "niqe"), device=None):
    """Optional. pip install pyiqa (needs torch). Returns list of enabled names."""
    try:
        import torch, pyiqa, cv2
    except Exception as e:
        print(f"[WARN] pyiqa unavailable ({type(e).__name__}); skipping {names}")
        return []
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enabled = []
    for n in names:
        try:
            m = pyiqa.create_metric(n, device=device)
        except Exception as e:
            print(f"[WARN] pyiqa metric {n} failed to load: {e}"); continue

        min_side = 96 if n == "niqe" else 32   # NIQE needs >=1 full 96x96 block

        def fn(f, img, m=m, n=n, min_side=min_side):
            if min(img.shape[:2]) < min_side:
                FAILURES[n] = FAILURES.get(n, 0) + 1
                return np.nan, np.nan, False, False
            def run(im):
                x = torch.from_numpy(np.ascontiguousarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))).permute(2, 0, 1)[None].float().div(255).to(device)
                with torch.no_grad():
                    return float(m(x).item())
            # pre-specified input rule (not a fallback): LIQE only, short side < 224 px -> isotropic bicubic
            # upscale to 224 px before inference (CLIP 224-px crops). Counted in UPSCALED.
            if n in UPSCALE_224 and min(img.shape[:2]) < 224:
                s_ = 224 / min(img.shape[:2])
                img = cv2.resize(img, None, fx=s_, fy=s_, interpolation=cv2.INTER_CUBIC)
                UPSCALED[n] = UPSCALED.get(n, 0) + 1
            try:
                v = run(img)
                if not np.isfinite(v): raise ValueError("non-finite output")
            except Exception as e:
                FAILURES[n] = FAILURES.get(n, 0) + 1
                FAIL_MSG.setdefault(n, f"{type(e).__name__}: {e}"[:200])
                return np.nan, np.nan, False, False
            return v, v, False, False
        register(n, fn, higher_is_better=not m.lower_better, needs_image=True)
        enabled.append(n)
    return enabled
