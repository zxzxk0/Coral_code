#!/usr/bin/env python
"""
H3: UCIQE / UIQM baseline audit against the UID2021 paper (arXiv 2204.08813, Table 9: SROCC on the whole 960-image
dataset UCIQE 0.6030, UIQM 0.5404, NIQE 0.3304).

Computes several UCIQE implementations on all 960 UID2021 images (60 raw 'Original' + 900 enhanced) and reports
pooled SROCC with MOS under (a) all 960  <- directly comparable with the published Table 9,
(b) 900 enhanced only (our setting), (c) subsampling sensitivity (mean over 100 random 20% subsets; NOT the paper protocol). This only audits BASELINE implementations; v2_C and its frozen calibration are untouched.

Variants
  r2       : legacy R2 code (OpenCV float Lab, L 0-100, saturation = C/L)            -> 'legacy_UCIQE' in H1/H2
  std      : v2 'published-scale' (L,C /100, saturation = C/sqrt(C^2+L^2), 384-px short side) -> 'feat_UCIQE_std'
  std_full : same formula as std but at full resolution (isolates the resize effect)
  hsv_sat  : r2 chroma/contrast terms + mean HSV saturation (a common re-implementation)
  r2_full8 : r2 formula on OpenCV uint8 Lab scaled back (L*100/255, a,b-128) (another common variant)
Outputs: H3_uciqe_audit/uciqe_audit.csv (+ per-image values)
"""
import argparse, sys
import os
from pathlib import Path
import numpy as np, pandas as pd, cv2
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C, v2
from qcs_eval.features import uciqe_r2, uiqm_legacy

DEF_UID = Path(os.environ.get("UID2021_DIR", "data/UID2021"))
W = (0.4680, 0.2745, 0.2576)


def _terms(L, Cc, sat):
    v = np.sort(L.ravel()); k = max(1, int(round(0.01 * len(v))))
    return W[0] * Cc.std() + W[1] * (v[-k:].mean() - v[:k].mean()) + W[2] * sat.mean()


def uciqe_std_full(bgr):
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[..., 0] / 100.0; Cc = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2) / 100.0
    return float(_terms(L, Cc, Cc / np.sqrt(Cc ** 2 + L ** 2 + 1e-12)))


def uciqe_hsv(bgr):
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[..., 0]; Cc = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    s = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float64) / 255.0
    return float(_terms(L, Cc, s))


def uciqe_u8(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[..., 0] * 100 / 255; a = lab[..., 1] - 128; b = lab[..., 2] - 128; Cc = np.sqrt(a * a + b * b)
    s = np.where(L > 0, Cc / np.maximum(L, 1e-12), 0)
    return float(_terms(L, Cc, s))


def read(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--uid", default=str(DEF_UID))
    ap.add_argument("--manifest", default=str(C.OUT / "A3a_dataset_audit" / "frozen" / "uid2021_all_sensitivity_manifest.csv"))
    ap.add_argument("--out", default=str(C.OUT / "H3_uciqe_audit"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(a.manifest); M = M[M.kind.isin(["raw", "enhanced"])]
    mos = pd.read_excel(Path(a.uid) / "mos.xlsx"); mos["key"] = mos.Name.astype(str).str.lower()
    M["key"] = M.path.map(lambda p: Path(p).name.lower())
    M = M.merge(mos[["key", "MOS"]], on="key", how="left", validate="one_to_one")
    print(f"[{'PASS' if len(M) == 960 and M.MOS.notna().all() else 'FAIL'}] {len(M)} UID images with MOS (expect 960)")
    rows = []
    for r in M.itertuples():
        img = read(r.path)
        rows.append(dict(key=r.key, kind=r.kind, MOS=r.MOS, r2=uciqe_r2(img), std=v2.uciqe_std(v2.resize_short(img)),
                         std_full=uciqe_std_full(img), hsv_sat=uciqe_hsv(img), r2_full8=uciqe_u8(img),
                         uiqm_legacy=uiqm_legacy(img), uiqm_std=v2.uiqm_std(v2.resize_short(img))))
    D = pd.DataFrame(rows); D.to_csv(out / "uciqe_per_image.csv", index=False)
    rng = np.random.default_rng(0); res = []
    for v in ["r2", "std", "std_full", "hsv_sat", "r2_full8", "uiqm_legacy", "uiqm_std"]:
        s960 = stats.spearmanr(D[v], D.MOS).correlation
        s900 = stats.spearmanr(D[D.kind == "enhanced"][v], D[D.kind == "enhanced"].MOS).correlation
        sub = [stats.spearmanr(*D.iloc[rng.choice(len(D), int(0.2 * len(D)), replace=False)][[v, "MOS"]].T.values).correlation
               for _ in range(100)]
        res.append(dict(variant=v, srocc_960_vs_paper=s960, srocc_900_enh=s900, subsample_sensitivity_20pct_x100=np.mean(sub)))
    R = pd.DataFrame(res); R.to_csv(out / "uciqe_audit.csv", index=False)
    import hashlib
    print(f"[INFO] mos.xlsx sha256 {hashlib.sha256((Path(a.uid) / 'mos.xlsx').read_bytes()).hexdigest()[:16]} "
          f"(record with the release; the official README mentions an updated 77-observer MOS)")
    print("\nUID2021 paper Table 9 (whole 960): UCIQE 0.603 | UIQM 0.540 | NIQE 0.330  -> compare with srocc_960_vs_paper")
    print(R.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSpearman between UCIQE variants (960 images):")
    print(D[["r2", "std", "std_full", "hsv_sat", "r2_full8"]].corr("spearman").to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
