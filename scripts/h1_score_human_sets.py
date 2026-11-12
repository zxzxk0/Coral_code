#!/usr/bin/env python
"""
H1: score every image of the frozen SAUD (dev) and UID2021 (confirm) manifests with QCS-v2 and baselines.
LABEL-FREE: reads only images listed in the A3b frozen manifests (hash-verified). Human scores are joined
in the next step (h2), after this file exists, so the metric outputs are fixed before any label is read.

Metrics: v2_C (candidate), v2_B, v2_A (ablations), qcs_v1 + qcs_v1_latent (R2 baseline), legacy UIQM/UCIQE/AG/
Tenengrad, published-scale UIQM_std/UCIQE_std, v2 sub-features, and (--pyiqa) general-purpose NR-IQA
baselines MUSIQ, NIQE, TOPIQ_NR, LIQE, HyperIQA, TReS (off-the-shelf pyiqa weights; no training on SAUD/UID2021).
Calibration: the frozen QCS-v2 calibration JSON (--v2-calib), hash-verified; its sha is written to every row.

Outputs (in --out):
  h1_scores.csv          one row per image (dataset, set, scene_uid, cluster_id, method, path, metrics...)
  h1_score_summary.csv   per dataset x method mean of each metric (sanity only; no labels)
  h1_log.txt
Usage:
  python scripts/h1_score_human_sets.py --v2-calib "<...>/_resub_v3/V2_calib/evalpool.json" [--pyiqa]
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C, v2, transform as T
from qcs_eval.features import compute_features

FROZEN = C.OUT / "A3a_dataset_audit" / "frozen"
EXPECTED_V2_SHA = "753c406c848c7217"
PYIQA_EXPECTED = {"musiq": 1, "niqe": -1, "topiq_nr": 1, "liqe": 1, "hyperiqa": 1, "tres": 1}   # metric definitions
PYIQA_SIGN = {}
LOG = []
def say(s=""): print(s); LOG.append(str(s))


def read(path, tries=4):
    last = None
    for t in range(tries):
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
            if img is None: raise IOError(f"decode failed ({buf.size} bytes)")
            return img
        except Exception as e:
            last = e; time.sleep(0.5 * (t + 1))
    raise IOError(str(last))


def _one(args):
    path, calib, v1 = args
    try:
        img = read(path)
        f = compute_features(img)                       # legacy R2 features (QCS-v1 inputs)
        g = v2.compute_v2_features(img)
    except Exception as e:
        return dict(path=path, error=f"{type(e).__name__}: {e}")
    r = dict(path=path, H_read=img.shape[0], W_read=img.shape[1])
    r.update({f"legacy_{k}": f[k] for k in ["UIQM", "UCIQE", "AG", "Tenengrad"]})
    r.update({f"feat_{k}": g[k] for k in v2.V2_FEATURES})
    for cand in ["v2_C", "v2_B", "v2_A"]:
        s, subs = v2.score(g, calib, cand, return_subscores=True)
        r[cand] = s
        if cand == "v2_C": r.update({f"v2_C_sub_{k}": x for k, x in subs.items()})
    q, z = T.apply(np.array([[f[k] for k in v1["order"]]]), v1, return_raw=True)
    r["qcs_v1"], r["qcs_v1_latent"] = float(q[0]), float(z[0])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-calib", required=True)
    ap.add_argument("--transform", default=str(C.OUT / "A0_repro" / "frozen_transform_r2.json"))
    ap.add_argument("--frozen", default=str(FROZEN))
    ap.add_argument("--pyiqa", action="store_true")
    ap.add_argument("--pyiqa-metrics", default="musiq,niqe,topiq_nr,liqe,hyperiqa,tres",
                    help="general-purpose NR-IQA baselines from pyiqa (orientation taken from the library definition)")
    ap.add_argument("--allow-other-calib", action="store_true", help="sensitivity runs only; output dir should differ")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(C.OUT / "H1_human_sets"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); fz = Path(a.frozen)

    # ---- frozen manifests, hash-verified
    H = json.loads((fz / "manifest_hashes.json").read_text())
    sets = {"saud_dev": "saud_dev_manifest.csv", "uid_confirm_primary": "uid2021_confirm_primary_manifest.csv",
            "uid_all_sensitivity": "uid2021_all_sensitivity_manifest.csv"}
    for k, fn in sets.items():
        ok = hashlib.sha256((fz / fn).read_bytes()).hexdigest() == H[fn]
        say(f"[{'PASS' if ok else 'FAIL'}] manifest hash {fn}")
        if not ok: sys.exit(1)
    S = pd.read_csv(fz / sets["saud_dev"]); U = pd.read_csv(fz / sets["uid_all_sensitivity"])
    prim = set(pd.read_csv(fz / sets["uid_confirm_primary"]).scene_uid)
    U["in_primary"] = U.scene_uid.isin(prim); S["in_primary"] = True
    M = pd.concat([S, U], ignore_index=True)
    M = M[M.kind == "enhanced"].copy()          # human scores exist for enhanced images; raw kept out of h1
    say(f"[INFO] images to score: SAUD {int((M.dataset == 'SAUD').sum())} | UID2021 {int((M.dataset == 'UID2021').sum())} "
        f"(primary {int(((M.dataset == 'UID2021') & M.in_primary).sum())})")
    exp = {"SAUD": 1000, "UID2021": 900}
    for d, n in exp.items():
        if int((M.dataset == d).sum()) != n: sys.exit(f"[FAIL] {d}: expected {n} enhanced images")

    if a.pyiqa:
        from qcs_eval import metrics as _MM
        if not hasattr(_MM, "UPSCALE_224"):
            sys.exit(f"[FAIL] outdated qcs_eval/metrics.py loaded from {_MM.__file__} -> replace it with the latest version "
                     f"(and delete qcs_eval/__pycache__) before scoring")
        say(f"[PASS] metrics.py version OK ({_MM.__file__}); LIQE-only 224-px rule: {sorted(_MM.UPSCALE_224)}")
    calib = v2.load_calibration(a.v2_calib)
    if calib["sha256"] != EXPECTED_V2_SHA and not a.allow_other_calib:
        sys.exit(f"[FAIL] unexpected v2 calibration sha {calib['sha256']} (protocol-frozen: {EXPECTED_V2_SHA}); "
                 f"use --allow-other-calib only for labelled sensitivity runs")
    say(f"[{'PASS' if calib['sha256'] == EXPECTED_V2_SHA else 'WARN'}] v2 calibration sha {'matches' if calib['sha256'] == EXPECTED_V2_SHA else 'DIFFERS from'} the protocol-frozen {EXPECTED_V2_SHA}")
    v1 = T.load(a.transform)
    say(f"[INFO] v2 calibration {a.v2_calib} sha={calib['sha256']} provisional={calib.get('provisional')} | desc: {calib['desc']}")
    say(f"[INFO] qcs_v1 transform sha={v1['sha256']}")

    with ProcessPoolExecutor(a.workers) as ex:
        R = pd.DataFrame(list(ex.map(_one, [(p, calib, v1) for p in M.path], chunksize=8)))
    if "error" in R.columns and R.error.notna().any():
        bad = R[R.error.notna()]; say(f"[WARN] {len(bad)} failed in parallel; retrying serially")
        R = pd.concat([R[R.error.isna()], pd.DataFrame([_one((p, calib, v1)) for p in bad.path])], ignore_index=True)
        if R.error.notna().any():
            R[R.error.notna()].to_csv(out / "h1_failed.csv", index=False); sys.exit(f"[FAIL] {int(R.error.notna().sum())} images unreadable, see h1_failed.csv")
        R = R.drop(columns="error")
    D = M.merge(R, on="path", how="left", validate="one_to_one")

    if a.pyiqa:
        from qcs_eval import metrics as MM
        want = [x.strip() for x in a.pyiqa_metrics.split(",") if x.strip()]
        names = MM.enable_pyiqa(tuple(want))
        miss = sorted(set(want) - set(names))
        if miss: say(f"[WARN] pyiqa metrics not available: {miss}")
        for n in names:
            lib = 1 if MM.REGISTRY[n]["hib"] else -1
            if lib != PYIQA_EXPECTED.get(n, lib):
                sys.exit(f"[FAIL] {n}: library orientation {lib} differs from expected {PYIQA_EXPECTED[n]}")
            PYIQA_SIGN[n] = lib
        for n in names:
            fn = MM.REGISTRY[n]["fn"]; vals = []
            for p in D.path:
                vals.append(fn({}, read(p))[0])
            D[n] = vals; say(f"[{'INFO' if not np.isnan(vals).any() else 'WARN'}] {n}: {int(np.isnan(vals).sum())} NaN"
                + (f" | upscaled to 224 px (pre-specified, LIQE only): {MM.UPSCALED.get(n, 0)}" if n in MM.UPSCALE_224 else "")
                + (f" | first failure: {MM.FAIL_MSG[n]}" if n in MM.FAIL_MSG else ""))

    # ---- debug checks
    say(f"[{'PASS' if D.v2_C.notna().all() else 'FAIL'}] every image has a v2_C score ({D.v2_C.notna().sum()}/{len(D)})")
    same = (D.H_read == D.H) & (D.W_read == D.W)
    say(f"[{'PASS' if same.all() else 'FAIL'}] decoded size == manifest size: {int(same.sum())}/{len(D)}")
    say(f"[INFO] v1 clipping: {(D.qcs_v1 >= 1).mean():.1%} at 1, {(D.qcs_v1 <= 0).mean():.1%} at 0")
    ties = D.groupby("scene_uid").v2_C.apply(lambda x: x.duplicated().sum()).sum()
    say(f"[INFO] exact v2_C ties within scenes: {int(ties)}")
    D["v2_calib_sha"] = calib["sha256"]; D["v1_sha"] = v1["sha256"]
    orient = {"v2_C": 1, "v2_B": 1, "v2_A": 1, "qcs_v1": 1, "qcs_v1_latent": 1, "legacy_UIQM": 1, "legacy_UCIQE": 1,
              "legacy_AG": 1, "legacy_Tenengrad": 1, "feat_UIQM_std": 1, "feat_UCIQE_std": 1}
    orient.update(PYIQA_SIGN)
    pd.Series(orient, name="higher_is_better_sign").to_csv(out / "h1_metric_orientation.csv")
    say(f"[INFO] orientation (metric definitions): +1 higher is better; pyiqa signs {PYIQA_SIGN}")
    D.to_csv(out / "h1_scores.csv", index=False)
    cols = ["v2_C", "v2_B", "v2_A", "qcs_v1", "legacy_Tenengrad", "legacy_AG", "feat_UIQM_std", "feat_UCIQE_std"] + \
           [c for c in PYIQA_SIGN if c in D]
    summ = D.groupby(["dataset", "method"])[cols].mean().round(4)
    summ.to_csv(out / "h1_score_summary.csv")
    say("\n== per-method means (label-free sanity) ==\n" + summ.to_string())
    (out / "h1_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\n[DONE] {out / 'h1_scores.csv'} ({len(D)} rows)")


if __name__ == "__main__":
    main()
