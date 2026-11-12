"""
A0: reproduce WACV Round-2 Table 1 with the new qcs_eval package and freeze the transform.

Step 1  (cache)  : R2 per-image feature cache -> refit Cholesky -> compare with paper Table 1
                   and with the R2 frozen npz. Writes frozen_transform_r2.json.
Step 2  (images) : recompute features from the images on disk for a seeded sample of stems
                   (same stems in every config) and compare with the cache.
                   -> proves the images on disk are the ones behind Table 1, and that
                      "Input" holds the same images as the R2 "Raw_Input".
Step 3  (sanity) : clipping/ties per config, effective weights, feature correlations.

Usage:
  python scripts/a0_reproduce_table1.py                 # all steps, 200-stem sample
  python scripts/a0_reproduce_table1.py --sample 0      # skip image step (no image download)
"""
import sys, argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import transform as T
from qcs_eval.features import FEATURES
from qcs_eval.io import score_folder, list_images, stem
from qcs_eval import configs as C

# rounding tolerances = half of the last printed digit in the paper
TOL = {"UIQM": 0.0051, "UCIQE": 0.0051, "AG": 0.0051, "Tenengrad": 0.51, "QCS": 0.00051}
log_lines = []


def say(m):
    print(m); log_lines.append(m)


def flag(ok):
    return "PASS" if ok else "FAIL"


def load_cache(path):
    d = pd.read_csv(path)
    if "UCIQE_ref_ratio" not in d.columns:
        raise SystemExit(f"[FAIL] {path} has no UCIQE_ref_ratio column; columns={list(d.columns)[:20]}")
    d["UCIQE"] = pd.to_numeric(d["UCIQE_ref_ratio"], errors="coerce")
    for c in FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "stem" not in d.columns:
        d["stem"] = d["Filename"].astype(str).str.rsplit(".", n=1).str[0].str.lower()
    d = d[d["Configuration"].isin(C.CONFIGS)]
    n_nan = int(d[FEATURES].isna().any(axis=1).sum())
    say(f"[{'PASS' if n_nan == 0 else 'WARN'}] cache rows with NaN features: {n_nan}")
    return d.dropna(subset=FEATURES).copy()


def step1(pool, out):
    say("\n== STEP 1: cache -> Table 1 ==")
    counts = pool.groupby("Configuration").size()
    for cfg in C.CONFIGS:
        n = int(counts.get(cfg, 0))
        if n != C.N_TEST:
            say(f"[FAIL] {cfg}: n={n} (expected {C.N_TEST})")
    dup = int(pool.duplicated(["Configuration", "stem"]).sum())
    say(f"[{flag(dup == 0)}] duplicated (config, stem) rows: {dup}")
    stem_sets = pool.groupby("Configuration")["stem"].apply(frozenset)
    same = stem_sets.nunique() == 1
    say(f"[{flag(same)}] all configs share the identical stem set")

    t = T.fit(pool[FEATURES].to_numpy(), FEATURES, "cholesky",
              calib_desc="R2 pool: 14 Table-1 configs x 4815 test images (calibration == evaluation)")
    T.save(t, out / "frozen_transform_r2.json")
    say(f"[INFO] frozen transform sha={t['sha256']}  p5={t['p_lo']:.6f} p95={t['p_hi']:.6f}  n={t['n_calib']}")

    # compare with R2 npz if present
    if C.R2_FROZEN_NPZ.exists():
        z = np.load(C.R2_FROZEN_NPZ, allow_pickle=True)
        rows = [dict(item="mu", max_abs_diff=float(np.max(np.abs(z["mu"] - np.array(t["mu"]))))),
                dict(item="Linv", max_abs_diff=float(np.max(np.abs(z["Linv"] - np.array(t["W"]))))),
                dict(item="p5", max_abs_diff=float(abs(float(z["p5"]) - t["p_lo"]))),
                dict(item="p95", max_abs_diff=float(abs(float(z["p95"]) - t["p_hi"])))]
        df = pd.DataFrame(rows); df["pass"] = df.max_abs_diff < 1e-9
        df.to_csv(out / "A0_transform_vs_r2npz.csv", index=False)
        say(f"[{flag(df['pass'].all())}] refit transform == R2 frozen npz  (max diff {df.max_abs_diff.max():.2e})")
    else:
        say(f"[WARN] R2 npz not found: {C.R2_FROZEN_NPZ}")

    rng = np.random.default_rng(42)
    rows = []
    for cfg in C.CONFIGS:
        d = pool[pool.Configuration == cfg]
        q = T.apply(d[FEATURES].to_numpy(), t)
        boots = np.array([rng.choice(q, len(q)).mean() for _ in range(1000)])
        half = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2
        r = dict(Configuration=cfg, n=len(d))
        paper = dict(zip(FEATURES + ["QCS"], C.TABLE1[cfg][1:]))
        repro = {**{f: float(d[f].mean()) for f in FEATURES}, "QCS": float(q.mean())}
        ok_all = True
        for k in FEATURES + ["QCS"]:
            r[f"{k}_paper"] = paper[k]; r[f"{k}_repro"] = repro[k]
            r[f"{k}_diff"] = repro[k] - paper[k]
            ok = abs(r[f"{k}_diff"]) <= TOL[k]; ok_all &= ok
        r["QCS_halfCI_repro"] = half
        r["QCS_clip0_pct"] = 100 * float(np.mean(q == 0)); r["QCS_clip1_pct"] = 100 * float(np.mean(q == 1))
        r["QCS_unique_frac"] = float(len(np.unique(q)) / len(q))
        r["all_match"] = ok_all
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab["Rank_repro"] = tab.QCS_repro.rank(ascending=False, method="min").astype(int)
    tab.to_csv(out / "A0_table1_repro.csv", index=False)
    for r in tab.itertuples():
        say(f"[{flag(r.all_match)}] {r.Configuration:<27} QCS paper={r.QCS_paper:.3f} repro={r.QCS_repro:.4f}"
            f"  clip0={r.QCS_clip0_pct:.1f}% clip1={r.QCS_clip1_pct:.1f}%")
    say(f"[{flag(tab.all_match.all())}] Table 1 reproduced within rounding: {int(tab.all_match.sum())}/{len(tab)}")
    return t


def step2(pool, t, out, n_sample, workers):
    say(f"\n== STEP 2: images on disk vs cache (sample={n_sample} stems/config) ==")
    stems = sorted(pool[pool.Configuration == "Retinex_+_LLaVA"].stem.unique())
    rng = np.random.default_rng(0)
    pick = set(rng.choice(stems, min(n_sample, len(stems)), replace=False))
    cache_dir = out / "feature_cache"
    rows = []
    for cfg in C.CONFIGS:
        spec = C.TABLE1[cfg][0]
        parts = spec if isinstance(spec, (list, tuple)) else [spec]
        folder = [C.BASE / x for x in parts] if len(parts) > 1 else C.BASE / parts[0]
        fname = "+".join(parts)
        missing = [C.BASE / x for x in parts if not (C.BASE / x).is_dir()]
        if missing:
            say(f"[FAIL] {cfg}: folder missing {missing}"); continue
        disk = list_images(folder)
        disk_stems = {stem(p) for p in disk}
        cache_stems = set(pool.loc[pool.Configuration == cfg, "stem"])
        only_cache, only_disk = cache_stems - disk_stems, disk_stems - cache_stems
        full_ok = not only_cache and not only_disk
        say(f"[{flag(full_ok)}] {cfg}: full stem set '{fname}' disk={len(disk_stems)} cache={len(cache_stems)}"
            f" only_in_cache={len(only_cache)} only_on_disk={len(only_disk)}"
            + ("" if full_ok else f"  e.g. cache {sorted(only_cache)[:2]} disk {sorted(only_disk)[:2]}"))
        n_hit = len(disk_stems & pick)
        if n_hit < len(pick):
            miss = sorted(pick - disk_stems)[:3]
            say(f"[WARN] {cfg}: {n_hit}/{len(pick)} sampled stems found in '{fname}'."
                f" cache stem e.g. {miss}  disk name e.g. {[p.name for p in disk[:3]]}")
        if n_hit == 0:
            say(f"[FAIL] {cfg}: no sampled stem exists in '{fname}' -> filename pattern differs; skipped")
            rows.append(dict(Configuration=cfg, folder=fname, n_sample=len(pick), n_matched=0, match=False))
            continue
        new = score_folder(folder, f"sample_{cfg}", cache_dir, stems=pick, workers=workers)
        if "error" in new.columns and new["error"].notna().any():
            bad = new[new["error"].notna()]
            say(f"[WARN] {cfg}: {len(bad)} unreadable files (cloud-only / corrupt?) e.g. {bad['error'].head(2).tolist()}")
            new = new[new["error"].isna()]
        if not set(FEATURES).issubset(new.columns) or len(new) == 0:
            say(f"[FAIL] {cfg}: 0 images could be scored; skipped")
            rows.append(dict(Configuration=cfg, folder=fname, n_sample=len(pick), n_matched=0, match=False))
            continue
        old = pool[(pool.Configuration == cfg) & pool.stem.isin(pick)]
        m = old[["stem"] + FEATURES].merge(new[["stem"] + FEATURES], on="stem", suffixes=("_cache", "_img"))
        r = dict(Configuration=cfg, folder=fname, n_sample=len(pick), n_matched=len(m))
        ok = len(m) == len(pick)
        for f in FEATURES:
            rel = np.abs(m[f + "_img"] - m[f + "_cache"]) / np.maximum(np.abs(m[f + "_cache"]), 1e-12)
            r[f"{f}_max_rel_diff"] = float(rel.max()) if len(m) else np.nan
            ok &= bool(len(m)) and r[f"{f}_max_rel_diff"] < 1e-6
        qi = T.apply(m[[f + "_img" for f in FEATURES]].to_numpy(), t) if len(m) else np.array([np.nan])
        qc = T.apply(m[[f + "_cache" for f in FEATURES]].to_numpy(), t) if len(m) else np.array([np.nan])
        r["QCS_max_abs_diff"] = float(np.max(np.abs(qi - qc)))
        r["match"] = ok
        rows.append(r)
        say(f"[{flag(ok)}] {cfg:<27} matched {len(m)}/{len(pick)}  max rel diff "
            + " ".join(f"{f}={r[f + '_max_rel_diff']:.1e}" for f in FEATURES))
    pd.DataFrame(rows).to_csv(out / "A0_images_vs_cache.csv", index=False)


def step3(pool, t, out):
    say("\n== STEP 3: sanity ==")
    raw_w, std_w = T.effective_weights(t)
    pd.DataFrame(dict(feature=list(raw_w), weight_per_unit=list(raw_w.values()),
                      standardized_share=list(std_w.values()))).to_csv(out / "A0_effective_weights.csv", index=False)
    say("[INFO] standardized effective weights (share of |w*sd|): "
        + ", ".join(f"{k}={v:+.3f}" for k, v in std_w.items()))
    neg = [k for k, v in std_w.items() if v < 0]
    if neg:
        say(f"[WARN] negative effective weight -> higher {neg} LOWERS QCS")
    corr_img = pool[FEATURES].corr(method="spearman")
    cfg_means = pool.groupby("Configuration")[FEATURES].mean()
    cfg_means["QCS"] = [T.apply(pool[pool.Configuration == c][FEATURES].to_numpy(), t).mean() for c in cfg_means.index]
    corr_cfg = cfg_means.corr(method="spearman")
    with open(out / "A0_feature_correlations.csv", "w") as fh:
        fh.write("# image-level Spearman (pooled 14x4815)\n"); corr_img.to_csv(fh)
        fh.write("\n# configuration-level Spearman (n=14), includes QCS\n"); corr_cfg.to_csv(fh)
    say(f"[INFO] config-level Spearman QCS~Tenengrad={corr_cfg.loc['QCS', 'Tenengrad']:.3f} "
        f"(paper/reviewer: 0.938), QCS~AG={corr_cfg.loc['QCS', 'AG']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(C.R2_FEATURE_CACHE))
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(C.OUT / "A0_repro"))
    ap.add_argument("--raw-folder", default=None, help="folder(s) for the Raw_Input row, comma-separated (default: Input/Val,Input/Test)")
    a = ap.parse_args()
    if a.raw_folder:
        rf = tuple(x.strip() for x in a.raw_folder.split(","))
        C.TABLE1["Raw_Input"] = (rf if len(rf) > 1 else rf[0],) + tuple(C.TABLE1["Raw_Input"][1:])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pool = load_cache(a.cache)
    t = step1(pool, out)
    step3(pool, t, out)
    if a.sample > 0:
        step2(pool, t, out, a.sample, a.workers)
    (out / "A0_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
