"""
Fit and freeze the QCS-v2 robust calibration (per-feature median and 1.4826*MAD).

Calibration sources (pre-registered; all are reported as a sensitivity analysis):
  evalpool  : the 14 Table-1 configurations on the evaluated split (PROVISIONAL; same images that
              are evaluated -> only for development / stress tests until Train outputs exist)
  raw       : raw inputs only (Input/Val+Test, or --folders for Train raw)
  folders   : any list of folders, e.g. Train outputs:  --folders "Train_Outputs/Raw;Train_Outputs/Retinex_w_o_post;..."

Writes _resub_v3/V2_calib/<name>.json (hash-verified on load) and <name>_features.csv,
and prints debug checks: feature ranges (UIQM_std expected ~1-6, UCIQE_std ~0.3-0.8),
calibrated z of the calibration set (median ~0, scale ~1), and cross-feature Spearman.

Usage:
  python scripts/v2_calibrate.py --source evalpool --per-config 300
  python scripts/v2_calibrate.py --source folders --folders "D:/Train_Outputs/Raw;D:/Train_Outputs/Retinex_w_o_post" --name train_v1
"""
import sys, os, argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C, v2
from qcs_eval.io import list_images, stem, read_bgr


VAL_PRIMARY = ["Retinex_plain_w_phys=0", "Retinex_w_o_post", "Retinex_+_Default_MHF-UIE", "Retinex_+_Rule_based",
               "RAUNE_plain_w_phys=0", "RAUNE_w_o_post", "RAUNE_+_Default_MHF-UIE", "RAUNE_+_Rule_based"]


def folder_of(cfg):
    spec = C.TABLE1[cfg][0]
    parts = spec if isinstance(spec, (list, tuple)) else [spec]
    return [C.BASE / p for p in parts] if len(parts) > 1 else C.BASE / parts[0]


def _read(path, tries=4):
    """np.fromfile + imdecode (safe for any Windows path) with retries: cloud-synced placeholders or a busy file can
    fail transiently; a real decode failure fails every try."""
    import time, cv2
    last = None
    for t in range(tries):
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            if buf.size == 0: raise IOError("empty read (0 bytes)")
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None: raise IOError(f"imdecode failed ({buf.size} bytes)")
            return img
        except Exception as e:
            last = e; time.sleep(0.5 * (t + 1))
    raise IOError(f"{type(last).__name__}: {last}")


def _one(args):
    path, label = args
    try:
        img = _read(path)
    except Exception as e:
        return dict(file=path, label=label, stem=stem(path), error=f"READ {e}")
    try:
        f = v2.compute_v2_features(img)
    except Exception as e:
        return dict(file=path, label=label, stem=stem(path), error=f"FEATURE {type(e).__name__}: {e}")
    f.update(file=path, label=label, stem=stem(path))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["evalpool", "raw", "folders", "valpool", "testpool"], default="evalpool",
                    help="testpool = PRIMARY: Raw + 8 non-adaptive configs on Test stems (use --fold A/B for cross-fitting)")
    ap.add_argument("--folders", default="")
    ap.add_argument("--per-config", type=int, default=300, help="images sampled per config/folder (0 = all)")
    ap.add_argument("--min-side", type=int, default=128, help="skip images whose short side is below this (clean-set rule)")
    ap.add_argument("--exclude-eval-dups", action="store_true",
                    help="drop images whose stem belongs to a Train duplicate group that also contains a Val/Test image "
                         "(uses _resub_v3/A0c_duplicates/dup_groups.csv); use for Train-based calibration")
    ap.add_argument("--stability", type=int, default=200, help="bootstrap resamples (over stems) for calibration-size stability; 0 = off")
    ap.add_argument("--fold", choices=["all", "A", "B"], default="all", help="cross-fitting fold of the Test split")
    ap.add_argument("--split-dedup", action="store_true",
                    help="sensitivity: also drop split stems that sit in an A0c group with the other split")
    ap.add_argument("--split", choices=["all", "Val", "Test"], default="all", help="restrict any source to one split's stems")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None, help="default: min(8, cpu-1); fewer workers = less I/O contention")
    a = ap.parse_args()
    out = C.OUT / "V2_calib"; out.mkdir(parents=True, exist_ok=True)
    name = a.name or a.source

    if a.source == "evalpool":
        groups = {cfg: folder_of(cfg) for cfg in C.CONFIGS}
    elif a.source in ("valpool", "testpool"):
        groups = {g: folder_of(g) for g in ["Raw_Input"] + VAL_PRIMARY}
    elif a.source == "raw":
        groups = {"Raw_Input": folder_of("Raw_Input")}
    else:
        groups = {Path(p).name: Path(p) for p in a.folders.split(";") if p.strip()}
    rng = np.random.default_rng(a.seed)
    lists = {g: list_images(f) for g, f in groups.items()}
    if a.source in ("valpool", "testpool") or a.split != "all":
        sp = {"valpool": "Val", "testpool": "Test"}.get(a.source, a.split)
        keep = {stem(p) for p in list_images(C.BASE / "Input" / sp)}
        other = "Test" if sp == "Val" else "Val"
        if a.split_dedup:
            hp = C.OUT / "A0c_duplicates" / "dup_groups.csv"
            if not hp.exists(): sys.exit(f"[FAIL] {hp} missing (run a0c_duplicate_audit.py)")
            G = pd.read_csv(hp); og = set(G.loc[G.split == other, "dup_group"])
            shared = set(G.loc[(G.split == sp) & G.dup_group.isin(og), "stem"].astype(str).str.lower())
            n0 = len(keep); keep -= shared
            print(f"[INFO] split rule (--split-dedup): {sp} stems {n0} -> {len(keep)} after removing {n0 - len(keep)} in A0c groups shared with {other}")
        else:
            print(f"[INFO] split rule: all {len(keep)} {sp} stems used (no cross-split A0c removal; use --split-dedup for the sensitivity run)")
        lists = {g: [p for p in l if stem(p) in keep] for g, l in lists.items()}
    if a.fold != "all":
        if a.source != "testpool" and a.split != "Test": sys.exit("[FAIL] --fold requires --source testpool or --split Test")
        from qcs_eval.folds import load_folds
        Fd = load_folds(); fs = set(Fd.loc[Fd.fold == a.fold, "stem"])
        lists = {g: [p for p in l if stem(p) in fs] for g, l in lists.items()}
        print(f"[INFO] fold {a.fold}: {len(fs)} Test stems ({Fd[Fd.fold == a.fold].cluster.nunique()} clusters)")
    paired = a.source in ("valpool", "testpool", "evalpool") or (a.source == "folders" and len(lists) > 1)
    if paired:
        # eligibility fixed on the RAW source image, then applied to every group (keeps the pool exactly paired)
        stem_sets = {g: {stem(p) for p in l} for g, l in lists.items()}
        if a.source in ("valpool", "testpool", "evalpool") and a.min_side > 0:
            from PIL import Image
            raw_l = lists["Raw_Input"]; small_raw = set()
            for p in raw_l:
                with Image.open(p) as im:
                    if min(im.size) < a.min_side: small_raw.add(stem(p))
            print(f"[INFO] eligibility on Raw source: {len(small_raw)} stems with short side < {a.min_side}px removed from ALL groups")
            stem_sets = {g: ss - small_raw for g, ss in stem_sets.items()}
        ref_name = "Raw_Input" if "Raw_Input" in stem_sets else next(iter(stem_sets)); ref = stem_sets[ref_name]
        for g, ss in stem_sets.items():
            if ss != ref:
                miss, extra = sorted(ref - ss), sorted(ss - ref)
                sys.exit(f"[FAIL] {g} stem set differs from {ref_name}: missing={len(miss)} extra={len(extra)} "
                         f"(examples missing={miss[:5]}, extra={extra[:5]})")
        print(f"[PASS] all {len(stem_sets)} calibration groups contain the same {len(ref)} stems")
        chosen = set(rng.choice(sorted(ref), min(a.per_config, len(ref)), replace=False)) if a.per_config else ref
        jobs = [(str(p), g) for g, l in lists.items() for p in l if stem(p) in chosen]
        assert len(jobs) == len(chosen) * len(lists), "paired job count mismatch"
        print(f"[INFO] {len(jobs)} image-configuration instances = {len(chosen)} unique source stems x {len(lists)} groups (paired)")
    else:
        jobs = []
        for g, l in lists.items():
            sel = l if not a.per_config else [l[i] for i in rng.choice(len(l), min(a.per_config, len(l)), replace=False)]
            jobs += [(str(p), g) for p in sel]
        print(f"[INFO] {len(jobs)} images from {len(groups)} group(s)")

    workers = a.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as ex:
        df = pd.DataFrame(list(ex.map(_one, jobs, chunksize=8)))
    if "error" in df.columns and df.error.notna().any():
        bad = df[df.error.notna()]
        print(f"[WARN] failed in parallel pass: {len(bad)} -> error types:")
        print(bad.error.str.slice(0, 90).value_counts().head(8).to_string())
        print(f"[INFO] retrying {len(bad)} serially ...")
        redo = pd.DataFrame([_one((r.file, r.label)) for r in bad.itertuples()])
        df = pd.concat([df[df.error.isna()], redo], ignore_index=True)
        still = df[df.error.notna()] if "error" in df.columns else df.iloc[:0]
        print(f"[{'PASS' if still.empty else 'WARN'}] after serial retry: {len(still)} still failing")
        if len(still):
            still[["label", "file", "error"]].to_csv(out / f"{name}_failed_files.csv", index=False)
            print(still.error.str.slice(0, 90).value_counts().head(8).to_string())
            print(still[["label", "file"]].head(5).to_string(index=False))
        df = df[df.error.isna()]
    # clean-set rule: short side >= min_side (measured on the original image)
    from PIL import Image
    small = []
    for p in df.file:
        with Image.open(p) as im:
            small.append(min(im.size) < a.min_side)
    df["small"] = small
    print(f"[INFO] per-image min-side<{a.min_side} rows: {int(df.small.sum())}")
    if paired and a.source in ("valpool", "testpool", "evalpool"):
        if df.small.any():
            print("[WARN] some OUTPUT images are below min-side although their Raw source is eligible -> kept (eligibility is defined on Raw)")
    else:
        df = df[~df.small]
    if paired:
        cnt = df.groupby("stem").label.nunique()
        bad = cnt[cnt != len(lists)]
        if len(bad): sys.exit(f"[FAIL] pairing broken after feature computation (unreadable files?): {len(bad)} stems, e.g. {list(bad.index[:5])}")
        print(f"[PASS] final pool paired: {df.stem.nunique()} stems x {len(lists)} groups = {len(df)} rows")
    if a.exclude_eval_dups:
        hp = C.OUT / "A0c_duplicates" / "dup_groups.csv"
        if not hp.exists():
            sys.exit(f"[FAIL] {hp} not found; run scripts/a0c_duplicate_audit.py first")
        G = pd.read_csv(hp)                       # only images that belong to a duplicate group (size > 1)
        ev_groups = set(G.loc[G.split.isin(["Val", "Test"]), "dup_group"])
        bad = set(G.loc[(G.split == "Train") & G.dup_group.isin(ev_groups), "stem"].astype(str).str.lower())
        n0 = len(df); df = df[~df.stem.astype(str).str.lower().isin(bad)]
        print(f"[INFO] excluded Train images sharing content with Val/Test: {n0 - len(df)} (of {len(bad)} flagged stems)")
    df.to_csv(out / f"{name}_features.csv", index=False)

    c = v2.fit_calibration(df, provisional=(a.source == "evalpool" and a.split == "all"),
                           desc=f"{a.source}: {', '.join(groups)} | n per group={a.per_config} | min_side={a.min_side}"
                                f" | exclude_eval_dups={a.exclude_eval_dups} | split={a.split} | split_dedup={a.split_dedup} | fold={a.fold}")
    v2.save_calibration(c, out / f"{name}.json")

    # ---- debug checks ----
    print("\n== feature ranges (calibration set) ==")
    print(df[v2.V2_FEATURES].describe(percentiles=[.05, .5, .95]).T[["min", "5%", "50%", "95%", "max"]].to_string(float_format=lambda x: f"{x:.4g}"))
    ok_u = 0.5 < df.UIQM_std.median() < 8; ok_c = 0.2 < df.UCIQE_std.median() < 1.0
    print(f"[{'PASS' if ok_u else 'WARN'}] UIQM_std median {df.UIQM_std.median():.3f} (published-scale ~1-6)")
    print(f"[{'PASS' if ok_c else 'WARN'}] UCIQE_std median {df.UCIQE_std.median():.3f} (published-scale ~0.3-0.8)")
    z = pd.DataFrame({f: (v2._tf(f, df[f]) - c["features"][f]["median"]) / c["features"][f]["scale"] for f in v2.V2_FEATURES})
    print(f"[{'PASS' if z.median().abs().max() < 1e-9 else 'FAIL'}] calibrated z medians ~0 (max |median| {z.median().abs().max():.1e})")
    print("[INFO] per-group candidate means and ranks (debug only; provisional calibrations are not for decisions):")
    for cand in v2.CANDIDATES:
        df[cand] = [v2.score(r, c, cand) for r in df[v2.V2_FEATURES].to_dict("records")]
    gm = df.groupby("label")[list(v2.CANDIDATES)].mean()
    ranks = gm.rank(ascending=False, method="min").astype(int).add_suffix("_rank")
    tab = pd.concat([gm, ranks], axis=1).sort_index()
    tab.to_csv(out / f"{name}_group_means.csv")
    print(tab.to_string(float_format=lambda x: f"{x:+.3f}"))
    # ---- calibration-size stability (measured, not assumed) ----
    if a.stability:
        rng2 = np.random.default_rng(a.seed + 1)
        units = df.stem.astype(str).str.lower().values; uu = np.unique(units)
        idx_by = {u: np.flatnonzero(units == u) for u in uu}
        def refit(sel_units):
            ii = np.concatenate([idx_by[u] for u in sel_units])
            return v2.fit_calibration(df.iloc[ii], desc="stability", provisional=True)["features"]
        recs = []
        for kind, k, B in [("bootstrap", len(uu), a.stability), ("half", len(uu) // 2, max(20, a.stability // 4))]:
            for b in range(B):
                fb = refit(rng2.choice(uu, k, replace=(kind == "bootstrap")))
                for f in v2.V2_FEATURES:
                    m0, s0 = c["features"][f]["median"], c["features"][f]["scale"]
                    recs.append(dict(kind=kind, n_units=k, b=b, feature=f, d_median_in_scale=(fb[f]["median"] - m0) / s0,
                                     scale_ratio=fb[f]["scale"] / s0))
        R = pd.DataFrame(recs)
        st = R.groupby(["kind", "n_units", "feature"]).agg(
            d_median_p2_5=("d_median_in_scale", lambda x: x.quantile(.025)), d_median_p97_5=("d_median_in_scale", lambda x: x.quantile(.975)),
            scale_ratio_p2_5=("scale_ratio", lambda x: x.quantile(.025)), scale_ratio_p97_5=("scale_ratio", lambda x: x.quantile(.975))).reset_index()
        st.to_csv(out / f"{name}_stability.csv", index=False)
        print(f"\n== calibration-size stability ({a.stability} bootstrap + half-size subsamples over {len(uu)} stems) ==")
        print(st.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
        cf = sorted({f for fs in v2.CANDIDATES["v2_C"].values() for f in fs})
        hb = st[(st.kind == "half") & st.feature.isin(cf)]
        worst_m = hb[["d_median_p2_5", "d_median_p97_5"]].abs().max().max()
        worst_s = max((hb.scale_ratio_p97_5 - 1).abs().max(), (1 - hb.scale_ratio_p2_5).abs().max())
        print(f"[INFO] v2_C features {cf}: half-size worst 95% range: |d median| {worst_m:.3f} scale units, scale ratio off by {worst_s:.1%} "
              f"(reported as-is; no pass/fail threshold was pre-registered)")
    print("\n[INFO] Spearman between features (image level):")
    print(df[v2.V2_FEATURES].corr("spearman").to_string(float_format=lambda x: f"{x:+.2f}"))
    print(f"\n[DONE] {out / (name + '.json')}  sha={c['sha256']}")


if __name__ == "__main__":
    main()
