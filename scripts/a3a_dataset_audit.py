#!/usr/bin/env python
"""
A3a (v2): LABEL-BLIND image-only audit of SAUD (dev) and UID2021 (confirm).

Never opens subjective-label files (SAUD sub_bt_scores.mat / sub_votes.mat, UID2021 mos.xlsx).
Their existence is recorded by NAME ONLY so the later loader knows where they are.

Changes vs the first draft
  * scene_uid = <group>/<scene>  (scene ids repeat across groups/categories -> n_scenes was undercounted)
  * files whose name prefix != folder scene (SAUD group_1/14,15,16 hold 2_*, 3_*, 4_*) are kept in the
    manifest with status="stray", NOT counted toward that folder's method set, and compared
    (sha256 + pixel) to the canonical file of the scene they name  -> tells whether they are harmless copies
  * explicit expected totals (SAUD 5x20 scenes x 10 methods; UID 6x10 scenes x 15 methods) -> PASS/FAIL
  * enhanced-vs-raw size agreement per scene
  * overlap: nearest-neighbour distance for EVERY raw (not only hits under threshold), cross-dataset
    SAUD<->UID and within-dataset raw<->raw (within-dataset near-dups would need a shared bootstrap cluster)
  * paper-ready dataset_summary.csv

Outputs (in --out):
  saud_manifest.csv, uid2021_manifest.csv   one row per image file (status canonical|stray)
  structure_issues.csv                      every anomaly
  stray_vs_canonical.csv                    stray file -> canonical file comparison
  raw_nn_cross.csv                          each SAUD raw: nearest UID raw (and reverse)
  raw_nn_within.csv                         within-dataset raw pairs under threshold
  label_files_found.csv                     label files by name only (never opened)
  dataset_summary.csv                       paper table (scenes, methods, images, resolution)
  A3a_log.txt                               console copy

Usage:
  python scripts/a3a_dataset_audit.py --saud "<...>/SAUD dataset" --uid "<...>/UID 2021" --out "<...>/_resub_v3/A3a_dataset_audit"
Deps: pandas pillow imagehash numpy
"""
from __future__ import annotations
import argparse, hashlib, re, sys
from collections import Counter
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

DEFAULT_SAUD = Path(os.environ.get("SAUD_DIR", "data/SAUD"))
DEFAULT_UID = Path(os.environ.get("UID2021_DIR", "data/UID2021"))
DEFAULT_OUT = Path(os.environ.get("CORAL_BASE", ".")) / "_resub_v3" / "A3a_dataset_audit"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_NAMES = {"sub_bt_scores.mat", "sub_votes.mat", "mos.xlsx"}
LABEL_EXTS = {".mat", ".xlsx", ".xls", ".csv"}           # never opened by this script

SAUD_METHODS = {"BL-TM", "GL-net", "histogram prior", "RayleighDistribution", "retinex-based",
                "RGHS", "two-step-based", "UDCP", "UWCNN", "Water-net"}
UID_METHODS = {"Bayesian", "CBF", "CHS", "GLN-CHE", "HP", "IBLA", "L2UWE", "RCP", "TS", "Ucolor",
               "UNTV", "UTV", "UWB-VCSE", "UWCNN", "VR"}
EXPECT = {"SAUD": dict(groups=5, scenes_per_group=20, methods=SAUD_METHODS),
          "UID2021": dict(groups=6, scenes_per_group=10, methods=UID_METHODS)}

LOG = []
def log(s=""):
    print(s); LOG.append(str(s))
def check(ok, msg):
    log(f"[{'PASS' if ok else 'FAIL'}] {msg}"); return ok


def is_image(p): return p.is_file() and p.suffix.lower() in IMG_EXTS
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def nat(s): return (len(s), s)


def img_info(p: Path, want_hash: bool):
    d = dict(H=None, W=None, mode=None, sha256=sha(p), phash=None, dhash=None, read_error=None)
    try:
        with Image.open(p) as im:
            d["W"], d["H"] = im.size; d["mode"] = im.mode
            if want_hash:
                rgb = im.convert("RGB")
                d["phash"], d["dhash"] = str(imagehash.phash(rgb)), str(imagehash.dhash(rgb))
    except Exception as e:
        d["read_error"] = repr(e)
    d["min_side"] = min(d["H"], d["W"]) if d["H"] else None
    return d


def find_label_files(root: Path, dataset: str):
    return [dict(dataset=dataset, name=p.name, path=str(p), size_bytes=p.stat().st_size)
            for p in root.rglob("*") if p.is_file() and (p.name in LABEL_NAMES or p.suffix.lower() in LABEL_EXTS)]


# ------------------------------------------------------------------ SAUD
def audit_saud(root: Path):
    bench = root / "Benchmark" if (root / "Benchmark").is_dir() else root
    E, R = bench / "Enhanced", bench / "Raw"
    for d in (E, R):
        if not d.is_dir(): sys.exit(f"[FAIL] missing SAUD folder: {d}")
    rows, issues = [], []
    groups = sorted({p.name for p in E.iterdir() if p.is_dir()} | {p.name for p in R.iterdir() if p.is_dir()}, key=nat)
    for g in groups:
        raws = {p.stem: p for p in (R / g).iterdir() if is_image(p)} if (R / g).is_dir() else {}
        sdirs = {p.name: p for p in (E / g).iterdir() if p.is_dir()} if (E / g).is_dir() else {}
        for sid in sorted(set(raws) ^ set(sdirs), key=nat):
            issues.append(dict(dataset="SAUD", scene_uid=f"{g}/{sid}", issue="raw_without_enhanced" if sid in raws else "enhanced_without_raw", detail=""))
        for sid in sorted(set(raws) | set(sdirs), key=nat):
            uid_ = f"{g}/{sid}"
            if sid in raws:
                rows.append(dict(dataset="SAUD", group=g, scene=sid, scene_uid=uid_, kind="raw", method="RAW",
                                 status="canonical", path=str(raws[sid]), **img_info(raws[sid], True)))
            if sid not in sdirs: continue
            files = sorted(p for p in sdirs[sid].iterdir() if is_image(p))
            canon = []
            for p in files:
                pre, _, meth = p.stem.partition("_")
                if pre == sid:
                    status = "canonical"; canon.append(meth)
                else:
                    status = "stray"
                    issues.append(dict(dataset="SAUD", scene_uid=uid_, issue="stray_file_other_scene",
                                       detail=f"{p.name} (names scene {g}/{pre})"))
                rows.append(dict(dataset="SAUD", group=g, scene=sid, scene_uid=uid_, kind="enhanced", method=meth,
                                 status=status, named_scene_uid=f"{g}/{pre}", path=str(p), **img_info(p, False)))
            _method_checks("SAUD", uid_, canon, SAUD_METHODS, issues)
    return pd.DataFrame(rows), issues


# ------------------------------------------------------------------ UID2021
def audit_uid(root: Path):
    rows, issues = [], []
    for cdir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        c = cdir.name
        for sdir in sorted((p for p in cdir.iterdir() if p.is_dir()), key=lambda p: nat(p.name)):
            m = re.fullmatch(rf"{re.escape(c)}-(\d+)", sdir.name)
            if not m:
                issues.append(dict(dataset="UID2021", scene_uid=f"{c}/{sdir.name}", issue="unexpected_scene_folder", detail="")); continue
            idx = m.group(1); uid_ = f"{c}/{idx}"
            canon, nraw = [], 0
            for p in sorted(q for q in sdir.iterdir() if is_image(q)):
                if re.fullmatch(rf"SI_{re.escape(c)}_{idx}", p.stem):
                    kind, meth, status = "raw", "RAW", "canonical"; nraw += 1
                else:
                    mm = re.fullmatch(rf"ERI_{re.escape(c)}_{idx}_(.+)", p.stem)
                    if mm: kind, meth, status = "enhanced", mm.group(1), "canonical"; canon.append(meth)
                    else:
                        kind, meth, status = "unknown", None, "stray"
                        issues.append(dict(dataset="UID2021", scene_uid=uid_, issue="stray_file_other_scene", detail=p.name))
                rows.append(dict(dataset="UID2021", group=c, scene=idx, scene_uid=uid_, kind=kind, method=meth,
                                 status=status, named_scene_uid=uid_, path=str(p), **img_info(p, kind == "raw")))
            if nraw != 1: issues.append(dict(dataset="UID2021", scene_uid=uid_, issue="raw_count_not_one", detail=str(nraw)))
            _method_checks("UID2021", uid_, canon, UID_METHODS, issues)
    return pd.DataFrame(rows), issues


def _method_checks(ds, uid_, canon, expected, issues):
    c = Counter(canon)
    for name, items in [("missing_methods", sorted(expected - set(canon))), ("extra_methods", sorted(set(canon) - expected)),
                        ("duplicate_methods", sorted(m for m, n in c.items() if n > 1))]:
        if items: issues.append(dict(dataset=ds, scene_uid=uid_, issue=name, detail=";".join(items)))


# ------------------------------------------------------------------ stray vs canonical
def stray_compare(df: pd.DataFrame):
    canon = df[(df.status == "canonical") & (df.kind == "enhanced")]
    cnt = canon.groupby(["scene_uid", "method"]).size()
    canon = canon.set_index(["scene_uid", "method"])
    out = []
    for r in df[df.status == "stray"].itertuples():
        key = (r.named_scene_uid, r.method)
        d = dict(stray_path=r.path, in_folder=r.scene_uid, names_scene=r.named_scene_uid, method=r.method)
        if key not in cnt.index:
            d.update(canonical_exists=False); out.append(d); continue
        if cnt[key] > 1:   # malformed dataset: ambiguous canonical file -> skip, already reported by duplicate_methods
            d.update(canonical_exists=True, canonical_ambiguous=True); out.append(d); continue
        c = canon.loc[key]
        d.update(canonical_exists=True, canonical_path=c.path, same_bytes=(c.sha256 == r.sha256),
                 same_size=(c.H == r.H and c.W == r.W))
        if not d["same_bytes"] and d["same_size"]:
            a = np.asarray(Image.open(r.path).convert("RGB"), np.int16); b = np.asarray(Image.open(c.path).convert("RGB"), np.int16)
            d.update(max_abs_pixel_diff=int(np.abs(a - b).max()), mean_abs_pixel_diff=float(np.abs(a - b).mean()))
        out.append(d)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ hashing
def ham(a, b): return bin(int(a, 16) ^ int(b, 16)).count("1")


def nn_cross(A: pd.DataFrame, B: pd.DataFrame, la, lb):
    rows = []
    for a in A.itertuples():
        d = [(ham(a.phash, b.phash), ham(a.dhash, b.dhash), b) for b in B.itertuples()]
        p_p, p_d, bp = min(d, key=lambda x: (x[0], x[1]))   # pHash-nearest
        d_p, d_d, bd = min(d, key=lambda x: (x[1], x[0]))   # dHash-nearest (may be a different image)
        rows.append(dict(query_set=la, query=a.scene_uid, query_path=a.path, nn_set=lb,
                         phash_nn=bp.scene_uid, phash_nn_path=bp.path, phash_dist=p_p, phash_nn_dhash_dist=p_d,
                         dhash_nn=bd.scene_uid, dhash_nn_path=bd.path, dhash_dist=d_d, dhash_nn_phash_dist=d_p,
                         nn_agree=(bp.path == bd.path)))
    return rows


def within_pairs(A: pd.DataFrame, ds, thr):
    rows, recs = [], list(A.itertuples())
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            p, d = ham(recs[i].phash, recs[j].phash), ham(recs[i].dhash, recs[j].dhash)
            rows.append(dict(dataset=ds, a=recs[i].scene_uid, b=recs[j].scene_uid, a_path=recs[i].path, b_path=recs[j].path,
                             phash_dist=p, dhash_dist=d, candidate=(p <= thr or d <= thr)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saud", default=str(DEFAULT_SAUD)); ap.add_argument("--uid", default=str(DEFAULT_UID))
    ap.add_argument("--out", default=str(DEFAULT_OUT)); ap.add_argument("--thresh", type=int, default=10,
                    help="pHash/dHash Hamming distance (of 64) flagged for visual review")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log("[INFO] LABEL-BLIND MODE: label files are listed by name only and never opened.")

    labels = pd.DataFrame(find_label_files(Path(a.saud), "SAUD") + find_label_files(Path(a.uid), "UID2021"))
    labels.to_csv(out / "label_files_found.csv", index=False)
    log(f"[INFO] label files found (NOT opened): {labels.name.tolist() if len(labels) else 'none'}")

    S, si = audit_saud(Path(a.saud)); U, ui = audit_uid(Path(a.uid))
    issues = pd.DataFrame(si + ui, columns=["dataset", "scene_uid", "issue", "detail"])
    assert not set(S.path.map(lambda p: Path(p).name)) & LABEL_NAMES and not set(U.path.map(lambda p: Path(p).name)) & LABEL_NAMES

    # ---- structure checks
    log("\n== EXPECTED STRUCTURE ==")
    summ = []
    for ds, df in [("SAUD", S), ("UID2021", U)]:
        e = EXPECT[ds]; can = df[df.status == "canonical"]; raw = can[can.kind == "raw"]; enh = can[can.kind == "enhanced"]
        n_sc = e["groups"] * e["scenes_per_group"]; n_m = len(e["methods"])
        check(df.group.nunique() == e["groups"], f"{ds}: groups {df.group.nunique()} (expected {e['groups']})")
        check(raw.scene_uid.nunique() == n_sc and len(raw) == n_sc, f"{ds}: raw {len(raw)} over {raw.scene_uid.nunique()} unique scenes (expected {n_sc})")
        check(len(enh) == n_sc * n_m, f"{ds}: canonical enhanced {len(enh)} (expected {n_sc}x{n_m}={n_sc*n_m})")
        per = enh.groupby("scene_uid").method.nunique()
        check(len(per) == n_sc and (per == n_m).all(), f"{ds}: every scene has all {n_m} methods exactly once "
              f"(scenes failing: {int((per != n_m).sum()) + n_sc - len(per)})")
        check(set(enh.method) == e["methods"], f"{ds}: method set matches expected")
        nstray = int((df.status == "stray").sum())
        log(f"[{'INFO' if nstray else 'PASS'}] {ds}: stray files (excluded from scene sets): {nstray}")
        nerr = int(df.read_error.notna().sum()); check(nerr == 0, f"{ds}: unreadable images {nerr}")
        # enhanced size == raw size within scene
        rs = raw.set_index("scene_uid")[["H", "W"]]
        m = enh.join(rs, on="scene_uid", rsuffix="_raw")
        mism = m[(m.H != m.H_raw) | (m.W != m.W_raw)]
        log(f"[{'PASS' if mism.empty else 'WARN'}] {ds}: enhanced size == raw size: {len(m)-len(mism)}/{len(m)}"
            + ("" if mism.empty else f" | mismatching methods: {mism.method.value_counts().to_dict()}"))
        for r in mism.itertuples(): issues.loc[len(issues)] = [ds, r.scene_uid, "enhanced_size_ne_raw", f"{r.method}: {r.H}x{r.W} vs {int(r.H_raw)}x{int(r.W_raw)}"]
        ms = raw.min_side
        summ.append(dict(dataset=ds, role={"SAUD": "dev", "UID2021": "confirm"}[ds], n_groups=df.group.nunique(),
                         n_scenes=raw.scene_uid.nunique(), n_methods=n_m, n_raw=len(raw), n_enhanced=len(enh),
                         n_stray_excluded=nstray, n_pairs_within_scene=int(n_sc * n_m * (n_m - 1) / 2),
                         raw_min_side_min=ms.min(), raw_min_side_median=ms.median(), raw_min_side_max=ms.max(),
                         raw_res_mode=raw.apply(lambda r: f"{int(r.W)}x{int(r.H)}", axis=1).mode().iat[0],
                         n_raw_below_128=int((ms < 128).sum()), n_distinct_raw_res=raw[["H", "W"]].drop_duplicates().shape[0]))
    summary = pd.DataFrame(summ)

    # ---- stray files
    log("\n== STRAY FILES vs CANONICAL ==")
    sc = stray_compare(pd.concat([S, U], ignore_index=True))
    sc.to_csv(out / "stray_vs_canonical.csv", index=False)
    if len(sc):
        log(sc.drop(columns=[c for c in ["stray_path", "canonical_path"] if c in sc]).to_string(index=False))
        nb = int(sc.get("same_bytes", pd.Series(dtype=bool)).fillna(False).sum())
        log(f"[INFO] {nb}/{len(sc)} stray files are byte-identical copies of the canonical file they name"
            + ("" if nb == len(sc) else " -> the rest DIFFER; canonical folder files are used, strays are excluded"))
    else:
        log("[PASS] none")

    # ---- overlap
    log(f"\n== RAW OVERLAP (pHash/dHash Hamming of 64; review threshold <= {a.thresh}) ==")
    rs, ru = S[(S.kind == "raw") & S.phash.notna()], U[(U.kind == "raw") & U.phash.notna()]
    cross = pd.DataFrame(nn_cross(rs, ru, "SAUD", "UID2021") + nn_cross(ru, rs, "UID2021", "SAUD"))
    cross["candidate"] = (cross.phash_dist <= a.thresh) | (cross.dhash_dist <= a.thresh)
    nd = int((~cross.nn_agree).sum())
    log(f"[INFO] pHash-NN and dHash-NN point to different images for {nd}/{len(cross)} queries (both pairs are saved)")
    cross.to_csv(out / "raw_nn_cross.csv", index=False)
    for q in ["SAUD", "UID2021"]:
        c = cross[cross.query_set == q]
        log(f"{q}->nearest other: pHash NN dist min {c.phash_dist.min()} / p5 {c.phash_dist.quantile(.05):.0f} / median {c.phash_dist.median():.0f}"
            f" | candidates {int(c.candidate.sum())}/{len(c)}")
    within = pd.concat([within_pairs(rs, "SAUD", a.thresh), within_pairs(ru, "UID2021", a.thresh)], ignore_index=True)
    within[within.candidate].to_csv(out / "raw_nn_within.csv", index=False)
    for ds in ["SAUD", "UID2021"]:
        w = within[within.dataset == ds]
        log(f"{ds} within raw pairs: {len(w)} | pHash min {w.phash_dist.min()} median {w.phash_dist.median():.0f} | candidates {int(w.candidate.sum())}")
    cand = cross[cross.candidate].sort_values(["phash_dist", "dhash_dist"])
    if len(cand): log("[WARN] cross-dataset candidates -> inspect visually before deciding:\n" + cand[["query_set", "query", "phash_nn", "phash_dist", "dhash_nn", "dhash_dist"]].head(30).to_string(index=False))
    else: log("[PASS] no SAUD<->UID2021 raw pair under threshold")
    summary["n_cross_overlap_candidates"] = [int(cross[(cross.query_set == d)].candidate.sum()) for d in summary.dataset]
    summary["n_within_raw_candidate_pairs"] = [int(within[within.dataset == d].candidate.sum()) for d in summary.dataset]

    # ---- write
    S.to_csv(out / "saud_manifest.csv", index=False); U.to_csv(out / "uid2021_manifest.csv", index=False)
    issues.to_csv(out / "structure_issues.csv", index=False); summary.to_csv(out / "dataset_summary.csv", index=False)
    log("\n== ISSUES ==")
    log(issues.groupby(["dataset", "issue"]).size().to_string() if len(issues) else "[PASS] none")
    log("\n== DATASET SUMMARY (paper) ==")
    log(summary.T.to_string(header=False))
    (out / "A3a_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    log(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
