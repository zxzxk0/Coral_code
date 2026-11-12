#!/usr/bin/env python
"""
A3b: freeze the label-blind image manifests for SAUD (dev) and UID2021 (confirm) from the A3a outputs.

Pre-registered rules (written before any label file is opened):
  R1  stray files are never used (canonical files only).
  R2  SAUD<->UID2021 raw content overlap (pHash OR dHash Hamming <= --thresh): the UID2021 scene is
      EXCLUDED from the primary confirm set (conservative: borderline candidates are excluded too).
      A sensitivity analysis on all 60 UID scenes is reported separately and labelled as such.
  R3  within-dataset raw candidates (<= --thresh): the two scenes are merged into one bootstrap cluster
      (union-find); nothing is dropped.
  R4  no label file is opened here.

Inputs : <a3a_dir>/saud_manifest.csv, uid2021_manifest.csv, raw_nn_cross.csv, raw_nn_within.csv
Outputs: <a3a_dir>/frozen/
  saud_dev_manifest.csv                  canonical images, cluster_id
  uid2021_confirm_primary_manifest.csv   PRIMARY confirm: overlap scenes removed (49 scenes expected)
  uid2021_all_sensitivity_manifest.csv   all 60 scenes + excluded_overlap / overlap_with (sensitivity only)
  overlap_pairs.csv                cross-dataset pairs used for R2 (deduplicated, symmetric)
  within_clusters.csv              merged within-dataset scenes (R3)
  review_sheet.png                 raw pairs side by side (documentation; decision does not depend on it)
  frozen_sets_summary.csv          paper numbers (scenes / clusters / pairs per set and per UID category)
  manifest_hashes.json                   sha256 of the three frozen manifests + scene counts
"""
import argparse, hashlib, json
import os
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

DEFAULT = Path(os.environ.get("CORAL_BASE", ".")) / "_resub_v3" / "A3a_dataset_audit"


def check(ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}"); return ok


class UF:
    def __init__(s, items): s.p = {i: i for i in items}
    def f(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def clusters(scenes, pairs):
    uf = UF(scenes)
    for a, b in pairs: uf.u(a, b)
    roots = {s: uf.f(s) for s in scenes}
    ids = {r: f"c{i:03d}" for i, r in enumerate(sorted(set(roots.values())))}
    return {s: ids[r] for s, r in roots.items()}


def thumb(p, h=160):
    im = Image.open(p).convert("RGB"); return im.resize((max(1, round(im.width * h / im.height)), h))


def sheet(pairs, path, title):
    if not pairs: return
    rows = []
    for la, pa, lb, pb, txt in pairs:
        a, b = thumb(pa), thumb(pb)
        r = Image.new("RGB", (a.width + b.width + 30, 190), "white"); r.paste(a, (0, 25)); r.paste(b, (a.width + 30, 25))
        ImageDraw.Draw(r).text((2, 4), f"{la}  |  {lb}   {txt}", fill="black"); rows.append(r)
    W = max(r.width for r in rows); out = Image.new("RGB", (W, 30 + 190 * len(rows)), "white")
    ImageDraw.Draw(out).text((4, 8), title, fill="black")
    for i, r in enumerate(rows): out.paste(r, (0, 30 + 190 * i))
    out.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3a", default=str(DEFAULT)); ap.add_argument("--thresh", type=int, default=10)
    a = ap.parse_args(); d = Path(a.a3a); out = d / "frozen"; out.mkdir(exist_ok=True)
    S = pd.read_csv(d / "saud_manifest.csv"); U = pd.read_csv(d / "uid2021_manifest.csv")
    X = pd.read_csv(d / "raw_nn_cross.csv")
    Wp = d / "raw_nn_within.csv"
    Wn = pd.read_csv(Wp) if Wp.exists() and Wp.stat().st_size > 1 else pd.DataFrame(columns=["dataset", "a", "b", "a_path", "b_path", "phash_dist", "dhash_dist"])

    # R1
    S = S[S.status == "canonical"].copy(); U = U[U.status == "canonical"].copy()
    check(len(S) == 1100 and len(U) == 960, f"canonical images SAUD {len(S)} (1100) / UID {len(U)} (960)")

    # R2: symmetric, deduplicated cross pairs from BOTH the pHash-NN and dHash-NN columns
    pr = []
    for r in X.itertuples():
        for nn, path, pdist, ddist in [(r.phash_nn, r.phash_nn_path, r.phash_dist, r.phash_nn_dhash_dist),
                                       (r.dhash_nn, r.dhash_nn_path, r.dhash_nn_phash_dist, r.dhash_dist)]:
            if pdist <= a.thresh or ddist <= a.thresh:
                s, u, sp, up = (r.query, nn, r.query_path, path) if r.query_set == "SAUD" else (nn, r.query, path, r.query_path)
                pr.append(dict(saud_scene=s, uid_scene=u, saud_path=sp, uid_path=up, phash_dist=pdist, dhash_dist=ddist))
    P = pd.DataFrame(pr, columns=["saud_scene", "uid_scene", "saud_path", "uid_path", "phash_dist", "dhash_dist"])
    P = P.sort_values(["phash_dist", "dhash_dist"]).drop_duplicates(["saud_scene", "uid_scene"]).reset_index(drop=True)
    P["tier"] = np.where(np.minimum(P.phash_dist, P.dhash_dist) <= 2, "near-identical", "borderline")
    P.to_csv(out / "overlap_pairs.csv", index=False)
    print(f"[INFO] cross-dataset overlap pairs: {len(P)} ({(P.tier == 'near-identical').sum()} near-identical, "
          f"{(P.tier == 'borderline').sum()} borderline) | UID scenes {P.uid_scene.nunique()} | SAUD scenes {P.saud_scene.nunique()}")
    check(P.uid_scene.is_unique and P.saud_scene.is_unique, "overlap is one-to-one (no UID scene matches two SAUD scenes)")
    ov = P.groupby("uid_scene").saud_scene.apply(lambda x: ";".join(sorted(x))).to_dict()
    U["excluded_overlap"] = U.scene_uid.map(lambda s: s in ov); U["overlap_with"] = U.scene_uid.map(ov)

    # R3
    for ds, df in [("SAUD", S), ("UID2021", U)]:
        w = Wn[Wn.dataset == ds]
        cl = clusters(sorted(df.scene_uid.unique()), list(zip(w.a, w.b)))
        df["cluster_id"] = df.scene_uid.map(cl).map(lambda c: f"{ds}_{c}")
        print(f"[INFO] {ds}: within-dataset candidate pairs {len(w)} -> {df.scene_uid.nunique()} scenes in {df.cluster_id.nunique()} clusters"
              + ("".join(f"\n       merged {r.a} + {r.b} (pHash {r.phash_dist}, dHash {r.dhash_dist})" for r in w.itertuples())))
    Wn.to_csv(out / "within_clusters.csv", index=False)

    # review sheet (documentation only; R2/R3 already decided)
    items = [("SAUD " + r.saud_scene, r.saud_path, "UID " + r.uid_scene, r.uid_path, f"p{r.phash_dist} d{r.dhash_dist} {r.tier}") for r in P.itertuples()]
    items += [(f"{r.dataset} {r.a}", r.a_path, f"{r.dataset} {r.b}", r.b_path, f"p{r.phash_dist} d{r.dhash_dist} within") for r in Wn.itertuples()]
    try:
        sheet(items, out / "review_sheet.png", f"A3b raw pairs (thresh {a.thresh}); decisions are rule-based, sheet is for the record")
    except Exception as e:
        print(f"[WARN] review sheet not written: {e}")

    # write + summary
    cols = ["dataset", "group", "scene", "scene_uid", "cluster_id", "kind", "method", "path", "H", "W", "sha256"]
    S[cols].to_csv(out / "saud_dev_manifest.csv", index=False)
    Uc = U[~U.excluded_overlap]
    U[cols + ["excluded_overlap", "overlap_with"]].to_csv(out / "uid2021_all_sensitivity_manifest.csv", index=False)
    Uc[cols].to_csv(out / "uid2021_confirm_primary_manifest.csv", index=False)
    old = out / "uid2021_confirm_manifest.csv"           # ambiguous name from the first version -> remove
    if old.exists(): old.unlink(); print(f"[INFO] removed ambiguous old file {old.name}")
    n_m = {"SAUD": 10, "UID2021": 15}
    def row(name, df, m):
        ns = df.scene_uid.nunique(); return dict(set=name, n_scenes=ns, n_clusters=df.cluster_id.nunique(), n_methods=m,
                                                 n_enhanced=int((df.kind == "enhanced").sum()), n_within_scene_pairs=ns * m * (m - 1) // 2)
    summ = [row("SAUD dev", S, 10), row("UID2021 confirm (primary, overlap removed)", Uc, 15), row("UID2021 all (sensitivity)", U, 15)]
    for c, g in U[U.kind == "raw"].groupby("group"):
        summ.append(dict(set=f"UID2021 category {c}", n_scenes=g.scene_uid.nunique(), n_scenes_kept=int((~g.excluded_overlap).sum())))
    summ = pd.DataFrame(summ); summ.to_csv(out / "frozen_sets_summary.csv", index=False)
    check((Uc.groupby("scene_uid").method.nunique() == 16).all(), "every kept UID scene has raw + 15 methods")
    check(not (set(Uc.scene_uid) & set(P.uid_scene)), "no overlapping UID scene left in the primary confirm set")
    prim = pd.read_csv(out / "uid2021_confirm_primary_manifest.csv")
    check(prim.scene_uid.nunique() == Uc.scene_uid.nunique() and not (set(prim.scene_uid) & set(P.uid_scene)),
          f"primary confirm FILE re-read: {prim.scene_uid.nunique()} scenes, no overlap scene inside")
    files = ["saud_dev_manifest.csv", "uid2021_confirm_primary_manifest.csv", "uid2021_all_sensitivity_manifest.csv"]
    h = {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files}
    h["n_scenes"] = {"saud_dev": int(S.scene_uid.nunique()), "uid_confirm_primary": int(prim.scene_uid.nunique()),
                     "uid_all_sensitivity": int(U.scene_uid.nunique())}
    h.update(thresh=a.thresh, rules="R1 canonical only; R2 UID scene excluded if raw overlaps SAUD; R3 within-dataset near-dups merged into one cluster")
    (out / "manifest_hashes.json").write_text(json.dumps(h, indent=2))
    print("\n== FROZEN SETS ==\n" + summ.to_string(index=False))
    print(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
