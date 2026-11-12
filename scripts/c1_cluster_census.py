"""Cluster composition of the 2,294 paired coral test images (for Section S7).
Run from the qcs_eval folder:  python scripts\\c1_cluster_census.py
Uses the same cluster file and the same raw short-side >= 128 px rule as c1_coral_rankings.py."""
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C
from qcs_eval.folds import load_folds
from qcs_eval.io import list_images, stem

F = load_folds(create=False)                                   # V2_calib/test_folds.csv
F["stem"] = F.stem.astype(str)
raw = {stem(p): p for p in list_images(C.BASE / "Input" / "Test")}
keep = []
for s in F.stem:
    p = raw.get(s) or raw.get(s.lower())
    if p is None: continue
    with Image.open(p) as im:
        if min(im.size) >= 128: keep.append(s)
D = F[F.stem.isin(keep)].copy()
D["kind"] = D.stem.str.lower().str.contains("_frame").map({True: "field video frame", False: "single image"})
print(f"[CHECK] test images {len(F)} (expect 2412) | after 128-px rule {len(D)} (expect 2294) | clusters {D.cluster.nunique()} (expect 1867)")
tab = D.groupby("kind").agg(images=("stem", "size"), clusters=("cluster", "nunique"), videos=("video", "nunique"))
print(tab.to_string())
mixed = D.groupby("cluster").kind.nunique(); print(f"[INFO] clusters mixing field frames and single images (A0c merges): {(mixed > 1).sum()}")
sz = D[D.kind == "field video frame"].groupby("cluster").size()
if len(sz): print(f"[INFO] field cluster size: median {sz.median():.0f}, max {sz.max()}, clusters with 1 frame {(sz == 1).sum()}")
tab.to_csv(C.OUT / "C1_coral_rankings" / "cluster_census.csv")
print("[DONE] cluster_census.csv written to _resub_v3/C1_coral_rankings")
