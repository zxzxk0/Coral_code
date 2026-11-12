"""Deterministic 2-fold split of the Test images for cross-fitted calibration.

Cluster = video id (stem text before '_frame', e.g. '3d_l443'); stems without '_frame' are their own
cluster. Clusters that share an A0c duplicate group are merged (union-find) so near-duplicate content
never lands in both folds. Clusters are shuffled with a fixed seed and assigned greedily to the fold with
fewer images (balanced by image count). Written once to _resub_v3/V2_calib/test_folds.csv and re-used.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from . import configs as C
from .io import list_images, stem

SEED = 20261021


def video_id(s):
    s = s.lower()
    return s.split("_frame")[0] if "_frame" in s else s


def make_folds(seed=SEED):
    stems = sorted({stem(p) for p in list_images(C.BASE / "Input" / "Test")})
    parent = {}
    def f(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def u(a, b): parent[f(a)] = f(b)
    for s in stems: f(video_id(s))
    hp = C.OUT / "A0c_duplicates" / "dup_groups.csv"
    n_merge = 0
    if hp.exists():
        G = pd.read_csv(hp); G = G[G.split == "Test"]; G["stem"] = G.stem.astype(str).str.lower()
        for _, g in G.groupby("dup_group"):
            vids = sorted({video_id(s) for s in g.stem if s in set(stems)})
            for v in vids[1:]:
                if f(v) != f(vids[0]): u(v, vids[0]); n_merge += 1
    df = pd.DataFrame({"stem": stems}); df["video"] = df.stem.map(video_id); df["cluster"] = df.video.map(f)
    sizes = df.cluster.value_counts()
    order = list(sizes.index); np.random.default_rng(seed).shuffle(order)
    order = sorted(order, key=lambda c: -sizes[c])  # big clusters first (stable sort keeps shuffled order among ties)
    load = {"A": 0, "B": 0}; assign = {}
    for c in order:
        k = "A" if load["A"] <= load["B"] else "B"; assign[c] = k; load[k] += sizes[c]
    df["fold"] = df.cluster.map(assign)
    return df, n_merge


def load_folds(create=True):
    p = C.OUT / "V2_calib" / "test_folds.csv"
    if p.exists():
        return pd.read_csv(p, dtype=str)
    if not create:
        raise FileNotFoundError(p)
    df, n_merge = make_folds()
    p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)
    print(f"[INFO] created {p}: {df.cluster.nunique()} clusters ({df.video.nunique()} videos, {n_merge} merges via A0c) | "
          f"fold A {int((df.fold == 'A').sum())} / fold B {int((df.fold == 'B').sum())} images | "
          f"largest cluster {df.cluster.value_counts().iat[0]}")
    return df
