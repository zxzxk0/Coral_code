"""
A0d: inspect A0c duplicate groups + diagnose the evaluated-set count.

Reads _resub_v3/A0c_duplicates/{hashes.csv, dup_groups.csv} (no re-hashing).

Writes _resub_v3/A0d_inspect/:
  group_sizes.csv          group id, size, #Train/#Val/#Test, #distinct species prefixes, #evaluated,
                           sizes (HxW) present, example names
  sheets/group_XXX.png     contact sheet (up to 24 tiles, 128px, name under each) for the 15 largest
                           groups and 10 random 2-member groups
  eval_set_diagnosis.txt   how many cache Raw_Input stems fall in Val / Test / nowhere, with examples
Usage: python scripts/a0d_inspect_groups.py
"""
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C

IN = C.OUT / "A0c_duplicates"
OUT = C.OUT / "A0d_inspect"


def species(s):
    return re.sub(r"_(square|medium|original|small|large)?\d*$", "", s)


def sheet(rows, path, tile=128, ncol=6):
    tiles = []
    for r in rows.itertuples():
        im = cv2.imread(r.file)
        if im is None:
            im = np.zeros((tile, tile, 3), np.uint8)
        h, w = im.shape[:2]; s = tile / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.full((tile + 30, tile, 3), 255, np.uint8)
        canvas[:im.shape[0], :im.shape[1]] = im
        cv2.putText(canvas, f"{r.split} {r.H}x{r.W}", (2, tile + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, r.stem[:24], (2, tile + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) % ncol:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)])
    cv2.imwrite(str(path), grid)


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "sheets").mkdir(exist_ok=True)
    H = pd.read_csv(IN / "hashes.csv")
    G = pd.read_csv(IN / "dup_groups.csv")
    G = G.merge(H[["file", "split"]].drop_duplicates(), on=["file", "split"], how="left")

    # ---- evaluated-set diagnosis ----
    cache = pd.read_csv(C.R2_FEATURE_CACHE, usecols=lambda c: c in {"Configuration", "Filename", "stem"})
    raw = cache[cache.Configuration == "Raw_Input"].copy()
    has_stem = "stem" in raw.columns
    raw["k"] = (raw["stem"] if has_stem else raw.Filename.astype(str).str.rsplit(".", n=1).str[0]).astype(str).str.lower()
    val = set(H[H.split == "Val"].stem.astype(str)); test = set(H[H.split == "Test"].stem.astype(str))
    ks = set(raw.k)
    lines = [f"cache rows Raw_Input: {len(raw)}, unique keys: {len(ks)}, key source: {'stem column' if has_stem else 'Filename'}",
             f"keys found in Val: {len(ks & val)}, in Test: {len(ks & test)}, in neither: {len(ks - val - test)}",
             f"Val files on disk: {len(val)}, Test files on disk: {len(test)}",
             f"Val files not in cache: {len(val - ks)}, Test files not in cache: {len(test - ks)}",
             f"cache key examples: {sorted(ks)[:5]}",
             f"Val stem examples:  {sorted(val)[:5]}",
             f"Test stem examples: {sorted(test)[:5]}",
             f"cache keys in neither (first 10): {sorted(ks - val - test)[:10]}",
             f"Val/Test files not in cache (first 15): {sorted((val | test) - ks)[:15]}"]
    (OUT / "eval_set_diagnosis.txt").write_text("\n".join(lines), encoding="utf-8")
    print("== EVALUATED-SET DIAGNOSIS ==\n  " + "\n  ".join(lines))
    G["evaluated"] = G.stem.astype(str).isin(ks) & G.split.isin(["Val", "Test"])

    # ---- group table ----
    rows = []
    for gid, g in G.groupby("dup_group"):
        rows.append(dict(group=gid, size=len(g), n_train=int((g.split == "Train").sum()), n_val=int((g.split == "Val").sum()),
                         n_test=int((g.split == "Test").sum()), n_evaluated=int(g.evaluated.sum()),
                         n_species=g.stem.astype(str).map(species).nunique(),
                         sizes=";".join(sorted({f"{h}x{w}" for h, w in zip(g.H, g.W)}))[:80],
                         examples=";".join(g.stem.astype(str).head(4))))
    T = pd.DataFrame(rows).sort_values("size", ascending=False)
    T.to_csv(OUT / "group_sizes.csv", index=False)
    print("\n== DUPLICATE GROUPS ==")
    print(f"  groups: {len(T)}, images in groups: {int(T['size'].sum())}")
    print(f"  size distribution: " + ", ".join(f"{k}: {v}" for k, v in T['size'].value_counts().sort_index().items() if k <= 5)
          + f", >5: {int((T['size'] > 5).sum())} groups ({int(T.loc[T['size'] > 5, 'size'].sum())} images)")
    print(f"  groups spanning >1 species name: {int((T.n_species > 1).sum())} ({int(T.loc[T.n_species > 1, 'size'].sum())} images)")
    print(f"  evaluated images inside any group: {int(T.n_evaluated.sum())}; inside groups of size >5: {int(T.loc[T['size'] > 5, 'n_evaluated'].sum())}")
    print("  15 largest groups:")
    print(T.head(15)[["group", "size", "n_train", "n_val", "n_test", "n_evaluated", "n_species", "sizes"]].to_string(index=False))

    rng = np.random.default_rng(0)
    pick = list(T.head(15).group) + list(rng.choice(T[T["size"] == 2].group, min(10, int((T["size"] == 2).sum())), replace=False))
    for i, gid in enumerate(pick):
        g = G[G.dup_group == gid].head(24)
        tag = "large" if i < 15 else "pair"
        sheet(g, OUT / "sheets" / f"{tag}_{i:02d}_group{gid}_n{int(T.loc[T.group == gid, 'size'].iloc[0])}.png")
    print(f"\n[DONE] {OUT}  (look at sheets/large_00*.png first)")


if __name__ == "__main__":
    main()
