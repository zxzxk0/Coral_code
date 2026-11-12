"""
A0b: resolution audit of the coral corpus (header-only reads, no pixel decoding).

Writes _resub_v3/A0b_sizes/:
  sizes_per_image.csv   split, stem, H, W, min_side, subset guess (public 'speciesname_squareN' vs other)
  sizes_summary.csv     per split x subset: n, min/p5/median/max of min_side, counts <96, <128, <256 px
Usage: python scripts/a0b_image_sizes.py
"""
import sys, re
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C
from qcs_eval.io import list_images, stem

SPLITS = {"Train": "Input/Train", "Val": "Input/Val", "Test": "Input/Test"}


def main():
    out = C.OUT / "A0b_sizes"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, rel in SPLITS.items():
        d = C.BASE / rel
        if not d.is_dir():
            print(f"[WARN] missing {d}"); continue
        for p in list_images(d, on_duplicate="keep"):
            with Image.open(p) as im:
                W, H = im.size
            s = stem(p)
            rows.append(dict(split=split, stem=s, file=p.name, ext=p.suffix.lower(), H=H, W=W, min_side=min(H, W),
                             square_thumb=bool(re.search(r"_square\d+$", s)),
                             subset="public_square" if re.search(r"_square\d+$", s) else "other"))
    df = pd.DataFrame(rows)
    # duplicates inside a split: same lower-cased stem, different file (extension or letter case)
    df["dup_in_split"] = df.duplicated(["split", "stem"], keep=False)
    dups = df[df.dup_in_split].sort_values(["split", "stem", "file"])
    dups.to_csv(out / "duplicate_stems_within_split.csv", index=False)
    for sp, x in df.groupby("split"):
        k = x[x.dup_in_split].stem.nunique()
        print(f"[{'PASS' if k == 0 else 'WARN'}] {sp}: {len(x)} files, {x.stem.nunique()} unique stems, {k} stems duplicated within split")
    if len(dups):
        same_size = dups.groupby(["split", "stem"]).apply(lambda g: g[["H", "W"]].drop_duplicates().shape[0] == 1)
        print(f"       duplicated stems with identical size: {int(same_size.sum())}/{len(same_size)}"
              f"; extensions involved: {dups.ext.value_counts().to_dict()}")
        print("       examples:\n" + dups.head(6)[["split", "file", "H", "W"]].to_string(index=False))
    df.to_csv(out / "sizes_per_image.csv", index=False)
    g = df.groupby(["split", "subset"]).min_side
    summ = pd.DataFrame({"n": g.size(), "min": g.min(), "p5": g.quantile(.05), "median": g.median(), "max": g.max(),
                         "lt96": g.apply(lambda x: int((x < 96).sum())), "lt128": g.apply(lambda x: int((x < 128).sum())),
                         "lt256": g.apply(lambda x: int((x < 256).sum()))}).reset_index()
    summ.to_csv(out / "sizes_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("\nmost common sizes (H x W):")
    print(df.groupby(["H", "W"]).size().sort_values(ascending=False).head(10).to_string())
    # debug checks
    n = df.drop_duplicates(["split", "stem"]).groupby("split").size().to_dict()   # unique stems
    vt = n.get("Val", 0) + n.get("Test", 0)
    print(f"\n[{'PASS' if vt == C.N_TEST else 'FAIL'}] Val+Test = {vt} (expected {C.N_TEST}, the evaluated split)")
    print(f"[{'PASS' if n.get('Train', 0) == 7236 else 'WARN'}] Train = {n.get('Train', 0)} (paper: 7,236)")
    sets = {k: set(v) for k, v in df.groupby("split").stem}
    for a, b in [("Train", "Val"), ("Train", "Test"), ("Val", "Test")]:
        ov = sets.get(a, set()) & sets.get(b, set())
        print(f"[{'PASS' if not ov else 'FAIL'}] stem overlap {a}/{b}: {len(ov)}" + (f" e.g. {sorted(ov)[:3]}" if ov else ""))
    print("\nfilename examples (for scene/cluster ID rules):")
    for (sp, sub), x in df.groupby(["split", "subset"]):
        print(f"  {sp:<5} {sub:<14} {x.stem.sample(min(4, len(x)), random_state=0).tolist()}")
    print(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
