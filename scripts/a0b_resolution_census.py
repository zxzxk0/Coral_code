"""
A0b: resolution census of the 4,815-image test split (and, optionally, any config folder).
Reads only image headers (PIL), no decoding.

Outputs in _resub_v3/A0b_resolution/:
  resolution_per_image.csv   stem, H, W, min_side, subset_guess (filename pattern), file
  resolution_summary.csv     counts per size bucket and per subset_guess
Debug: prints how many images fall below 96 px (NIQE undefined), 128, 256 px, and the
distinct (H,W) pairs that cover >=1% of images.

Usage: python scripts/a0b_resolution_census.py [--config Raw_Input]
"""
import sys, re, argparse
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C
from qcs_eval.io import list_images, stem

BUCKETS = [0, 96, 128, 256, 512, 1024, 10 ** 6]


def subset_guess(s):
    if re.search(r"_square\d+$", s): return "public_square_crop"
    if re.search(r"(frame|gopro|seg|video|^g[hx]\d)", s): return "custom_video"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Raw_Input")
    ap.add_argument("--out", default=str(C.OUT / "A0b_resolution"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec = C.TABLE1[a.config][0]
    parts = spec if isinstance(spec, (list, tuple)) else [spec]
    folder = [C.BASE / p for p in parts] if len(parts) > 1 else C.BASE / parts[0]
    rows = []
    for p in list_images(folder):
        with Image.open(p) as im:
            W, H = im.size
        s = stem(p)
        rows.append(dict(stem=s, H=H, W=W, min_side=min(H, W), subset_guess=subset_guess(s), file=str(p)))
    d = pd.DataFrame(rows)
    d["bucket"] = pd.cut(d.min_side, BUCKETS, right=False,
                         labels=["<96", "96-127", "128-255", "256-511", "512-1023", ">=1024"])
    d.to_csv(out / "resolution_per_image.csv", index=False)
    summ = d.groupby(["subset_guess", "bucket"], observed=False).size().unstack(fill_value=0)
    summ.to_csv(out / "resolution_summary.csv")
    n = len(d)
    print(f"[{'PASS' if n == C.N_TEST else 'WARN'}] {a.config}: {n} images (expected {C.N_TEST})")
    for t in (96, 128, 256):
        k = int((d.min_side < t).sum()); print(f"[INFO] min side < {t}px: {k} ({100 * k / n:.1f}%)")
    print("[INFO] most common (H,W):")
    vc = d.groupby(["H", "W"]).size().sort_values(ascending=False)
    for (h, w), c in vc[vc >= 0.01 * n].items():
        print(f"       {h}x{w}: {c} ({100 * c / n:.1f}%)")
    print("[INFO] subset_guess x bucket:\n" + summ.to_string())
    print("[INFO] filename examples per subset_guess:")
    for g, x in d.groupby("subset_guess"):
        print(f"       {g}: {x.stem.head(4).tolist()}")
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
