"""
A0c: content-level duplicate / near-duplicate audit of the coral corpus (Input/Train, Val, Test),
plus the list of on-disk Val+Test images that are missing from the evaluated 4,815 set.

Why: public filenames ('_square', '_medium', '_original') look like iNaturalist size variants of
the same photo. If one photo is present at several sizes under different names, we get
train/eval leakage (cross-split) or inflated n / broken cluster independence (within eval).

Method (no extra packages; OpenCV only):
  - pHash (32x32 DCT -> 8x8 low freq, median threshold, 64 bits) of
      (a) the full image and (b) its centre square crop
    The 75x75 'square' thumbnails are compared against the centre-crop hash of every image,
    so a thumbnail matches the larger version it was cut from.
  - Candidate pair if Hamming distance <= --thr (default 6 of 64).
  - Every candidate pair is VERIFIED: both images resized to 64x64 grey (centre crops when the
    pair involves a square thumbnail), Pearson correlation of pixels reported; pairs with
    corr >= 0.95 are flagged 'confirmed'.

Outputs in _resub_v3/A0c_duplicates/:
  hashes.csv                  split, stem, file, H, W, phash_full, phash_center
  dup_pairs.csv               a, b, split_a, split_b, kind (full/center), hamming, pixel_corr, confirmed
  dup_groups.csv              connected components of confirmed pairs (group id per image)
  eval_missing_from_4815.csv  Val/Test files on disk that are not in the evaluated set (R2 cache)
  summary printed + a0c_summary.txt
Usage:  python scripts/a0c_duplicate_audit.py [--thr 6] [--workers N]
"""
import sys, os, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C
from qcs_eval.io import list_images, stem

SPLITS = {"Train": "Input/Train", "Val": "Input/Val", "Test": "Input/Test"}
log = []


def say(m):
    print(m); log.append(m)


def center_square(img):
    h, w = img.shape[:2]; c = min(h, w)
    return img[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]


def phash(gray):
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def _hash_one(args):
    split, path = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return dict(split=split, file=path, error="unreadable")
    return dict(split=split, stem=Path(path).stem.lower(), file=path, H=img.shape[0], W=img.shape[1],
                phash_full=phash(img), phash_center=phash(center_square(img)))


def hamming_matrix(a, b):
    """a: (n,) uint64, b: (m,) uint64 -> (n,m) popcount of xor, chunked."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    out = np.zeros(x.shape, np.uint8)
    for s in range(0, 64, 8):
        out += POP8[((x >> np.uint64(s)) & np.uint64(255)).astype(np.uint8)]
    return out


POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def candidate_pairs(h_a, h_b, thr, same=False, chunk=2000):
    pairs = []
    for i in range(0, len(h_a), chunk):
        D = hamming_matrix(h_a[i:i + chunk], h_b)
        ii, jj = np.nonzero(D <= thr)
        for u, v in zip(ii + i, jj):
            if same and v <= u:
                continue
            pairs.append((int(u), int(v), int(D[u - i, v])))
    return pairs


def pixel_corr(pa, pb, center):
    A = cv2.imread(pa, cv2.IMREAD_GRAYSCALE); B = cv2.imread(pb, cv2.IMREAD_GRAYSCALE)
    if center:
        A, B = center_square(A), center_square(B)
    A = cv2.resize(A, (64, 64), interpolation=cv2.INTER_AREA).astype(float).ravel()
    B = cv2.resize(B, (64, 64), interpolation=cv2.INTER_AREA).astype(float).ravel()
    if A.std() < 1e-6 or B.std() < 1e-6:
        return np.nan
    return float(np.corrcoef(A, B)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=int, default=6)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(C.OUT / "A0c_duplicates"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for split, rel in SPLITS.items():
        d = C.BASE / rel
        if d.is_dir():
            jobs += [(split, str(p)) for p in list_images(d, on_duplicate="keep")]
        else:
            say(f"[WARN] missing {d}")
    workers = a.workers or max(1, (os.cpu_count() or 2) - 1)
    say(f"[INFO] hashing {len(jobs)} images on {workers} workers")
    with ProcessPoolExecutor(workers) as ex:
        H = pd.DataFrame(list(ex.map(_hash_one, jobs, chunksize=32)))
    if "error" in H.columns and H.error.notna().any():
        say(f"[WARN] unreadable files: {int(H.error.notna().sum())} e.g. {H[H.error.notna()].file.head(3).tolist()}")
        H = H[H.error.isna()].copy()
    H["is_square"] = H.stem.str.contains(r"_square\d+$", regex=True)
    H = H.reset_index(drop=True)
    H.drop(columns=[c for c in ["error"] if c in H.columns]).to_csv(out / "hashes.csv", index=False)

    # --- self-check: hash must be robust to resize, sensitive to content ---
    t = H.iloc[0]
    img = cv2.imread(t.file, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (max(16, img.shape[1] // 4), max(16, img.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    d_resize = bin(phash(img) ^ phash(small)).count("1")
    other = cv2.imread(H.iloc[len(H) // 2].file, cv2.IMREAD_GRAYSCALE)
    d_other = bin(phash(img) ^ phash(other)).count("1")
    say(f"[{'PASS' if d_resize <= a.thr < d_other else 'WARN'}] pHash self-check: same image 1/4 size -> {d_resize} bits, "
        f"unrelated image -> {d_other} bits (thr={a.thr})")

    full = H.phash_full.to_numpy(np.uint64); cen = H.phash_center.to_numpy(np.uint64)
    pairs = [(u, v, d, "full") for u, v, d in candidate_pairs(full, full, a.thr, same=True)]
    sq = np.nonzero(H.is_square.to_numpy())[0]
    if len(sq):
        for u, v, d in candidate_pairs(full[sq], cen, a.thr):   # square thumb vs centre crop of everything
            uu = int(sq[u])
            if uu != v:
                pairs.append((min(uu, v), max(uu, v), d, "center"))
    P = pd.DataFrame(pairs, columns=["i", "j", "hamming", "kind"]).drop_duplicates(["i", "j"])
    say(f"[INFO] candidate pairs (hamming <= {a.thr}): {len(P)}")
    P["pixel_corr"] = [pixel_corr(H.file[i], H.file[j], k == "center") for i, j, k in zip(P.i, P.j, P.kind)]
    P["confirmed"] = P.pixel_corr >= 0.95
    for c in ["stem", "split", "H", "W"]:
        P[c + "_a"] = H[c].to_numpy()[P.i]; P[c + "_b"] = H[c].to_numpy()[P.j]
    P["cross_split"] = P.split_a != P.split_b
    P.to_csv(out / "dup_pairs.csv", index=False)

    # connected components of confirmed pairs
    parent = list(range(len(H)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in P[P.confirmed][["i", "j"]].itertuples(index=False):
        parent[find(i)] = find(j)
    H["dup_group"] = [find(i) for i in range(len(H))]
    sizes = H.dup_group.map(H.dup_group.value_counts())
    G = H[sizes > 1][["dup_group", "split", "stem", "H", "W", "file"]].sort_values(["dup_group", "split"])
    G.to_csv(out / "dup_groups.csv", index=False)

    conf = P[P.confirmed]
    say("\n== DUPLICATES (confirmed: pHash match AND pixel corr >= 0.95) ==")
    say(f"  confirmed pairs: {len(conf)}  (cross-split: {int(conf.cross_split.sum())}, "
        f"within Train: {int(((conf.split_a == 'Train') & (conf.split_b == 'Train')).sum())}, "
        f"within Val+Test: {int((conf.split_a.isin(['Val', 'Test']) & conf.split_b.isin(['Val', 'Test'])).sum())})")
    for sa in ["Train", "Val", "Test"]:
        for sb in ["Train", "Val", "Test"]:
            if sa < sb or sa == sb:
                k = int((((conf.split_a == sa) & (conf.split_b == sb)) | ((conf.split_a == sb) & (conf.split_b == sa))).sum()) if sa != sb \
                    else int(((conf.split_a == sa) & (conf.split_b == sa)).sum())
                if k: say(f"    {sa}-{sb}: {k}")
    vid = lambda x: x.str.contains("_frame_", regex=False)
    both_vid = vid(conf.stem_a) & vid(conf.stem_b)
    say(f"  by source: video-frame pairs {int(both_vid.sum())} (consecutive frames are expected; they define clusters, not leakage "
        f"unless cross-split), public/other pairs {int((~both_vid).sum())}")
    say(f"  images in a duplicate group: {len(G)} in {G.dup_group.nunique()} groups")
    ev = H[H.split.isin(["Val", "Test"])]
    ev_groups = ev.dup_group.nunique()
    say(f"  Val+Test: {len(ev)} files -> {ev_groups} unique contents (after merging confirmed duplicates)")
    leak = ev.dup_group.isin(H[H.split == "Train"].dup_group)
    say(f"  [{'PASS' if not leak.any() else 'FAIL'}] Val+Test images whose content also appears in Train: {int(leak.sum())}")
    say(f"  by kind: {conf.kind.value_counts().to_dict()};  involving a square thumb: "
        f"{int((conf.stem_a.str.contains('_square') | conf.stem_b.str.contains('_square')).sum())}")
    if len(conf):
        say("  examples:\n" + conf.sort_values("pixel_corr", ascending=False).head(8)[
            ["stem_a", "split_a", "H_a", "W_a", "stem_b", "split_b", "H_b", "W_b", "kind", "hamming", "pixel_corr"]].to_string(index=False))
    near = P[~P.confirmed & (P.pixel_corr >= 0.85)]
    say(f"  unconfirmed near-matches (0.85<=corr<0.95, check by eye): {len(near)}")

    # --- on-disk eval images missing from the evaluated set ---
    if C.R2_FEATURE_CACHE.exists():
        cache = pd.read_csv(C.R2_FEATURE_CACHE, usecols=lambda c: c in {"Configuration", "Filename", "stem"})
        if "stem" not in cache.columns:
            cache["stem"] = cache.Filename.astype(str).str.rsplit(".", n=1).str[0].str.lower()
        evset = set(cache.loc[cache.Configuration == "Raw_Input", "stem"].astype(str).str.lower())
        miss = ev[~ev.stem.isin(evset)][["split", "stem", "H", "W", "file", "dup_group"]]
        miss = miss.assign(in_dup_group=miss.dup_group.map(H.dup_group.value_counts()) > 1)
        miss.to_csv(out / "eval_missing_from_4815.csv", index=False)
        say(f"\n[{'PASS' if len(miss) == len(ev) - len(evset) else 'WARN'}] Val+Test on disk={len(ev)}, evaluated={len(evset)}, "
            f"missing from evaluation={len(miss)}")
        if len(miss):
            say(miss[["split", "stem", "H", "W", "in_dup_group"]].to_string(index=False))
        extra = evset - set(ev.stem)
        if extra:
            say(f"[WARN] {len(extra)} evaluated stems not found on disk, e.g. {sorted(extra)[:3]}")
    (out / "a0c_summary.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
