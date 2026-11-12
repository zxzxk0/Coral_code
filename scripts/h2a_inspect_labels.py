#!/usr/bin/env python
"""
H2a: first look at the subjective-label files (allowed now: QCS-v2 calibration frozen, H1 scores written).
Prints structure only (keys, shapes, dtypes, a few values, sheet layout) so the loader can be written
exactly. Computes NO correlation with any metric.

Usage: python scripts/h2a_inspect_labels.py [--saud "<SAUD dataset root>"] [--uid "<UID 2021 root>"]
"""
import argparse, hashlib, sys
import os
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C

DEF_SAUD = Path(os.environ.get("SAUD_DIR", "data/SAUD"))
DEF_UID = Path(os.environ.get("UID2021_DIR", "data/UID2021"))


def show_array(name, x, depth=0):
    pad = "   " * depth
    if isinstance(x, np.ndarray):
        print(f"{pad}{name}: ndarray shape={x.shape} dtype={x.dtype}")
        if x.dtype == object:
            for i, e in enumerate(x.flat):
                if i >= 5: print(f"{pad}   ..."); break
                show_array(f"[{i}]", e, depth + 1)
        elif x.dtype.names:
            for fld in x.dtype.names:
                show_array(f".{fld}", x[fld], depth + 1)
        elif x.size:
            flat = x.ravel()
            if np.issubdtype(x.dtype, np.number):
                f = flat.astype(float)
                print(f"{pad}   min {np.nanmin(f):.4g} max {np.nanmax(f):.4g} mean {np.nanmean(f):.4g} nan {int(np.isnan(f).sum())}")
            print(f"{pad}   first values: {flat[:12].tolist()}")
            if x.ndim == 2 and x.shape[0] <= 12:
                print(f"{pad}   full array:\n{x}")
            elif x.ndim == 2:
                print(f"{pad}   first 3 rows:\n{x[:3]}")
    else:
        print(f"{pad}{name}: {type(x).__name__} = {str(x)[:200]}")


def load_mat(p):
    try:
        from scipy.io import loadmat
        d = loadmat(p, squeeze_me=False, struct_as_record=True)
        return {k: v for k, v in d.items() if not k.startswith("__")}, "scipy (v5)"
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(p, "r") as h:
            h.visititems(lambda k, v: out.__setitem__(k, np.array(v)) if isinstance(v, h5py.Dataset) else None)
        return out, "h5py (v7.3)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--saud", default=str(DEF_SAUD)); ap.add_argument("--uid", default=str(DEF_UID))
    a = ap.parse_args()
    files = sorted(Path(a.saud).rglob("*.mat")) + sorted(Path(a.uid).rglob("*.xls*")) + \
            sorted(p for p in list(Path(a.saud).rglob("*")) + list(Path(a.uid).rglob("*"))
                   if p.suffix.lower() in {".txt", ".md", ".pdf", ".csv"} and "tree" not in p.name.lower())
    for p in files:
        print("\n" + "=" * 100 + f"\n{p}\n  size {p.stat().st_size} bytes | sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
        if p.suffix.lower() == ".mat":
            d, how = load_mat(p); print(f"  loaded with {how}; variables: {list(d)}")
            for k, v in d.items(): show_array(k, v, 1)
        elif p.suffix.lower().startswith(".xls"):
            xl = pd.ExcelFile(p); print(f"  sheets: {xl.sheet_names}")
            for sh in xl.sheet_names:
                df = xl.parse(sh, header=None)
                print(f"\n  --- sheet '{sh}' raw shape {df.shape} (header=None) ---")
                print(df.head(12).to_string(max_colwidth=40))
                print("  ... last rows:"); print(df.tail(3).to_string(max_colwidth=40))
        elif p.suffix.lower() in {".txt", ".md", ".csv"}:
            print(p.read_text(errors="replace")[:1500])
        else:
            print("  (pdf; open manually if it documents the label order)")
    print("\n[DONE] structure only; no metric was correlated with any label.")


if __name__ == "__main__":
    main()
