"""
A2a qualitative figure: what the stress-test strengths look like.

Rows   = example images (randomly drawn, seeded, from the stress sample) x operator (USM, dehazing)
Columns= input (0), pipeline default (USM 0.35 / dehaze 0.25), beyond the pipeline bound (1.5 / 0.6),
         strongest tested setting (4 / 2)
Under each tile: AG (stress-fail) and LIQE (stress-pass) computed on the FULL image, exactly as in the
stress test; the tile itself is a native-resolution centre crop (no resizing, so halos stay visible).

Outputs (in --out):
  fig_stress_examples.pdf / .png   paper figure (5.4 in wide, 8 pt text)
  stress_examples_scores.csv       every annotated value + per-row direction (paper-ready numbers)
  strips/*.png                     full-resolution strips over the whole grid, for inspection
  stress_examples_debug.txt        PASS/WARN/FAIL checks

Usage (from the repo root, same environment as a2a_stress.py):
  python scripts/a2a_example_figure.py --sample <A2a_stress sample-0 folder>/a2a_sample_stems.csv --pyiqa
  python scripts/a2a_example_figure.py --stems STEM1,STEM2 --pyiqa          # choose images explicitly
  python scripts/a2a_example_figure.py --source Raw_Input --pyiqa           # operators on the raw input
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C, metrics as M
from qcs_eval.features import compute_features
from qcs_eval.io import list_images, stem, read_bgr
from qcs_eval.operators import OPERATORS

SHOW = {"usm": [0, 0.35, 1.5, 4.0], "dehaze": [0, 0.25, 0.6, 2.0]}
COLHEAD = {"usm": ["input (0)", "pipeline default (0.35)", "beyond bound (1.5)", "strongest (4)"],
           "dehaze": ["input (0)", "pipeline default (0.25)", "pipeline bound (0.6)", "strongest (2)"]}
OPNAME = {"usm": "USM", "dehaze": "Dehazing"}
log = []


def say(m):
    print(m); log.append(m)


def folder_of(cfg):
    spec = C.TABLE1[cfg][0]
    parts = spec if isinstance(spec, (list, tuple)) else [spec]
    return [C.BASE / p for p in parts] if len(parts) > 1 else C.BASE / parts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(C.OUT / "A2a_stress" / "a2a_sample_stems.csv"),
                    help="a2a_sample_stems.csv of stress sample 0 (images are drawn from this list)")
    ap.add_argument("--stems", default=None, help="comma-separated stems; overrides random selection")
    ap.add_argument("--source", default="Retinex_+_LLaVA", help="Raw_Input | Retinex_w_o_post | Retinex_+_LLaVA")
    ap.add_argument("--n", type=int, default=2, help="number of example images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=256, help="centre-crop size in native pixels")
    ap.add_argument("--pyiqa", action="store_true", help="annotate LIQE (needs pyiqa); otherwise Tenengrad is shown")
    ap.add_argument("--long", default=None, help="optional a2a_long.csv of the same run, to cross-check AG values")
    ap.add_argument("--ops", default="usm,dehaze", help="operators to show, e.g. usm (main figure) or dehaze (supplement)")
    ap.add_argument("--max-input-clip", type=float, default=None,
                    help="selection rule stated in the caption: skip images whose input already has more than this fraction "
                         "of pixels clipped (any channel >= 250 or <= 5), so over-processing is not hidden by an already saturated input")
    ap.add_argument("--ag-only", action="store_true", help="annotate only AG under each tile (scores are still computed and saved for both metrics)")
    ap.add_argument("--field-only", action="store_true",
                    help="draw only from the authors' own field-survey frames (stems containing '_frame'), "
                         "so the published figure contains no third-party (iNaturalist/ReefNet) imagery")
    ap.add_argument("--out", default=str(C.OUT / "A2a_stress" / "examples_figure"))
    a = ap.parse_args()
    out = Path(a.out); (out / "strips").mkdir(parents=True, exist_ok=True)
    keep_ops = [o.strip() for o in a.ops.split(",") if o.strip()]
    for o in list(SHOW):
        if o not in keep_ops: SHOW.pop(o)
    assert SHOW, "[FAIL] --ops selected no operator"

    # ---------------- debug 0: displayed strengths are on the stress grid
    for op, s_list in SHOW.items():
        grid = OPERATORS[op][1]
        bad = [s for s in s_list if not any(abs(s - g) < 1e-12 for g in grid)]
        assert not bad, f"[FAIL] {op} strengths {bad} not in the stress grid {grid}"
    say("[PASS] displayed strengths are all on the stress grid")

    second = "tenengrad"
    if a.pyiqa:
        en = M.enable_pyiqa(("liqe",))
        if "liqe" in en: second = "liqe"
        else: say("[WARN] LIQE unavailable -> Tenengrad shown instead (stress-fail, so the figure loses its PASS example)")
    say(f"[INFO] annotated metrics: AG (stress-fail) and {second.upper()} ({'stress-pass' if second == 'liqe' else 'stress-fail'})")

    files = {stem(p): p for p in list_images(folder_of(a.source))}
    if a.stems:
        pick = [s.strip() for s in a.stems.split(",") if s.strip()]
        how = "chosen explicitly (--stems)"
    else:
        pool = pd.read_csv(a.sample)["stem"].astype(str).tolist()
        pool = [s for s in pool if s in files]
        if a.field_only:
            pool = [s for s in pool if "_frame" in s.lower()]
            say(f"[INFO] --field-only: {len(pool)} field-survey frames in the sample")
        ok = []
        for s in pool:                                   # a crop larger than the image would hide nothing but break the layout
            h, w = read_bgr(files[s]).shape[:2]
            if min(h, w) < a.crop: continue
            if a.max_input_clip is not None:
                im0 = read_bgr(files[s]); clip = float(((im0 >= 250) | (im0 <= 5)).any(2).mean())
                if clip > a.max_input_clip: continue
            ok.append(s)
        rule = ("field-survey frames with " if a.field_only else "") + f"short side >= {a.crop}px" + (f" and input clipped-pixel fraction <= {a.max_input_clip:g}" if a.max_input_clip is not None else "")
        say(f"[INFO] {len(ok)}/{len(pool)} sample stems satisfy {rule}")
        assert ok, "[FAIL] no stem satisfies the selection rule"
        rng = np.random.default_rng(a.seed)
        pick = sorted(rng.choice(ok, min(a.n, len(ok)), replace=False).tolist())
        how = f"drawn at random (seed {a.seed}) from {Path(a.sample).name} among stems with {rule}"
    missing = [s for s in pick if s not in files]
    assert not missing, f"[FAIL] stems not found in {a.source}: {missing}"
    say(f"[INFO] images {pick} {how}, source={a.source}")
    for s in pick:
        im0 = read_bgr(files[s]); say(f"[INFO]   {s}: input clipped-pixel fraction {((im0 >= 250) | (im0 <= 5)).any(2).mean():.3f}, mean intensity {im0.mean():.1f}")

    def score(im, name):
        f = compute_features(im)
        return float(M.REGISTRY[name]["fn"](f, im)[0])

    rows, tiles = [], {}
    for sid in pick:
        img = read_bgr(files[sid])
        H, W = img.shape[:2]; c = min(a.crop, H, W)
        y0, x0 = (H - c) // 2, (W - c) // 2
        for op, (fn, grid) in OPERATORS.items():
            if op not in SHOW: continue
            strip = []
            for s_ in grid:
                im = fn(img, s_)
                if s_ == 0:
                    assert np.array_equal(im, img), f"[FAIL] {op} strength 0 is not the identity for {sid}"
                ag, m2 = score(im, "ag"), score(im, second)
                tile = im[y0:y0 + c, x0:x0 + c].copy()
                bar = np.full((22, c, 3), 255, np.uint8)
                cv2.putText(bar, f"{op}={s_:g} AG={ag:.1f} {second}={m2:.2f}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
                strip.append(np.vstack([tile, bar]))
                shown = any(abs(s_ - t) < 1e-12 for t in SHOW[op])
                if shown: tiles[(sid, op, s_)] = tile
                rows.append(dict(stem=sid, source=a.source, operator=op, strength=s_, shown_in_figure=shown,
                                 AG=ag, **{second.upper(): m2}, H=H, W=W, crop=c, crop_y0=y0, crop_x0=x0))
            cv2.imwrite(str(out / "strips" / f"{sid}__{a.source}__{op}.png"), np.hstack(strip))
    S = pd.DataFrame(rows)

    # ---------------- paper-ready summary: where each annotated score peaks on the full grid
    summ = []
    for (sid, op), g in S.groupby(["stem", "operator"]):
        g = g.sort_values("strength")
        for mcol in ["AG", second.upper()]:
            v = g[mcol].to_numpy(); s = g.strength.to_numpy()
            summ.append(dict(stem=sid, operator=op, metric=mcol, value_at_0=v[0], value_at_max=v[-1],
                             peak_strength=float(s[int(np.nanargmax(v))]),
                             rewards_strongest=bool(np.nanargmax(v) == len(v) - 1)))
    P = pd.DataFrame(summ)
    S.to_csv(out / "stress_examples_scores.csv", index=False)
    P.to_csv(out / "stress_examples_peaks.csv", index=False)
    for (op, mcol), g in P.groupby(["operator", "metric"]):
        say(f"[INFO] {OPNAME[op]:8s} {mcol:9s}: peak strength per image {g.peak_strength.tolist()}, "
            f"maximum at strongest setting in {int(g.rewards_strongest.sum())}/{len(g)} images")

    # ---------------- debug: the examples behave like the aggregate stress result (warn only; examples are not selected on this)
    for op in SHOW:
        ag = P[(P.operator == op) & (P.metric == "AG")]
        if op == "usm" and not ag.rewards_strongest.all():
            say("[WARN] AG does not peak at the strongest USM setting for every example (aggregate: AG fails under USM)")
        if second == "liqe":
            lq = P[(P.operator == op) & (P.metric == "LIQE")]
            if lq.rewards_strongest.any():
                say(f"[WARN] LIQE peaks at the strongest {op} setting for some example (aggregate: LIQE passes)")
    if a.long and Path(a.long).exists():
        L = pd.read_csv(a.long)
        cols = {c.lower(): c for c in L.columns}
        try:
            for met, col in [("ag", "AG"), (second, second.upper())]:
                q = L[(L["stem"].astype(str).isin(pick)) & (L["source"] == a.source) & (L["metric"] == met)]
                q = q[["stem", "operator", "strength", "score"]].copy(); q["strength"] = q.strength.round(6)
                s2 = S[["stem", "operator", "strength", col]].copy(); s2["strength"] = s2.strength.round(6)
                mrg = s2.merge(q, on=["stem", "operator", "strength"])
                if mrg.empty:
                    say(f"[WARN] {col}: no matching rows in a2a_long.csv"); continue
                d = (mrg[col] - mrg["score"]).abs().max()
                tol = 1e-6 if met == "ag" else 1e-3        # deep model: GPU/CPU nondeterminism allowed
                say(f"[{'PASS' if d < tol else 'FAIL'}] {col} matches a2a_long.csv (max |diff| = {d:.2e}, n={len(mrg)})")
        except Exception as e:
            say(f"[WARN] a2a_long.csv cross-check skipped ({type(e).__name__}: {e})")

    # ---------------- paper figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42, "image.composite_image": False})
    nr = len(pick) * len(SHOW)
    fig, ax = plt.subplots(nr, 4, figsize=(5.4, 1.48 * nr + 0.25), squeeze=False)
    r = 0
    for op in SHOW:
        for sid in pick:
            for j, s_ in enumerate(SHOW[op]):
                A = ax[r, j]
                A.imshow(cv2.cvtColor(tiles[(sid, op, s_)], cv2.COLOR_BGR2RGB), interpolation="none")   # embed the native pixels in the PDF, no resampling to the figure dpi
                A.set_xticks([]); A.set_yticks([])
                v = S[(S.stem == sid) & (S.operator == op) & (np.isclose(S.strength, s_))].iloc[0]
                A.set_xlabel(f"AG {v['AG']:.1f}" if a.ag_only else
                             (f"AG {v['AG']:.1f}  LIQE {v[second.upper()]:.2f}" if second == "liqe"
                              else f"AG {v['AG']:.1f}  Ten {v[second.upper()]:.0f}"), fontsize=7, labelpad=2)
                if r in (0, len(pick)): A.set_title(COLHEAD[op][j], fontsize=7, pad=3)
                if j == 0: A.set_ylabel(f"{OPNAME[op]}\nimage {pick.index(sid) + 1}", fontsize=7)
            r += 1
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    fig.savefig(out / "fig_stress_examples.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(out / "fig_stress_examples.png", dpi=400, bbox_inches="tight")
    say(f"[DONE] {out / 'fig_stress_examples.pdf'}")
    say(f"[INFO] caption facts: source={a.source}; images {how}; {a.crop}-px native centre crops; "
        f"scores on the full image; grid maxima USM 4, dehazing 2; pipeline bound 0.6")
    (out / "stress_examples_debug.txt").write_text("\n".join(log))


if __name__ == "__main__":
    main()
