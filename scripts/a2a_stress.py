"""
A2a: deterministic operator stress test (USM / dehaze / CLAHE sweeps).

For a seeded sample of stems shared by the chosen sources, every operator is applied at every
strength on every image, all registered metrics are computed, and the following are written to
_resub_v3/A2a_stress/:

  a2a_long.csv            one row per image x source x operator x strength x metric
                          (raw/latent value, reported score, clip flags)
  a2a_per_image.csv       one row per image x source x operator x metric
                          (monotone?, #violations, peak strength, peak drop in baseline-SD units,
                           clip-induced ties, Spearman of trajectory vs Tenengrad trajectory)
  a2a_summary.csv         aggregated per source x operator x metric (paper-ready numbers:
                          % monotone, % interior peak, mean change at each strength with 95% CI,
                          % images clipped at 1, % adjacent-strength ties caused by clipping)
  a2a_curves.png          mean change vs strength (baseline-SD units), one panel per operator x source
  a2a_debug.txt           PASS/WARN/FAIL checks

Debug checks run first: identity at strength 0, features at strength 0 equal the A0-verified cache,
operator direction sanity (USM raises Tenengrad, dehaze lowers dark channel, CLAHE raises L contrast).

Usage:
  python scripts/a2a_stress.py --n 200
  python scripts/a2a_stress.py --n 200 --pyiqa          # also MUSIQ + NIQE (pip install pyiqa)
  python scripts/a2a_stress.py --n 20 --quick           # smoke run
"""
import sys, argparse, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C, metrics as M
from qcs_eval.features import compute_features, FEATURES
from qcs_eval.io import list_images, stem, read_bgr
from qcs_eval.operators import OPERATORS
from qcs_eval.v2 import compute_v2_features

DEFAULT_SOURCES = ["Raw_Input", "Retinex_w_o_post", "Retinex_+_LLaVA"]
log = []


def say(m):
    print(m); log.append(m)


def folder_of(cfg):
    spec = C.TABLE1[cfg][0]
    parts = spec if isinstance(spec, (list, tuple)) else [spec]
    return [C.BASE / p for p in parts] if len(parts) > 1 else C.BASE / parts[0]


def dark_channel_mean(img):
    return float(cv2.erode(img.min(2), np.ones((15, 15), np.uint8)).mean())


def l_std(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].std())


def _worker(args):
    """One image x one operator: all strengths. Returns hand-crafted features + sanity stats."""
    path, src, sid, op, want_v2 = args
    fn, grid = OPERATORS[op]
    img = read_bgr(path)
    out = []
    for k, s in enumerate(grid):
        im = fn(img, s)
        f = compute_features(im)
        if want_v2:
            f.update(compute_v2_features(im))
        f.update(stem=sid, source=src, operator=op, strength=s, k=k, H=img.shape[0], W=img.shape[1],
                 identity_ok=bool(np.array_equal(im, img)) if k == 0 else None,
                 dark_mean=dark_channel_mean(im), L_std=l_std(im), mean_L=float(im.mean()))
        out.append(f)
    return out


def per_image_analysis(long):
    rows = []
    base = long[long.k == 0].groupby(["source", "operator", "metric"])
    sd0 = base["good"].std().rename("sd0").reset_index()
    sdl = base["latent_good"].std().rename("sd0_lat").reset_index()
    g = long.merge(sd0, on=["source", "operator", "metric"]).merge(sdl, on=["source", "operator", "metric"])
    ten = long[long.metric == "tenengrad"][["stem", "source", "operator", "k", "good"]].rename(columns={"good": "ten"})
    g = g.merge(ten, on=["stem", "source", "operator", "k"], how="left")
    for (sid, src, op, met), d in g.groupby(["stem", "source", "operator", "metric"], sort=False):
        d = d.sort_values("k")
        v = d.good.to_numpy(); lat = d.latent_good.to_numpy(); s = d.strength.to_numpy()
        tol = 1e-9 * (1 + np.abs(v).max())
        dv = np.diff(v)
        viol = int((dv < -tol).sum())
        kmax = int(np.argmax(v))
        sd = float(d.sd0.iloc[0]) if d.sd0.iloc[0] > 0 else np.nan
        drop = (v[kmax] - v[-1]) / sd if np.isfinite(sd) else np.nan
        tie_clip = int(((np.abs(dv) <= tol) & (np.abs(np.diff(lat)) > tol)).sum())
        # same analysis on the pre-clipping latent score (identical for unclipped metrics)
        tl = 1e-9 * (1 + np.abs(lat).max()); kl = int(np.argmax(lat))
        sdl = float(d.sd0_lat.iloc[0]) if d.sd0_lat.iloc[0] > 0 else np.nan
        dropl = (lat[kl] - lat[-1]) / sdl if np.isfinite(sdl) else np.nan
        viol_l = int((np.diff(lat) < -tl).sum())
        tv = d.ten.to_numpy()
        const = np.ptp(v) <= tol or np.ptp(tv) <= 1e-12
        rho = 1.0 if met == "tenengrad" else (np.nan if const else pd.Series(v).corr(pd.Series(tv), method="spearman"))
        rows.append(dict(stem=sid, source=src, operator=op, metric=met, n_steps=len(v), violations=viol,
                         monotone=viol == 0, peak_k=kmax, peak_strength=float(s[kmax]),
                         interior_peak=bool(kmax < len(v) - 1 and np.isfinite(drop) and drop > 0.1),
                         peak_drop_sd=drop, clip_ties=tie_clip, spearman_vs_tenengrad=rho,
                         change_at_max_sd=(v[-1] - v[0]) / sd if np.isfinite(sd) else np.nan,
                         monotone_latent=viol_l == 0, peak_strength_latent=float(s[kl]),
                         interior_peak_latent=bool(kl < len(lat) - 1 and np.isfinite(dropl) and dropl > 0.1),
                         peak_drop_latent_sd=dropl,
                         change_at_max_latent_sd=(lat[-1] - lat[0]) / sdl if np.isfinite(sdl) else np.nan))
    return pd.DataFrame(rows)


def summarize(long, per, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    sd0 = long[long.k == 0].groupby(["source", "operator", "metric"])["good"].std()
    base = long[long.k == 0].set_index(["stem", "source", "operator", "metric"])["good"]
    L = long.copy()
    L["delta_sd"] = (L.good - base.reindex(pd.MultiIndex.from_frame(L[["stem", "source", "operator", "metric"]])).to_numpy()) \
        / sd0.reindex(pd.MultiIndex.from_frame(L[["source", "operator", "metric"]])).to_numpy()
    rows = []
    for (src, op, met, k), d in L.groupby(["source", "operator", "metric", "k"]):
        x = d.delta_sd.dropna().to_numpy()
        b = np.array([rng.choice(x, len(x)).mean() for _ in range(n_boot)]) if len(x) else np.array([np.nan])
        rows.append(dict(source=src, operator=op, metric=met, k=k, strength=d.strength.iloc[0], n=len(x),
                         mean_delta_sd=x.mean() if len(x) else np.nan,
                         ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5),
                         mean_score=d.score.mean(), pct_clip_hi=100 * d.clip_hi.mean(), pct_clip_lo=100 * d.clip_lo.mean()))
    curve = pd.DataFrame(rows)
    agg = per.groupby(["source", "operator", "metric"]).agg(
        n_images=("stem", "nunique"), pct_monotone=("monotone", lambda x: 100 * x.mean()),
        pct_interior_peak=("interior_peak", lambda x: 100 * x.mean()),
        median_peak_strength=("peak_strength", "median"),
        mean_change_at_max_sd=("change_at_max_sd", "mean"),
        pct_monotone_latent=("monotone_latent", lambda x: 100 * x.mean()),
        pct_interior_peak_latent=("interior_peak_latent", lambda x: 100 * x.mean()),
        median_peak_strength_latent=("peak_strength_latent", "median"),
        mean_change_at_max_latent_sd=("change_at_max_latent_sd", "mean"),
        mean_clip_ties_per_image=("clip_ties", "mean"),
        mean_spearman_vs_tenengrad=("spearman_vs_tenengrad", "mean")).reset_index()
    return curve, agg


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# fixed metric -> colour so the same metric has the same colour in every figure (max 8 per figure)
METRIC_COLOR = {"qcs_v1_latent": "#6b6b6b", "v2_A": "#2a78d6", "v2_B": "#1baf7a", "v2_C": "#e34948", "v2_D": "#6250d6", "v2_D2": "#eda100",
                "tenengrad": "#eb6834", "ag": "#eda100", "uiqm": "#e87ba4", "uciqe": "#008300",
                "musiq": "#2a78d6", "niqe": "#1baf7a", "qcs_v1": "#6b6b6b"}
FIGURES = {"a2a_curves_v2.png": ["qcs_v1_latent", "v2_A", "v2_B", "v2_C", "v2_D", "v2_D2"],
           "a2a_curves_baselines.png": ["qcs_v1_latent", "tenengrad", "ag", "uiqm", "uciqe", "musiq", "niqe"]}


def plot(curve, sources, metrics, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ops = list(OPERATORS)
    fig, axes = plt.subplots(len(ops), len(sources), figsize=(4.2 * len(sources), 3.2 * len(ops)),
                             squeeze=False, sharey="row")
    cmap = {m: METRIC_COLOR.get(m, COLORS[i % len(COLORS)]) for i, m in enumerate(metrics)}
    pipeline_pt = {"usm": 0.18, "dehaze": 0.15, "clahe": 0.012}
    for i, op in enumerate(ops):
        for j, src in enumerate(sources):
            ax = axes[i][j]
            for m in metrics:
                d = curve[(curve.operator == op) & (curve.source == src) & (curve.metric == m)].sort_values("k")
                if d.empty: continue
                ax.plot(d.strength, d.mean_delta_sd, color=cmap[m], lw=2, marker="o", ms=4, label=m)
                ax.fill_between(d.strength, d.ci_lo, d.ci_hi, color=cmap[m], alpha=0.15, lw=0)
            ax.axhline(0, color="#999", lw=0.8)
            ax.axvline(pipeline_pt[op], color="#999", lw=0.8, ls="--")
            ax.set_title(f"{op}  |  {src}", fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            if j == 0: ax.set_ylabel("change vs strength 0\n(baseline SD units, higher=better)", fontsize=8)
            if i == len(ops) - 1: ax.set_xlabel("operator strength (dashed = pipeline operating point)", fontsize=8)
    axes[0][-1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def debug_checks(feat, cache, out):
    ok_all = True
    id_ok = feat[feat.k == 0].identity_ok.all()
    say(f"[{'PASS' if id_ok else 'FAIL'}] strength-0 output identical to input for every image/operator"); ok_all &= id_ok
    if cache is not None:
        f0 = feat[(feat.k == 0) & (feat.operator == "usm")][["stem", "source"] + FEATURES]
        m = f0.merge(cache, left_on=["stem", "source"], right_on=["stem", "Configuration"], suffixes=("", "_c"))
        rel = max(float((np.abs(m[f] - m[f + "_c"]) / np.abs(m[f + "_c"]).clip(1e-12)).max()) for f in FEATURES) if len(m) else np.nan
        ok = len(m) == len(f0) and rel < 1e-9
        say(f"[{'PASS' if ok else 'FAIL'}] strength-0 features == A0-verified cache ({len(m)}/{len(f0)} matched, max rel diff {rel:.1e})"); ok_all &= ok
    for op, col, sign, txt in [("usm", "Tenengrad", 1, "USM raises Tenengrad"), ("dehaze", "mean_L", 1, "pipeline 'dehaze' (divide by t<=1) raises brightness"),
                               ("clahe", "L_std", 1, "CLAHE raises L contrast")]:
        d = feat[feat.operator == op]
        lo = d[d.k == 0].set_index(["stem", "source"])[col]; hi = d[d.k == d.k.max()].set_index(["stem", "source"])[col]
        frac = float((np.sign(hi - lo.reindex(hi.index)) == sign).mean())
        say(f"[{'PASS' if frac > 0.9 else 'WARN'}] {txt} at max strength in {100 * frac:.1f}% of images")
    return ok_all


def dump_examples(files, sources, stems, out_dir, transform, crop=360):
    """Centre crop at native resolution (no resizing, so halos stay visible), strengths left->right,
    with QCS_v1 / latent / Tenengrad printed under each tile."""
    from qcs_eval import transform as T
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in stems:
        for src in sources:
            img = read_bgr(files[src][sid])
            H, W = img.shape[:2]; c = min(crop, H, W)
            y0, x0 = (H - c) // 2, (W - c) // 2
            for op, (fn, grid) in OPERATORS.items():
                tiles = []
                for s_ in grid:
                    im = fn(img, s_)
                    f = compute_features(im)
                    q, z = T.apply(np.array([[f[k] for k in transform["order"]]]), transform, return_raw=True)
                    tile = im[y0:y0 + c, x0:x0 + c].copy()
                    bar = np.full((44, c, 3), 255, np.uint8)
                    cv2.putText(bar, f"{op}={s_:g}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(bar, f"QCS={q[0]:.2f} lat={z[0]:+.2f} Ten={f['Tenengrad']:.0f}", (4, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                    tiles.append(np.vstack([tile, bar]))
                cv2.imwrite(str(out_dir / f"{sid}__{src}__{op}.png"), np.hstack(tiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    ap.add_argument("--transform", default=str(C.OUT / "A0_repro" / "frozen_transform_r2.json"))
    ap.add_argument("--cache", default=str(C.R2_FEATURE_CACHE))
    ap.add_argument("--pyiqa", action="store_true")
    ap.add_argument("--pyiqa-metrics", default="musiq,niqe,topiq_nr,liqe,hyperiqa,tres")
    ap.add_argument("--exclude-sample", default=None,
                    help="a2a_sample_stems.csv of a previous run; those stems are excluded (independent confirmation sample)")
    ap.add_argument("--split", choices=["all", "Val", "Test"], default="all",
                    help="sample stems only from this split (Stage 3: Test, since calibration uses Val)")
    ap.add_argument("--fold", choices=["all", "A", "B"], default="all",
                    help="sample only Test stems of this cross-fitting fold (calibrate on the OTHER fold)")
    ap.add_argument("--min-side", type=int, default=128, help="clean-set rule: skip images whose short side < this (0 = full set)")
    ap.add_argument("--v2-calib", default=None, help="calibration JSON from scripts/v2_calibrate.py -> also test v2_A..D")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap reps")
    ap.add_argument("--dump-examples", type=int, default=0,
                    help="save N example strips (one per operator/source) of the swept images for visual inspection")
    ap.add_argument("--only-dump", action="store_true", help="only write the example strips, skip scoring")
    ap.add_argument("--out", default=str(C.OUT / "A2a_stress"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sources = [s.strip() for s in a.sources.split(",")]

    t = M.make_qcs_v1(a.transform)
    say(f"[INFO] qcs_v1 transform sha={t['sha256']} (must equal A0 frozen_transform_r2.json)")
    provisional = False
    if a.v2_calib:
        c = M.make_v2(a.v2_calib)
        provisional = bool(c.get("provisional", False))
        say(f"[INFO] v2 candidates registered; calibration sha={c['sha256']} ({c['desc']}, n={c['n']})")
        if provisional:
            say("[DEV ONLY] calibration is PROVISIONAL (fitted on the evaluated pool): results are for debugging, "
                "NOT for the candidate pass/fail decision")
    if a.pyiqa:
        en = M.enable_pyiqa(tuple(x.strip() for x in a.pyiqa_metrics.split(",") if x.strip()))
        say(f"[INFO] pyiqa metrics enabled: {en}")
    metric_names = list(M.REGISTRY)
    say(f"[INFO] metrics: {metric_names}")

    # common stems across sources
    files = {s: {stem(p): p for p in list_images(folder_of(s))} for s in sources}
    common = sorted(set.intersection(*[set(v) for v in files.values()]))
    if a.split != "all":
        sp = {stem(p) for p in list_images(C.BASE / "Input" / a.split)}
        before = len(common); common = [x for x in common if x in sp]
        say(f"[INFO] split rule: kept {len(common)}/{before} stems from Input/{a.split}")
    if a.fold != "all":
        from qcs_eval.folds import load_folds
        Fd = load_folds(create=False); fs = set(Fd.loc[Fd.fold == a.fold, "stem"])
        before = len(common); common = [x for x in common if x in fs]
        say(f"[INFO] fold rule: kept {len(common)}/{before} stems from Test fold {a.fold}")
        if a.v2_calib:
            import json
            desc = json.load(open(a.v2_calib)).get("desc", "")
            if f"fold={a.fold}" in desc:
                sys.exit(f"[FAIL] calibration {a.v2_calib} was fitted on fold {a.fold}; use the other fold's calibration")
    if a.min_side > 0:
        from PIL import Image
        ref = sources[0]
        keep = []
        for sid in common:
            with Image.open(files[ref][sid]) as im:
                if min(im.size) >= a.min_side: keep.append(sid)
        say(f"[INFO] clean-set rule: {len(common) - len(keep)} stems with short side < {a.min_side}px excluded (checked on {ref})")
        common = keep
    if a.exclude_sample:
        prev = set()
        for f in a.exclude_sample.split(";"):
            if f.strip(): prev |= set(pd.read_csv(f.strip())["stem"].astype(str))
        before = len(common); common = [x for x in common if x not in prev]
        say(f"[INFO] independent sample: excluded {before - len(common)} stems used in {a.exclude_sample}")
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(common, min(a.n, len(common)), replace=False))
    say(f"[INFO] {len(common)} stems shared by {sources}; sampled {len(pick)} (seed={a.seed})")
    pd.Series(pick, name="stem").to_csv(out / "a2a_sample_stems.csv", index=False)
    if a.dump_examples:
        dump_examples(files, sources, pick[:a.dump_examples], out / "examples", transform=t)
        if a.only_dump:
            print(f"[DONE] examples in {out / 'examples'}"); return

    jobs = [(str(files[s][sid]), s, sid, op, bool(a.v2_calib)) for s in sources for sid in pick for op in OPERATORS]
    workers = a.workers or max(1, (os.cpu_count() or 2) - 1)
    say(f"[INFO] {len(jobs)} image x operator jobs on {workers} workers")
    feats = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_worker, jobs, chunksize=4), 1):
            feats.extend(r)
            if i % 200 == 0 or i == len(jobs): print(f"  {i}/{len(jobs)}")
    feat = pd.DataFrame(feats)
    feat.to_csv(out / "a2a_features.csv", index=False)

    cache = None
    if Path(a.cache).exists():
        cache = pd.read_csv(a.cache, usecols=lambda c: c in {"Configuration", "Filename", "stem", "UIQM", "UCIQE_ref_ratio", "AG", "Tenengrad"})
        cache = cache.rename(columns={"UCIQE_ref_ratio": "UCIQE"})
        if "stem" not in cache.columns:
            cache["stem"] = cache.Filename.astype(str).str.rsplit(".", n=1).str[0].str.lower()
    say("\n== DEBUG ==")
    debug_checks(feat, cache, out)
    sz = feat[(feat.k == 0) & (feat.operator == "usm")][["stem", "source", "H", "W"]]
    ms = sz[["H", "W"]].min(1)
    say(f"[INFO] image size (min side) over sampled images: min={ms.min()} p5={ms.quantile(.05):.0f} median={ms.median():.0f} max={ms.max()}"
        f"; <96px: {int((ms < 96).sum())}, <256px: {int((ms < 256).sum())} of {len(ms)}")
    if (ms < 96).any():
        say(f"[WARN] tiny images (min side <96, NIQE undefined): {sz[ms < 96].head(5).to_dict('records')}")
    sz.to_csv(out / "a2a_image_sizes.csv", index=False)

    # metrics
    rows = []
    img_metrics = [m for m in metric_names if M.REGISTRY[m]["needs_image"]]
    for r in feat.itertuples(index=False):
        f = r._asdict()
        img = None
        if img_metrics:
            fn, _ = OPERATORS[r.operator]
            img = fn(read_bgr(files[r.source][r.stem]), r.strength)
        for m in metric_names:
            spec = M.REGISTRY[m]
            if spec["needs_image"] and img is None: continue
            sc, lat, clo, chi = spec["fn"](f, img)
            sgn = 1 if spec["hib"] else -1
            rows.append(dict(stem=r.stem, source=r.source, operator=r.operator, strength=r.strength, k=r.k,
                             metric=m, higher_is_better=spec["hib"], latent=lat, score=sc, clip_lo=clo, clip_hi=chi,
                             good=sgn * sc, latent_good=sgn * lat))
    long = pd.DataFrame(rows)
    if M.FAILURES:
        say(f"[WARN] metric evaluations returned NaN (image too small / runtime error): {M.FAILURES}")
    long = long.dropna(subset=["score"])
    long.to_csv(out / "a2a_long.csv", index=False)

    per = per_image_analysis(long)
    per.to_csv(out / "a2a_per_image.csv", index=False)
    curve, agg = summarize(long, per, n_boot=200 if a.quick else 1000)
    curve.to_csv(out / "a2a_curves.csv", index=False)
    agg.to_csv(out / "a2a_summary.csv", index=False)
    for fname, wanted in FIGURES.items():
        ms = [m for m in wanted if m in metric_names]
        if len(ms) > 1:
            plot(curve, sources, ms, out / fname)
    say(f"[INFO] figures: {', '.join(FIGURES)}")

    # ---- stress gate (pre-registered): substantive turnover, not Tenengrad correlation ----
    G = agg[agg.operator.isin(["usm", "dehaze"])].copy()
    G["turnover_pct"] = G.pct_interior_peak_latent          # interior peak with drop > 0.1 baseline-SD (latent score)
    G["gate_turnover"] = G.turnover_pct >= 50
    G["gate_no_clip_ties"] = G.mean_clip_ties_per_image == 0
    gate = G.groupby("metric").agg(min_turnover_pct=("turnover_pct", "min"),
                                   all_turnover=("gate_turnover", "all"), no_clip_ties=("gate_no_clip_ties", "all"),
                                   mean_rho_vs_Ten_diag=("mean_spearman_vs_tenengrad", "mean")).reset_index()
    gate["PASS"] = gate.all_turnover & gate.no_clip_ties
    gate["decision_valid"] = not provisional
    gate.to_csv(out / "a2a_gate.csv", index=False)
    G.to_csv(out / "a2a_gate_detail.csv", index=False)
    say("\n== STRESS GATE (USM & dehaze, every source: >=50% images with turnover; no clipping ties) ==")
    say("   (Tenengrad rho is diagnostic only)" + ("   [DEV ONLY - provisional calibration]" if provisional else ""))
    for r in gate.sort_values("min_turnover_pct", ascending=False).itertuples():
        say(f"   {r.metric:<14} {'PASS' if r.PASS else 'fail'}  min turnover={r.min_turnover_pct:5.1f}%  "
            f"clipTies0={r.no_clip_ties}  rho_vs_Ten(diag)={r.mean_rho_vs_Ten_diag:+.2f}")

    say("\n== SUMMARY (per operator x source; % monotone / % interior peak / change at max strength in baseline SD) ==")
    for (op, src), d in agg.groupby(["operator", "source"]):
        say(f"-- {op} | {src}")
        for r in d.itertuples():
            lat = "" if r.metric != "qcs_v1" else (f"  [latent: monotone={r.pct_monotone_latent:5.1f}% peak={r.pct_interior_peak_latent:5.1f}%"
                                                   f" d_max={r.mean_change_at_max_latent_sd:+6.2f}sd]")
            say(f"   {r.metric:<10} monotone={r.pct_monotone:5.1f}%  peak={r.pct_interior_peak:5.1f}%  "
                f"d_max={r.mean_change_at_max_sd:+6.2f}sd{lat}  rho_vs_Ten={r.mean_spearman_vs_tenengrad:+.2f}  clipTies/img={r.mean_clip_ties_per_image:.2f}")
    (out / "a2a_debug.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
