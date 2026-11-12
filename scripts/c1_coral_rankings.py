#!/usr/bin/env python
"""
C1: does the choice of NR metric change the conclusion about which coral enhancement pipeline is best?

All 14 Table-1 configurations are scored on the SAME Test images (common stems of Input/Test and every config
folder; eligibility fixed on the Raw image: short side >= 128 px). Metrics: QCS-v2 (v2_C, frozen calibration
753c406c848c7217), v2_B, v2_A, QCS-v1 (frozen R2 transform), legacy AG/Tenengrad/UIQM/UCIQE, and (--pyiqa)
MUSIQ, NIQE, TOPIQ_NR, LIQE (LIQE-only 224-px rule), HyperIQA, TReS. NIQE is sign-flipped (higher = better).

  --phase score   : per-config feature/metric CSVs in <out>/scores/ (resumable: finished configs are skipped)
  --phase analyze : ranking analysis from those CSVs
  --phase all     : both

Analysis (all on the paired image set):
  config means per metric + cluster-bootstrap 95% CI (clusters = Test video clusters, qcs_eval/folds.py)
  rank matrix (14 configs x metrics) + bootstrap rank 95% interval + P(rank 1)
  Kendall tau-b between every pair of metric rankings (+ mean tau of each metric to the others)
  focus contrasts: Retinex_+_LLaVA vs Retinex_w_o_post / Retinex_+_Rule_based / Retinex_+_Default_MHF-UIE
                   (paired per-image win rate and mean difference in SD units, bootstrap CI)
Outputs (<out>): c1_config_means.csv, c1_rank_matrix.csv, c1_rank_ci.csv, c1_top1.csv, c1_kendall.csv,
                 c1_focus_contrasts.csv, c1_rank_heatmap.png, c1_kendall_heatmap.png, c1_log.txt
Caveat (write in the paper): v2_C's calibration was fitted on these configurations (in-sample reference
distribution); v2_C is a case-study metric here, not ground truth.
"""
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, cv2
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C, v2, transform as T
from qcs_eval.features import compute_features
from qcs_eval.io import list_images, stem

EXPECTED_V2_SHA = "753c406c848c7217"
HAND = ["v2_C", "v2_B", "v2_A", "qcs_v1", "legacy_AG", "legacy_Tenengrad", "legacy_UIQM", "legacy_UCIQE"]
DEEP = ["musiq", "niqe", "topiq_nr", "liqe", "hyperiqa", "tres"]
SIGN = {"niqe": -1}
FOCUS = [("Retinex_+_LLaVA", "Retinex_w_o_post"), ("Retinex_+_LLaVA", "Retinex_+_Rule_based"),
         ("Retinex_+_LLaVA", "Retinex_+_Default_MHF-UIE"), ("RAUNE_+_LLaVA", "RAUNE_w_o_post"),
         # physics constraint on vs off (A = physics on) and post-processing choice
         ("Retinex_w_o_post", "Retinex_plain_w_phys=0"), ("RAUNE_w_o_post", "RAUNE_plain_w_phys=0"),
         ("Retinex_+_Rule_based", "Retinex_+_Default_MHF-UIE")]
LOG = []
def say(s=""): print(s); LOG.append(str(s))


def read(path, tries=4):
    last = None
    for t in range(tries):
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
            if img is None: raise IOError(f"decode failed ({buf.size} bytes)")
            return img
        except Exception as e:
            last = e; time.sleep(0.5 * (t + 1))
    raise IOError(str(last))


def folders(cfg):
    spec = C.TABLE1[cfg][0]
    return [C.BASE / p for p in (spec if isinstance(spec, (list, tuple)) else [spec])]


def _hand(args):
    path, calib, v1 = args
    try:
        img = read(path); f = compute_features(img); g = v2.compute_v2_features(img)
    except Exception as e:
        return dict(path=path, error=f"{type(e).__name__}: {e}")
    r = dict(path=path, legacy_UIQM=f["UIQM"], legacy_UCIQE=f["UCIQE"], legacy_AG=f["AG"], legacy_Tenengrad=f["Tenengrad"])
    for c in ["v2_C", "v2_B", "v2_A"]: r[c] = v2.score(g, calib, c)
    q, _ = T.apply(np.array([[f[k] for k in v1["order"]]]), v1, return_raw=True); r["qcs_v1"] = float(q[0])
    return r


def eligible_stems():
    test = {stem(p): p for p in list_images(C.BASE / "Input" / "Test")}
    sets = [set(test)]
    for cfg in C.CONFIGS:
        sets.append({stem(p) for f in folders(cfg) for p in list_images(f)})
    common = set.intersection(*sets)
    from PIL import Image
    small = {s for s in common if min(Image.open(test[s]).size) < 128}
    say(f"[INFO] Input/Test {len(test)} | common to all 14 configs {len(common)} | short side <128 on Raw: {len(small)} "
        f"-> {len(common) - len(small)} eligible stems")
    return sorted(common - small)


def score(a, out):
    sd = out / "scores"; sd.mkdir(parents=True, exist_ok=True)
    calib = v2.load_calibration(a.v2_calib)
    if calib["sha256"] != EXPECTED_V2_SHA: sys.exit(f"[FAIL] v2 calibration sha {calib['sha256']} != {EXPECTED_V2_SHA}")
    v1 = T.load(a.transform)
    say(f"[PASS] v2 calibration {calib['sha256']} | qcs_v1 transform {v1['sha256']}")
    stems_ok = eligible_stems(); pd.Series(stems_ok, name="stem").to_csv(out / "c1_stems.csv", index=False)
    names = []
    if a.pyiqa:
        from qcs_eval import metrics as MM
        if not hasattr(MM, "UPSCALE_224"): sys.exit(f"[FAIL] outdated qcs_eval/metrics.py ({MM.__file__})")
        names = MM.enable_pyiqa(tuple(DEEP)); say(f"[INFO] pyiqa metrics: {names}")
    for cfg in C.CONFIGS:
        fp = sd / f"{cfg}.csv"
        if fp.exists() and not a.force:
            d = pd.read_csv(fp)
            if len(d) == len(stems_ok) and all(m in d for m in HAND + names):
                say(f"[SKIP] {cfg}: already scored ({len(d)})"); continue
        files = {stem(p): str(p) for f in folders(cfg) for p in list_images(f)}
        paths = [files[s] for s in stems_ok]
        t0 = time.time()
        with ProcessPoolExecutor(a.workers) as ex:
            R = pd.DataFrame(list(ex.map(_hand, [(p, calib, v1) for p in paths], chunksize=16)))
        if "error" in R and R.error.notna().any():
            bad = R[R.error.notna()]; say(f"[WARN] {cfg}: {len(bad)} failed in parallel, retrying serially")
            R = pd.concat([R[R.error.isna()], pd.DataFrame([_hand((p, calib, v1)) for p in bad.path])], ignore_index=True)
            if R.error.notna().any(): sys.exit(f"[FAIL] {cfg}: unreadable images, e.g. {R[R.error.notna()].path.iloc[0]}")
            R = R.drop(columns="error")
        for n in names:
            fn = MM.REGISTRY[n]["fn"]; R[n] = [fn({}, read(p))[0] for p in R.path]
        R["stem"] = R.path.map(stem); R["config"] = cfg
        nan = {m: int(R[m].isna().sum()) for m in HAND + names if R[m].isna().any()}
        R.to_csv(fp, index=False)
        say(f"[DONE] {cfg}: {len(R)} images in {time.time() - t0:.0f}s" + (f" | NaN {nan}" if nan else ""))


def kendall_matrix(Mn):
    ms = list(Mn.columns); K = pd.DataFrame(index=ms, columns=ms, dtype=float)
    for i in ms:
        for j in ms: K.loc[i, j] = stats.kendalltau(Mn[i], Mn[j]).statistic
    return K


def analyze(a, out):
    sd = out / "scores"
    D = pd.concat([pd.read_csv(sd / f"{c}.csv") for c in C.CONFIGS if (sd / f"{c}.csv").exists()], ignore_index=True)
    cfgs = [c for c in C.CONFIGS if c in set(D.config)]
    if len(cfgs) < len(C.CONFIGS): say(f"[WARN] missing configs: {sorted(set(C.CONFIGS) - set(cfgs))}")
    metrics = [m for m in HAND + DEEP if m in D.columns]
    for m in metrics: D[m] = D[m] * SIGN.get(m, 1)
    n_per = D.groupby("config").stem.nunique()
    ok = n_per.nunique() == 1 and (D.groupby("stem").config.nunique() == len(cfgs)).all()
    say(f"[{'PASS' if ok else 'FAIL'}] paired design: every config has the same {n_per.iloc[0]} stems")
    if not ok: sys.exit(1)
    if D[metrics].isna().any().any(): say(f"[WARN] NaN: {D[metrics].isna().sum()[lambda s: s > 0].to_dict()} (dropped per metric)")
    from qcs_eval.folds import load_folds
    Fd = load_folds(); clmap = dict(zip(Fd.stem, Fd.cluster))
    D["cluster"] = D.stem.map(clmap).fillna(D.stem)
    clusters = np.array(sorted(D.cluster.unique())); cid = {c: i for i, c in enumerate(clusters)}
    say(f"[INFO] {len(cfgs)} configs x {D.stem.nunique()} images | {len(clusters)} bootstrap clusters | metrics {metrics}")

    # cluster sums per config/metric -> fast bootstrap
    rng = np.random.default_rng(a.seed)
    W = np.stack([np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)) for _ in range(a.boot)])
    means, lo, hi, ranks_bs = {}, {}, {}, {}
    for m in metrics:
        S = np.zeros((len(cfgs), len(clusters))); N = np.zeros_like(S)
        for k, c in enumerate(cfgs):
            d = D[(D.config == c) & D[m].notna()]
            ix = d.cluster.map(cid).to_numpy()
            S[k] = np.bincount(ix, weights=d[m].to_numpy(), minlength=len(clusters)); N[k] = np.bincount(ix, minlength=len(clusters))
        pt = S.sum(1) / N.sum(1); bs = (S @ W.T) / (N @ W.T)            # configs x B
        means[m] = pt; lo[m] = np.percentile(bs, 2.5, axis=1); hi[m] = np.percentile(bs, 97.5, axis=1)
        ranks_bs[m] = (-bs).argsort(0).argsort(0) + 1                    # rank 1 = best
    Mn = pd.DataFrame(means, index=cfgs)
    CM = pd.concat({"mean": Mn, "lo": pd.DataFrame(lo, index=cfgs), "hi": pd.DataFrame(hi, index=cfgs)}, axis=1)
    CM.to_csv(out / "c1_config_means.csv")
    RK = Mn.rank(ascending=False, method="min").astype(int); RK.to_csv(out / "c1_rank_matrix.csv")
    rci = []
    for m in metrics:
        rb = ranks_bs[m]
        for k, c in enumerate(cfgs):
            rci.append(dict(metric=m, config=c, rank=int(RK.loc[c, m]), rank_lo=int(np.percentile(rb[k], 2.5)),
                            rank_hi=int(np.percentile(rb[k], 97.5)), p_rank1=float((rb[k] == 1).mean())))
    RC = pd.DataFrame(rci); RC.to_csv(out / "c1_rank_ci.csv", index=False)
    top = []
    for m in metrics:
        best = Mn[m].idxmax(); ties = int((Mn[m] == Mn[m].max()).sum())
        top.append(dict(metric=m, top1=best, p_rank1=float(RC[(RC.metric == m) & (RC.config == best)].p_rank1.iat[0]),
                        n_tied_at_top=ties, top3=", ".join(Mn[m].sort_values(ascending=False).index[:3])))
    top = pd.DataFrame(top)
    if (top.n_tied_at_top > 1).any(): say(f"[WARN] tied top-1 for: {top[top.n_tied_at_top > 1].metric.tolist()}")
    top.to_csv(out / "c1_top1.csv", index=False)
    K = kendall_matrix(Mn); K.to_csv(out / "c1_kendall.csv")
    off = K.where(~np.eye(len(K), dtype=bool))
    say("\n== top-1 configuration per metric ==\n" + top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    say(f"\n[INFO] distinct top-1 configs across {len(metrics)} metrics: {top.top1.nunique()} -> {sorted(top.top1.unique())}")
    say(f"[INFO] Kendall tau between metric rankings: median {np.nanmedian(off.values):+.2f}, min {np.nanmin(off.values):+.2f}, max {np.nanmax(off.values):+.2f}")
    say("[INFO] mean tau of each metric to the others:\n" + off.mean().sort_values().to_string(float_format=lambda x: f"{x:+.2f}"))
    if "Retinex_+_LLaVA" in cfgs:
        r = RC[RC.config == "Retinex_+_LLaVA"].set_index("metric")
        say("\n== Retinex_+_LLaVA rank per metric (95% bootstrap interval) ==\n" +
            "\n".join(f"   {m:<17} {int(r.loc[m, 'rank']):>2}  [{int(r.loc[m, 'rank_lo'])}-{int(r.loc[m, 'rank_hi'])}]  P(1st)={r.loc[m, 'p_rank1']:.2f}" for m in metrics))

    # focus contrasts (paired per image)
    fc = []
    P = D.pivot_table(index="stem", columns="config", values=metrics)
    cl_of = D.groupby("stem").cluster.first()
    for A, Bc in FOCUS:
        if A not in cfgs or Bc not in cfgs: continue
        for m in metrics:
            x = (P[(m, A)] - P[(m, Bc)]).dropna(); sdv = D[m].std()
            if not sdv > 0: sdv = np.nan   # constant metric -> SD-scaled difference undefined
            cl = cl_of[x.index].map(cid).to_numpy()
            s_ = np.bincount(cl, weights=x.to_numpy(), minlength=len(clusters)); n_ = np.bincount(cl, minlength=len(clusters))
            w_ = np.bincount(cl, weights=(x.to_numpy() > 0).astype(float), minlength=len(clusters))
            bsd = (W @ s_) / (W @ n_) / sdv; bsw = (W @ w_) / (W @ n_)
            fc.append(dict(A=A, B=Bc, metric=m, mean_diff_sd=x.mean() / sdv, diff_lo=np.percentile(bsd, 2.5), diff_hi=np.percentile(bsd, 97.5),
                           win_rate_A=(x > 0).mean(), win_lo=np.percentile(bsw, 2.5), win_hi=np.percentile(bsw, 97.5), n=len(x)))
    FC = pd.DataFrame(fc); FC.to_csv(out / "c1_focus_contrasts.csv", index=False)
    if len(FC):
        say("\n== focus contrasts (A - B; win rate = share of images where A scores higher) ==")
        for (A, Bc), g in FC.groupby(["A", "B"], sort=False):
            say(f"-- {A} vs {Bc}")
            say(g[["metric", "mean_diff_sd", "diff_lo", "diff_hi", "win_rate_A"]].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    figures(RK, K, metrics, out)
    (out / "c1_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\n[DONE] {out}")


def figures(RK, K, metrics, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    SURF, INK, SEC, MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    seq = LinearSegmentedColormap.from_list("seq", ["#0d366b", "#2a78d6", "#86b6ef", "#cde2fb"])
    order = RK.mean(1).sort_values().index
    R = RK.loc[order, metrics]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(metrics), 0.9 + 0.38 * len(R)), facecolor=SURF)
    ax.imshow(R.values, cmap=seq, vmin=1, vmax=len(R), aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            v = R.values[i, j]; ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="#ffffff" if v <= len(R) / 2 else INK)
    ax.set_xticks(range(len(metrics)), metrics, rotation=45, ha="right", fontsize=8, color=SEC)
    ax.set_yticks(range(len(R)), R.index, fontsize=8, color=SEC)
    ax.set_title("Configuration rank per metric (1 = best)\nrows ordered by mean rank", loc="left", fontsize=9, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(out / "c1_rank_heatmap.png", dpi=200, facecolor=SURF); plt.close(fig)
    div = LinearSegmentedColormap.from_list("div", ["#c2410c", "#eb6834", "#f3f2ee", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(metrics), 1.0 + 0.5 * len(metrics)), facecolor=SURF)
    ax.imshow(K.loc[metrics, metrics].values.astype(float), cmap=div, vmin=-1, vmax=1)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            v = float(K.iloc[i, j]); ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color="#ffffff" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(metrics)), metrics, rotation=45, ha="right", fontsize=8, color=SEC)
    ax.set_yticks(range(len(metrics)), metrics, fontsize=8, color=SEC)
    ax.set_title("Kendall tau between metric rankings\nof the 14 configurations", loc="left", fontsize=9, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(out / "c1_kendall_heatmap.png", dpi=200, facecolor=SURF); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["score", "analyze", "all"], default="all")
    ap.add_argument("--v2-calib", default=str(C.OUT / "V2_calib" / "evalpool.json"))
    ap.add_argument("--transform", default=str(C.OUT / "A0_repro" / "frozen_transform_r2.json"))
    ap.add_argument("--pyiqa", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(C.OUT / "C1_coral_rankings"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.phase in ("score", "all"): score(a, out)
    if a.phase in ("analyze", "all"): analyze(a, out)


if __name__ == "__main__":
    main()
