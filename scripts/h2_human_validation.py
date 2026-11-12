#!/usr/bin/env python
"""
H2: human-score validation of the frozen QCS-v2 (v2_C) against baselines on SAUD (dev) and UID2021 (confirm).

Inputs : H1 scores (h1_scores.csv, label-free, written earlier) + label files.
Labels : SAUD sub_bt_scores.mat (primary; Bradley-Terry scores estimated WITHIN each raw image's 10 results),
         SAUD sub_votes.mat (sensitivity label), UID2021 mos.xlsx (MOS; enhanced images only, the 60 'Original'
         rows are not used because H1 scores enhanced images only and SAUD has no raw scores).
Sets   : SAUD dev (100 scenes) | UID2021 confirm primary (49 scenes) | UID2021 all (60, sensitivity).

Statistics (all metric scores multiplied by the orientation sign fixed in H1; NIQE = -1):
  primary   ws_srocc  = mean over scenes of Spearman(metric, label) within the scene
            pair_acc  = share of within-scene pairs with a label difference where the metric orders them the same
                        way (metric ties count 0.5)
  secondary ws_krcc, pooled SROCC/KRCC, PLCC + RMSE after 5-parameter logistic mapping (pooled stats are
            flagged for SAUD because BT scores are only comparable within a scene)
  CIs       cluster bootstrap over the frozen cluster_id (B=--boot, percentile 95%); paired v2_C - baseline
            differences with two-sided bootstrap p and Holm correction across baselines (per set and statistic)
  incremental validity (EXPLORATORY; single fold split, no CI): within-scene rank-normalised label regressed on Tenengrad+AG ranks with / without the
            v2_C rank; grouped 5-fold CV by cluster -> delta R2; partial within-scene Spearman of v2_C given Ten+AG
  controls  label itself (must give 1.0) and a seeded random metric (~0 / ~0.5) as debug checks

Outputs (--out): h2_summary.csv, h2_vs_v2C.csv, h2_incremental.csv, h2_per_scene.csv, h2_per_category.csv,
                 h2_joined.csv, h2_log.txt
"""
import argparse, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning); warnings.filterwarnings("ignore", message="Covariance")
from itertools import combinations
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C

DEF_SAUD = Path(os.environ.get("SAUD_DIR", "data/SAUD"))
DEF_UID = Path(os.environ.get("UID2021_DIR", "data/UID2021"))
LOG = []
def say(s=""): print(s); LOG.append(str(s))
def check(ok, msg):
    say(f"[{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok: (OUT / "h2_log.txt").write_text("\n".join(LOG), encoding="utf-8"); sys.exit(1)

ABLATION = {"v2_B", "v2_A"}
ORIENTATION = {"v2_C": 1, "v2_B": 1, "v2_A": 1, "qcs_v1": 1, "qcs_v1_latent": 1, "legacy_Tenengrad": 1, "legacy_AG": 1,
               "legacy_UIQM": 1, "legacy_UCIQE": 1, "feat_UIQM_std": 1, "feat_UCIQE_std": 1, "musiq": 1, "niqe": -1,
               "topiq_nr": 1, "liqe": 1, "hyperiqa": 1, "tres": 1}


# ------------------------------------------------------------------ labels
def mat_pairs(p, var):
    from scipy.io import loadmat
    a = loadmat(p)[var]
    keys = [str(np.ravel(a[i, 0])[0]).replace("\\", "/").lower() for i in range(a.shape[0])]
    vals = [float(np.ravel(a[i, 1])[0]) for i in range(a.shape[0])]
    return pd.DataFrame({"key": keys, var: vals})


def saud_labels(root):
    bt = mat_pairs(root / "sub_bt_scores.mat", "sub_bt_scores")
    vt = mat_pairs(root / "sub_votes.mat", "sub_votes")
    check(len(bt) == 1000 and bt.key.is_unique, f"SAUD BT: {len(bt)} unique keys")
    check(set(bt.key) == set(vt.key), "SAUD BT and votes cover the same images")
    return bt.merge(vt, on="key").rename(columns={"sub_bt_scores": "label", "sub_votes": "label_votes"})


def uid_labels(root):
    df = pd.read_excel(root / "mos.xlsx")
    df.columns = [c.strip() for c in df.columns]
    df["key"] = df.Name.astype(str).str.lower()
    enh = df[df.Method.astype(str).str.lower() != "original"]
    check(len(df) == 960 and len(enh) == 900, f"UID MOS rows {len(df)} (960) | enhanced {len(enh)} (900)")
    return enh[["key", "MOS", "Method", "Scene", "Category"]].rename(columns={"MOS": "label"})


# ------------------------------------------------------------------ per-scene statistics
def scene_stats(g, m):
    x, y = g[m].to_numpy(float), g["label"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(y) == 0:
        return np.nan, np.nan, 0.0, 0
    sr = stats.spearmanr(x, y).correlation if np.ptp(x) > 0 else 0.0
    kr = stats.kendalltau(x, y).correlation if np.ptp(x) > 0 else 0.0
    i, j = np.triu_indices(len(x), 1)
    dy, dx = np.sign(y[i] - y[j]), np.sign(x[i] - x[j])
    keep = dy != 0
    hits = np.where(dx[keep] == 0, 0.5, (dx[keep] == dy[keep]).astype(float)).sum()
    return sr, kr, float(hits), int(keep.sum())


def logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def _fit_logistic(xs, y):
    p0 = [np.ptp(y), 1.0, 0.0, 0.0, y.mean()]
    b, _ = curve_fit(logistic5, xs, y, p0=p0, maxfev=20000)
    return b


def pooled(x, y, groups, folds=5, seed=0):
    """pooled SROCC/KRCC (rank-based, no fitting) + PLCC/RMSE of a 5-parameter logistic fitted on training
    folds only (grouped by cluster) and evaluated on the held-out fold predictions."""
    ok = np.isfinite(x) & np.isfinite(y); x, y, groups = x[ok], y[ok], np.asarray(groups)[ok]
    sr, kr = stats.spearmanr(x, y).correlation, stats.kendalltau(x, y).correlation
    cls = np.array(sorted(set(groups))); np.random.default_rng(seed).shuffle(cls)
    fold = pd.Series(groups).map({c: i % folds for i, c in enumerate(cls)}).to_numpy()
    pred = np.full(len(y), np.nan); n_lin = 0
    for k in range(folds):
        tr, te = fold != k, fold == k
        mu, sd = x[tr].mean(), x[tr].std() + 1e-12
        xt, xe = (x[tr] - mu) / sd, (x[te] - mu) / sd
        try:
            b = _fit_logistic(xt, y[tr]); pe = logistic5(xe, *b)
            if not np.all(np.isfinite(pe)): raise ValueError
        except Exception:                      # logistic did not converge on this fold -> linear map (counted)
            c = np.polyfit(xt, y[tr], 1); pe = np.polyval(c, xe); n_lin += 1
        pred[te] = pe
    pl = stats.pearsonr(pred, y)[0]; rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return sr, kr, pl, rmse, n_lin


def rank_within(df, col):
    return df.groupby("scene_uid")[col].rank(pct=True)


# ------------------------------------------------------------------ analysis of one set
def analyse(D, set_name, metrics, B, rng, pooled_ok):
    scenes = sorted(D.scene_uid.unique())
    cl = D.groupby("scene_uid").cluster_id.first()
    PS = []
    for m in metrics:
        for s, g in D.groupby("scene_uid"):
            sr, kr, h, n = scene_stats(g, m)
            PS.append(dict(set=set_name, metric=m, scene_uid=s, cluster_id=cl[s], srocc=sr, krcc=kr, hits=h, pairs=n))
    PS = pd.DataFrame(PS)
    clusters = sorted(cl.unique()); cidx = {c: i for i, c in enumerate(clusters)}
    # bootstrap weights over clusters (same draws for every metric -> paired differences)
    W = np.zeros((B, len(clusters)))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
    def boot(m):
        t = PS[PS.metric == m].set_index("scene_uid").loc[scenes]
        ci = t.cluster_id.map(cidx).to_numpy()
        # aggregate per cluster
        sr_sum = np.bincount(ci, weights=np.nan_to_num(t.srocc.to_numpy()), minlength=len(clusters))
        sr_n = np.bincount(ci, weights=t.srocc.notna().to_numpy().astype(float), minlength=len(clusters))
        h = np.bincount(ci, weights=t.hits.to_numpy(), minlength=len(clusters))
        n = np.bincount(ci, weights=t.pairs.to_numpy(), minlength=len(clusters))
        point = (sr_sum.sum() / sr_n.sum(), h.sum() / n.sum())
        bs = np.c_[(W @ sr_sum) / (W @ sr_n), (W @ h) / (W @ n)]
        return point, bs
    res, BS = [], {}
    for m in metrics:
        (ws, pa), bs = boot(m); BS[m] = bs
        t = PS[PS.metric == m]
        row = dict(set=set_name, metric=m, role=("candidate" if m == "v2_C" else "ablation" if m in ABLATION else
                   "control" if m.startswith("_") else "baseline"),
                   n_scenes=int(t.srocc.notna().sum()), n_clusters=len(clusters), n_pairs=int(t.pairs.sum()),
                   ws_srocc=ws, ws_srocc_lo=np.percentile(bs[:, 0], 2.5), ws_srocc_hi=np.percentile(bs[:, 0], 97.5),
                   pair_acc=pa, pair_acc_lo=np.percentile(bs[:, 1], 2.5), pair_acc_hi=np.percentile(bs[:, 1], 97.5),
                   ws_krcc=t.krcc.mean(), scenes_srocc_pos=float((t.srocc > 0).mean()))
        sr, kr, pl, rm, nl = pooled(D[m].to_numpy(float), D.label.to_numpy(float), D.cluster_id.to_numpy())
        row.update(pooled_srocc=sr, pooled_krcc=kr, plcc_logistic_cv=pl, rmse_logistic_cv=rm, logistic_folds_linear_fallback=nl,
                   pooled_valid=pooled_ok)
        res.append(row)
    R = pd.DataFrame(res)
    # paired differences v2_C - other
    VS = []
    for m in metrics:
        if m == "v2_C" or m.startswith("_"): continue
        for k, stat in enumerate(["ws_srocc", "pair_acc"]):
            d = BS["v2_C"][:, k] - BS[m][:, k]
            pt = float(R.loc[R.metric == "v2_C", stat].iat[0] - R.loc[R.metric == m, stat].iat[0])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            VS.append(dict(set=set_name, stat=stat, other=m, role=("ablation" if m in ABLATION else "baseline"),
                           diff=pt, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p_boot=max(p, 1.0 / B)))
    VS = pd.DataFrame(VS)
    if len(VS):
        for (st, role), g in VS.groupby(["stat", "role"]):
            order = g.p_boot.sort_values().index; mtot = len(g); run = 0.0
            for r_, ix in enumerate(order):
                run = max(run, min(1.0, (mtot - r_) * VS.at[ix, "p_boot"])); VS.at[ix, "p_holm"] = run
    return R, VS, PS


def incremental(D, set_name, folds=5, seed=0):
    from numpy.linalg import lstsq
    X = pd.DataFrame({"y": rank_within(D, "label"), "ten": rank_within(D, "legacy_Tenengrad"),
                      "ag": rank_within(D, "legacy_AG"), "c": rank_within(D, "v2_C"), "cl": D.cluster_id.values})
    X = X.dropna()
    cls = np.array(sorted(X.cl.unique())); np.random.default_rng(seed).shuffle(cls)
    fold_of = {c: i % folds for i, c in enumerate(cls)}; f = X.cl.map(fold_of).to_numpy()
    def cv_r2(cols):
        pred = np.zeros(len(X))
        for k in range(folds):
            tr, te = f != k, f == k
            A = np.c_[np.ones(tr.sum()), X.loc[tr, cols].to_numpy()]
            b = lstsq(A, X.y.to_numpy()[tr], rcond=None)[0]
            pred[te] = np.c_[np.ones(te.sum()), X.loc[te, cols].to_numpy()] @ b
        y = X.y.to_numpy(); return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r_base, r_full, r_c = cv_r2(["ten", "ag"]), cv_r2(["ten", "ag", "c"]), cv_r2(["c"])
    # partial Spearman (on within-scene ranks) of v2_C with label given Ten, AG
    def resid(v):
        A = np.c_[np.ones(len(X)), X[["ten", "ag"]].to_numpy()]
        return v - A @ lstsq(A, v, rcond=None)[0]
    pc = stats.pearsonr(resid(X.c.to_numpy()), resid(X.y.to_numpy()))[0]
    return dict(set=set_name, cv_r2_ten_ag=r_base, cv_r2_ten_ag_v2C=r_full, delta_r2=r_full - r_base,
                cv_r2_v2C_only=r_c, partial_r_ranks_v2C_given_ten_ag=pc, status="exploratory", folds=folds, n=len(X))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1", default=str(C.OUT / "H1_human_sets" / "h1_scores.csv"))
    ap.add_argument("--saud", default=str(DEF_SAUD)); ap.add_argument("--uid", default=str(DEF_UID))
    ap.add_argument("--boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out", default=str(C.OUT / "H2_human_validation"))
    ap.add_argument("--allow-other-calib", action="store_true")
    a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    H = pd.read_csv(a.h1)
    frozen_ok = H.v2_calib_sha.nunique() == 1 and H.v2_calib_sha.iat[0] == "753c406c848c7217"
    if not a.allow_other_calib:
        check(frozen_ok, "H1 scores use the frozen calibration 753c406c848c7217")
    else:
        say(f"[WARN] sensitivity run with calibration {H.v2_calib_sha.unique()}")
    orient = ORIENTATION
    op = Path(a.h1).parent / "h1_metric_orientation.csv"
    if op.exists():
        f = pd.read_csv(op, index_col=0).iloc[:, 0].to_dict()
        check(all(int(f[k]) == ORIENTATION[k] for k in f if k in ORIENTATION), "orientation in H1 file equals the fixed definition")
    else:
        say("[INFO] h1_metric_orientation.csv not found -> using the fixed in-code orientation (NIQE -1, others +1)")
    metrics = [m for m in ["v2_C", "v2_B", "v2_A", "qcs_v1", "qcs_v1_latent", "legacy_Tenengrad", "legacy_AG", "legacy_UIQM",
                           "legacy_UCIQE", "feat_UIQM_std", "feat_UCIQE_std", "musiq", "niqe", "topiq_nr", "liqe",
                           "hyperiqa", "tres"] if m in H.columns]
    for m in metrics: H[m] = H[m] * orient[m]
    say(f"[INFO] metrics: {metrics} | orientation applied (NIQE x -1 if present)")

    # ---- join SAUD
    S = H[H.dataset == "SAUD"].copy()
    S["key"] = (S.group.astype(str) + "/" + S.scene.astype(str) + "/" + S.path.map(lambda p: Path(p).name)).str.lower()
    SL = saud_labels(Path(a.saud))
    S = S.merge(SL, on="key", how="left", validate="one_to_one")
    check(S.label.notna().all() and len(S) == 1000, f"SAUD: all 1000 enhanced images matched a BT score ({int(S.label.notna().sum())})")
    # ---- join UID
    U = H[H.dataset == "UID2021"].copy(); U["key"] = U.path.map(lambda p: Path(p).name.lower())
    UL = uid_labels(Path(a.uid))
    U = U.merge(UL, on="key", how="left", validate="one_to_one")
    check(U.label.notna().all() and len(U) == 900, f"UID2021: all 900 enhanced images matched a MOS ({int(U.label.notna().sum())})")
    check((U.Method.str.lower() == U.method.str.lower()).all(), "UID2021: MOS 'Method' column agrees with manifest method")
    check((U.Scene.str.replace("_", "/") == U.scene_uid).all(), "UID2021: MOS 'Scene' agrees with scene_uid")
    say(f"[INFO] label ranges: SAUD BT {S.label.min():.3g}..{S.label.max():.3g} | votes {S.label_votes.min():.0f}..{S.label_votes.max():.0f} "
        f"| UID MOS {U.label.min():.3g}..{U.label.max():.3g}")
    say(f"[INFO] SAUD BT vs votes Spearman (pooled) {stats.spearmanr(S.label, S.label_votes).correlation:.3f}")

    # ---- controls (debug)
    allm = metrics + ["_label_control", "_random_control"]
    pd.concat([S, U]).to_csv(OUT / "h2_joined.csv", index=False)
    for nm, D in (("SAUD", S), ("UID", U)):   # null check of the pairwise/SROCC code: random metrics over 20 seeds
        nulls = []
        for sd in range(20):
            z = D.assign(_r=np.random.default_rng(100 + sd).normal(size=len(D)))
            st = [scene_stats(g, "_r") for _, g in z.groupby("scene_uid")]
            nulls.append((np.nanmean([x[0] for x in st]), sum(x[2] for x in st) / sum(x[3] for x in st)))
        nulls = np.array(nulls)
        say(f"[{'PASS' if abs(nulls[:, 1].mean() - 0.5) < 0.01 else 'FAIL'}] {nm} null check (20 random metrics): "
            f"mean ws_srocc {nulls[:, 0].mean():+.4f}, mean pair_acc {nulls[:, 1].mean():.4f}")

    sets = [("SAUD_dev", S, False), ("UID_confirm_primary", U[U.in_primary], True), ("UID_all_sensitivity", U, True)]
    S_votes = S.assign(label=S.label_votes); sets.append(("SAUD_dev_votes_label", S_votes, False))
    R_all, V_all, P_all, I_all = [], [], [], []
    for name, D, pok in sets:
        D = D.assign(_label_control=D.label, _random_control=np.random.default_rng(1).normal(size=len(D)))
        R, VS, PS = analyse(D, name, allm, a.boot, rng, pok)
        R_all.append(R); V_all.append(VS); P_all.append(PS); I_all.append(incremental(D, name))
        lc = R[R.metric == "_label_control"].iloc[0]; rc = R[R.metric == "_random_control"].iloc[0]
        check(abs(lc.ws_srocc - 1) < 1e-9 and abs(lc.pair_acc - 1) < 1e-9, f"{name}: label control gives 1.0")
        say(f"[{'PASS' if abs(rc.pair_acc - 0.5) < 0.05 else 'WARN'}] {name}: random control pair_acc {rc.pair_acc:.3f} (~0.5), ws_srocc {rc.ws_srocc:+.3f}")
    R = pd.concat(R_all); V = pd.concat(V_all); P = pd.concat(P_all); I = pd.DataFrame(I_all)
    R = R[~R.metric.str.startswith("_")]; P = P[~P.metric.str.startswith("_")]
    R.to_csv(OUT / "h2_summary.csv", index=False); V.to_csv(OUT / "h2_vs_v2C.csv", index=False)
    P.to_csv(OUT / "h2_per_scene.csv", index=False); I.to_csv(OUT / "h2_incremental.csv", index=False)

    # ---- per category (UID, exploratory)
    cat = []
    for c, g in U[U.in_primary].groupby("Category"):
        for m in metrics:
            st = [scene_stats(gg, m) for _, gg in g.groupby("scene_uid")]
            cat.append(dict(category=c, metric=m, n_scenes=len(st), ws_srocc=np.nanmean([x[0] for x in st]),
                            pair_acc=sum(x[2] for x in st) / max(1, sum(x[3] for x in st))))
    pd.DataFrame(cat).to_csv(OUT / "h2_per_category.csv", index=False)

    fmt = lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)
    for name, *_ in sets:
        r = R[R.set == name].sort_values("ws_srocc", ascending=False)
        rc = r[r.metric == "v2_C"].iloc[0]
        say(f"\n== {name} ==  (v2_C: scenes {rc.n_scenes}, clusters {rc.n_clusters}, pairs {rc.n_pairs})")
        short = r[r.n_scenes < rc.n_scenes]
        if len(short): say("  [NOTE] fewer scenes (NaN scores): " + ", ".join(f"{m} {n}" for m, n in zip(short.metric, short.n_scenes)))
        say(r[["metric", "role", "ws_srocc", "ws_srocc_lo", "ws_srocc_hi", "pair_acc", "pair_acc_lo", "pair_acc_hi",
               "pooled_srocc", "plcc_logistic_cv"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        v = V[(V.set == name) & (V.stat == "ws_srocc")]
        say("  v2_C minus other (ws_srocc): " + " | ".join(f"{o} {d:+.3f} [{l:+.3f},{h:+.3f}] pHolm {p:.3g}"
                                                            for o, d, l, h, p in v[["other", "diff", "lo", "hi", "p_holm"]].itertuples(index=False)))
    say("\n== incremental validity (EXPLORATORY: within-scene ranks, one grouped 5-fold split, no CI) ==\n" + I.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    gp = ["musiq", "niqe", "topiq_nr", "liqe", "hyperiqa", "tres"]
    miss = [m for m in gp if m not in metrics]
    say(f"[{'PASS' if not miss else 'WARN'}] general-purpose NR-IQA baselines present: {[m for m in gp if m in metrics]}"
        + (f" | MISSING: {miss} (rerun H1 with pyiqa before final tables)" if miss else ""))
    (OUT / "h2_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\n[DONE] {OUT}")


if __name__ == "__main__":
    main()
