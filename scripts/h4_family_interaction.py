#!/usr/bin/env python
"""
H4 (post hoc, exploratory): family-level benchmark interaction.

    G_b    = mean SROCC of the deep NR-IQA family  -  mean SROCC of the handcrafted family   (benchmark b)
    Delta  = G_SAUD - G_UID2021

Families are fixed in advance of this analysis (not of the data):
    deep        = MUSIQ, TOPIQ-NR, LIQE, HyperIQA, TReS
    handcrafted = UIQM, UCIQE, AG, Tenengrad
NIQE and the composites are excluded.

Inference: scene-cluster bootstrap inside each benchmark (the frozen cluster_id of H2). Within a benchmark every
metric shares the same resampled clusters, so family means keep their paired covariance. The two benchmarks have
disjoint scenes and are resampled independently. Two-sided bootstrap p = 2 * min(P(D<=0), P(D>=0)), floored at 1/B.

Input : _resub_v3/H2_human_validation/h2_per_scene.csv   (set, metric, scene_uid, cluster_id, srocc, ...)
Output: _resub_v3/H4_family_interaction/{h4_family_interaction.csv, h4_family_per_metric.csv, h4_debug.txt}

Usage (from the qcs_eval folder):
    python scripts\\h4_family_interaction.py
    python scripts\\h4_family_interaction.py --per-scene <path to h2_per_scene.csv> --boot 5000
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C

DEEP = ["musiq", "topiq_nr", "liqe", "hyperiqa", "tres"]
HAND = ["legacy_UIQM", "legacy_UCIQE", "legacy_AG", "legacy_Tenengrad"]
# Table 2 / Table S1 values, used only as a debug check that the per-scene file reproduces the paper
PAPER = {("SAUD_dev", "liqe"): 0.492, ("SAUD_dev", "legacy_AG"): 0.191,
         ("UID_confirm_primary", "liqe"): 0.339, ("UID_confirm_primary", "legacy_AG"): 0.513,
         ("UID_confirm_primary", "legacy_UCIQE"): 0.491, ("SAUD_dev", "tres"): 0.167}
log = []


def say(m):
    print(m); log.append(m)


def family_gap_boot(P, set_name, B, rng):
    """Point estimate and bootstrap draws of the family means and gap for one benchmark."""
    T = P[(P.set == set_name) & P.metric.isin(DEEP + HAND)].copy()
    clusters = np.sort(T.cluster_id.unique()); cidx = {c: i for i, c in enumerate(clusters)}
    K = len(clusters)
    W = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(B)]).astype(float)  # shared draws
    per_metric, draws = {}, {}
    for m in DEEP + HAND:
        t = T[T.metric == m]
        assert len(t), f"[FAIL] metric {m} missing in {set_name}"
        ci = t.cluster_id.map(cidx).to_numpy()
        s = np.bincount(ci, weights=np.nan_to_num(t.srocc.to_numpy()), minlength=K)
        n = np.bincount(ci, weights=t.srocc.notna().to_numpy().astype(float), minlength=K)
        per_metric[m] = s.sum() / n.sum()                      # mean within-scene SROCC (paper definition)
        draws[m] = (W @ s) / (W @ n)
    deep_pt = np.mean([per_metric[m] for m in DEEP]); hand_pt = np.mean([per_metric[m] for m in HAND])
    deep_b = np.mean([draws[m] for m in DEEP], axis=0); hand_b = np.mean([draws[m] for m in HAND], axis=0)
    return dict(K=K, n_scenes=T.scene_uid.nunique(), per_metric=per_metric,
                deep=deep_pt, hand=hand_pt, gap=deep_pt - hand_pt, deep_b=deep_b, hand_b=hand_b, gap_b=deep_b - hand_b)


def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def pval(x, B):
    return max(2 * min((x <= 0).mean(), (x >= 0).mean()), 1 / B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", default=str(C.OUT / "H2_human_validation" / "h2_per_scene.csv"))
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--out", default=str(C.OUT / "H4_family_interaction"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(a.per_scene)
    say(f"[INFO] per-scene file: {Path(a.per_scene).resolve()}  rows={len(P)}  sets={sorted(P.set.unique())}")
    need = {"set", "metric", "scene_uid", "cluster_id", "srocc"}
    assert need <= set(P.columns), f"[FAIL] columns missing: {need - set(P.columns)}"

    rng = np.random.default_rng(a.seed)
    res = {}
    for st in ["SAUD_dev", "UID_confirm_primary", "UID_all_sensitivity"]:
        if st not in set(P.set):
            say(f"[WARN] {st} not in file, skipped"); continue
        res[st] = family_gap_boot(P, st, a.boot, rng)
        r = res[st]
        say(f"[INFO] {st}: scenes {r['n_scenes']}, clusters {r['K']}, deep {r['deep']:.4f}, handcrafted {r['hand']:.4f}, gap {r['gap']:+.4f}")

    # ---- debug: per-metric means reproduce the paper (Table 2 / S1)
    for (st, m), v in PAPER.items():
        if st in res:
            got = res[st]["per_metric"][m]
            say(f"[{'PASS' if abs(got - v) < 0.0006 else 'FAIL'}] {st} {m}: {got:.4f} vs paper {v:.3f}")
    # ---- debug: cluster counts match H2 (SAUD 99, UID primary 48)
    for st, k in [("SAUD_dev", 99), ("UID_confirm_primary", 48)]:
        if st in res:
            say(f"[{'PASS' if res[st]['K'] == k else 'FAIL'}] {st} clusters {res[st]['K']} (expected {k})")

    rows = []
    for st, r in res.items():
        for name, pt, bd in [("deep family mean", r["deep"], r["deep_b"]), ("handcrafted family mean", r["hand"], r["hand_b"]),
                             ("gap G (deep - handcrafted)", r["gap"], r["gap_b"])]:
            lo, hi = ci(bd)
            rows.append(dict(benchmark=st, quantity=name, estimate=pt, ci_lo=lo, ci_hi=hi,
                             p_two_sided=pval(bd, a.boot) if name.startswith("gap") else np.nan,
                             B=a.boot, n_scenes=r["n_scenes"], n_clusters=r["K"]))
    for other in ["UID_confirm_primary", "UID_all_sensitivity"]:
        if "SAUD_dev" in res and other in res:
            D = res["SAUD_dev"]["gap_b"] - res[other]["gap_b"]
            pt = res["SAUD_dev"]["gap"] - res[other]["gap"]; lo, hi = ci(D)
            rows.append(dict(benchmark=f"SAUD_dev minus {other}", quantity="interaction Delta = G_SAUD - G_UID",
                             estimate=pt, ci_lo=lo, ci_hi=hi, p_two_sided=pval(D, a.boot), B=a.boot,
                             n_scenes=np.nan, n_clusters=np.nan))
            say(f"[RESULT] Delta (SAUD - {other}) = {pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p = {pval(D, a.boot):.4g}  (B={a.boot})")
    R = pd.DataFrame(rows)
    R.to_csv(out / "h4_family_interaction.csv", index=False)
    pd.DataFrame([dict(benchmark=st, metric=m, family="deep" if m in DEEP else "handcrafted", mean_ws_srocc=v)
                  for st, r in res.items() for m, v in r["per_metric"].items()]).to_csv(out / "h4_family_per_metric.csv", index=False)

    # ---- debug: the rounded Table 2 values give Delta ~ 0.491 on UID primary
    if "UID_confirm_primary" in res:
        d = res["SAUD_dev"]["gap"] - res["UID_confirm_primary"]["gap"]
        say(f"[{'PASS' if abs(d - 0.4912) < 0.003 else 'WARN'}] Delta from per-scene data {d:.4f} vs 0.4912 from rounded Table 2")
    (out / "h4_debug.txt").write_text("\n".join(log))
    say(f"[DONE] {out / 'h4_family_interaction.csv'}")


if __name__ == "__main__":
    main()
