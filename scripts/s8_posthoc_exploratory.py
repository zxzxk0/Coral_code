#!/usr/bin/env python
"""
S8 (exploratory, post hoc after the UID2021 confirmation result): component- and method-level analysis of the
benchmark crossover (supplementary Section S8, Table S16).
Input : _resub_v3/H2_human_validation/h2_joined.csv   (written by h2_human_validation.py; contains human labels, so it
        is not part of the public results folder)
Output: posthoc_Q1_components.csv, posthoc_Q2_method_means.csv, posthoc_Q2_method_level.csv, posthoc_method_level.png
        in the same folder.
Usage : python scripts\\s8_posthoc_exploratory.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C
os.chdir(C.OUT / "H2_human_validation")

import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = pd.read_csv("h2_joined.csv")
D = D[(D.dataset == "SAUD") | ((D.dataset == "UID2021") & D.in_primary)].copy()
D["set"] = np.where(D.dataset == "SAUD", "SAUD_dev", "UID_confirm_primary")
# signs so that "higher = predicted better" for raw artifact features
D["neg_noise_flat"] = -D.feat_noise_flat; D["neg_clip_frac"] = -D.feat_clip_frac
COMP = {"v2_C": "v2_C (total)", "v2_C_sub_color": "colour subscore", "v2_C_sub_detail": "detail subscore (AG)",
        "v2_C_sub_artifact": "artifact subscore", "feat_UIQM_std": "UIQM_std", "feat_UCIQE_std": "UCIQE_std",
        "neg_noise_flat": "−noise_flat", "neg_clip_frac": "−clip_frac", "legacy_AG": "AG", "legacy_Tenengrad": "Tenengrad",
        "legacy_UCIQE": "UCIQE (legacy)", "liqe": "LIQE", "musiq": "MUSIQ"}
B = 5000; rng = np.random.default_rng(20261025)


def ws(g, m):
    out = {}
    for s, x in g.groupby("scene_uid"):
        a, b = x[m].to_numpy(float), x.label.to_numpy(float)
        if np.ptp(b) == 0 or np.ptp(a) == 0: continue
        out[s] = stats.spearmanr(a, b).statistic
    return pd.Series(out)


rows = []
for st, g in D.groupby("set"):
    cl = g.groupby("scene_uid").cluster_id.first(); clusters = cl.unique()
    draws = [rng.choice(clusters, len(clusters), replace=True) for _ in range(B)]
    for m, lab in COMP.items():
        s = ws(g, m); c = cl[s.index]
        per_c = s.groupby(c.values).agg(["sum", "count"])
        bs = []
        for d in draws:
            sel = per_c.reindex(d).dropna(); bs.append(sel["sum"].sum() / sel["count"].sum())
        rows.append(dict(set=st, component=m, label=lab, ws_srocc=s.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         n_scenes=len(s), status="exploratory"))
Q1 = pd.DataFrame(rows)
Q1.to_csv("posthoc_Q1_components.csv", index=False)
piv = Q1.pivot(index="label", columns="set", values="ws_srocc").loc[list(COMP.values())]
print("== Q1 (EXPLORATORY) within-scene SROCC of components ==\n" + piv.to_string(float_format=lambda x: f"{x:+.3f}"))

# ---- Q2: method level
R = D.copy()
for m in ["label"] + list(COMP):
    R[m + "_r"] = R.groupby("scene_uid")[m].rank(pct=True)
M = R.groupby(["set", "method"])[[m + "_r" for m in ["label"] + list(COMP)]].mean().reset_index()
M.to_csv("posthoc_Q2_method_means.csv", index=False)
q2 = []
for st, g in M.groupby("set"):
    for m, lab in COMP.items():
        r = stats.spearmanr(g[m + "_r"], g.label_r)
        q2.append(dict(set=st, component=m, label=lab, method_level_spearman=r.statistic, p=r.pvalue, n_methods=len(g), status="exploratory"))
Q2 = pd.DataFrame(q2); Q2.to_csv("posthoc_Q2_method_level.csv", index=False)
print("\n== Q2 (EXPLORATORY) method-level Spearman (mean within-scene rank) ==\n" +
      Q2.pivot(index="label", columns="set", values="method_level_spearman").loc[list(COMP.values())].to_string(float_format=lambda x: f"{x:+.2f}"))
for st, g in M.groupby("set"):
    print(f"\n-- {st}: methods ordered by human preference (mean within-scene rank, 1 = best) --")
    show = g.sort_values("label_r", ascending=False)[["method", "label_r", "legacy_AG_r", "neg_noise_flat_r", "v2_C_sub_artifact_r", "liqe_r"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# ---- figure: method level, human rank vs AG rank and vs artifact subscore rank, per dataset
INK, SEC, MUT, GRID, AX, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COL = {"SAUD_dev": ("#2a78d6", "o"), "UID_confirm_primary": ("#eb6834", "s")}
fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor=SURF, sharey=True)
for ax, (m, lab) in zip(axs, [("legacy_AG", "AG (sharpness)"), ("v2_C_sub_artifact", "v2_C artifact subscore"), ("liqe", "LIQE")]):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(AX)
    ax.tick_params(colors=MUT)
    for st, (c, mk) in COL.items():
        g = M[M.set == st]; r = Q2[(Q2.set == st) & (Q2.component == m)].method_level_spearman.iat[0]
        ax.scatter(g[m + "_r"], g.label_r, c=c, marker=mk, s=46, edgecolors=SURF, linewidths=1.2,
                   label=f"{'SAUD' if st == 'SAUD_dev' else 'UID2021'} (ρ = {r:+.2f})", zorder=3)
    ax.set_title(lab, loc="left", color=INK, fontsize=10)
    ax.set_xlabel("method's mean within-scene rank by metric", color=SEC, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=SEC, loc="upper left", bbox_to_anchor=(0, -0.24), ncol=2)
axs[0].set_ylabel("method's mean within-scene rank by humans", color=SEC, fontsize=8.5)
fig.suptitle("EXPLORATORY (post-hoc): method-level agreement, one point per enhancement method", color=SEC, fontsize=9, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.94)); fig.savefig("posthoc_method_level.png", dpi=200, facecolor=SURF)
