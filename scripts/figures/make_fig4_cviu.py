#!/usr/bin/env python
"""Fig. 4: stress turnover vs human agreement. Every metric labelled via three label columns + leader lines
(no colour-only identity). Inputs: h2_summary.csv, a2a_gate_seed0.csv, a2a_gate_seed20261019.csv."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, ORANGE, INK, SEC, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
H = pd.read_csv("h2_summary.csv")
G = None
for f, tag in [("a2a_gate_seed0.csv", "0"), ("a2a_gate_seed20261019.csv", "1")]:
    g = pd.read_csv(f)[["metric", "min_turnover_pct", "PASS"]].rename(columns={"min_turnover_pct": "t" + tag, "PASS": "p" + tag})
    G = g if G is None else G.merge(g, on="metric")
assert (G.p0 == G.p1).all(), "debug: PASS/FAIL must agree across samples"
G["turn"] = (G.t0 + G.t1) / 2; G["pass_"] = G.p0
MAP = {"legacy_AG": "ag", "legacy_Tenengrad": "tenengrad", "legacy_UIQM": "uiqm", "legacy_UCIQE": "uciqe"}
NAME = {"legacy_AG": "AG", "legacy_Tenengrad": "Tenengrad", "legacy_UIQM": "UIQM", "legacy_UCIQE": "UCIQE", "qcs_v1": "QCS-v1",
        "v2_A": "v2_A", "v2_B": "v2_B", "v2_C": "v2_C", "musiq": "MUSIQ", "niqe": "NIQE", "topiq_nr": "TOPIQ-NR",
        "liqe": "LIQE", "hyperiqa": "HyperIQA", "tres": "TReS"}
fig, axs = plt.subplots(1, 2, figsize=(5.4, 3.4), sharex=True)
for ax, (st, title) in zip(axs, [("SAUD_dev", "SAUD"), ("UID_confirm_primary", "UID2021 primary")]):
    d = H[(H.set == st) & H.metric.isin(NAME)].copy()
    d["gm"] = d.metric.replace(MAP)
    d = d.merge(G, left_on="gm", right_on="metric", suffixes=("", "_g"))
    assert len(d) == 14, f"debug: {st} has {len(d)} metrics"
    rho = spearmanr(d.turn, d.ws_srocc)[0]          # [0] works on old and new SciPy
    ax.axvline(50, ls="--", lw=.8, color="0.55", zorder=1); ax.plot([0, 100], [0, 0], lw=.6, color="0.7", zorder=1)
    lo, hi = min(-0.02, d.ws_srocc.min() - 0.03), d.ws_srocc.max() + 0.05
    gap = (hi - lo) / 15
    # three label columns: left cluster (x<15), middle (15-50), right (>50)
    for sel, lx, ha in [(d.turn < 15, -6, "right"), ((d.turn >= 15) & (d.turn < 50), 36, "left"), (d.turn >= 50, 93, "left")]:
        g = d[sel].sort_values("ws_srocc")
        ys = g.ws_srocc.to_numpy().copy()
        for i in range(1, len(ys)): ys[i] = max(ys[i], ys[i - 1] + gap)          # push up to avoid overlap
        if len(ys) and ys[-1] > hi - gap / 2: ys -= ys[-1] - (hi - gap / 2)      # keep inside the frame
        for i in range(len(ys) - 2, -1, -1): ys[i] = min(ys[i], ys[i + 1] - gap)
        for (_, r), y in zip(g.iterrows(), ys):
            c = BLUE if r.pass_ else ORANGE
            ax.plot([r.turn, lx + (1 if ha == "right" else -1)], [r.ws_srocc, y], color=c, lw=.5, alpha=.6, zorder=2)
            ax.text(lx, y, NAME[r.metric], fontsize=7, va="center", ha=ha, color=INK)
    for _, r in d.iterrows():
        ax.scatter(r.turn, r.ws_srocc, marker="o" if r.pass_ else "s", s=24, color=BLUE if r.pass_ else ORANGE,
                   edgecolor="white", linewidth=.6, zorder=3)
    ax.set_xlim(-48, 122); ax.set_ylim(lo, hi); ax.set_xticks([0, 25, 50, 75, 100]); ax.spines["bottom"].set_bounds(0, 100)
    ax.set_title(f"{title} (Spearman $\\rho$ = {rho:+.2f})", fontsize=8.5)
    ax.set_xlabel("mean min. turnover (%)")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
axs[0].set_ylabel("within-scene SROCC")
fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, lw=0, ms=5, label="stress PASS"),
                    Line2D([], [], marker="s", color=ORANGE, lw=0, ms=5, label="stress FAIL"),
                    Line2D([], [], ls="--", color="0.55", lw=.8, label="50% gate")],
           loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, .92))
fig.savefig("fig4_stress_vs_human.pdf", bbox_inches="tight"); fig.savefig("fig4_stress_vs_human.png", dpi=300, bbox_inches="tight")
print("[PASS] Fig 4: 14 metrics labelled per panel")
