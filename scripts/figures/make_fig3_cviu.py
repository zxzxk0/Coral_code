#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size":8.5,"axes.titlesize":9,"axes.labelsize":8.5,
                     "legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,
                     "pdf.fonttype":42,"ps.fonttype":42})

import argparse, os
ap=argparse.ArgumentParser(); ap.add_argument("--csv",default="a2a_curves.csv",help="a2a_curves.csv of A2a_stress_deep_s0")
a=ap.parse_args()
print(f"[INFO] curves from {os.path.abspath(a.csv)}")
D=pd.read_csv(a.csv)
assert {"source","operator","metric","strength","mean_delta_sd","ci_lo","ci_hi"} <= set(D.columns), D.columns
z0=D[D.strength==0].mean_delta_sd.abs().max(); print(f"[{'PASS' if z0<1e-9 else 'FAIL'}] change at strength 0 is zero (max |.|={z0:.2e})")
SOURCE="Retinex_+_LLaVA"
OPS=["usm","dehaze"]
METRICS=["ag","liqe","v2_C"]
LAB={"ag":"AG (stress-fail)","liqe":"LIQE (stress-pass)","v2_C":"v2_C (stress-pass, in-sample)"}
MK={"ag":"s","liqe":"o","v2_C":"^"}

fig,axs=plt.subplots(1,2,figsize=(5.4,2.9),sharey=True)
for ax,op in zip(axs,OPS):
    for m in METRICS:
        z=D[(D.source==SOURCE)&(D.operator==op)&(D.metric==m)].sort_values("strength")
        if z.empty:
            raise ValueError(f"Missing rows for {SOURCE=} {op=} {m=}. Available metrics: {sorted(D.metric.unique())}")
        line,=ax.plot(z.strength,z.mean_delta_sd,marker=MK[m],ms=4,lw=1.4,label=LAB[m])
        ax.fill_between(z.strength,z.ci_lo,z.ci_hi,color=line.get_color(),alpha=.13,linewidth=0)
    ax.axhline(0,color="0.45",lw=.7)
    ax.set_title("Unsharp masking" if op=="usm" else "Dehazing")
    ax.set_xlabel("operator strength")
    ax.grid(alpha=.18)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axs[0].set_ylabel("change from strength 0\n(baseline SD units)")
axs[1].legend(frameon=False,loc="best")
fig.tight_layout()
fig.savefig("fig3_stress_curves.pdf",bbox_inches="tight")
fig.savefig("fig3_stress_curves.png",dpi=300,bbox_inches="tight")
print("[PASS] Fig 3")
