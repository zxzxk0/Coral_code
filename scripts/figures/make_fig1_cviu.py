#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size":8.5,"pdf.fonttype":42,"ps.fonttype":42})
fig, ax = plt.subplots(figsize=(5.4,2.35))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

items=[
(0.02,0.22,0.29,0.58,"AXIS 1\nHuman agreement","SAUD + UID2021\nSROCC + pair accuracy"),
(0.355,0.22,0.29,0.58,"AXIS 2\nStress robustness","USM + dehazing\n2 × 200-image samples"),
(0.69,0.22,0.29,0.58,"AXIS 3\nConclusion stability","14 pipelines × 14 metrics\n2,294 paired images")]
for x,y,w,h,t,b in items:
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012,rounding_size=.018",
                     linewidth=1,edgecolor="0.3",facecolor="0.97")
    ax.add_patch(p)
    ax.text(x+w/2,y+h*.66,t,ha="center",va="center",weight="bold",fontsize=8.7)
    ax.text(x+w/2,y+h*.32,b,ha="center",va="center",fontsize=8.1,linespacing=1.3)
for a,b in [(0.31,0.355),(0.645,0.69)]:
    ax.add_patch(FancyArrowPatch((a+.005,.51),(b-.005,.51),arrowstyle="-|>",mutation_scale=12,
                                 linewidth=1,color="0.35"))
ax.text(.5,.075,"Complementary evidence — no single axis is treated as ground truth",
        ha="center",va="center",fontsize=8.4,weight="bold")
fig.tight_layout()
fig.savefig("fig1_three_axis_overview.pdf",bbox_inches="tight")
fig.savefig("fig1_three_axis_overview.png",dpi=300,bbox_inches="tight")
print("[PASS] Fig 1")
