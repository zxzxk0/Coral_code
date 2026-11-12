"""
CVIU-sized Fig. 2, Fig. 5, Fig. 6a/6b.
Required in the same folder:
  h2_summary.csv
  c1_rank_matrix.csv
  c1_focus_contrasts_classified.csv
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

INK, SEC, MUT, GRID, AX = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":9,
    "axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8,
    "legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42
})
PASS=["tres","topiq_nr","liqe","hyperiqa","musiq","niqe","v2_C"]
FAIL=["legacy_AG","legacy_Tenengrad","legacy_UCIQE","legacy_UIQM","qcs_v1","v2_A","v2_B"]
NAME={"tres":"TReS","topiq_nr":"TOPIQ-NR","liqe":"LIQE","hyperiqa":"HyperIQA","musiq":"MUSIQ","niqe":"NIQE",
      "v2_C":"v2_C","legacy_AG":"AG","legacy_Tenengrad":"Tenengrad","legacy_UCIQE":"UCIQE","legacy_UIQM":"UIQM",
      "qcs_v1":"QCS-v1","v2_A":"v2_A","v2_B":"v2_B"}
def style(ax):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="x",color=GRID,lw=.6); ax.set_axisbelow(True)

H=pd.read_csv("h2_summary.csv")
H=H[H.set.isin(["SAUD_dev","UID_confirm_primary"]) & H.metric.isin(PASS+FAIL)]
S=H.pivot(index="metric",columns="set")
order=S[("ws_srocc","SAUD_dev")].sort_values().index.tolist()
fig,ax=plt.subplots(figsize=(5.4,4.7))
y=np.arange(len(order))
for st,c,mk,lab,off in [("SAUD_dev",BLUE,"o","SAUD (100 scenes)",.16),
                        ("UID_confirm_primary",ORANGE,"s","UID2021 primary (49 scenes)",-.16)]:
    v=S[("ws_srocc",st)].loc[order]; lo=S[("ws_srocc_lo",st)].loc[order]; hi=S[("ws_srocc_hi",st)].loc[order]
    ax.errorbar(v,y+off,xerr=[v-lo,hi-v],fmt=mk,ms=4.5,color=c,ecolor=c,elinewidth=1.1,mec="white",mew=.6,label=lab,zorder=3)
ax.set_yticks(y); ax.set_yticklabels([NAME[m]+("" if m in PASS else " †") for m in order])
ax.axvline(0,color=AX,lw=.8); style(ax)
ax.set_xlabel("within-scene SROCC with human scores (95% cluster-bootstrap CI)")
ax.legend(frameon=False,loc="lower left",bbox_to_anchor=(0,1.0),ncol=2)
ax.text(0,-.14,"† fails the over-processing stress test; metrics ordered by SAUD agreement.",
        transform=ax.transAxes,fontsize=8,color=MUT)
fig.tight_layout(); fig.savefig("fig2_human_crossover.pdf",bbox_inches="tight"); plt.close(fig)

R=pd.read_csv("c1_rank_matrix.csv",index_col=0)
CFG={"Raw_Input":"Raw input","Retinex_plain_w_phys=0":"Retinex, physics off","Retinex_w_o_post":"Retinex, physics on",
"Retinex_+_Default_MHF-UIE":"Retinex + Default MHF","Retinex_+_Rule_based":"Retinex + Rule","Retinex_+_LLaVA":"Retinex + LLaVA",
"RAUNE_plain_w_phys=0":"RAUNE, physics off","RAUNE_w_o_post":"RAUNE, physics on","RAUNE_+_Default_MHF-UIE":"RAUNE + Default MHF",
"RAUNE_+_Rule_based":"RAUNE + Rule","RAUNE_+_LLaVA":"RAUNE + LLaVA","RAUNE_unsupervised":"RAUNE unsupervised",
"CCL_NET":"CCL-Net","NUCE":"NUCE"}
R=R[PASS+FAIL]; R=R.loc[R.median(axis=1).sort_values(kind="stable").index]
cmap=LinearSegmentedColormap.from_list("b",["#104281","#3987e5","#9ec5f4","#f0efec"])
fig,ax=plt.subplots(figsize=(5.4,5.1))
im=ax.imshow(R.values,cmap=cmap,vmin=1,vmax=14,aspect="auto")
for i in range(14):
    for j in range(14):
        v=R.values[i,j]; ax.text(j,i,str(v),ha="center",va="center",fontsize=8,
            color="white" if v<=5 else INK,fontweight="bold" if v==1 else "normal")
ax.set_xticks(range(14)); ax.set_xticklabels([NAME[c] for c in R.columns],rotation=55,ha="right")
ax.set_yticks(range(14)); ax.set_yticklabels([CFG[c] for c in R.index])
ax.axvline(6.5,color="white",lw=2.5)
ax.text(3,-.85,"stress PASS",ha="center",fontsize=8,color=SEC)
ax.text(10,-.85,"stress FAIL",ha="center",fontsize=8,color=SEC)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb=fig.colorbar(im,ax=ax,fraction=.03,pad=.02,ticks=[1,7,14]); cb.set_label("rank (1 = best)",color=SEC); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig("fig5_rank_heatmap.pdf",bbox_inches="tight"); plt.close(fig)

F=pd.read_csv("c1_focus_contrasts_classified.csv")
morder=PASS+FAIL
robust=[
("RAUNE_w_o_post","RAUNE_plain_w_phys=0","RAUNE physics on vs off","robust failure"),
("Retinex_+_Rule_based","Retinex_+_Default_MHF-UIE","Retinex Rule vs Default MHF","robust"),
("Retinex_+_LLaVA","Retinex_+_Default_MHF-UIE","Retinex LLaVA vs Default MHF","robust"),
("RAUNE_+_LLaVA","RAUNE_w_o_post","RAUNE LLaVA vs no post","robust")]
sensitive=[
("Retinex_+_LLaVA","Retinex_+_Rule_based","Retinex LLaVA vs Rule","direction reverses"),
("Retinex_w_o_post","Retinex_plain_w_phys=0","Retinex physics on vs off","magnitude differs"),
("Retinex_+_LLaVA","Retinex_w_o_post","Retinex LLaVA vs no post","mixed")]

def forest(pairs,outfile):
    n=len(pairs)
    fig,axs=plt.subplots(1,n,figsize=(5.4,4.7),sharey=True)
    axs=np.atleast_1d(axs); yy=np.arange(14)[::-1]
    for ax,(A,B,title,cls) in zip(axs,pairs):
        g=F[(F.A==A)&(F.B==B)].set_index("metric").loc[morder]
        for (m,r),yv in zip(g.iterrows(),yy):
            c,mk=(BLUE,"o") if m in PASS else (ORANGE,"s")
            null=r.diff_lo<=0<=r.diff_hi
            ax.plot([r.diff_lo,r.diff_hi],[yv,yv],color=c,lw=1.1)
            ax.plot(r.mean_diff_sd,yv,mk,ms=4.2,color="white" if null else c,mec=c,mew=1.0)
        ax.axvline(0,color=INK,lw=.7); ax.axhline(6.5,color=GRID,lw=.8); style(ax)
        ax.set_title(title.replace(" vs ","\nvs "),fontsize=8.1,fontweight="bold",pad=14)
        ax.text(.5,1.01,cls,transform=ax.transAxes,ha="center",fontsize=7.6,color=SEC)
        lim=max(abs(g.diff_lo.min()),abs(g.diff_hi.max()))*1.12; ax.set_xlim(-lim,lim)
    axs[0].set_yticks(yy); axs[0].set_yticklabels([NAME[m] for m in morder],fontsize=8)
    for ax in axs[1:]: ax.tick_params(labelleft=False)
    handles=[Line2D([],[],marker="o",color=BLUE,lw=1.1,ms=4,label="stress PASS"),
             Line2D([],[],marker="s",color=ORANGE,lw=1.1,ms=4,label="stress FAIL"),
             Line2D([],[],marker="o",color=SEC,mfc="white",lw=0,ms=4,label="95% CI includes 0")]
    fig.legend(handles=handles,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.0))
    fig.text(.5,.01,"standardized paired difference (A − B, metric SD units); each panel has its own x scale",
             ha="center",fontsize=8,color=SEC)
    fig.tight_layout(rect=(0,.04,1,.93)); fig.savefig(outfile,bbox_inches="tight"); plt.close(fig)

forest(robust,"fig6a_robust_contrasts.pdf")
forest(sensitive,"fig6b_sensitive_contrasts.pdf")
print("[PASS] CVIU-sized Fig. 2, 5, 6a, 6b written.")
