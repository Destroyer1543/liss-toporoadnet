"""APLS_r validation figure: per-tile agreement with graph APLS + method-ranking agreement."""
import numpy as np, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True})
d = np.load("/data/road_liss4/apls_ranking.npz", allow_pickle=True)
names=[str(n) for n in d["names"]]; mr=d["mean_r"]; mg=d["mean_g"]
pr=d["pool_r"]; pg=d["pool_g"]; rho=float(d["spearman"]); pear=float(d["pearson"])
OURS="#1f7a8c"; ALT="#c98a2b"

fig,(a,b)=plt.subplots(1,2,figsize=(7.4,3.2))
# panel A: per-tile scatter
a.scatter(pr,pg,s=10,alpha=0.35,color=OURS,edgecolors="none")
lim=[0,max(pr.max(),pg.max())*1.05]
a.plot(lim,lim,ls="--",lw=0.9,color="#888")
a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("APLS$_r$ (raster proxy)"); a.set_ylabel("graph APLS")
a.set_title("Per-tile agreement",fontsize=10.5)
a.text(0.05,0.92,f"Spearman $\\rho$ = {rho:.2f}",transform=a.transAxes,fontsize=9.5,color="#333")
# panel B: method ranking under both metrics
x=np.arange(len(names)); w=0.38
a2=b
a2.bar(x-w/2,mr,w,label="APLS$_r$",color=OURS)
a2.bar(x+w/2,mg,w,label="graph APLS",color=ALT)
a2.set_xticks(x); a2.set_xticklabels(names,fontsize=8.5,rotation=15,ha="right")
a2.set_ylabel("mean over test tiles"); a2.set_title("Same method ranking",fontsize=10.5)
a2.legend(frameon=False,fontsize=8.5,loc="upper right")
plt.tight_layout()
for e in ("pdf","png"): plt.savefig(f"paper_figs/fig_apls.{e}",dpi=200,bbox_inches="tight")
print("saved fig_apls; rankings:",
      " > ".join([names[i] for i in np.argsort(-np.array(mr))]), "||",
      " > ".join([names[i] for i in np.argsort(-np.array(mg))]))
