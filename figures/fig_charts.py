"""F5 human observability, F7 main results, F8 LORO, band ablation — clean paper charts."""
import numpy as np, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
                     "grid.linewidth":0.6,"axes.axisbelow":True})
OUT=pathlib.Path("paper_figs")
OURS="#1f7a8c"; BASE="#8a9aa8"; ABL="#c98a2b"; BAD="#c0532e"

# ---------- F7: main results — T2 APLS across models, ranked ----------
models=[("Ours (full)",0.2303,"ours"),("− spectral stem",0.2251,"abl"),
        ("− reconstruction",0.2104,"abl"),("− topology heads",0.2093,"abl"),
        ("U-Net",0.1981,"base"),("D-LinkNet",0.1933,"base"),
        ("− deep-sup.",0.1758,"abl"),("RoadFormer",0.1085,"base")]
labels=[m[0] for m in models]; vals=[m[1] for m in models]
cmap={"ours":OURS,"abl":ABL,"base":BASE}; cols=[cmap[m[2]] for m in models]
fig,ax=plt.subplots(figsize=(6.4,3.4))
y=np.arange(len(models))[::-1]
ax.barh(y,vals,color=cols,height=0.7)
for yi,v in zip(y,vals): ax.text(v+0.004,yi,f"{v:.3f}",va="center",fontsize=9,color="#333")
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=9.5)
ax.set_xlabel("APLS (held-out test, visible-roads tier)"); ax.set_xlim(0,0.26)
ax.axvline(0.1981,ls="--",lw=0.9,color="#666"); ax.text(0.1981,len(models)-0.4,"U-Net",fontsize=7.5,color="#666",ha="center")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=OURS,label="ours"),Patch(color=ABL,label="ablation"),Patch(color=BASE,label="baseline")],
          loc="lower right",frameon=False,fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT/"fig7_results.pdf"); plt.savefig(OUT/"fig7_results.png",dpi=200); plt.close()

# ---------- band ablation: all bands needed (T2 APLS) ----------
bands=["G+R+NIR\n(full)","− G","− NIR","− R"]; bv=[0.2303,0.0855,0.0435,0.0292]
fig,ax=plt.subplots(figsize=(3.4,3.0))
bc=[OURS,BAD,BAD,BAD]
ax.bar(range(4),bv,color=bc,width=0.66)
for i,v in enumerate(bv): ax.text(i,v+0.005,f"{v:.3f}",ha="center",fontsize=9,color="#333")
ax.set_xticks(range(4)); ax.set_xticklabels(bands,fontsize=9)
ax.set_ylabel("APLS (test)"); ax.set_ylim(0,0.26)
plt.tight_layout(); plt.savefig(OUT/"fig_band.pdf"); plt.savefig(OUT/"fig_band.png",dpi=200); plt.close()

# ---------- F5: human observability — recall vs road width ----------
wlab=["9–14 m\n(1–2 px)","14–23 m","> 23 m"]; rec=[0.19,0.43,0.59]; share=[97,2,1]
fig,ax=plt.subplots(figsize=(4.2,3.2))
x=np.arange(3)
b=ax.bar(x,rec,color=["#c0532e","#d9944e","#2f8f6f"],width=0.62)
for xi,r,s in zip(x,rec,share): ax.text(xi,r+0.015,f"{r:.2f}",ha="center",fontsize=10,color="#222")
ax.set_xticks(x); ax.set_xticklabels(wlab,fontsize=9.5)
ax.set_ylabel("human recall"); ax.set_ylim(0,0.7)
ax.set_title("Humans miss ~80% of the thinnest roads",fontsize=10.5,pad=8)
plt.tight_layout(); plt.savefig(OUT/"fig5_observability.pdf"); plt.savefig(OUT/"fig5_observability.png",dpi=200); plt.close()

# ---------- F8: LORO cross-region — IoU per held-out scene vs density ----------
reg=[("jun23",0.171,0.068),("aug23",0.137,0.020),("jun24",0.123,0.018),
     ("oct23",0.068,0.013),("jan26",0.059,0.007),("may25",0.002,0.010)]
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.6,3.1))
names=[r[0] for r in reg]; ious=[r[1] for r in reg]; dens=[r[2] for r in reg]
a1.bar(range(6),ious,color=["#1f7a8c"]*5+["#c0532e"],width=0.68)
a1.axhline(np.mean(ious),ls="--",lw=0.9,color="#666")
a1.text(5.4,np.mean(ious)+0.005,f"mean {np.mean(ious):.3f}",fontsize=8,color="#666",ha="right")
a1.set_xticks(range(6)); a1.set_xticklabels(names,fontsize=8.5,rotation=0)
a1.set_ylabel("IoU on held-out region"); a1.set_title("Per-region performance",fontsize=10)
a1.annotate("fails",(5,0.002),(5,0.06),fontsize=8,color="#c0532e",ha="center",
            arrowprops=dict(arrowstyle="->",color="#c0532e",lw=1))
a2.scatter(dens,ious,c=["#1f7a8c"]*5+["#c0532e"],s=55,zorder=3)
for n,d,i in reg: a2.annotate(n,(d,i),(4,4),textcoords="offset points",fontsize=7.5,color="#555")
a2.set_xlabel("scene road density"); a2.set_ylabel("IoU")
a2.set_title("Score tracks density",fontsize=10)
plt.tight_layout(); plt.savefig(OUT/"fig8_loro.pdf"); plt.savefig(OUT/"fig8_loro.png",dpi=200); plt.close()

print("saved fig7_results, fig_band, fig5_observability, fig8_loro (png+pdf)")
