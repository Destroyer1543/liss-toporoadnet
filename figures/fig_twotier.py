"""F3 — two-tier evaluation protocol: all-OSM (tier-1) vs visible-roads (tier-2) + ignore band."""
import numpy as np, pathlib, json, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

maj={p.stem:np.load(p) for p in pathlib.Path("/data/road_liss4/eval_masks").glob("*.npz")}
stems=json.load(open("/data/road_liss4/annot_tiles.json"))
# find a tile whose scene has a major mask + decent structure
demo=None
for s in stems:
    sc,ti=s.split("__");
    if sc in maj: demo=(s,sc,ti); break
s,sc,ti=demo
y=int(re.search(r"y(\d+)",ti).group(1)); x=int(re.search(r"x(\d+)",ti).group(1))
d=np.load(f"/data/road_liss4/liss4_tiles/test/{s}.npz")
img=d["image"].astype(np.float32); allm=d["mask"].astype(bool)
mj=maj[sc]["major"][y:y+256,x:x+256].astype(bool)
ig=maj[sc]["ignore"][y:y+256,x:x+256].astype(bool)
def fc(img):
    ch=[img[2],img[1],img[0]]
    return np.stack([np.clip((c-np.percentile(c,2))/(np.percentile(c,98)-np.percentile(c,2)+1e-6),0,1) for c in ch],-1)
rgb=fc(img)
def ov(base,m,c):
    o=base.copy(); o[m]=c; return o

YEL=[1.0,0.86,0.0]; IGN=[0.10,0.75,0.95]   # scored road = yellow, ignored = cyan
fig,axes=plt.subplots(1,3,figsize=(7.6,2.9))
panels=[
    (ov(rgb,allm,YEL), "Tier 1 — all OSM roads"),
    (ov(ov(rgb,mj,YEL),ig,IGN), "Tier 2 — visible roads\n(+ ignore band)"),
    (rgb, "LISS-IV input"),
]
order=[2,0,1]
for ax,i in zip(axes,order):
    p,t=panels[i]; ax.imshow(p); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,fontsize=10,pad=5)
    for sp in ax.spines.values(): sp.set_edgecolor("#ccc")
# tiny legend for tier-2 colors
from matplotlib.patches import Patch
axes[2].legend(handles=[Patch(color=YEL,label="scored road"),
                        Patch(color=IGN,label="ignored (sub-res)")],
               loc="lower center",bbox_to_anchor=(0.5,-0.32),ncol=2,frameon=False,fontsize=8)
plt.subplots_adjust(wspace=0.05,left=0.005,right=0.995,top=0.9,bottom=0.14)
plt.savefig("paper_figs/fig3_twotier.pdf",bbox_inches="tight")
plt.savefig("paper_figs/fig3_twotier.png",dpi=200,bbox_inches="tight")
print("saved fig3_twotier  demo:",s)
