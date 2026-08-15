"""F6 — label-granularity: predictions from full / major-only / highway-only models."""
import sys; sys.path.insert(0,"src")
import numpy as np, torch, pathlib, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing
from model import LISSTopoRoadNet

dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load(run):
    m=LISSTopoRoadNet().to(dev)
    m.load_state_dict(torch.load(f"runs/{run}/best.pt",map_location=dev,weights_only=False)["model"]); m.eval(); return m
mods=[("All roads",load("exp15_toporoad_full")),
      ("Major only",load("exp30_toporoad_majoronly")),
      ("Highways only",load("exp31_toporoad_highwayonly"))]
stems=json.load(open("/data/road_liss4/annot_tiles.json"))
pick=[stems[0],stems[4]]
def fc(img):
    ch=[img[2],img[1],img[0]]
    return np.stack([np.clip((c-np.percentile(c,2))/(np.percentile(c,98)-np.percentile(c,2)+1e-6),0,1) for c in ch],-1)
def ov(rgb,m,c):
    o=rgb.copy(); o[m.astype(bool)]=c; return o
ROAD=[0.16,0.86,0.55]; cl=np.ones((3,3))
cols=["Input"]+[m[0] for m in mods]
fig,axes=plt.subplots(len(pick),len(cols),figsize=(len(cols)*2.0,len(pick)*2.0))
for r,stem in enumerate(pick):
    d=np.load(f"/data/road_liss4/liss4_tiles/test/{stem}.npz"); img=d["image"].astype(np.float32); rgb=fc(img)
    x=torch.from_numpy(img[None]).to(dev); panels=[rgb]
    with torch.no_grad():
        for _,m in mods:
            pm=binary_closing((torch.sigmoid(m(x)["fused_logit"])[0,0]>0.5).cpu().numpy(),cl)
            panels.append(ov(rgb,pm,ROAD))
    for c,p in enumerate(panels):
        ax=axes[r,c]; ax.imshow(p); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor("#ccc")
        if r==0: ax.set_title(cols[c],fontsize=10.5,pad=5)
plt.subplots_adjust(wspace=0.04,hspace=0.04,left=0.005,right=0.995,top=0.92,bottom=0.005)
plt.savefig("paper_figs/fig6_granularity.pdf",bbox_inches="tight")
plt.savefig("paper_figs/fig6_granularity.png",dpi=200,bbox_inches="tight")
print("saved fig6_granularity")
