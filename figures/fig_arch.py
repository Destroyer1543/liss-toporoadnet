"""F1 — LISS-TopoRoadNet architecture. Cream 3D cuboids, orthogonal straight connectors,
grouping panels for encoder/decoder, larger real input + head outputs."""
import numpy as np, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

A = pathlib.Path("paper_figs/assets")
plt.rcParams.update({"font.family": "DejaVu Sans"})
EDGE = "#3b4a57"

def _sh(c,f):
    c=np.array(mc.to_rgb(c)); return tuple(np.clip(c*f,0,1))

def block(ax, x, ybase, w, h, chan, face, z=4, label=None):
    d=min(chan,1.0); dx,dy=0.34*d,0.5*d
    ax.add_patch(Polygon([(x,ybase+h),(x+w,ybase+h),(x+w+dx,ybase+h+dy),(x+dx,ybase+h+dy)],
                 closed=True, fc=_sh(face,.90), ec=EDGE, lw=1.1, zorder=z))
    ax.add_patch(Polygon([(x+w,ybase),(x+w+dx,ybase+dy),(x+w+dx,ybase+h+dy),(x+w,ybase+h)],
                 closed=True, fc=_sh(face,.78), ec=EDGE, lw=1.1, zorder=z))
    ax.add_patch(Polygon([(x,ybase),(x+w,ybase),(x+w,ybase+h),(x,ybase+h)],
                 closed=True, fc=face, ec=EDGE, lw=1.1, zorder=z+.2))
    if label: ax.text(x+w/2, ybase-0.30, label, ha="center", va="top", fontsize=12.6, weight="bold", color="#222")
    return x+w, x+w+dx           # front-right x, depth-right x

def himg(ax, path, cx, cy, zoom, frame="#333", lw=1.6):
    ab=AnnotationBbox(OffsetImage(mpimg.imread(A/path),zoom=zoom),(cx,cy),frameon=True,
                      bboxprops=dict(edgecolor=frame,lw=lw),zorder=9)
    ax.add_artist(ab)

def panel(ax, x0, x1, y0, y1, fc, ec, title, subtitle=None):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0.02,rounding_size=0.18",
                 fc=fc, ec=ec, lw=1.3, zorder=1))
    ax.text((x0+x1)/2, y0-0.28, title, ha="center", va="top", fontsize=13.8, weight="bold", color=ec)
    if subtitle: ax.text((x0+x1)/2, y0-0.82, subtitle, ha="center", va="top", fontsize=11.5, weight="bold", color=ec, style="italic")

def farrow(ax,x0,x1,y,c=EDGE,lw=1.8,z=6):
    ax.annotate("",(x1,y),(x0,y),arrowprops=dict(arrowstyle="-|>",color=c,lw=lw,shrinkA=0,shrinkB=0),zorder=z)
def hline(ax,x0,x1,y,c,lw=1.2,z=3): ax.plot([x0,x1],[y,y],color=c,lw=lw,zorder=z,solid_capstyle="round")
def vline(ax,x,y0,y1,c,lw=1.2,z=3): ax.plot([x,x],[y0,y1],color=c,lw=lw,zorder=z,solid_capstyle="round")

fig,ax=plt.subplots(figsize=(13.8,8.7))
ax.set_xlim(0,27.5); ax.set_ylim(-6.2,9.2); ax.axis("off")
CREAM="#f3ddc0"; ENCC="#cfe1ec"; BOT="#e2cfe8"; DECC="#cfe8da"; SKP="#c58fa8"
PANE_E="#eef4f8"; PANE_D="#eef7f1"; EC_E="#5b7f97"; EC_D="#5b9779"
base=2.9; mid=base+1.6

# ---- input (larger)
himg(ax,"in_falsecolor.png",1.65,mid,0.26,frame="#111",lw=2.0)
ax.text(1.65,base-0.85,"LISS-IV G/R/NIR\n256$\\times$256",ha="center",va="top",fontsize=12.1,weight="bold",color="#222")
farrow(ax,2.25,3.42,mid)

# ---- spectral stem
fr,dr=block(ax,3.5,base,0.55,3.1,0.35,CREAM,label="spectral\nstem")
farrow(ax,dr+0.05,dr+0.75,mid)

# ---- ENCODER panel (3 close blocks + cross-scale fusion arrows)
ex=dr+0.9
enc=[("64$^2$",2.7,0.55,ENCC),("32$^2$",2.05,0.75,ENCC),("16$^2$",1.45,1.0,BOT)]
encfr=[]; gap=0.62
for lab,h,ch,col in enc:
    fr,dr=block(ax,ex,base,0.58,h,ch,col,label=lab); encfr.append((ex,fr,dr,base+h)); ex=dr+gap
panel(ax, encfr[0][0]-0.35, encfr[-1][2]+0.32, base-0.85, base+3.6, PANE_E, EC_E,
      "Dual-path encoder", subtitle="cross-scale fusion  ⇅")
# cross-scale fusion: vertical double-arrows between adjacent encoder blocks
for a, b in zip(encfr, encfr[1:]):
    xm=(a[2]+b[0])/2
    ax.annotate("",(xm,base+2.4),(xm,base+1.15),arrowprops=dict(arrowstyle="<|-|>",color="#c0532e",lw=1.6),zorder=7)

# encoder -> decoder
farrow(ax,encfr[-1][2]+0.05,encfr[-1][2]+0.75,mid)

# ---- DECODER panel
dx0=encfr[-1][2]+0.9
dec=[("32$^2$",2.05,0.75,DECC),("64$^2$",2.7,0.55,DECC),("256$^2$",3.1,0.35,DECC)]
decfr=[]
for lab,h,ch,col in dec:
    fr,dr=block(ax,dx0,base,0.58,h,ch,col,label=lab); decfr.append((dx0,fr,dr,base+h)); dx0=dr+gap
panel(ax, decfr[0][0]-0.32, decfr[-1][2]+0.32, base-0.85, base+3.6, PANE_D, EC_D,
      "Topology decoder", subtitle="deep-supervised")

# ---- skip connections: orthogonal straight lines over the top
skips=[(3.5+0.55/2, base+3.1, decfr[2][0]+0.28, decfr[2][3], 7.55),
       (encfr[0][0]+0.28, encfr[0][3], decfr[1][0]+0.28, decfr[1][3], 6.9),
       (encfr[1][0]+0.28, encfr[1][3], decfr[0][0]+0.28, decfr[0][3], 6.25)]
for xe,ye,xd,yd,yt in skips:
    vline(ax,xe,ye,yt,SKP,1.1); hline(ax,xe,xd,yt,SKP,1.1)
    ax.annotate("",(xd,yd),(xd,yt),arrowprops=dict(arrowstyle="-|>",color=SKP,lw=1.1),zorder=3)
ax.text((3.5+decfr[2][0])/2,7.8,"skip connections",ha="center",fontsize=11.5,weight="bold",color="#a5678a",style="italic")

# ---- task heads (larger images, orthogonal bus)
farrow(ax,decfr[-1][2]+0.05,decfr[-1][2]+0.75,mid)
bus=decfr[-1][2]+0.95
hc=mid
heads=[("out_skel.png","centre-line",hc+2.15,"#c0532e"),
       ("out_mask.png","road",       hc+1.07,"#c0532e"),
       ("out_width.png","width",      hc,     "#c0532e"),
       ("out_aff.png","direction",   hc-1.07,"#c0532e"),
       ("out_recon.png","reconstruct",hc-2.15,"#5b9779")]
ys=[h[2] for h in heads]; vline(ax,bus,min(ys),max(ys),EDGE,1.5)
tx=bus+1.35
for path,lab,hy,col in heads:
    ax.annotate("",(tx-0.42,hy),(bus,hy),arrowprops=dict(arrowstyle="-|>",color=EDGE,lw=1.3),zorder=6)
    himg(ax,path,tx,hy,0.11,frame=col,lw=1.8)
    ax.text(tx+0.62,hy,lab,ha="left",va="center",fontsize=12.1,weight="bold",color="#222")
ax.text(tx+0.1, hc-2.75, "Task heads", ha="center", va="top", fontsize=13.8, weight="bold", color="#c0532e")

# note that the block labels are pixel resolutions
ax.text((encfr[0][0]+decfr[-1][2])/2, 0.7, "block labels: feature-map size in pixels (channel width grows as resolution falls)",
        ha="center", va="top", fontsize=10.9, weight="bold", color="#666", style="italic")

# =================== SUB-COMPONENT DETAIL ROW ===================
yT, yB = -0.7, -6.0
xL, xR = 0.4, tx+2.4
gp=0.55; w3=(xR-xL-2*gp)/3
PA=(xL, xL+w3); PB=(xL+w3+gp, xL+2*w3+gp); PC=(xL+2*w3+2*gp, xR)
PANE="#f6f3ec"; cyA=(yT+yB)/2 + 0.15
def dpanel(x0,x1,title,ec):
    ax.add_patch(FancyBboxPatch((x0,yB),x1-x0,yT-yB,boxstyle="round,pad=0.02,rounding_size=0.15",
                 fc=PANE,ec=ec,lw=1.4,zorder=1))
    ax.text((x0+x1)/2, yT+0.18, title, ha="center", va="bottom", fontsize=14.4, weight="bold", color=ec)
def sbox(cx,cy,w,h,fc,txt=None,fs=11.5,tc="#222"):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc=fc,ec=EDGE,lw=1.2,zorder=5))
    if txt: ax.text(cx,cy,txt,ha="center",va="center",fontsize=fs,weight="bold",color=tc,zorder=6)
def darrow(x0,y0,x1,y1,c=EDGE,lw=1.8):
    ax.annotate("",(x1,y1),(x0,y0),arrowprops=dict(arrowstyle="-|>",color=c,lw=lw),zorder=6)
def dnote(px,txt): ax.text(px, yB+0.42, txt, ha="center", fontsize=10.1, weight="bold", color="#5a6b7d", style="italic")

# --- Panel A: spectral stem
dpanel(*PA,"Spectral stem",EC_E)
P0=PA[0]
for i,(lab,col) in enumerate([("G","#3a9a55"),("R","#c0392b"),("NIR","#7a5230")]):
    sbox(P0+1.05,cyA+1.4-i*1.4,0.95,0.95,col,txt=lab,fs=12.6,tc="white")
darrow(P0+1.55,cyA,P0+2.15,cyA)
sbox(P0+2.85,cyA,1.35,1.35,"#efe0c8",txt="1$\\times$1\nconv",fs=12.6)
darrow(P0+3.55,cyA,P0+4.1,cyA)
sbox(P0+4.55,cyA,0.75,2.5,CREAM)
dnote((PA[0]+PA[1])/2, "mixes bands at full resolution")

# --- Panel B: cross-scale fusion
dpanel(*PB,"Cross-scale fusion",EC_E)
P0=PB[0]; sc=[("64$^2$",2.2,ENCC),("32$^2$",1.6,ENCC),("16$^2$",1.15,BOT)]
scpos=[P0+1.0,P0+2.75,P0+4.5]
for (lab,h,col),sx in zip(sc,scpos):
    sbox(sx,cyA,1.0,h,col,txt=lab,fs=12.6)
# both arrow rows sit inside the smallest block's height (16$^2$, half-height ~0.57)
for a2,b2 in zip(scpos,scpos[1:]):
    darrow(a2+0.5,cyA+0.30,b2-0.5,cyA+0.30,c="#c0532e")     # down to coarser (strided)
    darrow(b2-0.5,cyA-0.30,a2+0.5,cyA-0.30,c="#5b7f97")     # up to finer (upsample)
ax.text((PB[0]+PB[1])/2, cyA+1.55, "$\\downarrow$ strided conv", ha="center", fontsize=10.3, weight="bold", color="#c0532e")
ax.text((PB[0]+PB[1])/2, cyA-1.7, "$\\uparrow$ upsample", ha="center", fontsize=10.3, weight="bold", color="#5b7f97")
dnote((PB[0]+PB[1])/2, "features shared across scales")

# --- Panel C: multi-task heads (orthogonal routing: horizontal, vertical bus, horizontal)
dpanel(*PC,"Multi-task heads",EC_D)
P0=PC[0]
sbox(P0+0.85,cyA,0.72,2.5,DECC)
ax.text(P0+0.85,cyA-1.45,"shared\nfeatures",ha="center",va="top",fontsize=9.8,weight="bold",color="#555")
busx=P0+1.9; hbus=[cyA+1.55,cyA+0.52,cyA-0.52,cyA-1.55]
hline(ax,P0+1.21,busx,cyA,EDGE,1.6)
vline(ax,busx,min(hbus),max(hbus),EDGE,1.6)
for lab,yy in zip(["road","centre-line","width","direction"],hbus):
    ax.annotate("",(P0+2.55,yy),(busx,yy),arrowprops=dict(arrowstyle="-|>",color=EDGE,lw=1.5),zorder=6)
    sbox(P0+3.05,yy,0.95,0.68,"#efe0c8",txt="1$\\times$1",fs=10.9)
    ax.text(P0+3.62,yy,lab,ha="left",va="center",fontsize=10.9,weight="bold",color="#222")
dnote((PC[0]+PC[1])/2, "one 1$\\times$1 conv per task")

ax.set_xlim(0, tx+2.9)     # crop the empty right margin to fit the content
plt.tight_layout(pad=0.2)
for ext in ("png","pdf"): plt.savefig(f"paper_figs/fig1_architecture.{ext}",dpi=200,bbox_inches="tight")
print("saved fig1_architecture")
