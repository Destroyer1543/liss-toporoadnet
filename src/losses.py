"""Losses: Focal Tversky + Dice (region), soft-clDice (topology), skeleton BCE,
width Huber (on GT skeleton), affinity BCE, mask/reconstruction consistency."""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def soft_dice_loss(prob, target):
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + EPS
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + EPS
    return (1 - num / den).mean()


def focal_tversky_loss(prob, target, alpha=0.7, beta=0.3, gamma=0.75):
    tp = (prob * target).sum(dim=(1, 2, 3))
    fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
    fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
    tv = (tp + EPS) / (tp + alpha * fn + beta * fp + EPS)
    # clamp: d/dx x^gamma is infinite at x=0 for gamma<1 (empty-tile batches)
    return ((1 - tv).clamp_min(1e-4) ** gamma).mean()


# ------------------------------------------------------------------ soft clDice

def _soft_erode(img):
    return -F.max_pool2d(-img, 3, 1, 1)


def _soft_dilate(img):
    return F.max_pool2d(img, 3, 1, 1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img, iters=10):
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(prob, target, iters=10):
    skel_p = soft_skeletonize(prob, iters)
    skel_t = soft_skeletonize(target, iters)
    tprec = ((skel_p * target).sum(dim=(1, 2, 3)) + EPS) / (skel_p.sum(dim=(1, 2, 3)) + EPS)
    tsens = ((skel_t * prob).sum(dim=(1, 2, 3)) + EPS) / (skel_t.sum(dim=(1, 2, 3)) + EPS)
    cl = 2 * tprec * tsens / (tprec + tsens + EPS)
    return (1 - cl).mean()


# ------------------------------------------------------------------ combined

class TopoRoadLoss(nn.Module):
    """stage='warmup' -> seg loss only; stage='full' -> everything."""

    def __init__(self, l_seg=1.0, l_cldice=0.5, l_skel=1.0, l_width=0.2,
                 l_aff=0.5, l_cons=0.3, l_aux=0.4, l_blob=0.0,
                 skel_pos_weight=20.0, cldice_iters=10):
        super().__init__()
        self.w = dict(seg=l_seg, cldice=l_cldice, skel=l_skel,
                      width=l_width, aff=l_aff, cons=l_cons, aux=l_aux, blob=l_blob)
        self.skel_pos_weight = skel_pos_weight
        self.cldice_iters = cldice_iters

    def forward(self, out, batch, stage="full"):
        y = batch["mask"]
        prob = torch.sigmoid(out["fused_logit"])
        losses = {}
        losses["seg"] = focal_tversky_loss(prob, y) + 0.5 * soft_dice_loss(prob, y)
        if self.w["blob"] > 0:
            # anti-blob: penalize road prob in the 2-px band just outside true
            # roads (dilate(y)\y), where over-wide predictions bleed. Targeted FP.
            y_dil = F.max_pool2d(y, 5, 1, 2)
            band = (y_dil - y).clamp(0, 1)
            denom = band.sum(dim=(1, 2, 3)).clamp_min(1.0)
            losses["blob"] = ((prob * band).sum(dim=(1, 2, 3)) / denom).mean()
        if "aux_logit" in out and self.w["aux"] > 0:
            # deep supervision at 1/4 scale; max-pool GT so 1-px roads survive
            y4 = F.max_pool2d(y, 4)
            p4 = torch.sigmoid(out["aux_logit"])
            losses["aux"] = focal_tversky_loss(p4, y4) + 0.5 * soft_dice_loss(p4, y4)
        if stage != "warmup":
            losses["cldice"] = soft_cldice_loss(prob, y, self.cldice_iters)
            if "skel_logit" in out and "skel" in batch:
                pw = torch.tensor(self.skel_pos_weight, device=y.device)
                losses["skel"] = F.binary_cross_entropy_with_logits(
                    out["skel_logit"], batch["skel"], pos_weight=pw)
                skel_px = batch["skel"] > 0.5
                if skel_px.any():
                    losses["width"] = F.huber_loss(
                        out["width"][skel_px], batch["width"][skel_px])
                losses["aff"] = F.binary_cross_entropy_with_logits(
                    out["aff_logit"], batch["aff"])
            if "m_rec" in out:
                losses["cons"] = (torch.sigmoid(out["mask_logit"]) - out["m_rec"]).abs().mean()
        total = sum(self.w[k] * v for k, v in losses.items())
        losses["total"] = total
        return total, {k: v.detach().item() for k, v in losses.items()}
