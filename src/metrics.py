import numpy as np
from skimage.morphology import skeletonize

EPS = 1e-6


def raster_metrics(prob, target, thr=0.5):
    p = (prob >= thr).astype(np.float64)
    t = (target >= 0.5).astype(np.float64)
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    iou = tp / (tp + fp + fn + EPS)
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec}


def cldice_metric(prob, target, thr=0.5):
    vals = []
    for p1, t1 in zip(prob[:, 0], target[:, 0]):
        p = p1 >= thr
        t = t1 >= 0.5
        if not t.any():
            continue
        sp = skeletonize(p)
        st = skeletonize(t)
        tprec = (sp & t).sum() / (sp.sum() + EPS)
        tsens = (st & p).sum() / (st.sum() + EPS)
        vals.append(2 * tprec * tsens / (tprec + tsens + EPS))
    return float(np.mean(vals)) if vals else float("nan")
