import json, pathlib, numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
d = np.load("data/rebuttal/gt_test_cache.npz")
gt = np.unpackbits(d["gt"], axis=-1).astype(bool)
t = []
for m in gt:
    if m.sum() < 10:
        continue
    t.append(distance_transform_edt(m)[skeletonize(m)])
t = np.concatenate(t)
res = {"n_px": int(len(t)), "target_mean": float(t.mean()), "target_median": float(np.median(t)),
       "mae_const_mean": float(np.abs(t - t.mean()).mean()), "mae_const_median": float(np.abs(t - np.median(t)).mean()),
       "frac_lt1.5": float((t < 1.5).mean())}
for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 99)):
    s = (t >= lo) & (t < hi)
    res[f"mae_const_median_t{lo}-{hi}"] = float(np.abs(t[s] - np.median(t)).mean())
print(res); pathlib.Path("data/rebuttal/width_baseline.json").write_text(json.dumps(res, indent=1))
