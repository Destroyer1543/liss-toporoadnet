import datetime, hashlib, json, pathlib

R = pathlib.Path("data/rebuttal")
S = pathlib.Path("evaluation")
from btrack_screen import FAM, ORIG

sel = json.loads((R / "btrack_screen__full.json").read_text())
code = ["vector_apls.py", "vector_apls_run.py", "decode_hysteresis.py", "decode_bridges.py", "infer_scene_variants.py",
        "infer_scene_probs.py", "infer_scene_mosaic.py", "btrack_screen.py", "aggregate_families.py", "btrack_final.py"]
man = {
    "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "selected": {f: sel[f]["_selected"] for f in FAM},
    "original": ORIG, "runs": FAM,
    "evaluation": {"scenes_test": ["R2F24AUG2023064073010100066SSANSTUC00GTDD", "RAF23OCT2023035685010100064SSANSTUC00GTDA"],
                   "scenes_external": ["R2F13DEC2025076029011100053SSANSTUC00GTDA", "R2F25JAN2026076640011000053SSANSTUC00GTDD",
                                       "RAF18JAN2026047300011100053SSANSTUC00GTDC", "RAF20DEC2025046888011000052SSANSTUC00GTDD"],
                   "scales_px": [256, 1024], "tiers": [2, 1], "buffer_px": 2, "primary": "T2 raster-GT APLS @ 1.5 km"},
    "success_rule": "ours(selected) beats every baseline family(selected) on T2 strict APLS at 1.5 km on BOTH test scenes "
                    "and on the pooled external scenes, 95% paired block-bootstrap CI excluding 0; no test scene where ours "
                    "is significantly worse than any baseline; segmentation mask (IoU/APLS_r) unchanged by the output contract.",
    "code_sha256": {c: hashlib.sha256((S / c).read_bytes()).hexdigest()[:16] for c in code if (S / c).exists()},
}
(R / "BTRACK_FROZEN_MANIFEST.json").write_text(json.dumps(man, indent=1))
md = pathlib.Path("rebuttal/BTRACK_FROZEN_MANIFEST.md")
md.write_text("# B-track frozen manifest (written mechanically by btrack_manifest.py before the test/external evaluation)\n\n"
              "```json\n" + json.dumps(man, indent=1) + "\n```\n")
print("FROZEN:", man["selected"])
