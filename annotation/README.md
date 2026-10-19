# LISS-IV Road Observability Study — annotation app

A tiny client/server web app. Students shade the roads they can see on satellite tiles;
each tile is saved to the server as they go. **Python stdlib only — no pip installs.**

## Files
- `server.py`   — backend (serves the app + receives submissions). stdlib only.
- `index.html`  — the annotation UI (brush shading, ~24 tiles, progress + timer).
- `tiles.json`  — the 24 image tiles (embedded, 8 highway / 8 major / 8 minor).
- `submissions/`— created at runtime; one `<student>.jsonl` per annotator (one line per tile).

## Run it (on any machine with python3)
    cd annotation
    python3 server.py            # or: python3 server.py 9000  for a custom port
Then share  `http://<this-machine-ip>:8000`  with your 4-5 annotators (same Wi-Fi/LAN,
or a public IP / port-forward if remote). Each spends ~20-30 min, does as many tiles as
they can, clicks **Finish & submit** when done. Data is saved per tile, so partial
sessions are never lost. `GET /health` shows how many students have submitted.

## Collect + score
Submissions accumulate in `annotation/submissions/`. Copy that folder back to the repo
machine (where the ground truth lives) and run:

    python3 scripts/score_annotations.py annotation/submissions

Prints human recall vs road width + per-class recall (highway/major/minor). Ground truth:
`data/annot_gt/` (T00-T23, never shown to students).

## Expected result
Recall high on highway/major tiles, collapsing on minor tiles -> a human "observability
floor" that motivates the two-tier eval and major-only label filtering.
