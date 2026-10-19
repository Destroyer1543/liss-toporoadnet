import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

API = "https://bhoonidhi-api.nrsc.gov.in"
COLLECTIONS = ["ResourceSat-2_LISS4-MX70_L2", "ResourceSat-2A_LISS4-MX70_L2"]


def _post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def auth():
    user, pw = os.environ.get("BHOONIDHI_USER"), os.environ.get("BHOONIDHI_PASS")
    if not user or not pw:
        sys.exit("Set BHOONIDHI_USER and BHOONIDHI_PASS")
    tok = _post(f"{API}/auth/token",
                {"userId": user, "password": pw, "grant_type": "password"})
    return tok["access_token"]


def search(token, bbox, start, end, limit=200):
    lon0, lat0, lon1, lat1 = bbox
    poly = [[[lon0, lat0], [lon0, lat1], [lon1, lat1], [lon1, lat0], [lon0, lat0]]]
    payload = {
        "collections": COLLECTIONS,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "intersects": {"type": "Polygon", "coordinates": poly},
        "limit": limit,
        "filter-lang": "cql2-json",
    }
    return _post(f"{API}/data/search", payload, token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", nargs=4, type=float, required=True,
                    metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="search + manifest only, no downloads")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    token = auth()
    res = search(token, args.aoi, args.start, args.end)
    feats = res.get("features", res.get("results", []))
    manifest = []
    for f in feats:
        props = f.get("properties", {})
        manifest.append({
            "id": f.get("id"),
            "collection": f.get("collection"),
            "datetime": props.get("datetime"),
            "online": props.get("Online", props.get("online")),
            "cloud": props.get("cloudcover", props.get("CloudCover")),
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    online = [m for m in manifest if str(m["online"]).upper().startswith("Y")]
    print(f"found {len(manifest)} scenes, {len(online)} online; manifest -> {out/'manifest.json'}")
    if args.dry_run:
        return
    for i, m in enumerate(online):
        dest = out / f"{m['id']}.zip"
        if dest.exists():
            continue
        url = f"{API}/download?id={m['id']}&collection={m['collection']}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        print(f"[{i+1}/{len(online)}] {m['id']}")
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        time.sleep(1)


if __name__ == "__main__":
    main()
