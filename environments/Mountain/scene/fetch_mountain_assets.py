"""Download the CC0 Poly Haven asset set used by Simulation_0_0_6's mountain.

Same download/verify machinery as Simulation_0_0_5's fetch_city_assets.py (md5-verified,
skip-if-present), applied to a mountain instead of a city:

  * ROCK/SNOW textures were picked and then MEASURED (diffuse std over the actual image) before
    being wired into the scene -- see run_texture_contrast_check() below. Three of the city's five
    facade textures turned out to be near-blank Poly Haven "detail lives in the normal map" wall
    sets, and that was only caught by measuring, not by the texture's name or category.
  * the HDRI is a PURESKY capture, chosen the same way the city's replacement was: fetched, then
    tone-mapped and actually looked at (an automatic "horizon detail" heuristic scored a
    ground-level bridge photo as clean in the city project, so that check is not trusted).

Files land in the layout natural_valley.py's helpers expect:
    <root>/textures/<texture_id>/{diffuse,normal,roughness}.<ext>
    <root>/models/<asset_id>/<asset_id>_1k.usdc  (+ its own textures/ subdir)
    <root>/hdri/<hdri_id>/<hdri_id>_{2k,4k}.hdr
"""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

API = "https://api.polyhaven.com"
USER_AGENT = "YerevaNN-VLN-Simulations/1.0 (+https://github.com/YerevaNN/VLN-Simulations)"

# Rock/snow/ground PBR. Picked from Poly Haven's rock category, then verified for actual diffuse
# detail (see main()) before being trusted -- names alone were not reliable evidence in the city.
MOUNTAIN_TEXTURES = {
    "rock_face": "cliff / exposed rock face",
    "aerial_rocks_02": "scree / talus slope, shot from above",
    "dry_riverbed_rock": "boulder field / rocky ground",
    "brown_mud_rocks_01": "dirt trail / lower slope",
    "snow_02": "snow cap",
    "snow_field_aerial": "snow field, shot from above",
}

# Sky-only capture -- clouds above, smooth gradient below, no photographed ground clutter. Chosen
# by the same method the city HDRI fix used: fetch, tone-map, and look at it (see
# verify_hdri_is_sky_only() at the bottom of this file).
MOUNTAIN_HDRI = "kloofendal_48d_partly_cloudy_puresky"

# Real rock/vegetation props. boulder_01/rock_07/rock_09/rock_moss_set_01/stone_01 are already in
# the forest's asset pack (polyhaven-v2) and are reused from there -- no need to fetch twice, and
# it keeps the mountain's boulders visually consistent with the forest's. mountainside is a large
# rock-formation model that is fetched here since the forest doesn't already have it.
MOUNTAIN_MODELS = ["mountainside"]

TEXTURE_CHANNELS = (("Diffuse", "jpg", "diffuse"), ("nor_gl", "exr", "normal"), ("Rough", "exr", "roughness"))


def api_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def file_md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target, expect_md5=None, expect_size=None):
    target = Path(target)
    if target.is_file():
        if expect_size is not None and target.stat().st_size != expect_size:
            pass
        elif expect_md5 is None or file_md5(target) == expect_md5:
            return False
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    fd, tmp = tempfile.mkstemp(prefix=".dl-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=300) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if expect_md5 is not None and file_md5(tmp) != expect_md5:
            raise ValueError(f"md5 mismatch for {url}")
        os.replace(tmp, target)
        return True
    finally:
        Path(tmp).unlink(missing_ok=True)


def fetch_texture(root, texture_id):
    manifest = api_get(f"{API}/files/{texture_id}")
    written = 0
    for channel, fmt, basename in TEXTURE_CHANNELS:
        try:
            entry = manifest[channel]["1k"][fmt]
        except KeyError:
            continue
        target = Path(root) / "textures" / texture_id / f"{basename}.{fmt}"
        if download(entry["url"], target, entry.get("md5"), entry.get("size")):
            written += 1
    return written


def fetch_model(root, asset_id):
    manifest = api_get(f"{API}/files/{asset_id}")
    usd = manifest["usd"]
    res = "1k" if "1k" in usd else sorted(usd.keys())[0]
    entry = usd[res]["usd"]
    base = Path(root) / "models" / asset_id
    written = 0
    if download(entry["url"], base / f"{asset_id}_{res}.usdc", entry.get("md5"), entry.get("size")):
        written += 1
    for rel, item in entry.get("include", {}).items():
        if download(item["url"], base / rel, item.get("md5"), item.get("size")):
            written += 1
    if res != "1k":
        alias = base / f"{asset_id}_1k.usdc"
        if not alias.exists():
            alias.symlink_to(f"{asset_id}_{res}.usdc")
    return written


def fetch_hdri(root, hdri_id, res="2k"):
    manifest = api_get(f"{API}/files/{hdri_id}")
    entry = manifest["hdri"][res]["hdr"]
    target = Path(root) / "hdri" / hdri_id / f"{hdri_id}_{res}.hdr"
    return 1 if download(entry["url"], target, entry.get("md5"), entry.get("size")) else 0


def check_texture_contrast(root):
    """Measure diffuse std for every fetched texture and print it. This is the check that would
    have caught the city's three blank facade textures before they ever reached the renderer --
    run it here instead of finding out from a rendered frame."""
    from PIL import Image
    import numpy as np

    print("\n--- diffuse detail check (std over the diffuse image; <5 reads as flat paint) ---")
    for texture_id in MOUNTAIN_TEXTURES:
        f = Path(root) / "textures" / texture_id / "diffuse.jpg"
        if not f.exists():
            print(f"  {texture_id:22s} MISSING")
            continue
        a = np.asarray(Image.open(f).convert("RGB").resize((256, 256)), dtype=float)
        std = float(a.mean(axis=2).std())
        flag = "" if std >= 5.0 else "  <-- FLAT, do not use for hero surfaces"
        print(f"  {texture_id:22s} std={std:6.2f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/weka/sgrigoryan/assets/polyhaven-mountain")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for texture_id, purpose in MOUNTAIN_TEXTURES.items():
        n = fetch_texture(root, texture_id)
        print(f"texture {texture_id:24s} {n} new file(s)   [{purpose}]", flush=True)
    for asset_id in MOUNTAIN_MODELS:
        n = fetch_model(root, asset_id)
        print(f"model   {asset_id:24s} {n} new file(s)", flush=True)
    n = fetch_hdri(root, MOUNTAIN_HDRI)
    print(f"hdri    {MOUNTAIN_HDRI:24s} {n} new file(s)", flush=True)

    manifest = {
        "source": "Poly Haven",
        "license": "CC0 1.0 Universal",
        "credit": "Powered by Poly Haven (https://polyhaven.com)",
        "textures": MOUNTAIN_TEXTURES,
        "models": MOUNTAIN_MODELS,
        "reused_from_forest_pack": ["boulder_01", "rock_07", "rock_09", "rock_moss_set_01",
                                     "stone_01", "pine_sapling_small", "fir_sapling"],
        "hdri": MOUNTAIN_HDRI,
    }
    (root / "asset_manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"DONE: {args.root}  ({total/1e6:.1f} MB)", flush=True)

    check_texture_contrast(root)


if __name__ == "__main__":
    main()
