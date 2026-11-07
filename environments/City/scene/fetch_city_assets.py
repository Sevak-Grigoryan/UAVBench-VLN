"""Download the CC0 Poly Haven asset set used by Simulation_0_0_5's city.

Why this exists: NVIDIA's Isaac asset library has NO city building content -- its ArchVis tree is
interior furniture (Conference/Seating/Kitchen/...), Industrial/Buildings holds only a Warehouse,
and the single outdoor environment (Rivermark) measured ~19 minutes to stream with broken upstream
texture references. So the city is AUTHORED from real assets instead, exactly the way the forest
in Simulations_0_0_4 got its quality: real PBR textures + real props.

Files are written in the same on-disk layout natural_valley.py's helpers expect, so
create_pbr_material()/create_asset_material()/asset_path() work against this root unchanged:
    <root>/textures/<texture_id>/{diffuse,normal,roughness}.<ext>
    <root>/models/<asset_id>/<asset_id>_1k.usdc  (+ its own textures/ subdir)

Downloads are md5-verified against the Poly Haven API manifest and skipped when already present,
so re-running is cheap and safe.
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

# PBR textures: (poly haven id, what it is used for)
CITY_TEXTURES = {
    "aerial_asphalt_01": "road surface (authored for aerial views)",
    "asphalt_02": "secondary road / parking",
    "cobblestone_floor_01": "plaza / sidewalk",
    "brick_4": "building facade - brick",
    "beige_wall_001": "building facade - painted concrete",
    "white_stucco": "building facade - stucco",
    "ceramic_roof_01": "roof - ceramic",
    "clay_roof_tiles": "roof - clay tile",
    # tower / modern facades -- these are literal building-facade textures, which is what makes
    # authored prismatic geometry read as a real building rather than a coloured box
    "rectangular_facade_tiles": "tower facade - panel grid",
    "concrete_tile_facade": "tower facade - concrete tile",
    "concrete_panels": "tower facade - concrete panels",
    "concrete_wall_008": "plain concrete wall",
    "metal_plate": "metal trim / bus-bay detail",
}

# HDRI sky. MUST be a sky-only ("puresky") capture.
#
# This was 'cedar_bridge_2' -- described as an urban midday sky, and it does light the scene
# correctly, but it is a GROUND-LEVEL photograph taken standing on a bridge: the image contains
# the bridge deck, its railings, roadside guardrails and buildings. A DomeLight projects its
# texture at infinity, so from a drone at 52m all of that was painted around the horizon, reading
# as a warehouse and a guardrail looming over the city. It was the single biggest reason the
# environment looked like it stopped at a hard edge.
#
# A "_puresky" HDRI has clouds above and a smooth neutral gradient below, which blends into the
# far ground instead of fighting it. Check any replacement by tone-mapping it and LOOKING at it --
# an automatic "horizon detail" metric scored cedar_bridge_2 as clean, so that check does not work.
CITY_HDRI = "kloofendal_48d_partly_cloudy_puresky"

# Real street props, instanced through the city
CITY_MODELS = [
    "street_lamp_01",
    "street_lamp_02",
    "fire_hydrant",
    "painted_wooden_bench",
    "modular_street_seating",
    "metal_trash_can",
    "concrete_road_barrier",
    "planter_box_01",
    "modular_electricity_poles",
    "covered_car",
    "modular_fire_escape",
]

# which API channel -> which on-disk basename natural_valley.texture_path() globs for
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
            pass  # size mismatch -> re-download
        elif expect_md5 is None or file_md5(target) == expect_md5:
            return False  # already good
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
    # the .usdc references these by the relative paths used as dict keys
    for rel, item in entry.get("include", {}).items():
        if download(item["url"], base / rel, item.get("md5"), item.get("size")):
            written += 1
    # natural_valley.asset_path() always asks for <id>_1k.usdc; alias if the only res differs
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/weka/sgrigoryan/assets/polyhaven-city")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for texture_id, purpose in CITY_TEXTURES.items():
        n = fetch_texture(root, texture_id)
        print(f"texture {texture_id:24s} {n} new file(s)   [{purpose}]", flush=True)
    for asset_id in CITY_MODELS:
        n = fetch_model(root, asset_id)
        print(f"model   {asset_id:24s} {n} new file(s)", flush=True)
    n = fetch_hdri(root, CITY_HDRI)
    print(f"hdri    {CITY_HDRI:24s} {n} new file(s)", flush=True)

    manifest = {
        "source": "Poly Haven",
        "license": "CC0 1.0 Universal",
        "credit": "Powered by Poly Haven (https://polyhaven.com)",
        "textures": CITY_TEXTURES,
        "models": CITY_MODELS,
        "hdri": CITY_HDRI,
    }
    (root / "asset_manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"DONE: {args.root}  ({total/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
