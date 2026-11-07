"""Download the CC0 Poly Haven asset set used by Simulation_0_0_7's coastline.

Same download/verify machinery as the mountain's fetch_mountain_assets.py (md5-verified,
skip-if-present, then MEASURE diffuse contrast before trusting a texture -- names alone were not
reliable evidence in the city, where 3 of 5 facade textures turned out to be near-blank).

    aerial_beach_01 was tried and DROPPED: measured std 3.47 (flat). coast_sand_05 (std 7.29) is
    used for the beach instead.

Files land in the layout natural_valley.py's helpers expect:
    <root>/textures/<texture_id>/{diffuse,normal,roughness}.<ext>
    <root>/hdri/<hdri_id>/<hdri_id>_{2k,4k}.hdr
grass/rock PROPS are reused directly from the forest's own pack (polyhaven-v2) -- no need to
re-download species already proven to look right.
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

# Sand/rock/grass PBR. Picked from Poly Haven's coastal-adjacent textures, then verified for
# actual diffuse detail (see main()) before being trusted.
COASTAL_TEXTURES = {
    "coast_sand_05": "beach / wet sand",
    "cliff_side": "cliff face -- primary rock band",
    "coast_land_rocks_02": "cliff face -- secondary rock variety",
    "rocky_terrain_02": "rocky clifftop ground",
    "dry_ground_rocks": "rocky clifftop ground -- secondary",
}

# Sky-only capture, same one already verified (fetched, tone-mapped, and looked at) for the
# mountain project -- reused rather than re-verified, since puresky HDRIs from the same source
# don't need re-checking each time.
COASTAL_HDRI = "kloofendal_48d_partly_cloudy_puresky"

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


def fetch_hdri(root, hdri_id, res="2k"):
    manifest = api_get(f"{API}/files/{hdri_id}")
    entry = manifest["hdri"][res]["hdr"]
    target = Path(root) / "hdri" / hdri_id / f"{hdri_id}_{res}.hdr"
    return 1 if download(entry["url"], target, entry.get("md5"), entry.get("size")) else 0


def check_texture_contrast(root):
    from PIL import Image
    import numpy as np

    print("\n--- diffuse detail check (std over the diffuse image; <5 reads as flat paint) ---")
    for texture_id in COASTAL_TEXTURES:
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
    ap.add_argument("--root", default="/mnt/weka/sgrigoryan/assets/polyhaven-coastal")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for texture_id, purpose in COASTAL_TEXTURES.items():
        n = fetch_texture(root, texture_id)
        print(f"texture {texture_id:24s} {n} new file(s)   [{purpose}]", flush=True)
    n = fetch_hdri(root, COASTAL_HDRI)
    print(f"hdri    {COASTAL_HDRI:24s} {n} new file(s)", flush=True)

    manifest = {
        "source": "Poly Haven",
        "license": "CC0 1.0 Universal",
        "credit": "Powered by Poly Haven (https://polyhaven.com)",
        "textures": COASTAL_TEXTURES,
        "reused_from_forest_pack": ["grass_bermuda_01", "shrub_02", "boulder_01", "rock_07",
                                     "rock_09", "rock_moss_set_01", "stone_01"],
        "hdri": COASTAL_HDRI,
    }
    (root / "asset_manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"DONE: {args.root}  ({total/1e6:.1f} MB)", flush=True)

    check_texture_contrast(root)


if __name__ == "__main__":
    main()
