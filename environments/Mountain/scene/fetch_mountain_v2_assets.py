"""Download the CC0 Poly Haven assets used by Mountain v2 (Easy Mountain dataset, 2026-10-20).

Re-uses the md5-verified downloader of City/scene/fetch_city_assets.py; same on-disk layout as the other packs:
    <root>/textures/<texture_id>/...   <root>/models/<asset_id>/<asset_id>_1k.usdc   <root>/hdri/<hdri_id>/<id>_2k.hdr
and measures every sky's sun (elevation, image longitude, colour) into <root>/hdri/sky_info.json, the way the Coastal v2
sky table was measured, so the distant sun light points where the sky image has its sun.

The old Simulation_0_0_6 pack (assets/polyhaven-mountain) is left untouched.

    python3 fetch_mountain_v2_assets.py --root /mnt/weka/sgrigoryan/assets/polyhaven-mountain-v2
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "City/scene"))
import fetch_city_assets as fca  # noqa: E402

MODELS = {
    # trees (conifer forest below the treeline, broadleaf in the valley)
    "fir_tree_01": "mature fir", "pine_tree_01": "mature pine", "fir_sapling_medium": "young fir",
    "pine_sapling_medium": "young pine", "fir_sapling": "fir sapling", "pine_sapling_small": "pine sapling",
    "tree_small_02": "small broadleaf tree", "jacaranda_tree": "broadleaf tree", "dead_tree_trunk_02": "dead trunk",
    "shrub_02": "shrub", "fern_02": "fern", "grass_bermuda_01": "grass tuft",
    # rocks
    "boulder_01": "boulder", "rock_07": "rock", "rock_09": "rock", "rock_moss_set_01": "mossy rocks",
    "rock_moss_set_02": "mossy rocks", "stone_01": "stone", "rock_face_01": "rock face", "rock_face_02": "rock face",
    "namaqualand_boulder_02": "boulder", "namaqualand_boulder_03": "boulder", "namaqualand_boulder_04": "boulder",
    "namaqualand_boulder_05": "boulder", "namaqualand_rocks_01": "rock cluster", "tree_stump_01": "stump",
    # camp / hut detail
    "wooden_picnic_table": "picnic table", "wooden_barrels_01": "barrels", "wooden_crate_01": "crate",
}
TEXTURES = {
    "snow_02": "snow cap", "snow_03": "wind-packed snow", "snow_field_aerial": "snow field from above",
    "rock_face": "cliff face", "rock_face_03": "cliff face (grey)", "cliff_side": "cliff (warm)", "dark_rock": "dark rock",
    "lichen_rock": "lichen rock", "aerial_rocks_02": "scree", "aerial_rocks_04": "scree (grey)",
    "aerial_ground_rock": "rocky ground", "rocky_trail": "rocky trail", "brown_mud_rocks_01": "dirt slope",
    "forest_ground_04": "forest floor", "leafy_grass": "alpine meadow", "sparse_grass": "sparse grass",
    "withered_grass": "autumn grass", "aerial_grass_rock": "grass with rocks", "river_small_rocks": "river bed",
    "gravel_road": "gravel road", "asphalt_02": "mountain road", "large_sandstone_blocks": "dam / wall stone",
    "stacked_stone_wall": "stone walls", "brown_planks_05": "timber", "corrugated_iron": "hut roofs",
    "gravel_concrete": "concrete", "clay_roof_tiles": "village roofs",
}
HDRI = ["kloofendal_48d_partly_cloudy_puresky", "kloofendal_43d_clear_puresky", "kloofendal_38d_partly_cloudy_puresky",
        "qwantani_noon_puresky", "qwantani_mid_morning_puresky", "qwantani_morning_puresky", "qwantani_afternoon_puresky",
        "qwantani_late_afternoon_puresky", "syferfontein_18d_clear_puresky", "qwantani_sunset_puresky",
        "kloofendal_overcast_puresky", "kloofendal_misty_morning_puresky", "kloofendal_28d_misty_puresky",
        "pizzo_pernice_puresky", "rocky_ridge_puresky", "table_mountain_1_puresky", "snow_field_puresky"]


def read_rgbe(path):
    """Minimal Radiance .hdr (RGBE, new-style RLE) reader -> float32 HxWx3 (imageio's default plugin clips to 8 bit)."""
    import numpy as np

    data = Path(path).read_bytes()
    pos = 0
    while True:
        end = data.index(b"\n", pos)
        line = data[pos:end]
        pos = end + 1
        if line.startswith((b"-Y", b"+Y")):
            _, h, _, w = line.split()
            h, w = int(h), int(w)
            break
    out = np.zeros((h, w, 4), np.uint8)
    buf = np.frombuffer(data, np.uint8)
    for y in range(h):
        if buf[pos] == 2 and buf[pos + 1] == 2:
            pos += 4
            for ch in range(4):
                x = 0
                while x < w:
                    n = int(buf[pos]); pos += 1
                    if n > 128:
                        n -= 128
                        out[y, x:x + n, ch] = buf[pos]; pos += 1
                    else:
                        out[y, x:x + n, ch] = buf[pos:pos + n]; pos += n
                    x += n
        else:
            out[y] = buf[pos:pos + 4 * w].reshape(w, 4); pos += 4 * w
    e = out[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)
    return out[..., :3].astype(np.float32) * scale[..., None]


def measure_sky(hdr):
    """Sun position + colour of a latlong HDR: brightest smoothed region (elevation from the image row, longitude u)."""
    import numpy as np

    img = read_rgbe(hdr)
    h, w = img.shape[:2]
    lum = img[..., :3].mean(axis=2)
    top = lum[: h // 2]
    k = int(np.argmax(top))
    r, c = divmod(k, w)
    win = img[max(0, r - 3): r + 4, max(0, c - 3): c + 4, :3].reshape(-1, 3).mean(axis=0)
    rgb = (win / max(1e-6, win.max())).tolist()
    med = float(np.median(top))
    return {"sun_elev_deg": round(90.0 - (r + 0.5) / h * 180.0, 1), "sun_u_deg": round((c + 0.5) / w * 360.0, 1),
            "peak": round(float(top[r, c]), 1), "sky_median": round(med, 3), "sun_to_sky": round(float(top[r, c]) / max(med, 1e-6), 1),
            "sun_rgb": [round(v, 3) for v in rgb]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/weka/sgrigoryan/assets/polyhaven-mountain-v2")
    root = Path(ap.parse_args().root)
    root.mkdir(parents=True, exist_ok=True)
    got = {"source": "Poly Haven", "license": "CC0 1.0 Universal", "models": {}, "textures": {}, "hdri": []}
    for t, why in TEXTURES.items():
        try:
            fca.fetch_texture(root, t)
            got["textures"][t] = why
            print("texture", t, flush=True)
        except Exception as e:
            print("FAILED texture", t, e, flush=True)
    for m, why in MODELS.items():
        try:
            fca.fetch_model(root, m)
            got["models"][m] = why
            print("model", m, flush=True)
        except Exception as e:
            print("FAILED model", m, e, flush=True)
    info = {}
    for hid in HDRI:
        try:
            fca.fetch_hdri(root, hid)
            info[hid] = measure_sky(root / "hdri" / hid / f"{hid}_2k.hdr")
            got["hdri"].append(hid)
            print("hdri", hid, info[hid], flush=True)
        except Exception as e:
            print("FAILED hdri", hid, e, flush=True)
    (root / "hdri" / "sky_info.json").write_text(json.dumps(info, indent=1))
    (root / "asset_manifest.json").write_text(json.dumps(got, indent=1))
    print("done", len(got["models"]), "models", len(got["textures"]), "textures", len(got["hdri"]), "skies")


if __name__ == "__main__":
    main()
