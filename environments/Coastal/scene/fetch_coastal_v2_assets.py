"""Download the CC0 Poly Haven assets used by Coastal v2 (Easy Coastal, 2026-10-13).

Re-uses the md5-verified downloader of City/scene/fetch_city_assets.py; same on-disk layout, so
natural_valley's create_pbr_material()/asset_path() helpers work against this root unchanged:
    <root>/textures/<texture_id>/{diffuse,normal,roughness}.<ext>
    <root>/models/<asset_id>/<asset_id>_1k.usdc
    <root>/hdri/<hdri_id>/...

The old Simulation_0_0_7 pack (assets/polyhaven-coastal) is left untouched.

    python3 fetch_coastal_v2_assets.py --root /mnt/weka/sgrigoryan/assets/polyhaven-coastal-v2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "City/scene"))
import fetch_city_assets as fca  # noqa: E402

MODELS = {
    # ships (Poly Haven "smugglers_cove") -- historic sailing ships at anchor / moored as museum ships
    "dutch_ship_large_01": "three-masted Dutch merchant ship",
    "dutch_ship_large_02": "three-masted Dutch ship, second variant",
    "dutch_ship_medium": "two-masted Dutch ship",
    "ship_pinnace": "small pinnace / rowing tender",
    # coast
    "coastal_cliff_01": "coastal cliff",
    "coastal_cliff_02": "coastal cliff",
    "coastal_cliff_04": "coastal cliff",
    "coast_rocks_01": "shore rocks", "coast_rocks_02": "shore rocks", "coast_rocks_03": "shore rocks",
    "coast_rocks_05": "shore rocks",
    "coast_land_rocks_03": "land rocks", "coast_land_rocks_04": "land rocks",
    "sand_rocks_small_01": "small sand rocks",
    # sea markers
    "ocean_buoy": "ocean buoy", "lateral_sea_marker": "lateral sea marker",
    # harbour detail
    "lifebuoy": "lifebuoy", "wooden_barrels_01": "wooden barrels", "wooden_crate_01": "wooden crate",
    "wooden_crate_02": "wooden crate", "plastic_crate_01": "fish crate", "plastic_crate_02": "fish crate",
    "wooden_lantern_01": "lantern", "wooden_picnic_table": "picnic table",
    "outdoor_table_chair_set_01": "cafe table set", "wooden_bucket_01": "bucket",
    "modular_fort_01": "stone fort", "bronze_whale_statue": "whale statue",
}
TEXTURES = {
    "coast_sand_05": "beach sand", "coast_sand_01": "beach sand (variant)", "damp_beach_sand": "wet sand",
    "cliff_side": "cliff rock", "coast_land_rocks_02": "rock", "rocky_terrain_02": "rocky ground",
    "aerial_grass_rock": "grass", "brown_planks_05": "pier deck planks", "dark_planks": "old pier planks",
    "blue_painted_planks": "boat-house walls", "corrugated_iron": "shed roofs / walls",
    "green_metal_rust": "rusty hull / containers", "blue_metal_plate": "ship deck plating",
    "gravel_concrete": "quay surface", "large_sandstone_blocks": "harbour wall", "clean_pebbles": "pebble beach",
}
HDRI = ["kloofendal_48d_partly_cloudy_puresky"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/weka/sgrigoryan/assets/polyhaven-coastal-v2")
    root = Path(ap.parse_args().root)
    root.mkdir(parents=True, exist_ok=True)
    got = {"models": {}, "textures": {}, "hdri": HDRI}
    for t, why in TEXTURES.items():
        fca.fetch_texture(root, t)
        got["textures"][t] = why
        print("texture", t, flush=True)
    for m, why in MODELS.items():
        try:
            fca.fetch_model(root, m)
            got["models"][m] = why
            print("model", m, flush=True)
        except Exception as e:  # keep going; a missing optional model is reported, not fatal
            print("FAILED model", m, e, flush=True)
    for h in HDRI:
        fca.fetch_hdri(root, h)
    (root / "asset_manifest.json").write_text(json.dumps(got, indent=1))
    print("done", len(got["models"]), "models", len(got["textures"]), "textures")


if __name__ == "__main__":
    main()
