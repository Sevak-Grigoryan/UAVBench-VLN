"""Probe NVIDIA's prebuilt outdoor environments before committing a mission design to them.

Simulation_0_0_5 needs a city at the SAME quality as 0_0_4's forest. The forest looks good because
it is a real authored environment with real photogrammetry assets -- so the city needs real
authored content too, not the primitive coloured boxes Simulations_0_0_3 used.

Isaac/Environments/Outdoor/Rivermark is a candidate, but nothing is known about it here: an
earlier attempt to load it (job 250419) was cancelled after 8 minutes, and these assets stream
over HTTPS (the Warehouse asset alone measured 282s to load). So before designing anything:
measure the load time, get the bounding box, dump the top-level prim structure so mission
landmarks can be found, and render probe frames from several altitudes so the thing can actually
be LOOKED at.

Cheap to run, and it decides the whole design -- same verify-before-build discipline used for the
forest (terrain profile, asset heights and tree density were all measured before any mission ran).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import hq_bridge

OUT_DIR = Path("/mnt/weka/sgrigoryan/Simulation_0_0_5/_probe")
CANDIDATES = [
    "Isaac/Environments/Outdoor/Rivermark/rivermark.usd",
]
# (x, y, z, yaw_deg, pitch_deg) probe views -- a low street-level look, a mid oblique, and a high
# overview, so both the detail and the overall layout are visible.
PROBE_VIEWS = [
    (0.0, -60.0, 8.0, 90.0, 5.0, "street_low"),
    (0.0, -120.0, 40.0, 90.0, 20.0, "oblique_mid"),
    (0.0, -260.0, 140.0, 90.0, 35.0, "overview_high"),
    (120.0, 120.0, 60.0, 225.0, 25.0, "corner_view"),
]


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = "/mnt/weka/sgrigoryan/Simulation_0_0_5/_runs/probe"

    # Build the HQ app but with an EMPTY world -- we only want the referenced environment.
    class ProbeBridge(hq_bridge.HQBridge):
        def _setup_world(self):
            UsdGeom, stage = self.UsdGeom, self.stage
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())

    bridge = ProbeBridge(run_dir)
    print(f"[{time.time()-t0:.1f}s] ProbeBridge booted, render_mode={bridge.current_render_mode()}",
          flush=True)

    from pxr import Usd, UsdGeom, Gf

    report = {}
    for rel in CANDIDATES:
        url = f"{bridge.assets_root.rstrip('/')}/{rel}"
        name = rel.split("/")[-2]
        print(f"[{time.time()-t0:.1f}s] referencing {url}", flush=True)
        t_load = time.time()
        prim = bridge.stage.DefinePrim(f"/World/{name}", "Xform")
        prim.GetReferences().AddReference(url)
        # let the payload stream in
        for i in range(600):
            bridge.app.update()
            if i % 100 == 99:
                print(f"    [{time.time()-t0:.1f}s] streaming... update {i+1}", flush=True)
        load_s = time.time() - t_load

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        n_prims = sum(1 for _ in Usd.PrimRange(prim))
        info = {
            "url": url,
            "load_seconds": round(load_s, 1),
            "prim_count": n_prims,
            "bbox_empty": rng.IsEmpty(),
        }
        if not rng.IsEmpty():
            mn, mx = rng.GetMin(), rng.GetMax()
            info["bbox_min"] = [round(v, 2) for v in mn]
            info["bbox_max"] = [round(v, 2) for v in mx]
            info["size_m"] = [round(mx[i] - mn[i], 2) for i in range(3)]
        print(f"[{time.time()-t0:.1f}s] {name}: {json.dumps(info)}", flush=True)

        # top-level children, so mission landmarks can be located
        kids = []
        for child in prim.GetChildren():
            crng = cache.ComputeWorldBound(child).ComputeAlignedRange()
            entry = {"name": child.GetName(), "type": child.GetTypeName()}
            if not crng.IsEmpty():
                cmn, cmx = crng.GetMin(), crng.GetMax()
                entry["center"] = [round((cmn[i] + cmx[i]) / 2.0, 1) for i in range(3)]
                entry["size"] = [round(cmx[i] - cmn[i], 1) for i in range(3)]
            kids.append(entry)
        info["children"] = kids
        print(f"[{time.time()-t0:.1f}s] {len(kids)} top-level children:", flush=True)
        for k in kids[:40]:
            print(f"    {k}", flush=True)
        report[name] = info

        from PIL import Image

        for (px, py, pz, yaw, pitch, label) in PROBE_VIEWS:
            bridge.set_camera_pose(px, py, pz, yaw, pitch)
            frame = bridge.process_camera_data()
            out = OUT_DIR / f"{name}_{label}.png"
            Image.fromarray(frame).save(out)
            print(f"[{time.time()-t0:.1f}s] probe {label}: mean={frame.mean():.1f} -> {out}",
                  flush=True)

    (OUT_DIR / "probe_report.json").write_text(json.dumps(report, indent=2))
    print(f"[{time.time()-t0:.1f}s] PROBE DONE -> {OUT_DIR}/probe_report.json", flush=True)
    bridge.shutdown()


if __name__ == "__main__":
    main()
