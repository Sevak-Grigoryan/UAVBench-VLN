"""GPU probe: do depth + semantic/instance segmentation annotators work alongside the RealTimePathTracing rgb
capture, and do labels on a parent Xform propagate to its children? Writes PNGs + a JSON report to --out."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "scene"), str(HERE.parents[1] / "common/pipeline")]
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
import hq_bridge

class Probe(hq_bridge.HQBridge):
    def _setup_world(self):
        super()._setup_world()
        from pxr import UsdGeom, Gf
        st = self.stage
        for i, (name, x) in enumerate((("ship", -3.0), ("person", 0.0), ("crate", 3.0))):
            root = UsdGeom.Xform.Define(st, f"/World/Obj_{name}")
            for j in range(2):
                c = UsdGeom.Cube.Define(st, f"/World/Obj_{name}/part{j}")
                c.CreateSizeAttr(1.0); c.AddTranslateOp().Set(Gf.Vec3d(x, 6.0 + j * 1.2, 0.5 + j))
        from isaacsim.core.utils.semantics import add_labels
        add_labels(st.GetPrimAtPath("/World/Obj_ship"), ["ship"], "class")
        add_labels(st.GetPrimAtPath("/World/Obj_person"), ["person"], "class")
        add_labels(st.GetPrimAtPath("/World/Obj_crate"), ["crate"], "class")

b = Probe(a.out / "kit")
rep = b.rep
rep_ann = {}
for name, kw in (("distance_to_image_plane", {}), ("semantic_segmentation", {"colorize": False}),
                 ("instance_id_segmentation", {"colorize": False})):
    try:
        an = rep.AnnotatorRegistry.get_annotator(name, init_params=kw) if kw else rep.AnnotatorRegistry.get_annotator(name)
        an.attach(b.render_product); rep_ann[name] = an
    except Exception as e:
        print("ATTACH FAIL", name, e, flush=True)
b.set_camera_pose(0.0, 0.0, 2.0, 90.0, 10.0)
rep_out = {}
t0 = time.time()
for attempt in range(12):
    rgb = b.process_camera_data()
    print("attempt", attempt, "rgb shape", rgb.shape, flush=True)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        break
    for _ in range(4):
        b.app.update()
t_rgb = time.time() - t0
rgb = rgb[..., :3]
from PIL import Image
Image.fromarray(rgb).save(a.out / "rgb.png")
for name, an in rep_ann.items():
    d = an.get_data()
    data = d["data"] if isinstance(d, dict) else d
    info = d.get("info", {}) if isinstance(d, dict) else {}
    arr = np.asarray(data)
    rep_out[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "min": float(np.nanmin(arr[np.isfinite(arr)])) if arr.size else None,
                     "max": float(np.nanmax(arr[np.isfinite(arr)])) if arr.size else None, "info": {k: str(v)[:600] for k, v in info.items()},
                     "unique": [int(u) for u in np.unique(arr)[:20]] if arr.dtype.kind in "iu" else None}
    if name == "distance_to_image_plane":
        dd = np.nan_to_num(arr, nan=0, posinf=0); Image.fromarray((np.clip(dd / 30, 0, 1) * 255).astype(np.uint8)).save(a.out / "depth.png")
    else:
        Image.fromarray(((arr.astype(np.int64) * 47) % 255).astype(np.uint8)).save(a.out / f"{name}.png")
t0 = time.time()
for _ in range(3): b.process_camera_data(); [an.get_data() for an in rep_ann.values()]
rep_out["time_per_frame_s_with_aux"] = (time.time() - t0) / 3
rep_out["time_first_rgb_s"] = t_rgb
(a.out / "probe.json").write_text(json.dumps(rep_out, indent=1))
print("PROBE DONE", json.dumps(rep_out)[:3000], flush=True)
b.shutdown()
