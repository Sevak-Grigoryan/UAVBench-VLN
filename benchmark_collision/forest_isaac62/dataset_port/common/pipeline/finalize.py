"""
FINALIZE phase: stitch a mission's rendered slices into trajectory.mp4, check it, and publish the
sample to All_Simulations. Runs on the login node (ffmpeg only, no GPU).

    python finalize.py --run <Env>/_runs/<mission> --dest All_Simulations/<Difficulty>/<Env>/Simulation_K

Refuses to publish unless: the flight finished (flight/FLIGHT_OK), the slices cover frames
[0, N) exactly once with no gaps or overlaps, every slice has the frame count it claims, and the
stitched video has exactly as many frames as telemetry.json. Stitching is ffmpeg concat with
stream copy (no re-encode), valid because every slice uses the same encoder settings.

dest receives ONLY trajectory.mp4, action.json, telemetry.json, meta.json (+ depth.mkv, segmentation.mkv for
Coastal v2). Mountain (2026-10-20, review item C11): segmentation / depth are rendered and used for the annotations but
NOT published -- exactly the four files.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

FFMPEG = "/mnt/weka/shared-cache/miniforge3/bin/ffmpeg"
FFPROBE = "/mnt/weka/shared-cache/miniforge3/bin/ffprobe"


def nframes(p):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    a = ap.parse_args()
    flight, slices_dir = a.run / "flight", a.run / "slices"
    if not (flight / "FLIGHT_OK").exists():
        sys.exit("flight not finished OK")
    if a.dest.exists() and any(a.dest.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {a.dest}")
    n = len(json.loads((flight / "telemetry.json").read_text()))

    infos = sorted(((json.loads(p.read_text()), p.with_suffix(".mp4")) for p in slices_dir.glob("slice_*.json") if not p.name.endswith(".mask.json")),
                   key=lambda x: x[0]["start"])
    infos = [(i, p) for i, p in infos if p.exists()]
    pos = 0
    for info, mp4 in infos:
        if info["start"] != pos:
            sys.exit(f"slice coverage broken at frame {pos} (next slice starts at {info['start']})")
        got = nframes(mp4)
        if got != info["frames"]:
            sys.exit(f"{mp4.name}: {got} frames, expected {info['frames']}")
        pos = info["end"]
    if pos != n:
        sys.exit(f"slices cover [0,{pos}) but the flight has {n} frames")

    out = a.run / "trajectory.mp4"
    lst = slices_dir / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for _, p in infos))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    got = nframes(out)
    if got != n:
        sys.exit(f"stitched video has {got} frames, telemetry has {n}")

    # depth + segmentation slices (Coastal v2): stitch the same way and check every frame is there
    aux = {}
    if all(p.with_suffix(".depth.mkv").exists() and p.with_suffix(".seg.mkv").exists() for _, p in infos):
        for key, suf, dst in (("depth", ".depth.mkv", "depth.mkv"), ("segmentation", ".seg.mkv", "segmentation.mkv")):
            lst_a = slices_dir / f"concat_{key}.txt"
            lst_a.write_text("".join(f"file '{p.with_suffix(suf).resolve()}'\n" for _, p in infos))
            o = a.run / dst
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst_a), "-c", "copy", str(o)], check=True)
            got_a = nframes(o)
            if got_a != n:
                sys.exit(f"{dst}: {got_a} frames, telemetry has {n}")
            aux[dst] = o
    mask_px = {}
    for _, p in infos:
        mp = p.with_suffix(".mask.json")
        if mp.exists():
            mask_px.update(json.loads(mp.read_text()))

    meta = json.loads((flight / "meta.json").read_text())
    vi = meta.get("environment", {}).get("valley_info")
    if isinstance(vi, dict):  # drop the ~650 KB scene inventory, keep the short summary fields
        meta["environment"]["valley_info"] = {k: v for k, v in vi.items() if len(json.dumps(v, default=str)) <= 400}
    meta["render"] = {
        "renderer": infos[0][0]["renderer"], "rt_subframes": infos[0][0]["rt_subframes"],
        "method": "flight first (physics + PX4 only), then the recorded camera poses rendered in parallel slices",
        "slices": len(infos), "gpu_render_seconds": round(sum(i["render_s"] for i, _ in infos), 1),
        "nodes": sorted({i["node"] for i, _ in infos}),
    }
    meta["files"] = {"trajectory.mp4": "onboard gimbal camera video, 30 fps",
                     "action.json": "stick actions, one entry per change",
                     "telemetry.json": "per-frame pose, velocity, camera pose, PX4 state",
                     "meta.json": "this file"}
    telemetry = None
    env_name = meta.get("environment", {}).get("name")
    enrich_py = Path(__file__).resolve().parents[2] / str(env_name) / "enrich.py"
    if env_name in ("Coastal", "Mountain") and enrich_py.exists():  # new-generation annotations (NEXT_GENERATION_REQUIREMENTS.md)
        import importlib.util
        spec = importlib.util.spec_from_file_location("env_enrich", enrich_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not aux:
            sys.exit(f"{env_name} sample without depth/segmentation slices")
        telemetry = json.loads((flight / "telemetry.json").read_text())
        meta, telemetry = mod.enrich(meta, telemetry, a.run, mask_px)
    a.dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, a.dest / "trajectory.mp4")
    shutil.copy(flight / "action.json", a.dest / "action.json")
    if telemetry is None:
        shutil.copy(flight / "telemetry.json", a.dest / "telemetry.json")
    else:
        (a.dest / "telemetry.json").write_text(json.dumps(telemetry))
    if env_name != "Mountain":
        for dst, o in aux.items():
            shutil.copy(o, a.dest / dst)
    else:
        meta.pop("render_aux", None)
    # one common meta.json structure for every environment + consistency gate (data-consistency audit 2026-10-21)
    import meta_schema
    tel_ = telemetry if telemetry is not None else json.loads((flight / "telemetry.json").read_text())
    meta_schema.normalize(meta, tel_)
    acts_ = json.loads((flight / "action.json").read_text())
    acts_ = acts_["actions"] if isinstance(acts_, dict) else acts_
    bad = meta_schema.check(meta)
    nv_ = nframes(a.dest / "trajectory.mp4")
    if len(tel_) != meta["frames"] or nv_ != meta["frames"] or len(acts_) != meta["actions"]:
        bad.append(f"counts: telemetry {len(tel_)}, video {nv_}, frames {meta['frames']}, "
                   f"actions {len(acts_)} vs meta {meta['actions']}")
    if bad:
        sys.exit(f"meta.json inconsistent, not published: {bad}")
    (a.dest / "meta.json").write_text(json.dumps(meta, indent=1))
    if env_name == "Mountain":
        extra = sorted(p.name for p in a.dest.iterdir() if p.name not in ("trajectory.mp4", "action.json", "telemetry.json", "meta.json"))
        if extra:
            sys.exit(f"unexpected files in {a.dest}: {extra}")
    (a.run / "PUBLISHED").write_text(str(a.dest))
    print(f"published {n} frames from {len(infos)} slices -> {a.dest}")


if __name__ == "__main__":
    main()
