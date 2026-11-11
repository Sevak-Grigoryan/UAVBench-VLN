"""Episode window of a flown benchmark mission: the frames between the end of the pre-roll and the expert's STOP.
    python window.py <work>/flight        -> prints "first_frame end_frame" (end exclusive)"""
import json
import sys
from pathlib import Path


def episode_window(flight_dir):
    flight_dir = Path(flight_dir)
    meta = json.loads((flight_dir / "meta.json").read_text())
    tele = json.loads((flight_dir / "telemetry.json").read_text())
    beats = meta["mission"]["beats"]
    ep = [i for i, b in enumerate(beats) if b.get("phase") == "episode"]
    ev = {e["beat"]: e for e in meta["events"]}
    t0, t1 = ev[ep[0]]["t_start"], ev[ep[-1]]["t_end"]
    f0 = next(f["frame"] for f in tele if f["t"] >= t0 - 1e-6)
    # the frame AT t1 already carries the post-roll's first stick decision (the pilot advances and decides in the same
    # frame): the episode ends with the frame before it
    f1 = max(f["frame"] for f in tele if f["t"] < t1 - 1e-6) + 1
    return f0, f1, t0, t1


if __name__ == "__main__":
    f0, f1, _, _ = episode_window(sys.argv[1])
    print(f0, f1)
