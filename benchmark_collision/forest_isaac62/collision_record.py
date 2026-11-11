"""Flight collision.json (run_flight.py, times from the start of recording) -> episode collision record (times and physics steps
from the episode start), the format the evaluator reads: reference/collision.json of a published episode, and the collision
file every evaluated agent flight must come with."""

KEEP = ("prohibited", "allowed_landing", "proximity")


def _shift(r, t0, step0):
    if r is None:
        return None
    return dict(r, t=round(r["t"] - t0, 6), episode_step=r["step"] - step0, flight_t=r["t"], flight_step=r["step"])


def episode_record(col, t0, t1, step0):
    """t0, t1: episode start / end (flight time, s); step0: physics step of the episode-start frame."""
    def inside(r):
        return r is not None and r["step"] > step0 and r["t"] <= t1 + 1e-6

    fp = col.get("first_prohibited")
    inv = col.get("first_invalid")
    term = col.get("termination")
    termination = None
    if term and term["t"] <= t1 + 1e-6:
        termination = {k: v for k, v in term.items() if k != "record"}
        termination.update(t=round(term["t"] - t0, 6), episode_step=term["step"] - step0, flight_t=term["t"])
    return {
        "version": col["version"],
        "rule": "first prohibited UAV-environment contact during the episode = failure; landing-gear contact with the designated "
                "landing surface inside its footprint during the land beat is allowed; internal UAV pairs are ignored",
        "monitor_verified": col["monitor_verified"], "monitor_verified_by": col["monitor_verified_by"],
        "api": col["api"], "physics": col["physics"], "touch_separation_m": col["touch_separation_m"],
        "drone_colliders": col["drone_colliders"], "uav_coverage": col["uav_coverage"],
        "environment_coverage": {k: v for k, v in col["environment_coverage"].items() if k != "audit"},
        "landing_exception": col.get("landing_exception"),
        "episode": {"flight_t0_s": t0, "flight_t1_s": t1, "flight_step0": step0},
        "first_prohibited": _shift(fp, t0, step0) if inside(fp) else None,
        "first_allowed_landing": _shift(col.get("first_allowed_landing"), t0, step0) if inside(col.get("first_allowed_landing")) else None,
        "invalid": _shift(inv, t0, step0) if inv is not None and inv["t"] <= t1 + 1e-6 else None,
        "contacts": [_shift(r, t0, step0) for r in col["events"] if inside(r) and r["verdict"] in KEEP],
        "termination": termination,
    }
