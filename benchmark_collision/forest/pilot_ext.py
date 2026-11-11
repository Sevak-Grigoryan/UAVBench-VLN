"""The dataset stick pilot (dataset_codebase/common/pipeline/pilot.py, unchanged) plus one beat type used by the collision tests:

  sticks   {sticks: (x1, y1, x2, y2), s, gimbal}    hold these sticks for s seconds (decided every 0.5 s like every other beat)

Centred sticks = PX4 Position mode holds position and height, so sticks (0, 0, 0, 0) is a hover that needs no look-at point."""
import pilot as base

DECIDE_EVERY_S = base.DECIDE_EVERY_S


class Pilot(base.Pilot):
    def _decide(self, t, s):
        b = self.beat
        if b["type"] == "sticks":
            bs = self.beat_state
            bs.setdefault("t0", t)
            if t - bs["t0"] >= b["s"] - 1e-9:
                self._advance(t, s)
                return self._decide(t, s) if not self.done else ((0.0, 0.0, 0.0, 0.0), ("pitch", base.GIMBAL_TRANSIT_DEG))
            return tuple(b["sticks"]), ("pitch", b.get("gimbal", base.GIMBAL_TRANSIT_DEG))
        return super()._decide(t, s)

    def _budget(self, b, s):
        if b["type"] == "sticks":
            return b["s"] + 5.0
        return super()._budget(b, s)
