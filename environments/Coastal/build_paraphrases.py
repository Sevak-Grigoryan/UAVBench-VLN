"""Merge the LLM-written paraphrases (paraphrases_src_a/b.py) with each mission's canonical instruction into
paraphrases_easy.json, and check them automatically:
  * every number in a paraphrase must be within 10 % of a number in the canonical instruction;
  * circle missions must keep the direction (clockwise vs counter-clockwise / anticlockwise);
  * follow-shore missions must keep the heading (east / west) and the side (left / right);
  * every paraphrase must mention the task (circle/orbit/loop, hover/inspect, over, coast, helideck).
Writes only entries that pass; prints the failures."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
import coastal_v2 as cv  # noqa: E402
import missions as cm  # noqa: E402
import pilot  # noqa: E402
import paraphrases_src_a  # noqa: E402
import paraphrases_src_b  # noqa: E402
import paraphrases_src_p1  # noqa: E402

SRC = {**paraphrases_src_a.P, **paraphrases_src_b.P}
# paraphrase #1 replaced by the freer rewrite
SRC = {k: (paraphrases_src_p1.P1.get(k, v[0]),) + tuple(v[1:]) for k, v in SRC.items()}
# v2.1 (2026-10-14): every mission except the frozen, already published ones changed (varied take-off / landing places)
import paraphrases_v21_a  # noqa: E402
import paraphrases_v21_b  # noqa: E402
SRC = {k: v for k, v in SRC.items() if int(k) in cv.FROZEN}
SRC.update({**paraphrases_v21_a.P, **paraphrases_v21_b.P})
TASK_WORDS = {"circle": ("circle", "orbit", "loop", "lap"), "inspect": ("hover", "inspect", "watch", "observe", "look", "facing"),
              "pass_over": ("over", "above"), "follow_shore": ("coast", "shore"), "land_on_ship": ("helideck", "helicopter deck")}


def nums(t):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]


def check(m, texts):
    can = m["instruction"]
    cn = nums(can)
    errs = []
    for i, t in enumerate(texts):
        lo = t.lower()
        short = i == len(texts) - 1
        for v in nums(t):
            if not any(abs(v - c) <= 0.1 * max(c, 1) for c in cn):
                errs.append(f"number {v} not in canonical: {t[:70]}")
        if m["task_type"] == "circle":
            ccw = "counter-clockwise" in can
            if ccw and not ("counter-clockwise" in lo or "anticlockwise" in lo):
                errs.append(f"direction lost (ccw): {t[:70]}")
            if not ccw and ("counter-clockwise" in lo or "anticlockwise" in lo):
                errs.append(f"direction wrong (cw): {t[:70]}")
        if m["task_type"] == "follow_shore":
            way = "east" if "coastline east" in can else "west"
            if way not in lo:
                errs.append(f"heading lost: {t[:70]}")
            side = "left" if "on your left" in can else "right"
            if side not in lo and not short:
                errs.append(f"side lost: {t[:70]}")
        if not any(w in lo for w in TASK_WORDS[m["task_type"]]):
            errs.append(f"task not mentioned: {t[:70]}")
    return errs


def main():
    out, bad = {}, 0
    for k, v in sorted(SRC.items()):
        mid = f"{cm.PREFIX}{k}"
        _, m, _ = cm.mission_and_plan(int(k), cv, pilot)
        e = check(m, list(v))
        if e:
            bad += 1
            print(mid, *e, sep="\n   ")
            continue
        out[mid] = {"canonical": m["instruction"], "paraphrases": list(v[:3]), "short": v[3], "author": "LLM (Claude), " + ("2026-10-13 (paraphrase 1 rewritten freer)" if int(k) in cv.FROZEN else "2026-10-14 (v2.1 missions)")}
    (HERE / "paraphrases_easy.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"written {len(out)} missions, {bad} failed checks")


if __name__ == "__main__":
    main()
