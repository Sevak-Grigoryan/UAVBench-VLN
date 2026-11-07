"""
Build Mountain/paraphrases_easy.json from the LLM-written paraphrase sources (paraphrases_src_*.py, each a dict
PARA = {mission_id: {"paraphrases": [p1, p2, p3], "short": s}}) and CHECK them (Coastal review 2026-10-20):

  C6  every paraphrase differs from the original: word-sequence similarity (difflib) < 0.8 AND word-set Jaccard < 0.8;
      paraphrases also differ from each other (< 0.85)
  C7  every number of the original (distances, heights, seconds) is in every paraphrase and in the short version
  C5  no instruction text (original, paraphrase or short) appears in two simulations, and no two simulations have
      near-identical originals (similarity >= 0.9)

    python3 build_paraphrases.py            # needs _offline/<idx>.json (tools/offline_all.py) for the canonical texts
"""
import difflib
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def sim(a, b):
    wa, wb = words(a), words(b)
    seq = difflib.SequenceMatcher(None, wa, wb).ratio()
    jac = len(set(wa) & set(wb)) / max(1, len(set(wa) | set(wb)))
    return seq, jac


def numbers(t):
    return sorted(set(re.findall(r"\b\d+\b", t)))


def main():
    canon = {}
    for f in sorted(glob.glob(str(HERE / "_offline" / "*.json"))):
        r = json.loads(Path(f).read_text())
        if r.get("ok"):
            canon[f"easy_mountain_{r['idx']:03d}"] = r["instruction"]
    para = {}
    for src in sorted(HERE.glob("paraphrases_src_*.py")):
        spec = importlib.util.spec_from_file_location(src.stem, src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        para.update(mod.PARA)
    bad, out = [], {}
    for mid, c in sorted(canon.items()):
        d = para.get(mid)
        if not d:
            bad.append((mid, "missing"))
            continue
        if d.get("canonical") != c:  # written for another (older) version of this mission's instruction
            bad.append((mid, "paraphrases were written for a different instruction"))
            continue
        ps, sh = d["paraphrases"], d["short"]
        for k, p in enumerate(ps):
            seq, jac = sim(c, p)
            if seq >= 0.8 or jac >= 0.8:
                bad.append((mid, f"paraphrase {k + 1} too close to the original: seq {seq:.2f} jaccard {jac:.2f}"))
            for j, q in enumerate(ps[k + 1:], k + 1):
                s2 = sim(p, q)
                if max(s2) >= 0.85:
                    bad.append((mid, f"paraphrases {k + 1} and {j + 1} too close: {s2[0]:.2f}/{s2[1]:.2f}"))
            miss = [n for n in numbers(c) if n not in numbers(p)]
            if miss:
                bad.append((mid, f"paraphrase {k + 1} lacks numbers {miss}"))
        miss = [n for n in numbers(c) if n not in numbers(sh)]
        if miss:
            bad.append((mid, f"short lacks numbers {miss}"))
        out[mid] = {"canonical": c, "paraphrases": ps, "short": sh}
    # C5 across simulations
    seen = {}
    for mid, d in out.items():
        for t in [d["canonical"], *d["paraphrases"], d["short"]]:
            key = " ".join(words(t))
            if key in seen and seen[key] != mid:
                bad.append((mid, f"text also used by {seen[key]}: {t[:80]}"))
            seen[key] = mid
    ids = sorted(out)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            s = difflib.SequenceMatcher(None, words(out[a]["canonical"]), words(out[b]["canonical"])).ratio()
            if s >= 0.9:
                bad.append((a, f"original near-identical to {b} ({s:.2f})"))
    for b in bad:
        print("PROBLEM", *b)
    badm = {b[0] for b in bad}
    out = {k: v for k, v in out.items() if k not in badm}  # only checked entries are published
    (HERE / "paraphrases_easy.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} missions written, {len(bad)} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
