"""exit 0 iff paraphrases_easy.json has entries for this mission whose canonical text matches the current mission."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
mid = sys.argv[1]
p = HERE / "paraphrases_easy.json"
if not p.exists():
    sys.exit(1)
d = json.loads(p.read_text()).get(mid)
if not d or len(d.get("paraphrases", [])) < 2 or not d.get("short"):
    sys.exit(1)
import mountain_v2 as cv, missions as cm, pilot
_, m, _ = cm.mission_and_plan(cm.index_of(mid), cv, pilot)
sys.exit(0 if d.get("canonical") == m["instruction"] else 2)
