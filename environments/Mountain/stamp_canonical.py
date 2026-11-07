"""Record, in a paraphrase source file, the exact current instruction each entry was written for (run ONCE right after
writing a batch, from the same _offline/<idx>.json texts the paraphrases were written from)."""
import importlib.util, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = HERE / sys.argv[1]
spec = importlib.util.spec_from_file_location("m", src); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
out = {}
for mid, d in mod.PARA.items():
    r = json.loads((HERE / "_offline" / f"{int(mid.split('_')[-1]):03d}.json").read_text())
    out[mid] = {"canonical": d.get("canonical", r["instruction"]), "paraphrases": d["paraphrases"], "short": d["short"]}
head = src.read_text().split("PARA = {")[0]
src.write_text(head + "PARA = " + json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(len(out), "entries stamped")
