"""Mitigation demos on the existing beach frames (no Blender needed). Test text only; nothing explicit is rendered.
    python3 rating_demo.py      -> out/rating_demo/<case>/{clip.mp4,poster.jpg,content_rating.json}
"""
import json, os, shutil, sys
from dreamrender.rating import rate, scene_mitigations
from dreamrender.mitigate import veil_look
from dreamrender.pipeline import run, post_scene

BASE = "out/beach"
CASES = {
    "censor_bar": "I was naked on the beach at night and everyone could see me.",
    "reaction_cut": "[TEST] Someone on the beach was stabbed.",
    "fog_veil": "[TEST] The dream was about suicide.",
    "dark_fog_pg": "[TEST] I was a kid again and demons were hunting me across the beach.",
    "abort": "[TEST] A child and something sexual.",
}
look0 = json.load(open(os.path.join(BASE, "manifest.json")))["look"]
for name, text in CASES.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    keep = os.path.join("out/rating_demo", name)
    out = os.path.join(os.environ.get("DEMO_WORK", "/tmp/rating_demo"), name)   # frames are big: work outside the project folder
    os.makedirs(out, exist_ok=True); os.makedirs(keep, exist_ok=True)
    cr = rate({"input": {"transcript": text}, "scenes": [{"text": text}]}, use_llm=False)
    json.dump(dict(text=text, **cr), open(os.path.join(keep, "content_rating.json"), "w"), indent=2)
    print(name, cr["assessed"], cr["target"], cr["outcome"], [m["kind"] for m in cr["mitigations"]])
    if cr["outcome"] == "abort":
        continue
    if not os.path.exists(os.path.join(out, "frames")):
        shutil.copytree(os.path.join(BASE, "frames"), os.path.join(out, "frames"))
    mits = scene_mitigations(cr, 0)
    look = veil_look(dict(look0), mits)
    run({"seed": 7}, look, out, skip_render=True, duration=6.0)
    print("  applied", post_scene(out, mits, look, seed=7))
    for f in ("clip.mp4", "poster.jpg", "dream.jpg"):
        shutil.copy(os.path.join(out, f), os.path.join(keep, f))
