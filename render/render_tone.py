"""Round 2 renders: dark fog and fog that changes mid-dream, on existing scene frames."""
import json, os, sys
from dataclasses import asdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from examples import TONE_DREAMS
from dreamrender.parse import parse
from dreamrender.stylize import look_from_spec
from dreamrender.pipeline import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
for name in (sys.argv[1:] or TONE_DREAMS):
    scene, text = TONE_DREAMS[name]
    spec = parse(text, seed=sum(map(ord, name)))
    spec.setting = scene
    look = look_from_spec(asdict(spec))
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    m = run(spec, look, d, skip_render=True, duration=8.0 if spec.timeline else 6.0,
            frames_dir=os.path.join(OUT, scene, "frames"))
    print(name, round(look["fog"], 2), look["fog_tone"], spec.timeline, m["clip"], flush=True)
