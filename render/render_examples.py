"""Render every example dream end to end. Usage: python3 render_examples.py [names...]"""
import json, os, sys, time
from dataclasses import asdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from examples import DREAMS
from dreamrender.parse import parse
from dreamrender.stylize import look_from_spec
from dreamrender.pipeline import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FRAMES = {"mountain": 22, "stairs": 20}
names = sys.argv[1:] or list(DREAMS)
skip = os.environ.get("SKIP_RENDER") == "1"
for name in names:
    spec = parse(DREAMS[name], seed=sum(map(ord, name)))
    look = look_from_spec(asdict(spec))
    t = time.time()
    m = run(spec, look, os.path.join(OUT, name), frames=FRAMES.get(name, 16), res=(960, 540), samples=32,
            duration=6.0, skip_render=skip)
    print(f"{name}: {time.time() - t:.0f}s {m['clip']}", flush=True)
