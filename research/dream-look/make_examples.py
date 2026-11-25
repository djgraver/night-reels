import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dreamlook import Depth, DreamParams, load, render, render_video, save, _radial  # noqa

S = sys.argv[1]
SRC = f"{S}/src"
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
D = Depth(f"{SRC}/midas_small.onnx")
W = 1200

SCENES = [
    dict(id="street", src="da_1.png", focus=(0.82, 0.62),
         report="I was on a long street in London waiting for a red bus that never quite arrived. Everything is hazy now.",
         params=dict(clarity=0.25, mood="neutral")),
    dict(id="house", src="v1_11.img", focus=(0.55, 0.55),
         report="My childhood home, except it was made of glass, at dusk. Honestly it was really vivid, like real life.",
         params=dict(clarity=0.85, mood="warm")),
    dict(id="stairs", src="da_15.png", focus=(0.45, 0.6),
         report="A palace with a staircase that kept going. It was in black and white, like an old film.",
         params=dict(clarity=0.4, style="bw", indoor=True)),
    dict(id="river", src="v1_4.img", focus=(0.45, 0.45),
         report="A big building across a grey river. I felt so sad looking at it and I can't remember why.",
         params=dict(clarity=0.15, mood="sad")),
    dict(id="sunset", src="da_16.png", focus=(0.45, 0.55),
         report="I knew I was dreaming. Friends on bikes at sunset, everything glowing and sharp.",
         params=dict(clarity=0.8, lucid=True, mood="warm")),
    dict(id="night", src="v1_12.img", focus=(0.4, 0.55),
         report="A city at night with blue lights. Something was following me. Nightmare.",
         params=dict(clarity=0.35, nightmare=True, mood="dread")),
    dict(id="sunflowers", src="da_2.png", focus=(0.5, 0.45),
         report="A field of sunflowers, faded like a photo from my gran's album.",
         params=dict(clarity=0.3, style="faded", mood="warm")),
    dict(id="office", src="da_10.png", focus=(0.4, 0.55),
         report="My office but empty, and the light was wrong. Eerie.",
         params=dict(clarity=0.3, mood="eerie", indoor=True)),
]

meta = {"scenes": [], "ladder": [], "styles": [], "stages": [], "videos": []}


def cache_depth(src):
    img = load(f"{SRC}/{src}", W)
    return img, D(img)


for sc in SCENES:
    img, dist = cache_depth(sc["src"])
    p = DreamParams(focus=sc["focus"], **sc["params"]).resolve()
    save(f"{OUT}/{sc['id']}_before.jpg", img)
    save(f"{OUT}/{sc['id']}_after.jpg", render(img, p, dist))
    d = p.as_dict()
    meta["scenes"].append(dict(id=sc["id"], report=sc["report"], input=sc["params"],
                               resolved={k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}))
    print("scene", sc["id"])

# Clarity ladder on the street
img, dist = cache_depth("da_1.png")
for c in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
    p = DreamParams(clarity=c, focus=(0.82, 0.62)).resolve()
    save(f"{OUT}/ladder_{int(c*100):03d}.jpg", render(img, p, dist))
    meta["ladder"].append(dict(clarity=c, fog=round(p.fog, 2), fps=round(p.out_fps, 1),
                               sat=round(p.saturation, 2), beta=round(p.fog_beta, 2)))

# Style grid on the stairs
img, dist = cache_depth("da_15.png")
for st in ["natural", "bw", "sepia", "faded", "noir", "vivid"]:
    p = DreamParams(clarity=0.45, style=st, indoor=True, focus=(0.45, 0.6)).resolve()
    save(f"{OUT}/style_{st}.jpg", render(img, p, dist))
    meta["styles"].append(st)
for mood in ["calm", "warm", "eerie", "sad"]:
    p = DreamParams(clarity=0.45, mood=mood, indoor=True, focus=(0.45, 0.6)).resolve()
    save(f"{OUT}/mood_{mood}.jpg", render(img, p, dist))
p = DreamParams(clarity=0.45, nightmare=True, mood="dread", indoor=True, focus=(0.45, 0.6)).resolve()
save(f"{OUT}/mood_nightmare.jpg", render(img, p, dist))
p = DreamParams(clarity=0.45, lucid=True, indoor=True, focus=(0.45, 0.6)).resolve()
save(f"{OUT}/mood_lucid.jpg", render(img, p, dist))

# Stage-by-stage on the river scene: switch stages on cumulatively
img, dist = cache_depth("v1_4.img")
save(f"{OUT}/stage_0_source.jpg", img)
dv = (dist * 255).astype(np.uint8)
save(f"{OUT}/stage_1_depth.jpg", cv2.cvtColor(cv2.applyColorMap(255 - dv, cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB) / 255.0)
full = DreamParams(clarity=0.2, mood="calm", focus=(0.45, 0.45)).resolve()


def staged(n):
    q = DreamParams(clarity=0.2, mood="calm", focus=(0.45, 0.45)).resolve()
    if n < 2: q.periph_blur = 0
    if n < 3: q.fog = 0
    if n < 4: q.saturation = 1.0; q.tint = (1, 1, 1)
    if n < 5: q.bloom = 0; q.halation = 0
    if n < 6: q.contrast = 1; q.black_lift = 0; q.exposure = 0; q.vignette = 0
    if n < 7: q.grain = 0
    return q


names = ["periphery", "fog", "colour", "glow", "tone_vignette", "grain"]
for n, nm in zip(range(2, 8), names):
    save(f"{OUT}/stage_{n}_{nm}.jpg", render(img, staged(n), dist))

# Videos
VIDS = [
    dict(id="dolphins", src="vid_davis_dolphins.mp4", params=dict(clarity=0.3, mood="calm", focus=(0.5, 0.5)),
         report="Swimming next to dolphins, the water was green and slow."),
    dict(id="wheel_clear", src="v2vid_ferris_wheel.mp4", params=dict(clarity=0.85, mood="warm", focus=(0.5, 0.45)),
         report="A ferris wheel at a fair. Really clear, I remember every light."),
    dict(id="wheel_fog", src="v2vid_ferris_wheel.mp4", params=dict(clarity=0.15, mood="warm", focus=(0.5, 0.45)),
         report="A ferris wheel at a fair, I think. Very foggy."),
    dict(id="coaster", src="vid_davis_rollercoaster.mp4", params=dict(clarity=0.35, nightmare=True, mood="dread", focus=(0.5, 0.4)),
         report="A rollercoaster I couldn't get off. Nightmare."),
    dict(id="seasnake_bw", src="vid_davis_seasnake.mp4", params=dict(clarity=0.4, style="bw", focus=(0.5, 0.5)),
         report="Something long swimming under me. Black and white."),
]
for v in (VIDS if not os.environ.get("NOVID") else []):
    p = DreamParams(**v["params"]).resolve()
    os.system(f'ffmpeg -y -loglevel error -i "{SRC}/{v["src"]}" -t 6 -vf scale=960:-2 -c:v libx264 -pix_fmt yuv420p -crf 26 -an "{OUT}/vid_{v["id"]}_before.mp4"')
    render_video(f"{SRC}/{v['src']}", f"{OUT}/vid_{v['id']}_after.mp4", p, D, max_w=960, max_seconds=6)
    meta["videos"].append(dict(id=v["id"], report=v["report"], input=v["params"], fps=round(p.out_fps, 1)))
    print("video", v["id"])

if not os.environ.get("NOVID"): json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
