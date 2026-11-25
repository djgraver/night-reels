"""Build the review page (site/index.html) from out/*/manifest.json. Prints the Artifact files map as JSON."""
import html, json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
SITE = os.path.join(ROOT, "site")
sys.path.insert(0, ROOT)
from examples import DREAMS, TONE_DREAMS  # noqa: E402

TITLES = {
    "hallway": "The hallway that kept going", "beach": "Glass sea, huge moon", "mountain": "Every detail, the valley of lakes",
    "station": "The man under the clock", "flooded_city": "Streetlights under the water", "forest": "The red door in the woods",
    "stairs": "The staircase into the clouds",
    "d_1pyxr9f": "Grandmother's hallway, then the sea", "d_1poy5kg": "Followed through the rain, then falling",
    "d_1qiwynd": "Dad waving across the grass",
    "hallway_dark": "The hallway filling with smoke", "forest_shift": "When the forest went dark",
    "beach_shift": "When the fog lifted",
}
WHY = {
    "hallway_dark": "Same hallway frames as above. \u201cNightmare\u201d, \u201cblack smoke\u201d and \u201clights going out\u201d set the fog tone near black: the fog swallows the walls, darkness closes in, and only real light sources still glow.",
    "forest_shift": "The first beat (\u201cpeaceful and soft and warm\u201d) keeps the fog its own colour. The turn (\u201cthen suddenly it went dark, the fog turned black\u201d) swells the fog to black halfway through the clip.",
    "beach_shift": "The reverse: \u201ceverything was black, I couldn't see anything\u201d opens in black fog, then \u201cthe fog lifted and the moon came out\u201d thins and pales it.",
    "hallway": "“couldn't see” and “couldn't see who” pull clarity down, so fog is heavy; the stranger stays a silhouette and the light under the end door glows through.",
    "beach": "No clarity words, so this is the default dream fog. The moon is treated as a light, so it punches through the haze.",
    "mountain": "“Crystal clear”, “every detail” and “sharp” push clarity to the top: almost no fog, 12 fps, little grain. This is the control for what “clear” looks like.",
    "station": "“Black and white, like an old film” switches on mono with a film black point, flicker, dust and scratches. Fog stays at the default.",
    "flooded_city": "“Everything was purple” becomes a tint carried into the fog colour and the mid-tones, so the lamps glow violet.",
    "forest": "“Really hazy” and “couldn't see very far” give the heaviest fog; the red door keeps its colour because it sits in the attention island.",
    "stairs": "“So vivid and saturated, like a painting” is read as a colour cue, not a clarity cue, so it saturates without clearing the fog.",
}


def load(name):
    p = os.path.join(OUT, name, "manifest.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def chips(spec, look):
    c = []
    c.append(("fog", f"{look['fog']:.2f}"))
    tone = look.get("fog_tone", 0.5)
    c.append(("tone", "black" if tone > 0.8 else "dark" if tone > 0.62 else "pale" if tone < 0.25 else "light" if tone < 0.4 else "scene colour"))
    tl = look.get("timeline") or []
    if tl:
        def nm(t):
            return "black" if t > 0.8 else "dark" if t > 0.62 else "pale" if t < 0.4 else "own colour"
        c.append(("shifts", " \u2192 ".join(dict.fromkeys(nm(k["tone"]) for k in tl))))
    c.append(("fps", f"{look['fps']:.0f}"))
    if look.get("style", "natural") != "natural":
        c.append(("style", look["style"]))
    if spec.get("tint_name"):
        c.append(("tint", spec["tint_name"]))
    c.append(("scene", spec["setting"].replace("_", " ")))
    c.append(("light", spec["time_of_day"]))
    return c


def cue_list(spec):
    out = []
    for e in spec.get("clarity_evidence", []):
        out.append(("clear" if e.startswith("+") else "hazy", e[1:]))
    for e in spec.get("tone_evidence", []):
        k, _, v = e.partition(":")
        out.append(("dark" if k == "dark" else "light", v))
    for e in spec.get("style_evidence", []):
        k, _, v = e.partition(":")
        out.append(("style", v))
    return out


def section(name, files, app=False):
    m = load(name)
    if not m:
        return ""
    if app:
        first = m["scenes"][0]
        spec, look = first["spec"], first["look"]
        text = json.load(open(os.path.join(ROOT, "app_examples", f"{name}.json")))["input"]["transcript"]
        scenes = " then ".join(s["spec"]["setting"].replace("_", " ") for s in m["scenes"])
        why = f"From the app's dream spec v1: {len(m['scenes'])} scene{'s' if len(m['scenes']) > 1 else ''} ({scenes}), rendered one shot each and joined with fog dissolves."
    else:
        spec, look = m["spec"], m["look"]
        text = DREAMS[name] if name in DREAMS else TONE_DREAMS[name][1]
        why = WHY.get(name, "")
    base = f"media/{name}"
    for f in ("clip.mp4", "poster.jpg", "raw.jpg", "dream.jpg", "fog_dial.jpg"):
        files[f"{base}/{f}"] = os.path.join(OUT, name, f)
    chip_html = "".join(f'<span class="chip"><i>{html.escape(k)}</i>{html.escape(v)}</span>' for k, v in chips(spec, look))
    cues = "".join(f'<mark class="cue {k}">{html.escape(v)}</mark>' for k, v in cue_list(spec)) or '<span class="none">no clarity or style words, defaults apply</span>'
    return f"""
<article class="dream" id="{name}">
  <header class="dream-head">
    <h2>{html.escape(TITLES.get(name, name))}</h2>
    <blockquote>{html.escape(text)}</blockquote>
    <div class="cues"><span class="lbl">Heard</span>{cues}</div>
    <div class="chips">{chip_html}</div>
    <p class="why">{html.escape(why)}</p>
  </header>
  <div class="media">
    <figure class="clip">
      <video src="{base}/clip.mp4" poster="{base}/poster.jpg" muted loop playsinline autoplay preload="metadata"></video>
      <figcaption>Clip, {look['fps']:.0f} fps with held frames</figcaption>
    </figure>
    <figure class="compare" style="--pos:50%">
      <img src="{base}/raw.jpg" alt="Render before the dream pass" loading="lazy">
      <img class="top" src="{base}/dream.jpg" alt="The same frame after the dream pass" loading="lazy">
      <input type="range" min="0" max="100" value="50" aria-label="Compare render and dream pass" id="cmp-{name}">
      <figcaption><span>Render</span><span>Dream pass</span></figcaption>
    </figure>
  </div>
  <figure class="dial">
    <img src="{base}/fog_dial.jpg" alt="The same frame at four fog levels" loading="lazy">
    <figcaption class="dial-scale"><span>fog 0.10</span><span>0.35</span><span>0.60</span><span>0.85</span></figcaption>
  </figure>
</article>"""


def main():
    files = {}
    ex = "".join(section(n, files) for n in ["mountain", "station", "forest", "flooded_city", "hallway", "beach", "stairs"])
    tone = "".join(section(n, files) for n in ["hallway_dark", "forest_shift", "beach_shift"] if load(n))
    app = "".join(section(n, files, app=True) for n in ["d_1pyxr9f", "d_1poy5kg", "d_1qiwynd"])
    tpl = open(os.path.join(ROOT, "page_template.html")).read()
    page = tpl.replace("<!--TONE-->", tone).replace("<!--EXAMPLES-->", ex).replace("<!--APP-->", app or '<p class="pending">These are still rendering.</p>')
    os.makedirs(SITE, exist_ok=True)
    open(os.path.join(SITE, "index.html"), "w").write(page)
    print(json.dumps(files))


if __name__ == "__main__":
    main()
