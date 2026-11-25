"""Dream description -> DreamSpec.

Rule-based cue extraction so the pipeline runs offline. A production version
would ask an LLM (Claude) for the same JSON schema with the lexicon below as
guidance, and keep this parser as the deterministic fallback.
"""
from __future__ import annotations
import json, re
import numpy as np
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- lexicons
SETTINGS = {
    "hallway":      ["hallway", "corridor", "hall", "doors", "house", "grandmother", "grandma", "apartment", "room", "school"],
    "beach":        ["beach", "sea", "ocean", "shore", "waves", "sand", "coast"],
    "mountain":     ["mountain", "ridge", "valley", "peak", "cliff", "summit", "hills", "lakes"],
    "station":      ["station", "platform", "train", "railway", "tracks", "subway"],
    "flooded_city": ["flooded", "underwater", "street", "city", "streetlights", "buildings", "town"],
    "forest":       ["forest", "woods", "trees", "woodland", "pines", "jungle"],
    "stairs":       ["staircase", "stairs", "steps", "clouds", "sky", "floating", "flying"],
}
TIMES = {
    "night": ["night", "midnight", "dark", "moon", "moonlight", "stars"],
    "dusk":  ["dusk", "sunset", "evening", "twilight", "golden hour"],
    "dawn":  ["dawn", "sunrise", "early morning"],
    "day":   ["morning", "noon", "midday", "afternoon", "sunny", "sunlight", "daylight", "bright sun"],
}
DEFAULT_TIME = {"hallway": "night", "beach": "night", "mountain": "day", "station": "dusk",
                "flooded_city": "dusk", "forest": "dusk", "stairs": "day"}

CLEAR_CUES = ["crystal clear", "clear", "clearly", "sharp", "in focus", "every detail", "high definition",
              "hd", "lucid", "vivid", "real life", "like real", "realistic", "detailed", "precise",
              "could see everything", "remember exactly"]
FOG_CUES = ["hazy", "haze", "foggy", "fog", "misty", "mist", "blurry", "blurred", "fuzzy", "vague",
            "murky", "dim", "smoky", "couldn't see", "could not see", "can't remember", "cannot remember",
            "don't remember", "half remember", "faint", "dreamy", "washed out", "unclear", "obscured"]
INTENSIFIERS = ["very", "really", "completely", "so", "super", "extremely", "totally", "perfectly",
                "incredibly", "crystal", "absolutely", "unbelievably"]
NEGATIONS = ["not", "wasn't", "weren't", "isn't", "never", "no", "nothing", "hardly", "barely"]

STYLE_CUES = {
    "bw":        ["black and white", "black & white", "b&w", "b/w", "monochrome", "greyscale", "grayscale",
                  "no colour", "no color", "colourless", "colorless"],
    "sepia":     ["sepia", "old photograph", "old photo", "faded photo", "brownish"],
    "vivid":     ["vivid", "saturated", "vibrant", "colourful", "colorful", "neon", "technicolor", "like a painting"],
    "film":      ["old film", "film", "movie", "cinema", "8mm", "super 8", "grainy", "film grain", "silent film"],
    "vhs":       ["vhs", "videotape", "camcorder", "home video", "90s video"],
}
COLOURS = {
    "red": (0.85, 0.12, 0.10), "crimson": (0.7, 0.05, 0.1), "orange": (0.95, 0.5, 0.15),
    "yellow": (0.95, 0.85, 0.3), "gold": (0.95, 0.75, 0.3), "green": (0.2, 0.7, 0.35),
    "blue": (0.2, 0.4, 0.95), "teal": (0.15, 0.65, 0.65), "purple": (0.55, 0.25, 0.85),
    "violet": (0.6, 0.3, 0.9), "pink": (0.95, 0.45, 0.7), "white": (0.95, 0.95, 0.95),
}
OBJECTS = ["door", "moon", "figure", "person", "man", "woman", "someone", "clock", "stairs", "staircase",
           "light", "lamp", "streetlights", "tree", "boat", "train", "window", "mother", "father", "stranger"]
FIGURE_WORDS = {"figure", "person", "man", "woman", "someone", "mother", "father", "stranger", "people", "child"}
MOTIONS = {"falling": ["falling", "fell"], "flying": ["flying", "flew", "floating", "floated", "hovering"],
           "running": ["running", "ran", "chased", "chasing"], "walking": ["walking", "walked", "wandering"],
           "swimming": ["swimming", "swam"], "still": ["standing", "stood", "sitting", "frozen"]}
MOODS = {"eerie": ["eerie", "strange", "uncanny", "weird", "empty", "alone", "lonely", "silent"],
         "scary": ["scary", "afraid", "terrified", "fear", "nightmare", "chased", "dread"],
         "calm": ["calm", "peaceful", "quiet", "still", "gentle", "safe", "warm"],
         "joyful": ["happy", "joy", "laughing", "beautiful", "amazing", "wonderful"]}


# fog tone: 0 = pale milky fog, 0.5 = fog takes the scene's own colour, 1 = black smoke / ink
DARK_CUES = ["black", "couldn't see anything", "could not see anything", "nightmare", "pitch black", "pitch dark", "black fog", "black smoke", "smoke", "darkness", "dark",
             "shadow", "shadows", "terrified", "afraid", "scared", "fear", "dread", "panic", "trapped", "chased",
             "chasing", "couldn't breathe", "suffocating", "drowning", "monster", "evil", "dead", "death", "dying",
             "blood", "screaming", "scream", "went black", "turned black", "ominous", "menacing", "cold", "lost",
             "following me", "something was behind", "someone was behind", "storm"]
LIGHT_CUES = ["bright", "white", "glowing", "sunny", "sunlight", "sunshine", "warm", "peaceful", "calm", "happy",
              "safe", "soft", "gentle", "heaven", "heavenly", "golden", "light", "lifted", "cleared", "clear",
              "beautiful", "joy", "laughing", "waving", "love"]
SHIFT_WORDS = ["then", "suddenly", "until", "but then", "after that", "and then", "all of a sudden", "later",
               "at first", "slowly", "gradually"]
TONE_VERBS = {  # explicit changes of the fog itself
    +1: ["went dark", "went black", "turned black", "turned dark", "got darker", "got dark", "grew dark",
         "darkness fell", "fog turned black", "smoke filled", "lights went out", "everything went black"],
    -1: ["fog lifted", "the fog lifted", "mist lifted", "it cleared", "cleared up", "got brighter", "grew brighter",
         "light came", "the sun came out", "moon came out", "everything became clear", "became clear", "dawn broke"],
}


@dataclass
class DreamSpec:
    text: str
    setting: str = "forest"
    time_of_day: str = "dusk"
    clarity: float = 0.35          # 0 = barely remembered, 1 = crystal clear. Typical dream ~0.35
    clarity_evidence: list = field(default_factory=list)
    palette: str = "color"         # color | bw | sepia
    saturation: float = 0.75       # multiplier after stylisation
    tint: tuple | None = None      # global colour cast named by the dreamer ("everything was purple")
    tint_name: str | None = None
    film: bool = False
    vhs: bool = False
    objects: list = field(default_factory=list)
    object_colours: dict = field(default_factory=dict)
    figures: int = 0
    motion: str = "drift"
    mood: str = "eerie"
    style_evidence: list = field(default_factory=list)
    fog_tone: float = 0.5          # 0 pale, 0.5 scene colour, 1 black
    tone_evidence: list = field(default_factory=list)
    timeline: list = field(default_factory=list)   # [{"at": 0..1, "fog": f, "tone": t, "because": str}]
    seed: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def prompt(self) -> str:
        """Prompt a production image/video model would receive (the fog is applied after)."""
        bits = [f"photorealistic {self.setting.replace('_', ' ')} at {self.time_of_day}"]
        bits += [f"{self.object_colours.get(o, '')} {o}".strip() for o in self.objects]
        if self.figures:
            bits.append("a distant featureless human figure")
        bits.append(f"{self.mood} atmosphere, 35mm, natural light, slight motion")
        if self.palette == "bw":
            bits.append("black and white film photograph")
        return ", ".join(bits)


def _norm(text: str) -> str:
    t = text.lower().replace("’", "'")
    return re.sub(r"\s+", " ", t)


def _find(t: str, phrase: str):
    return [m.start() for m in re.finditer(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", t)]


def _window_before(t: str, pos: int, n_words: int = 4) -> list[str]:
    return re.findall(r"[a-z'&]+", t[:pos])[-n_words:]


def _score(t: str, cues, sign: float, evidence: list) -> float:
    total, taken = 0.0, []
    for cue in sorted(cues, key=len, reverse=True):      # longest first so "crystal clear" wins over "clear"
        for p in _find(t, cue):
            if any(a <= p < b for a, b in taken):
                continue
            if cue == "vivid" and "colo" in t[max(0, p - 30):p + 30]:
                continue                                    # "vivid colours" is a palette cue, not clarity
            taken.append((p, p + len(cue)))
            before = _window_before(t, p)
            w = 1.0 + 0.5 * sum(b in INTENSIFIERS for b in before)
            if cue.startswith(("crystal", "every", "could see")):
                w += 0.5
            s = sign
            if any(b in NEGATIONS or b.endswith("n't") for b in before):
                s = -sign                                   # "wasn't clear" -> fog
            total += s * w
            evidence.append(("+" if s > 0 else "-") + cue)
    return total


def parse(text: str, seed: int | None = None) -> DreamSpec:
    t = _norm(text)
    spec = DreamSpec(text=text.strip())
    spec.seed = seed if seed is not None else abs(hash(t)) % 10_000

    # setting: score every lexicon, first-mention breaks ties
    best, best_score = "forest", 0.0
    for name, words in SETTINGS.items():
        hits = [p for w in words for p in _find(t, w)]
        score = len(hits) + (0.01 * (len(t) - min(hits)) / max(len(t), 1) if hits else 0)
        if score > best_score:
            best, best_score = name, score
    spec.setting = best

    spec.time_of_day = DEFAULT_TIME[best]
    for tod, words in TIMES.items():
        if any(_find(t, w) for w in words):
            spec.time_of_day = tod
            break

    # clarity: start at "typical dream" and move with evidence
    ev: list = []
    raw = _score(t, CLEAR_CUES, +1, ev) + _score(t, FOG_CUES, -1, ev)
    spec.clarity_evidence = ev
    spec.clarity = float(min(1.0, max(0.0, 0.35 + 0.22 * raw)))

    sev: list = []
    for style, cues in STYLE_CUES.items():
        for c in cues:
            if _find(t, c):
                sev.append(f"{style}:{c}")
                if style == "bw":
                    spec.palette = "bw"
                elif style == "sepia":
                    spec.palette = "sepia"
                elif style == "vivid":
                    spec.saturation = 1.25
                elif style == "film":
                    spec.film = True
                elif style == "vhs":
                    spec.vhs = True
                break
    m = re.search(r"(?:everything|the whole \w+|all|the sky|the light|the air) (?:was|were|looked|glowed|turned) (?:all |kind of |so |a |deep |dark |bright )*([a-z]+)", t)
    if m and m.group(1) in COLOURS and spec.palette == "color":
        spec.tint, spec.tint_name = COLOURS[m.group(1)], m.group(1)
        sev.append(f"tint:{m.group(1)}")
    spec.style_evidence = sev

    for o in OBJECTS:
        for p in _find(t, o):
            key = "figure" if o in FIGURE_WORDS else ("stairs" if o == "staircase" else o)
            if key not in spec.objects:
                spec.objects.append(key)
            prev = _window_before(t, p, 2)
            for w in prev:
                if w in COLOURS:
                    spec.object_colours[key] = w
    spec.figures = sum(1 for w in FIGURE_WORDS if _find(t, w))
    spec.figures = min(spec.figures, 3)

    for mot, words in MOTIONS.items():
        if any(_find(t, w) for w in words):
            spec.motion = mot
            break
    mood_scores = {m: sum(len(_find(t, w)) for w in ws) for m, ws in MOODS.items()}
    if max(mood_scores.values()) > 0:
        spec.mood = max(mood_scores, key=mood_scores.get)
    spec.fog_tone, spec.tone_evidence = tone_of(t, spec.mood)
    spec.timeline = fog_timeline(t, spec.clarity, spec.fog_tone)
    return spec


def tone_of(t: str, mood: str = "eerie"):
    ev = []
    dark = light = 0.0
    taken = []
    for cue in sorted(DARK_CUES, key=len, reverse=True):
        for p in _find(t, cue):
            if any(a <= p < b for a, b in taken):
                continue
            if cue == "black" and t[p:p + 16].startswith(("black and white", "black & white")):
                continue
            taken.append((p, p + len(cue)))
            neg = any(b in NEGATIONS or b.endswith("n't") for b in _window_before(t, p, 3))
            if neg and cue not in ("couldn't breathe",):
                continue
            dark += 1.0; ev.append("dark:" + cue)
    for cue in sorted(LIGHT_CUES, key=len, reverse=True):
        for p in _find(t, cue):
            if any(a <= p < b for a, b in taken):
                continue
            if cue == "white" and t[max(0, p - 10):p].endswith(("black and ", "black & ")):
                continue
            taken.append((p, p + len(cue)))
            light += 0.7; ev.append("light:" + cue)
    for sign, verbs in TONE_VERBS.items():
        for v in verbs:
            if _find(t, v):
                ev.append(("dark:" if sign > 0 else "light:") + v)
    tone = 0.5 + 0.14 * dark - 0.1 * light + {"scary": 0.15, "eerie": 0.05, "calm": -0.05, "joyful": -0.1}.get(mood, 0)
    return float(np.clip(tone, 0.05, 0.97)), ev


def _segments(t: str):
    """Split a dream into beats at sentence ends and at 'then' / 'suddenly' style turns."""
    parts = re.split(r"(?<=[.!?;])\s+|,?\s+(?=(?:and then|but then|then|suddenly|all of a sudden|until|after that)\b)", t)
    segs, carry = [], ""
    for p in parts:
        p = (carry + " " + p).strip() if carry else (p or "").strip()
        if len(p.split()) < 4:
            carry = p
            continue
        carry = ""
        segs.append(p)
    if carry:
        if segs:
            segs[-1] += " " + carry
        else:
            segs.append(carry)
    return segs


def fog_timeline(t: str, clarity: float, tone: float):
    """Per-beat fog density and tone, so the fog can change during the dream.
    Each beat is scored on its own; explicit change verbs ('went black', 'fog lifted') carry forward."""
    from .stylize import clarity_to_fog
    segs = _segments(t)
    if len(segs) < 2:
        return []
    base_fog = clarity_to_fog(clarity)
    n = len(segs)
    keys, cur_tone, cur_fog = [], None, base_fog
    for i, seg in enumerate(segs):
        st, sev = tone_of(seg)
        ev_c: list = []
        raw = _score(seg, CLEAR_CUES, +1, ev_c) + _score(seg, FOG_CUES, -1, ev_c)
        explicit = 0
        for sign, verbs in TONE_VERBS.items():
            if any(_find(seg, v) for v in verbs):
                explicit = sign
        if cur_tone is None:
            cur_tone = st if sev else tone
        else:
            cur_tone = 0.5 * cur_tone + 0.5 * st if sev else cur_tone
        if abs(raw) >= 1:
            seg_fog = clarity_to_fog(float(np.clip(0.35 + 0.22 * raw, 0, 1)))
            if i == 0 or abs(seg_fog - cur_fog) > 0.15:
                cur_fog = seg_fog if i else 0.5 * (seg_fog + base_fog)
        if explicit > 0:
            cur_tone = max(cur_tone, 0.9); cur_fog = max(cur_fog, 0.7)
        elif explicit < 0:
            cur_tone = min(cur_tone, 0.3); cur_fog = min(cur_fog, 0.3)
        keys.append(dict(at=round((i + 0.5) / n, 3), fog=round(float(cur_fog), 3), tone=round(float(cur_tone), 3),
                         because=seg[:80]))
    # flat timelines carry no information
    if max(k["tone"] for k in keys) - min(k["tone"] for k in keys) < 0.12 and \
       max(k["fog"] for k in keys) - min(k["fog"] for k in keys) < 0.12:
        return []
    return keys


if __name__ == "__main__":
    import sys
    s = parse(" ".join(sys.argv[1:]) or "I was in a hazy forest and there was a red door")
    print(s.to_json()); print(s.prompt())
