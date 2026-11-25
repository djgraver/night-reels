"""Adapter: the app's "dream spec v1" (see docs/dream-spec.md) -> pipeline inputs.

The app owns the *look* (the dreamer's clarity and style words, plus any hand overrides);
the pipeline's own parser still reads the transcript for scene content (setting, objects, figures).
"""
from __future__ import annotations
from .parse import parse, COLOURS, tone_of, fog_timeline
from .stylize import fog_to_clarity, hex_to_rgb, look_from_spec

APP_SETTING = {"corridor": "hallway", "room": "hallway", "water": "beach", "underwater": "flooded_city",
               "city": "street", "forest": "forest", "field": "field", "mountains": "mountain", "open": "stairs"}
APP_STYLE = {"natural": "natural", "mono": "mono", "noir": "noir", "sepia": "sepia", "vhs": "vhs",
             "vivid": "vivid", "pastel": "pastel"}
APP_LIGHT = {"night": "night", "dusk": "dusk", "dawn": "dawn", "day": "day", "overcast": "day",
             "underwater": "dusk", "interior": "night"}
APP_MOOD = {"fear": "scary", "calm": "calm", "sad": "eerie", "joy": "joyful", "uncanny": "eerie", "longing": "calm"}


def from_app_spec(app: dict):
    look_in = {**app.get("look", {}), **(app.get("overrides") or {})}
    text = (app.get("input") or {}).get("transcript") or app.get("summary") or app.get("prompt", "")
    spec = parse(text, seed=int(app.get("seed", 0)) % 10_000)
    scenes = app.get("scenes") or []
    if scenes:
        s0 = scenes[0]
        mapped = APP_SETTING.get(s0.get("setting", ""))
        # flooded / swimming city dreams use the flooded street; an open sky with stairs uses the staircase
        low = text.lower()
        if s0.get("setting") == "city" and any(w in low for w in ("flood", "underwater", "swim")):
            mapped = "flooded_city"
        if s0.get("setting") in ("open", "field") and "stairs" not in s0.get("elements", []) and spec.setting != "stairs":
            mapped = spec.setting if spec.setting != "forest" else mapped
        if mapped and (spec.setting == "forest" and "forest" not in low or mapped != "stairs"):
            spec.setting = mapped
        for e in s0.get("elements", []):
            key = {"crowd": "figure"}.get(e, e)
            if key not in spec.objects:
                spec.objects.append(key)
        if "figure" in s0.get("elements", []) or "crowd" in s0.get("elements", []):
            spec.figures = max(spec.figures, 1)
        if s0.get("motion"):
            spec.motion = s0["motion"]
    if look_in.get("light") in APP_LIGHT:
        spec.time_of_day = APP_LIGHT[look_in["light"]]
    if "fog" in look_in:
        spec.clarity = fog_to_clarity(float(look_in["fog"]))
    moods = app.get("mood") or []
    if moods:
        spec.mood = APP_MOOD.get(moods[0], spec.mood)
    look = look_from_spec(spec.__dict__)
    for k in ("fog", "fps", "grain", "echo", "vignette", "saturation"):
        if k in look_in and look_in[k] is not None:
            look[k] = float(look_in[k])
    if look_in.get("style") in APP_STYLE:
        look["style"] = APP_STYLE[look_in["style"]]
        look["film"] = look["style"] in ("mono", "noir", "sepia") or look.get("film", False)
    if look_in.get("tint"):
        look["tint"] = hex_to_rgb(look_in["tint"])
    if look_in.get("fog_tone") is not None:          # 0 pale .. 0.5 scene colour .. 1 black
        look["fog_tone"] = float(look_in["fog_tone"])
    if look_in.get("fog_timeline"):                  # [{at, fog, tone}] over the whole dream
        look["timeline"] = look_in["fog_timeline"]
    return spec, look


def from_app_spec_multi(app: dict):
    """One (spec, look) per app scene (1-4). Each scene is parsed from its own text; the look is shared."""
    base_spec, look = from_app_spec(app)
    out = []
    scenes = app.get("scenes") or [{}]
    full = ((app.get("input") or {}).get("transcript") or "")
    for i, sc in enumerate(scenes):
        text = sc.get("text") or full
        sp = parse(text, seed=(int(app.get("seed", 0)) + 977 * i) % 10_000)
        low = (text + " " + full).lower() if i == 0 else text.lower()
        setting = APP_SETTING.get(sc.get("setting", ""), sp.setting)
        if sc.get("setting") == "city" and any(w in low for w in ("flood", "underwater", "swim")):
            setting = "flooded_city"
        if sc.get("setting") == "open" and "stairs" not in sc.get("elements", []) and sp.setting != "stairs":
            setting = sp.setting
        if sc.get("setting") == "city" and sc.get("motion") == "falling" and not sc.get("elements"):
            setting = "street"
        sp.setting = setting
        els = sc.get("elements", [])
        sp.objects = list(dict.fromkeys(sp.objects + [{"crowd": "figure"}.get(e, e) for e in els]))
        sp.figures = max(sp.figures, 1 if ("figure" in els or "crowd" in els) else 0)
        sp.motion = sc.get("motion", sp.motion)
        light = sc.get("light") or (app.get("look") or {}).get("light")
        if light in APP_LIGHT:
            sp.time_of_day = APP_LIGHT[light]
        if setting == "beach" and sp.time_of_day not in ("night", "dusk"):
            sp.time_of_day = "night"
        sp.clarity, sp.palette = base_spec.clarity, base_spec.palette
        lk = dict(look, time_of_day=sp.time_of_day)
        lk.pop("timeline", None)
        user_tone = ((app.get("overrides") or {}).get("fog_tone") if app.get("overrides") else None)
        if sc.get("fog_tone") is not None:
            lk["fog_tone"] = float(sc["fog_tone"])
        elif user_tone is None:
            lk["fog_tone"] = tone_of(text.lower(), base_spec.mood)[0]        # each scene gets its own fog tone
        tl = fog_timeline(text.lower(), fog_to_clarity(lk["fog"]), lk["fog_tone"])
        if tl:
            lk["timeline"] = tl
        out.append((sp, lk, float(sc.get("duration_s", 6))))
    return out
