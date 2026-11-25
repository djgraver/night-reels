"""Content rating stage: dream spec v1 -> `content_rating` (assessed rating, mitigations, abort decision).

Policy: render/CONTENT_POLICY.md (target: Australian M; PG when a child is involved).

Two raters produce the same thing, a list of *findings* (scene, element, impact, tags):
  * llm_findings()   Claude reads the transcript + scenes and classifies (structured output)
  * rule_findings()  offline lexicon rater; always runs, and acts as a floor under the LLM
Findings from both are merged (the stricter wins), then decide() applies the policy in plain code,
so the policy is testable and the LLM never decides on its own whether something renders.

    from dreamrender.rating import rate
    cr = rate(app_spec)            # -> dict, written to spec["content_rating"]
    cr["outcome"]                  # "render" | "render_mitigated" | "abort"
"""
from __future__ import annotations
import json, os, re
from typing import Literal, Optional

LEVELS = ["none", "very_mild", "mild", "moderate", "strong", "high"]
LV = {k: i for i, k in enumerate(LEVELS)}
RATING_OF = {0: "G", 1: "G", 2: "PG", 3: "M", 4: "MA15+", 5: "R18+"}
ELEMENTS = ["themes", "violence", "sex", "nudity", "language", "drug_use", "horror"]
TAGS = ["affection", "swimwear", "implied_sex", "explicit_sex", "nudity_sexual", "nudity_nonsexual",
        "sexual_violence", "fight", "weapon", "death", "blood", "gore", "torture",
        "suicide", "self_harm", "threat", "drug_mild", "drug_hard", "instructional",
        "coarse", "strong_coarse", "abuse", "child_involved"]
LLM_MODEL = os.environ.get("DREAM_RATER_MODEL", "claude-opus-5-5")

MESSAGES = {
    "abort": ("This dream stays in your journal as words. Dreams go wherever they go, and that says nothing "
              "about you, but we don't turn dreams that put children together with sexual or violent "
              "content into pictures."),
    "mitigated": ("A few moments are veiled (fog, a censor bar or a cut to your reaction) so the video stays "
                  "around an M rating. Your full dream is saved exactly as you wrote it."),
    "support": ("If anything in this dream is close to home, Lifeline is there any time on 13 11 14."),
}


# =========================================================================== rule rater (offline floor)
def _rx(*words):
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.I)

# (element, impact, tags, pattern). Most specific first doesn't matter: every match is a finding.
LEXICON = [
    # sex / nudity
    ("sex", "very_mild", ["affection"], _rx(r"kiss(?:ed|es|ing)?", r"made out", r"making out", r"hug(?:ged|ging)?", r"cuddl\w*", r"flirt\w*", r"held hands", r"on a date")),
    ("sex", "mild", ["swimwear"], _rx(r"bikinis?", r"swimsuits?", r"swim(?:ming)? trunks", r"underwear", r"lingerie", r"in (?:my|his|her|their) undies")),
    ("sex", "moderate", ["implied_sex"], _rx(r"slept with", r"sleeping with", r"in bed with", r"hook(?:ed|ing) up",
                                             r"made love", r"had sex", r"having sex", r"sex", r"sexual(?:ly)?", r"something sexual", r"seduc\w+", r"sexy", r"turned on", r"aroused")),
    ("sex", "strong", ["explicit_sex"], _rx(r"sex(?:ual)? act", r"oral sex", r"intercourse", r"orgasm\w*", r"porn\w*", r"genitals?", r"penis", r"vagina",
                                            r"blow ?job", r"masturbat\w+", r"explicit(?:ly)? sexual", r"threesome", r"orgy")),
    ("nudity", "moderate", ["nudity_nonsexual"], _rx(r"naked", r"nude", r"nudity", r"topless", r"no clothes on", r"without (?:any )?clothes", r"undress\w*", r"stripped")),
    ("violence", "high", ["sexual_violence"], _rx(r"rap(?:e|ed|ing|ist)", r"sexual(?:ly)? assault\w*", r"molest\w*", r"grop(?:e|ed|ing)", r"forced (?:me|him|her|them) to (?:have sex|undress|touch)")),
    # violence
    ("violence", "mild", ["fight"], _rx(r"punch\w*", r"slap\w*", r"kick(?:ed|ing)?", r"fist ?fight", r"fight(?:ing)?", r"fought", r"shov\w+", r"push(?:ed)? (?:me|him|her|them)", r"wrestl\w+", r"tackl\w+")),
    ("violence", "moderate", ["weapon"], _rx(r"guns?", r"gunfights?", r"gunfire", r"gunshots?", r"pistol", r"rifle", r"shoot(?:ing|out)?", r"shot at", r"knife", r"knives", r"sword\w*", r"explo(?:sion|ded|sive)s?", r"bomb\w*",
                                             r"attack\w*", r"beat(?:en)? up", r"war ?zone", r"battle", r"car chase", r"crash(?:ed)?", r"(?:was|got|been|being) shot")),
    ("violence", "strong", ["abuse"], _rx(r"abus(?:e|ed|ing|ive)", r"beat(?:ing)? (?:me|him|her|them) (?:up|badly)", r"touch(?:ed|ing)? (?:me|him|her|them) inappropriately", r"inappropriate(?:ly)? touch\w*")),
    ("violence", "moderate", ["death"], _rx(r"kill(?:ed|ing|s)?", r"murder\w*", r"shot (?:him|her|them|me|dead)", r"died", r"dead", r"dying")),
    ("violence", "strong", ["blood"], _rx(r"blood\w*", r"bleed\w*", r"bled", r"wound(?:ed|s)?", r"stab(?:bed|bing)?", r"slit", r"bullet holes?")),
    ("violence", "high", ["gore"], _rx(r"gore", r"gory", r"guts", r"entrails", r"decapitat\w+", r"behead\w*", r"dismember\w*", r"mutilat\w+", r"severed", r"corpses?", r"dead bod(?:y|ies)")),
    ("violence", "high", ["torture"], _rx(r"tortur\w+")),
    # themes / horror
    ("horror", "mild", ["threat"], _rx(r"monsters?", r"ghosts?", r"chas(?:ed|ing) me", r"following me", r"scar(?:ed|y)", r"spiders?", r"lost", r"trapped", r"nightmare")),
    ("horror", "moderate", ["threat"], _rx(r"demons?", r"possess\w+", r"zombies?", r"hunt(?:ed|ing) me", r"drown\w*", r"buried alive", r"something (?:evil|horrible)", r"terrif\w+", r"screaming")),
    ("themes", "strong", ["suicide"], _rx(r"suicid\w*", r"kill(?:ed|ing)? (?:my|him|her|them)sel(?:f|ves)", r"end(?:ed|ing)? (?:my|his|her|their) li(?:fe|ves)", r"hang(?:ed|ing) (?:my|him|her|them)sel(?:f|ves)", r"overdos\w+")),
    ("themes", "strong", ["self_harm"], _rx(r"self[- ]harm\w*", r"cut(?:ting)? (?:my|him|her|them)sel(?:f|ves)", r"hurt(?:ing)? (?:my|him|her|them)sel(?:f|ves)")),
    # drugs
    ("drug_use", "mild", ["drug_mild"], _rx(r"weed", r"joint", r"stoned", r"high on", r"pills?", r"drunk", r"smok(?:ed|ing) pot")),
    ("drug_use", "moderate", ["drug_hard"], _rx(r"heroin", r"meth", r"ice pipe", r"cocaine", r"coke", r"snort\w*", r"inject\w*", r"needles?", r"syringes?", r"crack pipe")),
    ("drug_use", "strong", ["instructional"], _rx(r"how to (?:make|cook) (?:meth|drugs|a bomb)", r"cook(?:ed|ing)? meth")),
    # language (only matters for text the app shows: title, summary, captions)
    ("language", "mild", ["coarse"], _rx(r"damn", r"hell", r"crap", r"bloody", r"arse", r"piss\w*")),
    ("language", "moderate", ["coarse"], _rx(r"shit\w*", r"bastards?", r"bitch\w*", r"dick", r"fuck\w*")),
    ("language", "strong", ["strong_coarse"], _rx(r"cunts?", r"motherfuck\w*")),
]

CHILD_SURE = _rx(r"child(?:ren)?", r"kids?", r"bab(?:y|ies)", r"toddlers?", r"infants?", r"little (?:girl|boy|sister|brother|one)s?", r"school ?(?:girl|boy|kid)s?",
                 r"primary school", r"kindergarten", r"preschool\w*", r"minors?", r"under ?age", r"teen(?:s|age|ager|agers)?", r"tweens?",
                 r"i was (?:a kid|little|a child|young again|small again)", r"(?:1[0-7]|[1-9])[- ]years?[- ]old")
CHILD_MAYBE = _rx(r"boys?", r"girls?", r"sons?", r"daughters?", r"nieces?", r"nephews?", r"students?", r"young (?:man|woman|person)")
# words that make a "maybe" person an adult in context ("girlfriend" is already excluded by \b)
ADULT_HINT = _rx(r"(?:grown|adult) (?:son|daughter)", r"girls? at (?:the|a) (?:bar|club|pub|office|work)", r"boys? at (?:the|a) (?:bar|club|pub|office|work)")

SEXUAL_TAGS = {"implied_sex", "explicit_sex", "nudity_sexual", "nudity_nonsexual", "sexual_violence", "swimwear"}
VIOLENT_TAGS = {"weapon", "death", "blood", "gore", "torture", "sexual_violence", "abuse"}


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if s.strip()]


def _scene_texts(spec):
    scenes = spec.get("scenes") or []
    full = ((spec.get("input") or {}).get("transcript") or spec.get("summary") or "")
    out = [(i, sc.get("text") or "") for i, sc in enumerate(scenes)]
    covered = " ".join(t for _, t in out)
    # transcript text not covered by any scene still counts (scene -1 = whole dream)
    if full and (not out or len(covered) < 0.8 * len(full)):
        out.append((-1, full))
    for k in ("title", "summary"):
        if spec.get(k):
            out.append((-2, spec[k]))         # -2 = shown text only, not rendered
    return out


def rule_findings(spec, people=None):
    people = people if people is not None else (spec.get("people") or [])
    minors = [p for p in people if p.get("minor") is True]
    unknown_age = [p for p in people if "minor" in p and p.get("minor") is None and p.get("relation") != "self"]
    as_child = bool((spec.get("dreamer") or {}).get("as_child"))
    findings, child_present, child_sure = [], bool(unknown_age), as_child
    for scene, text in _scene_texts(spec):
        for sent in _sentences(text):
            sure = as_child or bool(CHILD_SURE.search(sent)) or any(_name_in(p, sent) for p in minors)
            maybe = bool(CHILD_MAYBE.search(sent)) and not ADULT_HINT.search(sent)
            child_present |= sure or maybe
            child_sure |= sure
            named = [p.get("id") for p in people if _name_in(p, sent)]
            for el, imp, tags, rx in LEXICON:
                m = rx.search(sent)
                if not m:
                    continue
                tags = list(tags)
                if el == "nudity" and any(t in SEXUAL_TAGS - {"nudity_nonsexual", "swimwear"} for _, _, tt, r2 in LEXICON if r2.search(sent) for t in tt):
                    tags, imp = ["nudity_sexual"], "strong"
                # "dead tired", "I died laughing" etc. are common; keep death mild unless a person is the object
                if "death" in tags and re.search(r"\bdead (?:tired|end|silence|of night)|died laughing|dying to\b", sent, re.I):
                    continue
                if sure and (set(tags) & (SEXUAL_TAGS - {"swimwear"} | VIOLENT_TAGS) or (el == "violence" and LV[imp] >= LV["moderate"])):
                    tags.append("child_involved")
                findings.append(dict(scene=scene, element=el, impact=imp, tags=tags, person_ids=named,
                                     note=f"matched '{m.group(0).lower()}'", source="rules",
                                     child_maybe=maybe and not sure))
    return dict(findings=findings, child_present=child_present or bool(minors), child_certain=child_sure or bool(minors))


def _name_in(person, sent):
    """A person is in a sentence if their name, how the dreamer referred to them, or a recorded mention appears."""
    names = [person.get("name"), person.get("who")] + [m.get("phrase") for m in person.get("mentions") or []]
    for n in names:
        n = (n or "").strip()
        if n and re.search(r"\b" + re.escape(n) + r"\b", sent, re.I):
            return True
    return False


# =========================================================================== LLM rater
SYSTEM_PROMPT = """You are the content rater for a dream journal app that turns a dreamer's own description of a dream into a short, foggy, low-framerate video.

Your only job is classification. You never judge the dreamer: dreams are involuntary and often dark, violent or sexual. You list what the dream contains so that downstream code can decide what to veil. Downstream code, not you, decides whether anything renders.

Rate each scene (index from 0; use -1 for content in the transcript that is not in any scene) against the six Australian classifiable elements plus horror, using the Australian impact scale: none, very_mild (G), mild (PG), moderate (M), strong (MA15+), high (R18+). Rate what the dream *depicts*, as it would be shown on screen, not how upsetting it was for the dreamer.

Calibration (Australian Classification Guidelines for Films 2012, with BBFC 12A and MPA PG-13 as cross-checks):
- violence: mild = scuffles, slapstick, a punch; moderate = action-movie fights, guns fired, explosions, a death that is not lingered on, no blood detail; strong = blood, wounds, stabbing, detailed injury, sustained brutality; high = gore, torture, dismemberment. Any sexual violence is tagged sexual_violence and rated high.
- sex: very_mild = kissing, hugging, flirting (tag affection); mild = swimwear or underwear, bikini level (tag swimwear); moderate = sex that is implied but not shown, making out (implied_sex); strong = sexual activity described or shown (explicit_sex).
- nudity: any nudity is at least moderate; tag nudity_sexual when it is in a sexual context, else nudity_nonsexual (e.g. the classic 'naked in public' dream).
- themes: moderate = grief, illness, death of a loved one, menace; strong = suicide, self-harm (tag suicide / self_harm), sustained abuse, discrimination that the dream endorses.
- horror: mild = a monster, being lost or chased; moderate = sustained threat, demons, drowning, being hunted; strong = intense, prolonged terror with graphic detail.
- drug_use: mild = alcohol or cannabis, pills; moderate = hard drug use (drug_hard); strong = instructional detail (instructional).
- language: only coarse words the dreamer quotes. mild = damn/hell/crap; moderate = shit, the occasional f-word; strong = c-word, aggressive or repeated very strong language (strong_coarse).

Children: set child_present true if anyone in the dream is, or appears to be, under 18, including the dreamer being a child or teenager in the dream, or any person listed with minor=true. Ambiguous words (girl, boy, son, daughter, student) mean a child only when the context says so; decide from context and say so in child_reason. Tag a finding child_involved when the sexual, nude or violent content involves, is directed at, is witnessed up close by, or alludes to a child. Allusions count: an implied or vague reference is still child_involved. Set child_sexual_or_violent true if any finding is child_involved and is sexual, nude, or violence at moderate or above.

People: if a listed person (by id) takes part in a finding, put their id in person_ids.

Notes must be neutral, at most 12 words, and contain no graphic detail. Return nothing that repeats explicit content from the transcript.
"""

TagT = Literal[tuple(TAGS)]  # type: ignore[valid-type]


def _models():
    from pydantic import BaseModel, Field

    class Finding(BaseModel):
        scene: int
        element: Literal[tuple(ELEMENTS)]  # type: ignore[valid-type]
        impact: Literal[tuple(LEVELS)]     # type: ignore[valid-type]
        tags: list[TagT] = Field(default_factory=list)
        person_ids: list[str] = Field(default_factory=list)
        note: str = ""

    class LLMRating(BaseModel):
        findings: list[Finding]
        child_present: bool
        child_reason: str
        child_sexual_or_violent: bool
        overall_tone: Literal["light", "neutral", "dark", "nightmare"]

    return Finding, LLMRating


def _llm_input(spec, people):
    keep = dict(transcript=(spec.get("input") or {}).get("transcript"), title=spec.get("title"),
                scenes=[dict(index=i, text=s.get("text"), setting=s.get("setting"), elements=s.get("elements"))
                        for i, s in enumerate(spec.get("scenes") or [])],
                mood=spec.get("mood"),
                people=[dict(id=p.get("id"), name=p.get("name") or p.get("who"), relation=p.get("relation"),
                             minor=p.get("minor"), scenes=p.get("scenes")) for p in people],
                dreamer_as_child=bool((spec.get("dreamer") or {}).get("as_child")))
    return "Rate this dream.\n\n<dream>\n" + json.dumps(keep, ensure_ascii=False, indent=1) + "\n</dream>"


def llm_findings(spec, people=None, client=None, model=LLM_MODEL):
    """Claude classifies the dream. Returns None if no credentials / the call fails / it declines."""
    people = people if people is not None else (spec.get("people") or [])
    _, LLMRating = _models()
    try:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        resp = client.beta.messages.parse(
            model=model, max_tokens=8000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _llm_input(spec, people)}],
            output_format=LLMRating,
        )
    except Exception as e:                       # no key, network, API error: the rules floor still runs
        return dict(error=f"{type(e).__name__}: {e}"[:300])
    if getattr(resp, "stop_reason", None) == "refusal" or resp.parsed_output is None:
        return dict(error="declined")
    r = resp.parsed_output
    f = [dict(x.model_dump(), source="llm") for x in r.findings]
    return dict(findings=f, child_present=r.child_present, child_certain=r.child_present,
                child_sexual_or_violent=r.child_sexual_or_violent, tone=r.overall_tone,
                child_reason=r.child_reason, model=getattr(resp, "model", model))


# =========================================================================== policy
def decide(findings, child_present=False, child_certain=None, people=None, tone=None, child_sexual_or_violent=False,
           dreamer=None, scene_people=None):
    """Findings -> content_rating. Pure function: all policy lives here (see CONTENT_POLICY.md).
    people: spec people[] (minor true/false/null, likeness); dreamer: spec dreamer {as_child, emotion, photo_ref};
    scene_people: {scene index: [people ids]} from spec scenes[].people."""
    people, dreamer, scene_people = people or [], dreamer or {}, scene_people or {}
    if dreamer.get("as_child"):
        child_present = child_certain = True
    child_certain = child_present if child_certain is None else child_certain
    target = "PG" if child_present else "M"
    limit = LV["mild"] if target == "PG" else LV["moderate"]
    photo_ids = {p.get("id") for p in people if (p.get("photo") or p.get("has_photo") or p.get("likeness") == "photo") and p.get("minor") is not True}
    minor_ids = {p.get("id") for p in people if p.get("minor") is True}
    unknown_ids = {p.get("id") for p in people if "minor" in p and p.get("minor") is None and p.get("relation") != "self"}

    impact = {e: "none" for e in ELEMENTS}
    abort_reasons, mits, extra = [], [], set()
    for f in findings:
        el, lv, tags, sc = f["element"], LV[f["impact"]], set(f.get("tags", [])), f.get("scene", -1)
        if lv > LV[impact[el]]:
            impact[el] = f["impact"]
        sexual = el in ("sex", "nudity") and (tags & SEXUAL_TAGS or lv >= LV["moderate"])
        violent = el == "violence" and (lv >= LV["moderate"] or tags & VIOLENT_TAGS)
        pids = set(f.get("person_ids", []))
        involves_minor = "child_involved" in tags or bool(pids & minor_ids) or (dreamer.get("as_child") and sc != -2)
        if sexual and pids & unknown_ids:                # age unknown ("my son"): treated as a child for sexual content
            involves_minor = True
        # ---- hard stop: children + sexual / nude / violent content (including allusions)
        if involves_minor and (sexual or violent or "sexual_violence" in tags):
            abort_reasons.append(f"child involved in {el} content (scene {sc})")
            continue
        if child_certain and sexual and (lv >= LV["moderate"] or tags & (SEXUAL_TAGS - {"swimwear"})):
            abort_reasons.append(f"sexual content in a dream with a child present (scene {sc})")
            continue
        if sc == -2:                                    # title/summary text: only language matters there
            if el == "language" and (lv > limit or "strong_coarse" in tags):
                mits.append(dict(scene=-2, kind="soften_text", reason="coarse language in title or summary"))
            continue
        face = {k: v for k, v in (("face_ref", dreamer.get("photo_ref")), ("emotion", dreamer.get("emotion"))) if v}

        def add(kind, reason, at=None, **kw):
            if kind == "reaction_cut":
                kw = {**face, **kw}
            mits.append(dict(scene=sc, kind=kind, reason=reason, **({"at": at} if at else {}), **kw))
        # ---- sex & nudity: bikini level renders; anything further gets the comical bar
        if el == "nudity" or (el == "sex" and (lv > LV["mild"] if target == "M" else lv > LV["very_mild"] and "swimwear" not in tags)):
            add("censor_bar", f"{el}: {f['impact'].replace('_', ' ')}", at=[0.15, 1.0])
        # ---- violence: action-movie level renders; blood, gore, death on screen -> cut to the dreamer's reaction
        if "sexual_violence" in tags:
            add("fade_to_fog", "sexual violence is never shown", at=[0.0, 1.0], tone=0.95)
            add("reaction_cut", "sexual violence: show only the dreamer's reaction", at=[0.25, 1.0])
        elif el == "violence":
            if tags & {"blood", "gore", "torture"} or lv > limit:
                add("reaction_cut", f"violence: {', '.join(sorted(tags & VIOLENT_TAGS)) or f['impact']}", at=[0.45, 1.0])
            elif "death" in tags:
                add("reaction_cut", "a death is shown as the dreamer's reaction", at=[0.55, 1.0])
            elif target == "PG" and lv >= LV["moderate"]:
                add("reaction_cut", "PG cap: violence shown as reaction", at=[0.5, 1.0])
        # ---- themes & horror
        if tags & {"suicide", "self_harm"}:
            add("fade_to_fog", "suicide or self-harm: no method or place is shown", at=[0.2, 1.0], tone=0.6)
            extra.add("support")
        elif el in ("themes", "horror") and lv > limit:
            add("dark_fog", f"{el} above {target}: veil in dark fog", at=[0.3, 1.0], tone=0.9)
        # ---- drugs
        if el == "drug_use" and (tags & {"drug_hard", "instructional"} or lv > limit):
            add("fade_to_fog", "drug use: no paraphernalia or method detail", at=[0.3, 1.0], tone=0.5)
        # ---- language in captions
        if el == "language" and (lv > limit or "strong_coarse" in tags):
            add("soften_text", "coarse language")
        # ---- uploaded likenesses never appear in sexual or violent-victim content
        in_scene = set(scene_people.get(sc, [])) if sc >= 0 else set().union(*scene_people.values()) if scene_people else set()
        for pid in (pids | in_scene) & photo_ids:
            if sexual or violent:
                add("generic_face", "an uploaded likeness is never used in sexual or violent content", target=pid)

    if child_sexual_or_violent and not abort_reasons:
        abort_reasons.append("rater flagged child + sexual/violent content")
    max_lv = max([LV[v] for v in impact.values()] + [0])
    assessed = RATING_OF[max_lv]
    if any("sexual_violence" in f.get("tags", []) for f in findings):
        assessed = "R18+"
    if abort_reasons:
        assessed = "RC"
    # dedupe mitigations (same scene + kind): keep the earliest start
    seen = {}
    for m in mits:
        k = (m["scene"], m["kind"], m.get("target"))
        if k not in seen or (m.get("at", [1])[0] < seen[k].get("at", [1])[0]):
            seen[k] = m
    mits = [] if abort_reasons else sorted(seen.values(), key=lambda m: (m["scene"], m["kind"]))
    outcome = "abort" if abort_reasons else ("render_mitigated" if mits else "render")
    msg = MESSAGES["abort"] if abort_reasons else (MESSAGES["mitigated"] if mits else None)
    if "support" in extra and not abort_reasons:
        msg = (msg + " " if msg else "") + MESSAGES["support"]
    return dict(version=1, assessed=assessed, target=target, outcome=outcome,
                child_involved=bool(child_present), impact=impact, mitigations=mits,
                dreamer_message=msg, tone=tone, abort_reasons=abort_reasons)


def merge_findings(*sets):
    """Union of findings; per (scene, element) the stricter impact wins and tags are unioned."""
    out = {}
    for fs in sets:
        for f in fs or []:
            k = (f.get("scene", -1), f["element"])
            if k not in out:
                out[k] = dict(f, tags=list(f.get("tags", [])), person_ids=list(f.get("person_ids", [])), sources=[f.get("source")])
                continue
            o = out[k]
            if LV[f["impact"]] > LV[o["impact"]]:
                o["impact"] = f["impact"]
            o["tags"] = sorted(set(o["tags"]) | set(f.get("tags", [])))
            o["person_ids"] = sorted(set(o["person_ids"]) | set(f.get("person_ids", [])))
            o["sources"] = sorted(set(o["sources"]) | {f.get("source")})
    return list(out.values())


def rate(spec, people=None, use_llm=None, client=None):
    """Full stage. use_llm: None = try Claude if credentials exist, True = require it, False = rules only."""
    people = people if people is not None else (spec.get("people") or [])
    rules = rule_findings(spec, people)
    llm = None
    if use_llm is None:
        use_llm = client is not None or bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if use_llm:
        llm = llm_findings(spec, people, client=client)
    if llm and "findings" in llm:
        # The LLM resolves ambiguous words ("girl"), so its child judgement wins over a *maybe*,
        # but an unambiguous child word in the text always counts (the floor).
        child_present = llm["child_present"] or rules["child_certain"]
        child_certain = child_present
        findings = merge_findings(llm["findings"], [f for f in rules["findings"] if not f.get("child_maybe") or llm["child_present"]])
        rater = f"llm:{llm.get('model', LLM_MODEL)}+rules"
        extra = dict(tone=llm.get("tone"), child_sexual_or_violent=llm.get("child_sexual_or_violent", False))
    else:
        child_present, child_certain = rules["child_present"], rules["child_certain"]
        findings = merge_findings(rules["findings"])
        rater = "rules" + (f" (llm unavailable: {llm['error']})" if llm and "error" in llm else "")
        extra = {}
    scene_people = {i: list(sc.get("people") or []) for i, sc in enumerate(spec.get("scenes") or [])}
    cr = decide(findings, child_present=child_present, child_certain=child_certain, people=people,
                dreamer=spec.get("dreamer"), scene_people=scene_people, **extra)
    cr["rater"] = rater
    cr["findings"] = [dict(scene=f["scene"], element=f["element"], impact=f["impact"], tags=f["tags"],
                           sources=f.get("sources")) for f in findings]
    return cr


# =========================================================================== spec rewrite
NEGATIVE_GUARD = "nudity, explicit content, blood, gore, graphic injury, weapons pointed at camera"
NEGATIVE_GUARD_PG = ", sexualised poses, revealing clothing, violence, frightening imagery"
GRAWLIX = re.compile(r"\b(?:fuck\w*|motherfuck\w*|cunts?|shit\w*|bastards?|bitch\w*)\b", re.I)


def soften(text):
    pool = "#@$%&!*"
    return GRAWLIX.sub(lambda m: "".join(pool[(i * 3 + len(m.group(0))) % len(pool)] for i in range(len(m.group(0)))), text or "")


def apply_to_spec(spec, cr):
    """Returns a copy of the app spec with content_rating, a guarded negative prompt and softened text."""
    s = json.loads(json.dumps(spec))
    s["content_rating"] = {k: v for k, v in cr.items() if k != "findings"}
    neg = s.get("negative_prompt") or ""
    guard = NEGATIVE_GUARD + (NEGATIVE_GUARD_PG if cr["target"] == "PG" else "")
    s["negative_prompt"] = (neg + ", " if neg else "") + guard
    if any(m["kind"] == "soften_text" for m in cr["mitigations"]):
        for k in ("title", "summary"):
            if s.get(k):
                s[k] = soften(s[k])
    if cr["outcome"] == "abort":
        s["scenes"] = []
        s["prompt"] = ""
    return s


def scene_mitigations(cr, i):
    """Mitigations for rendered scene i (whole-dream findings, scene -1, apply to every scene)."""
    return [m for m in cr.get("mitigations", []) if m["scene"] in (i, -1)]


def main():
    import argparse, sys
    ap = argparse.ArgumentParser(description="Rate a dream spec (JSON file, or --text)")
    ap.add_argument("spec", nargs="?"); ap.add_argument("--text"); ap.add_argument("--rules-only", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(a.spec)) if a.spec else {"input": {"transcript": a.text}, "scenes": [{"text": a.text}]}
    cr = rate(spec, use_llm=False if a.rules_only else None)
    json.dump(cr, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
