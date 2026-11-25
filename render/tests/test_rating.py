"""Content rating tests. Run: cd dream-render && python3 -m pytest -q tests
Abort cases use abstract test text only ("[TEST] ..."): nothing explicit lives in this file."""
import json, os, subprocess, sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dreamrender.rating import rate, decide, apply_to_spec, soften, llm_findings, _models, scene_mitigations
from dreamrender import mitigate


def spec_of(text, **kw):
    return dict({"input": {"transcript": text}, "scenes": [{"text": text}], "title": "t"}, **kw)


def kinds(cr):
    return sorted({m["kind"] for m in cr["mitigations"]})


# ---------------------------------------------------------------- rules rater: renders as-is (<= M)
@pytest.mark.parametrize("text", [
    "I was walking on a beach at night and the moon was huge.",
    "We were at the beach and she was in a bikini, and we kissed.",           # men's-magazine / bikini level
    "There was a shootout in the street and a car exploded behind me.",      # action-movie violence
    "Two guys had a fist fight in the bar.",
    "A monster was chasing me through a hallway and I was scared.",
    "Demons were hunting me through the house.",                             # moderate horror is M
    "A girl at the bar kissed me.",                                          # 'girl' in adult context
    "I was a kid again and we had a water fight at the pool.",               # child present, PG-safe
    "My kids were building sandcastles in their swimsuits.",
])
def test_renders_unmitigated(text):
    cr = rate(spec_of(text), use_llm=False)
    assert cr["outcome"] == "render", (text, cr["mitigations"])


# ---------------------------------------------------------------- mitigations
@pytest.mark.parametrize("text,expect", [
    ("I was naked at school in front of everyone.", "censor_bar"),
    ("I was in bed with my old boss and we had sex.", "censor_bar"),
    ("[TEST] He stabbed the guard and there was blood.", "reaction_cut"),
    ("[TEST] Someone was killed in front of me.", "reaction_cut"),
    ("[TEST] The dream was about suicide.", "fade_to_fog"),
    ("[TEST] Someone injected heroin.", "fade_to_fog"),
    ("[TEST] There was gore everywhere.", "reaction_cut"),
])
def test_mitigated(text, expect):
    cr = rate(spec_of(text), use_llm=False)
    assert cr["outcome"] == "render_mitigated"
    assert expect in kinds(cr)
    assert cr["dreamer_message"]


def test_sexual_violence_never_shown_for_adults():
    cr = rate(spec_of("[TEST] sexual assault."), use_llm=False)
    assert cr["outcome"] == "render_mitigated" and cr["assessed"] == "R18+"
    assert {"fade_to_fog", "reaction_cut"} <= set(kinds(cr))


def test_suicide_adds_support_line_gently():
    cr = rate(spec_of("[TEST] The dream was about suicide."), use_llm=False)
    assert "13 11 14" in cr["dreamer_message"]
    assert "you should" not in cr["dreamer_message"].lower()


# ---------------------------------------------------------------- children: PG cap and abort
@pytest.mark.parametrize("text", [
    "[TEST] A child was there and something sexual was implied.",
    "[TEST] A kid was shot.",
    "[TEST] Someone abused a child.",
    "[TEST] A toddler and nudity.",
    "[TEST] I was a kid again and there was blood.",
    "[TEST] A 12-year-old and a knife.",
])
def test_abort_children(text):
    cr = rate(spec_of(text), use_llm=False)
    assert cr["outcome"] == "abort", (text, cr)
    assert cr["assessed"] == "RC" and cr["mitigations"] == []
    assert "says nothing about you" in cr["dreamer_message"]


def test_child_present_caps_at_pg():
    cr = rate(spec_of("I lost my little boy in a supermarket. [TEST] Later, a horror scene with demons hunting me."), use_llm=False)
    assert cr["target"] == "PG"
    assert cr["outcome"] == "render_mitigated" and "dark_fog" in kinds(cr)


def test_sexual_content_anywhere_in_a_dream_with_a_child_aborts():
    s = {"input": {"transcript": "x"}, "scenes": [{"text": "My daughter's kindergarten concert."},
                                                 {"text": "[TEST] Later I had sex with my partner."}]}
    assert rate(s, use_llm=False)["outcome"] == "abort"


def test_ambiguous_child_word_caps_but_does_not_abort():
    cr = rate(spec_of("[TEST] Two adults fought with swords while my daughter watched TV in another room."), use_llm=False)
    assert cr["target"] == "PG" and cr["outcome"] == "render_mitigated"


def test_minor_from_people_list():
    s = spec_of("[TEST] Sam was shot.", people=[{"id": "p1", "name": "Sam", "minor": True}])
    assert rate(s, use_llm=False)["outcome"] == "abort"


def test_uploaded_likeness_gets_generic_face():
    s = spec_of("[TEST] Alex and I had sex.", people=[{"id": "p2", "name": "Alex", "has_photo": True}])
    cr = rate(s, use_llm=False)
    assert any(m["kind"] == "generic_face" and m["target"] == "p2" for m in cr["mitigations"])


# ---------------------------------------------------------------- language & spec rewrite
def test_language_softened_in_title_only_when_strong():
    s = spec_of("calm", title="[TEST] what the cunt")
    cr = rate(s, use_llm=False)
    out = apply_to_spec(s, cr)
    assert "cunt" not in out["title"] and out["content_rating"]["outcome"] == "render_mitigated"
    assert soften("oh shit") != "oh shit"


def test_apply_to_spec_guards_prompt_and_strips_abort():
    s = spec_of("[TEST] A kid was shot.", prompt="p", negative_prompt="cartoon")
    cr = rate(s, use_llm=False)
    out = apply_to_spec(s, cr)
    assert out["scenes"] == [] and out["prompt"] == "" and "gore" in out["negative_prompt"]
    assert "findings" not in out["content_rating"]


def test_whole_dream_findings_apply_to_every_scene():
    cr = decide([{"scene": -1, "element": "nudity", "impact": "moderate", "tags": ["nudity_nonsexual"]}])
    assert scene_mitigations(cr, 0) and scene_mitigations(cr, 3)


# ---------------------------------------------------------------- LLM rater (mocked client) and merge
class FakeClient:
    def __init__(self, parsed=None, stop="end_turn", exc=None):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))
        self.parsed, self.stop, self.exc = parsed, stop, exc

    def _parse(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop, model=kw["model"])


def llm_out(findings, child=False, csv=False, tone="neutral"):
    Finding, LLMRating = _models()
    return LLMRating(findings=[Finding(**f) for f in findings], child_present=child, child_reason="",
                     child_sexual_or_violent=csv, overall_tone=tone)


def test_llm_call_shape():
    c = FakeClient(llm_out([]))
    llm_findings(spec_of("hello"), client=c)
    kw = c.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in kw["betas"] and "thinking" not in kw


def test_llm_finds_what_rules_miss():
    c = FakeClient(llm_out([{"scene": 0, "element": "violence", "impact": "strong", "tags": ["blood"], "note": "injury"}]))
    cr = rate(spec_of("[TEST] something vague happened to the guard"), client=c)
    assert cr["rater"].startswith("llm:") and "reaction_cut" in kinds(cr)


def test_llm_cannot_lower_the_rules_floor():
    c = FakeClient(llm_out([], child=False))
    cr = rate(spec_of("[TEST] A kid was shot."), client=c)
    assert cr["outcome"] == "abort"


def test_llm_resolves_ambiguous_girl_as_adult():
    c = FakeClient(llm_out([{"scene": 0, "element": "sex", "impact": "very_mild", "tags": ["affection"]}], child=False))
    cr = rate(spec_of("[TEST] The girl from work and I made out."), client=c)
    assert cr["target"] == "M" and cr["outcome"] == "render"


def test_rules_alone_cap_ambiguous_girl_at_pg():
    cr = rate(spec_of("[TEST] The girl from work and I made out."), use_llm=False)
    assert cr["target"] == "PG" and cr["outcome"] == "render"


def test_llm_child_flag_aborts():
    c = FakeClient(llm_out([], child=True, csv=True))
    assert rate(spec_of("[TEST] abstract"), client=c)["outcome"] == "abort"


@pytest.mark.parametrize("client", [FakeClient(stop="refusal"), FakeClient(exc=RuntimeError("no network"))])
def test_llm_failure_falls_back_to_rules(client):
    cr = rate(spec_of("I was naked at school."), client=client)
    assert cr["rater"].startswith("rules") and "censor_bar" in kinds(cr)


# ---------------------------------------------------------------- mitigation renders
def test_veil_look_builds_timeline():
    lk = mitigate.veil_look({"fog": 0.5, "fog_tone": 0.5}, [{"kind": "dark_fog", "at": [0.4, 1], "tone": 0.9}])
    assert lk["timeline"][-1]["fog"] > 0.9 and lk["timeline"][-1]["tone"] >= 0.9
    assert mitigate.veil_look({"fog": 0.5}, [])["fog"] == 0.5


def test_censor_bar_and_veil_change_pixels():
    fr = (np.random.default_rng(0).random((270, 480, 3)) * 255).astype(np.uint8)
    assert np.abs(mitigate.draw_censor_bar(fr, (0.5, 0.5), 1.0).astype(int) - fr).mean() > 5
    v = mitigate.veil_frame(fr, 1.0, 1.0)
    assert v.std() < fr.std() * 0.3 and v.mean() < fr.mean()


def test_apply_to_clip_end_to_end(tmp_path):
    src, dst = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24:duration=2",
                    "-pix_fmt", "yuv420p", src], check=True)
    mits = [{"kind": "censor_bar", "at": [0.1, 1], "reason": "test"}, {"kind": "reaction_cut", "at": [0.5, 1], "reason": "test"}]
    done = mitigate.apply_to_clip(src, dst, mits, {"fog": 0.5, "fps": 8})
    assert {d["kind"] for d in done} == {"censor_bar", "reaction_cut"} and os.path.getsize(dst) > 1000


# ---------------------------------------------------------------- people[] / dreamer fields from the app (faces thread schema)
def test_dreamer_as_child_holds_pg_and_aborts_on_violence():
    s = spec_of("[TEST] There was a gunfight.", dreamer={"as_child": True})
    cr = rate(s, use_llm=False)
    assert cr["target"] == "PG" and cr["outcome"] == "abort"
    s = spec_of("A monster was under the bed.", dreamer={"as_child": True})
    assert rate(s, use_llm=False)["outcome"] == "render"


def test_unknown_age_person_is_conservative():
    p = [{"id": "k_son", "who": "my son", "relation": "son", "minor": None}]
    cr = rate(spec_of("My son was at the beach.", people=p), use_llm=False)
    assert cr["target"] == "PG"
    assert rate(spec_of("[TEST] My son and nudity.", people=p), use_llm=False)["outcome"] == "abort"


def test_photo_likeness_in_scene_gets_generic_face_and_reaction_uses_dreamer_face():
    p = [{"id": "p_1", "who": "my ex", "relation": "other", "minor": False, "likeness": "photo", "photo_ref": "local:p_1"}]
    s = {"input": {"transcript": "x"}, "people": p, "dreamer": {"photo_ref": "local:self", "emotion": "fear"},
         "scenes": [{"text": "[TEST] There was blood on the floor.", "people": ["p_1"]}]}
    cr = rate(s, use_llm=False)
    assert any(m["kind"] == "generic_face" and m["target"] == "p_1" for m in cr["mitigations"])
    rc = [m for m in cr["mitigations"] if m["kind"] == "reaction_cut"][0]
    assert rc["face_ref"] == "local:self" and rc["emotion"] == "fear"


def test_mentions_phrase_matches_person():
    p = [{"id": "c_1", "who": "a little girl", "minor": True, "mentions": [{"phrase": "the girl"}]}]
    assert rate(spec_of("[TEST] The girl was shot.", people=p), use_llm=False)["outcome"] == "abort"
