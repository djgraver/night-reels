# dreamrender

Turns a dream description into a dream-fogged still and a short low-framerate clip.

```
dream text / app spec v1
   │
   ├─ parse.py      clarity cues (+"crystal clear", −"hazy", negation aware), style cues (b&w, sepia, vivid, film, vhs),
   │                colour tints ("everything was purple"), setting, time of day, objects, figures, motion, mood
   ├─ app_spec.py   or: the app's dream spec v1 (look block wins, scenes[] → one shot per scene)
   │
   ├─ scenes.py     base imagery. Blender Cycles, procedural, no downloaded assets. 9 scenes:
   │                hallway, beach (night sea + moon), mountain, station, flooded_city, forest, stairs (into clouds),
   │                street (rainy night, or falling between buildings), field (tall grass at sunset, Cycles hair)
   │                Per frame: beauty PNG (AgX) + depth EXR (camera distance) + subject screen position.
   │
   ├─ stylize.py    the dream look (numpy/OpenCV, ~0.3 s per 960×540 frame)
   │                 1 slow low-frequency warp            6 edges dissolve into fog, then vignette
   │                 2 depth fog with light scattering    7 dissolve in/out
   │                   (lights glow through, subject       8 memory colour: desaturate, milky blacks, mood split-tone, tint
   │                    keeps an "attention island")      9 explicit styles: mono / noir / sepia / vhs / vivid / pastel
   │                 3 soft focus: sharp subject, vague   10 grain, film dust and scratches for film styles
   │                   periphery, depth defocus
   │                 4 bloom + red halation + Orton glow
   │                 5 chromatic aberration
   │
   └─ animate.py    6–12 fps with irregular holds at 24 fps, echo trails, double exposure between frames,
                    fog and warp keep drifting under held frames, fade up from fog and back. ffmpeg → H.264 mp4.
```

One number drives most of it: **fog** (0–1). The parser's clarity maps onto it
(typical dream 0.55, "crystal clear" 0.08, "really hazy" 0.9) and fog then sets
fog distance, blur, glow, warp, saturation, grain, echo and fps. The app's
`look` block uses the same names (fog, fps, grain, echo, vignette, style,
saturation, tint), so the dreamer's hand overrides pass straight through.

## Run

```
pip install bpy==5.3.0 opencv-python-headless OpenEXR numpy      # ffmpeg on PATH
python3 render_examples.py                                       # 7 example dreams -> out/<name>/
python3 -m dreamrender.pipeline --text "I was in a hazy forest and there was a red door" --out out/mine
python3 -m dreamrender.pipeline --app-spec app_examples/d_1pyxr9f.json --out out/d_1pyxr9f
SKIP_RENDER=1 python3 render_examples.py                         # re-stylize existing frames only
python3 -m dreamrender.serve --port 8787                         # POST /jobs (spec) -> poll GET /jobs/<id>
```

Each output folder has `raw.jpg` (render before the dream pass), `dream.jpg`,
`fog_dial.jpg` (same frame at fog 0.10 / 0.35 / 0.60 / 0.85), `depth.jpg`,
`clip.mp4`, `poster.jpg`, `spec.json`, `manifest.json`.

Cost on this 4-core CPU box: 15–40 s per 960×540 frame (32 samples, denoised),
so 5–10 minutes per 16-frame clip. The stylizer itself is real-time-ish.

## Content rating (runs before anything is drawn)

`dreamrender/rating.py` rates the dream spec (Claude `claude-opus-5-5` classifies, an offline word-list rater is the
floor, `decide()` applies the policy in `CONTENT_POLICY.md`) and writes `content_rating` into the spec:
target M (PG when a child is in the dream), outcome `render | render_mitigated | abort`, mitigations, and a gentle
`dreamer_message`. `pipeline.run_app_spec()` is the gated entry point (the CLI and `serve.py` use it): abort raises
`ContentDeclined` (serve reports `status: failed, declined: true` with the message); otherwise `mitigate.py`
adds fog veils before rendering and censor bars / reaction-face cuts to each scene clip after.

```
python3 -m dreamrender.rating --text "I was naked on the beach" --rules-only   # just the rating
python3 -m pytest -q tests                                                      # 47 tests
python3 rating_demo.py                                                          # demo clips -> out/rating_demo/
```
Set `ANTHROPIC_API_KEY` (or `ant auth login`) to use the Claude rater; without it the rules rater runs alone.

## What production would swap

The fog stage, the spec and the clip assembly are meant to survive. The base
imagery is the stand-in: procedural Blender scenes were the only way to get
pseudo-realistic frames with depth here (no GPU, no image-model API, no access
to photo libraries from this sandbox). In production:

* **Scene understanding**: an LLM (Claude) fills the same DreamSpec JSON; this parser stays as the offline fallback and for live cue highlighting.
* **Base stills**: a text-to-image model (FLUX / SDXL class) prompted with `spec.prompt()` plus a negative prompt against painterly / HDR looks. Keep it photographic; the dream look comes after.
* **Base motion**: image-to-video (Veo / Kling / Runway / Wan class) from that still, 4–6 s, slow camera. Or cheaper: depth-estimate the still (Depth Anything V2) and do 2.5D parallax.
* **Depth**: Depth Anything V2 / MiDaS on every generated frame feeds `stylize()` exactly as the Blender depth does now.
* **Subject position**: a detector or the model's attention map gives the attention-island focus point.
* **Stylizer**: port `stylize()` to a GPU shader (it is all blurs, remaps and per-pixel maths) so the app can preview fog changes live.

## Faces (added 2026-11-25)

`dreamrender/faces.py` + `faces_demo.py`: one ordinary photo -> the dream identity states
(known_clear, known_different, stranger/known_unclear, obscured), photo-warp expressions, ArcFace identity score,
and a composite into a rendered scene through `stylize()` with face-aware fog.

```
sh fetch_face_models.sh                     # MediaPipe landmarker + segmenter, InsightFace w600k_r50 (non-commercial licence)
apt-get install -y libegl1 libgles2         # MediaPipe needs these headless
pip install mediapipe scikit-image onnxruntime
python3 faces_demo.py --photo me.jpg --scene hallway --emotion sad --out out/faces
```

Honest limits: this is a demonstration and fallback, not the production look. Warping cannot change a photo's
mouth (a smiling photo keeps smiling; painting the mouth over read as a smear, so it is off by default),
cannot change pose or angle, and the cut-out reads as a collage. Production uses an identity-conditioned
image/video model; see the production plan doc linked from the project thread.

## Fog tone and fog that changes (round 2)

* `look.fog_tone` 0..1: 0 pale milky fog, 0.5 fog in the scene's own colour, 1 black smoke. Parsed from the dream's words
  (`parse.tone_of`): dread, smoke, darkness, being chased push toward black; calm, warm, bright push toward pale.
  Black fog closes in faster, dims near surfaces too, swallows the sky and the frame edges, and only real light sources still glow through it.
* `look.timeline` `[{at, fog, tone}]`: the dream is split into beats (sentences, "then", "suddenly"), each scored on its own;
  explicit turns ("it went dark", "the fog lifted") force the change. `animate.py` eases between beats per frame.
* App spec: `look.fog_tone`, `look.fog_timeline`, `scenes[i].fog_tone` are accepted as overrides.
* `python3 render_tone.py` re-styles existing scene frames for the tone examples (hallway_dark, forest_shift, beach_shift).
