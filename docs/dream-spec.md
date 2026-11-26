# Dream spec v1 (app → render pipeline hand-off)

The app turns a typed or spoken dream into this JSON and hands it to a renderer.
Source of truth: `index.html` (repo root) (search "DREAM SPEC" and "RENDERER SEAM").

```jsonc
{
  "version": 1,
  "id": "d_xxxx", "created_at": "ISO time", "seed": 123,
  "input": { "mode": "voice" | "text", "transcript": "..." },
  "title": "The hallway that kept going", "summary": "...",
  "look": {                      // global treatment, derived from the dreamer's own words
    "fog": 0.0-1.0,              // default 0.55; "crystal clear/vivid/sharp" -> 0.08-0.18; "hazy/blurry/can't remember" -> 0.62-0.9
    "fps": 4-24,                 // default 8; clear dreams 12; "slow motion" 6; "fast/rushing" 14
    "grain": 0-1, "echo": 0-1,   // echo = frame ghosting / trails
    "vignette": 0-1,
    "style": "natural|mono|noir|sepia|vhs|vivid|pastel",  // only non-natural when the dreamer names it
    "saturation": 0-1.5, "tint": "#rrggbb" | null,         // tint from colour words ("everything was red")
    "light": "night|dusk|dawn|day|overcast|underwater|interior",
    "fog_tone": 0-1,             // optional: 0 pale milk, 0.5 scene colour, 1 black smoke (app sends it only when the "Fog colour" dial is moved; also scenes[i].fog_tone)
    "fog_timeline": [{"at":0-1,"fog":0-1,"tone":0-1}]  // optional, built by the pipeline from turns like "then the fog lifted"
  },
  "mood": ["fear","calm","sad","joy","uncanny","longing"],
  "scenes": [ { "text": "...", "setting": "corridor|room|water|city|forest|field|mountains|open|underwater",
                "elements": ["moon","sun","stars","house","door","stairs","figure","crowd","birds","rain","snow","trees","road","glow"],
                "light": "...", "motion": "drift|falling|flying|chase|recede", "duration_s": 6 } ],   // 1-4 scenes
  "cues": [ { "kind": "clarity|style|mood|setting|motion|light", "phrase": "hazy", "effect": "more fog", "at": [start,end] } ],
  "prompt": "text-to-image/video prompt built from scenes + look",
  "negative_prompt": "cartoon, illustration, crisp HDR, ...",
  "overrides": { /* look keys the dreamer tuned by hand; these win over look */ }
}
```

Effective look = `{...look, ...overrides}`.

## Renderer interface

```ts
interface DreamRenderer { id: string; label: string;
  render(spec, {onProgress, signal}): Promise<RenderResult> }
type RenderResult = { kind: "video", src: string } | { kind: "image", src: string }
                  | { kind: "live", start(canvas), stop(), update(spec) }   // the in-browser placeholder
```

The app tries `PipelineRenderer` first (POST spec JSON to `endpoint`, expects `{kind:"video"|"image", src}`),
and falls back to the in-browser placeholder. Setting `PipelineRenderer.endpoint` is the only change needed to plug in.

## Getting real videos into the app (updated)

The published artifact runs in a sandbox that can only load its own files: no fetch or video from other hosts.
So an HTTP endpoint can't be reached from the artifact itself. Two routes are wired in:

1. **Pre-rendered clips (works today).** Publish clips beside the page and list them in `renders/index.json`:
   `{ "<spec.id>" or "t_<fnv1a hash of transcript, base36>": { "src": "renders/<id>/clip.mp4", "poster": "renders/<id>/poster.jpg" } }`.
   The three built-in examples have fixed ids: d_1pyxr9f (hallway → sea), d_1poy5kg (noir city chase → falling), d_1qiwynd (clear field at sunset).
   When a dream matches, the app shows the rendered video with a "Rendered video / Live preview" switch.
2. **Async job API** (for a page hosted outside the sandbox): `POST {endpoint}/jobs` with the spec → `{job_id}`;
   `GET {endpoint}/jobs/{id}` → `{status: queued|running|done|failed, progress?, eta_s?, result?: {src, poster}, error?}`. Polls every 5 s.

## People and faces (added 2026-11-25)

```jsonc
"people": [ {
  "id": "self" | "p_<n>" | "k_<relation>" | "s_<n>" | "c_<n>",   // your People list / "my dad" with no list entry / stranger / found only by Claude
  "who": "my dad",                     // as the dreamer said it
  "ref": "p_1" | null,                 // People-list entry when matched
  "relation": "father|mother|sister|...|stranger|self|other",
  "identity": "known_clear" | "known_different" | "known_unclear" | "obscured" | "stranger" | "self",
  "identity_source": "said" | "claude" | "default",
  "face_clarity": 0-1,                 // how far the face resolves through the fog (clear .9, different .75, stranger .35, unclear .2, hidden .08)
  "emotion": "neutral|joy|sad|fear|anger|surprise" | null,
  "minor": true | false | null,        // null = unknown; the content-rating stage treats true as PG-only
  "likeness": "photo" | "generic",     // photo only with a photo on file AND the person's consent ticked AND not a minor
  "photo_ref": "local:p_1" | null,     // the image never travels inside the spec; the renderer fetches it separately
  "scenes": [0], "mentions": [ {"phrase": "my dad", "at": [s, e]} ]
} ],
"dreamer": { "seen": false, "as_child": false, "emotion": "fear" | null, "photo_ref": "local:self" | null },
"scenes": [ { ..., "people": ["p_1", "s_1"] } ]
```

- identity states: **known_clear** it's them, face seen; **known_different** you knew it was them but the face was someone else's
  (render: keep hair, build, clothes, voice cues; swap the face for a different one); **known_unclear** "can't remember her face"
  (face dissolves, never resolves); **obscured** face hidden, turned away, in shadow; **stranger** someone you don't know.
- `dreamer.as_child` ("I was a kid again") holds the whole dream to PG (rating thread rule).
- `dreamer.emotion` + `dreamer.photo_ref` feed reaction-shot cuts used as a content mitigation.
- Face-specific rating rules (agreed with the rating thread): an uploaded likeness is never used in sexual or violent-victim content
  (rater emits a `generic_face` mitigation naming the people id); photos of under-18s are never accepted.
- Phrases about a face ("couldn't see his face", "can't remember her face") are people cues and no longer thicken the scene fog.
