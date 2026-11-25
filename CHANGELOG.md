# Changelog

Night Reels versions are plain numbers: v1, v2, v3… Each version is pushed to `main`.

## v1 — 2026-11-25

First commit of everything built so far.

- **App** (`app/dream-log.html`): single-page dream journal. Type or speak a dream, get a dream spec, an in-browser fog
  preview, a People panel for your own photo and people you know, and a "does this feel like your dream?" check.
  Storage is deliberately in-memory only until tests with real people.
- **Dream spec v1** (`docs/dream-spec.md`): the hand-off from the app to the renderer (look, scenes, people, content rating).
- **Render pipeline** (`render/`): text or app spec → procedural Blender scenes → depth-aware dream fog → low-fps clip.
  Fog amount, fog tone (pale to black) and fog that shifts mid-dream.
- **Content rating** (`render/dreamrender/rating.py`, `mitigate.py`, `CONTENT_POLICY.md`): Australian M target,
  PG with children, abort on any child-related sexual or violent content; censor bars, reaction-face cuts and fog veils.
  47 tests.
- **Faces prototype** (`render/dreamrender/faces.py`): known / different-face / unclear / obscured identity states from one photo.
- **Research** (`research/dream-look/`): the reference dream-look renderer behind the clarity dial.
