"""Faces demo: one photo -> dream identity states and expressions, composited into a rendered dream scene.

python3 faces_demo.py [--photo me.jpg] [--scene hallway] [--frame 6] --out out/faces
Without --photo it uses scikit-image's public-domain NASA test portrait (astronaut.png).
Writes: states_raw.jpg, states_dream.jpg (light fog), states_dark.jpg (dark fog), expressions.jpg, scores.json.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dreamrender import faces as F                     # noqa: E402
from dreamrender.stylize import load_beauty, load_depth, stylize, save   # noqa: E402

FACE_FOG = {          # how the dream fog treats each identity state (see FACES.md, "fog and faces")
    "known_clear": dict(island=True, face_fog=0.0),     # attention island sits on the face
    "known_different": dict(island=True, face_fog=0.15),
    "stranger": dict(island=False, face_fog=0.45),      # the face never comes into focus
    "obscured": dict(island=False, face_fog=0.3),
}
LABEL = {"known_clear": "known, clear (sad)", "known_different": "known, different face",
         "stranger": "stranger / can't recall face", "obscured": "known, face hidden"}


def label(img, text):
    im = F._u8(img).copy()
    cv2.rectangle(im, (0, im.shape[0] - 34), (im.shape[1], im.shape[0]), (0, 0, 0), -1)
    cv2.putText(im, text, (12, im.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo")
    ap.add_argument("--scene", default="hallway")
    ap.add_argument("--frame", type=int, default=6)
    ap.add_argument("--emotion", default="sad")
    ap.add_argument("--out", default="out/faces")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.photo:
        photo = cv2.cvtColor(cv2.imread(a.photo), cv2.COLOR_BGR2RGB)
    else:
        import skimage.data
        photo = skimage.data.astronaut()
    s = 1024 / max(photo.shape[:2])
    photo = cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)

    kit = F.FaceKit()
    face = kit.landmarks(photo)
    if face is None:
        sys.exit("no face found in the photo")
    ref = kit.identity(photo, face)
    masks = kit.person_mask(photo)
    pmask = cv2.resize(masks["person"], photo.shape[1::-1])
    pmask = np.clip((pmask - 0.45) / 0.3, 0, 1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((pmask > 0.5).astype(np.uint8))
    if n > 1:                                   # keep the largest blob (the person), drop background leaks
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pmask = pmask * cv2.GaussianBlur((lab == keep).astype(np.float32), (0, 0), 3)
    frames = sorted(glob.glob(f"out/{a.scene}/frames/f*.png"))
    fpng = frames[min(a.frame, len(frames) - 1)]
    bg, bgd = load_beauty(fpng), load_depth(fpng.replace(".png", "_depth.exr"))

    scores, raw_row, dream_row, dark_row = {}, [], [], []
    for st in F.STATES:
        emo = a.emotion if st in ("known_clear", "known_different") else "neutral"
        amt = 2.2
        p = F.express(photo, face, emo, amt) if st in ("known_clear", "known_different") else F._f32(photo)
        if st == "known_different":
            p = F.different_face(p, face, 2.0)
        elif st == "stranger":
            p = F.stranger_face(p, face)
        elif st == "obscured":
            p = F.obscured_face(p, face)
        p = F._f32(p)
        sim = F.cosine(ref, kit.identity(p, face))
        scores[st] = round(sim, 3)

        # extra fog over the face for states where it should not resolve
        if FACE_FOG[st]["face_fog"] > 0:
            m = F.face_mask(face, p.shape, 1.1, 0.8)[..., None] * FACE_FOG[st]["face_fog"]
            p = p * (1 - m) + cv2.GaussianBlur(p, (0, 0), face.iod * 0.2) * m
        img, dep, (x0, y0, pw, ph) = F.composite(bg, bgd, p, pmask, x=0.7, height=0.95, dist=2.0)
        fx = (x0 + face.center[0] * pw / p.shape[1]) / bg.shape[1]
        fy = (y0 + face.center[1] * ph / p.shape[0]) / bg.shape[0]
        focus = (fx, fy) if FACE_FOG[st]["island"] else None
        look = dict(fog=0.5, style="natural", saturation=0.85, grain=0.3, vignette=0.5, mood="eerie",
                    time_of_day="night", fog_tone=0.5)
        light = stylize(img, dep, look, t=1.0, seed=7, frame_index=a.frame, focus=focus)
        dark = stylize(img, dep, dict(look, fog=0.6, fog_tone=0.9, mood="fear"), t=1.0, seed=7,
                       frame_index=a.frame, focus=focus)
        raw_row.append(label(img, LABEL[st] + f"  ID {sim:.2f}"))
        dream_row.append(label(light, LABEL[st]))
        dark_row.append(label(dark, LABEL[st] + "  dark fog"))

    def grid(row):
        return np.vstack([np.hstack(row[:2]), np.hstack(row[2:])])
    for name, row in (("states_raw", raw_row), ("states_dream", dream_row), ("states_dark", dark_row)):
        cv2.imwrite(f"{a.out}/{name}.jpg", cv2.cvtColor(grid(row), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # expressions close-up, with what MediaPipe's own blendshapes read back from the warped photo
    cx, cy = face.center.astype(int)
    r = int(face.iod * 2.3)
    tiles, expr = [], {}
    for emo in F.EMOTIONS:
        o = F.express(photo, face, emo, 2.2)
        fo = kit.landmarks(o)
        top = sorted(fo.blend.items(), key=lambda kv: -kv[1])[:3] if fo else []
        expr[emo] = dict(id=round(F.cosine(ref, kit.identity(o)), 3), blendshapes={k: round(v, 2) for k, v in top})
        crop = F._u8(o)[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        tiles.append(label(cv2.resize(crop, (320, 320)), emo))
    cv2.imwrite(f"{a.out}/expressions.jpg", cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(dict(photo=a.photo or "skimage astronaut (public domain, NASA)", scene=fpng, identity=scores,
                   expressions=expr, mouth_open=round(F.mouth_open(face), 3)),
              open(f"{a.out}/scores.json", "w"), indent=2)
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
