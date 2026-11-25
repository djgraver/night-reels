"""Mitigation renders for the content rating stage (see rating.py and CONTENT_POLICY.md).

  veil_look(look, mits)        fade_to_fog / dark_fog -> a look["timeline"] that swells to thick (dark) fog
                               at the moment to hide (stylize.look_at_time interpolates it per frame)
  apply_to_clip(mp4, ...)      censor_bar   -> comical tilted black bar + mosaic over the subject
                               reaction_cut -> cut to a fogged close-up of the dreamer reacting
  apply_to_still(jpg, ...)     the same for poster / dream stills

Clip post-processing works on the finished per-scene mp4 (decode -> overlay -> encode), so it does not
depend on how animate.py builds clips.
"""
from __future__ import annotations
import json, math, os, subprocess
import numpy as np
import cv2

CONTAINER_FPS = 24


# =========================================================================== fog veil (pre-render)
def veil_look(look: dict, mits: list) -> dict:
    """Thick fog over the hidden part of the scene. dark_fog / sexual violence -> black smoke; others -> the scene's own fog."""
    veils = [m for m in mits if m["kind"] in ("fade_to_fog", "dark_fog")]
    if not veils:
        return look
    start = min(m.get("at", [0.3, 1])[0] for m in veils)
    tone = max(float(m.get("tone", 0.5)) for m in veils)
    base_fog, base_tone = float(look.get("fog", 0.55)), float(look.get("fog_tone", 0.5))
    keys = [k for k in (look.get("timeline") or []) if k["at"] < start]
    if not keys:
        keys = [{"at": 0.0, "fog": base_fog, "tone": base_tone}]
    k0 = dict(keys[-1]); k0["at"] = max(start - 0.02, 0.0)
    swell = min(start + 0.15, 1.0)
    keys += [k0, {"at": swell, "fog": 0.97, "tone": max(tone, k0["tone"])}, {"at": 1.0, "fog": 0.97, "tone": max(tone, k0["tone"])}]
    out = dict(look, timeline=keys)
    if start <= 0.0:                     # the whole scene is hidden: start in the veil
        out.update(fog=0.97, fog_tone=max(tone, base_tone))
    return out


# =========================================================================== censor bar
def _text(img, s, center, scale, thick, color):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def draw_censor_bar(frame: np.ndarray, focus, u: float, seed: int = 0, label: str = "CENSORED") -> np.ndarray:
    """frame: uint8 HxWx3 RGB. u: seconds since the bar appeared (pop-in + wobble).
    Comical rather than clinical: a chunky tilted bar that pops in with an overshoot and wobbles slightly."""
    h, w = frame.shape[:2]
    fx, fy = focus if focus else (0.5, 0.55)
    cx, cy = fx * w, min(fy + 0.06, 0.9) * h
    out = frame.copy()
    # 1) mosaic behind the bar, soft-edged, so nothing reads around it
    mw, mh = int(0.30 * w), int(0.46 * h)
    x0, y0 = int(max(cx - mw / 2, 0)), int(max(cy - mh / 2, 0))
    x1, y1 = int(min(cx + mw / 2, w)), int(min(cy + mh / 2, h))
    if x1 - x0 > 8 and y1 - y0 > 8:
        roi = out[y0:y1, x0:x1]
        cell = max(6, w // 48)
        small = cv2.resize(roi, (max(1, roi.shape[1] // cell), max(1, roi.shape[0] // cell)), interpolation=cv2.INTER_AREA)
        mos = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]].astype(np.float32)
        e = ((xx - roi.shape[1] / 2) / (roi.shape[1] / 2)) ** 2 + ((yy - roi.shape[0] / 2) / (roi.shape[0] / 2)) ** 2
        a = np.clip((1.15 - e) / 0.3, 0, 1)[..., None]
        out[y0:y1, x0:x1] = (mos * a + roi * (1 - a)).astype(np.uint8)
    # 2) the bar: pop-in scale with overshoot, slight wobble, drop shadow
    pop = 1.0 if u > 0.35 else (1 - math.cos(u / 0.35 * math.pi * 1.5) * math.exp(-u * 9)) * min(1, u / 0.12 + 0.2)
    ang = -7 + 2.2 * math.sin(u * 5.3 + seed)
    bw, bh = 0.36 * w * pop, 0.105 * h * pop
    if bw < 4:
        return out
    layer = np.zeros((h, w, 3), np.uint8); mask = np.zeros((h, w), np.uint8)
    box = cv2.boxPoints(((cx, cy), (bw, bh), ang)).astype(np.int32)
    shadow = box + np.array([int(0.006 * w), int(0.01 * h)])
    sh = out.copy(); cv2.fillConvexPoly(sh, shadow, (0, 0, 0), cv2.LINE_AA)
    out = cv2.addWeighted(sh, 0.35, out, 0.65, 0)
    cv2.fillConvexPoly(layer, box, (12, 12, 14), cv2.LINE_AA)
    cv2.fillConvexPoly(mask, box, 255, cv2.LINE_AA)
    # text drawn upright on its own canvas, then rotated with the bar
    tcan = np.zeros((h, w, 3), np.uint8)
    _text(tcan, label, (cx, cy), 1.25 * pop * w / 960, max(1, int(2.5 * w / 960)), (245, 245, 240))
    M = cv2.getRotationMatrix2D((cx, cy), -ang, 1.0)
    tcan = cv2.warpAffine(tcan, M, (w, h))
    layer = np.maximum(layer, tcan)
    m = (mask.astype(np.float32) / 255)[..., None] * 0.96
    return (layer * m + out * (1 - m)).astype(np.uint8)


# =========================================================================== reaction shot
def reaction_plate(w, h, bg_rgb=(0.25, 0.27, 0.32), emotion="shock", face_img=None):
    """A plain close-up of the dreamer as a beauty frame (float RGB 0..1) + depth, ready for stylize().
    face_img: optional uploaded photo of the dreamer (float RGB); otherwise a soft, anonymous head
    whose wide eyes and open mouth read as a reaction once the dream fog is on it."""
    bg = np.ones((h, w, 3), np.float32) * np.array(bg_rgb, np.float32) * 0.6
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.full((h, w), 9.0, np.float32)
    cx, cy = 0.5 * w, 0.5 * h
    if face_img is not None:
        fh, fw = face_img.shape[:2]
        s = max(h * 0.95 / fh, w * 0.5 / fw)
        fi = cv2.resize(face_img, (int(fw * s), int(fh * s)), interpolation=cv2.INTER_AREA)
        x0, y0 = int(cx - fi.shape[1] / 2), int(cy - fi.shape[0] / 2)
        xs, ys = max(x0, 0), max(y0, 0)
        xe, ye = min(x0 + fi.shape[1], w), min(y0 + fi.shape[0], h)
        e = ((xx - cx) / (0.22 * w)) ** 2 + ((yy - cy) / (0.48 * h)) ** 2
        a = np.clip((1.2 - e) / 0.4, 0, 1)[..., None]
        canvas = bg.copy(); canvas[ys:ye, xs:xe] = fi[ys - y0:ye - y0, xs - x0:xe - x0]
        img = canvas * a + bg * (1 - a)
        depth = np.where(a[..., 0] > 0.3, 1.2, 9.0).astype(np.float32)
        return np.clip(img, 0, 1), depth
    skin = np.array([0.62, 0.50, 0.43], np.float32)
    hair = np.array([0.10, 0.08, 0.07], np.float32)
    img = bg.copy()
    # shoulders
    sh = ((xx - cx) / (0.36 * w)) ** 2 + ((yy - (cy + 0.62 * h)) / (0.30 * h)) ** 2 < 1
    img[sh] = np.array([0.16, 0.17, 0.2], np.float32); depth[sh] = 1.6
    neck = (np.abs(xx - cx) < 0.045 * w) & (yy > cy + 0.18 * h) & (yy < cy + 0.42 * h)
    img[neck] = skin * 0.7; depth[neck] = 1.4
    # head, lit from upper left
    rx, ry = 0.105 * w, 0.30 * h
    e = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    head = e < 1
    light = np.clip(1.05 - 0.55 * (((xx - (cx - 0.5 * rx)) / (2 * rx)) ** 2 + ((yy - (cy - 0.5 * ry)) / (2 * ry)) ** 2), 0.35, 1.0)
    img[head] = (skin[None] * light[head][:, None])
    depth[head] = 1.2
    hcap = head & (yy < cy - 0.42 * ry + 0.25 * ry * ((xx - cx) / rx) ** 2)
    img[hcap] = hair; img[(e < 1.12) & (e >= 1) & (yy < cy)] = hair
    # features
    def ell(c, ax, col, ang=0, th=-1):
        cv2.ellipse(img, (int(c[0]), int(c[1])), (max(1, int(ax[0])), max(1, int(ax[1]))), ang, 0, 360, tuple(float(v) for v in col), th, cv2.LINE_AA)
    wide = {"shock": 1.0, "fear": 1.1, "sad": 0.7}.get(emotion, 1.0)
    for sgn in (-1, 1):
        ex, ey = cx + sgn * 0.42 * rx, cy - 0.08 * ry
        ell((ex, ey), (0.2 * rx, 0.085 * ry * wide), (0.86, 0.84, 0.8))
        ell((ex + sgn * 0.02 * rx, ey), (0.085 * rx, 0.075 * ry * min(wide, 1)), (0.12, 0.1, 0.09))
        ell((ex - 0.03 * rx, ey - 0.02 * ry), (0.025 * rx, 0.02 * ry), (0.95, 0.95, 0.95))
        bx, by = ex, ey - (0.2 if emotion != "sad" else 0.15) * ry
        cv2.ellipse(img, (int(bx), int(by)), (int(0.22 * rx), int(0.05 * ry)), 0, 200, 340, (0.14, 0.11, 0.1), max(2, int(w / 300)), cv2.LINE_AA)
    cv2.line(img, (int(cx), int(cy + 0.02 * ry)), (int(cx - 0.05 * rx), int(cy + 0.22 * ry)), tuple(float(v) for v in skin * 0.6), max(1, int(w / 480)), cv2.LINE_AA)
    mo = {"shock": (0.13, 0.11), "fear": (0.16, 0.08), "sad": (0.14, 0.03)}.get(emotion, (0.13, 0.11))
    ell((cx, cy + 0.45 * ry), (mo[0] * rx, mo[1] * ry), (0.18, 0.08, 0.08))
    img = cv2.GaussianBlur(img, (0, 0), w / 700)
    return np.clip(img, 0, 1), depth


def reaction_frames(n, w, h, look, seed=0, bg_rgb=(0.25, 0.27, 0.32), emotion="shock", face_img=None, t0=0.0):
    """n stylized uint8 frames of a slow push-in on the reaction plate, at least half-veiled in fog."""
    from .stylize import stylize
    try:
        from .stylize import look_at_time
    except ImportError:
        look_at_time = lambda l, u: l
    plate, depth = reaction_plate(w, h, bg_rgb, emotion, face_img)
    lk = dict(look); lk.pop("timeline", None)
    lk["fog"] = float(np.clip(max(lk.get("fog", 0.55), 0.5), 0, 0.8))
    fps = float(np.clip(look.get("fps", 8), 4, 24))
    out, cache_t = [], None
    for j in range(n):
        t = t0 + j / CONTAINER_FPS
        tq = math.floor(t * fps) / fps                       # the picture updates at the dream's fps
        if tq != cache_t:
            s = 1.0 + 0.06 * (j / max(n - 1, 1))
            M = cv2.getRotationMatrix2D((w / 2, h * 0.47), 0, s)
            im = cv2.warpAffine(plate, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            dp = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            fr = stylize(im, dp, lk, t=tq, seed=seed + 7, frame_index=j, focus=(0.5, 0.42))
            cur = (np.clip(fr, 0, 1) * 255).astype(np.uint8); cache_t = tq
        out.append(cur)
    return out


def veil_frame(fr: np.ndarray, a: float, tone: float = 0.5) -> np.ndarray:
    """Swallow a frame into fog: heavy blur, contrast crushed toward the fog colour (tone 1 = black smoke)."""
    h, w = fr.shape[:2]
    f = fr.astype(np.float32)
    small = cv2.resize(f, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_AREA)
    blur = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.2), (w, h), interpolation=cv2.INTER_CUBIC)
    mean = f.reshape(-1, 3).mean(0)
    fog = mean * (1.0 - 0.85 * max(tone - 0.5, 0) * 2) + (1 - tone) * 30
    target = blur * 0.18 + fog * 0.82
    return np.clip(f * (1 - a) + target * a, 0, 255).astype(np.uint8)


def load_face(ref):
    """dreamer.photo_ref ("local:self") -> float RGB image, if the host saved it as $DREAM_FACES_DIR/<id>.jpg|png.
    Photos never travel inside the spec; without one the anonymous reaction plate is used."""
    d = os.environ.get("DREAM_FACES_DIR")
    if not ref or not d:
        return None
    key = os.path.basename(str(ref).split(":", 1)[-1])
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = os.path.join(d, key + ext)
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return None


# =========================================================================== clip post-processing
def _probe(mp4):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames",
                        "-of", "json", mp4], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s.get("nb_frames") or 0)


def _focus_track(frames_dir, n):
    try:
        fm = json.load(open(os.path.join(frames_dir, "render_meta.json")))["frames"]
        pts = [f.get("focus") for f in fm if f.get("focus")]
    except Exception:
        pts = []
    if not pts:
        return [None] * n
    return [pts[min(int(j / max(n - 1, 1) * (len(pts) - 1) + 0.5), len(pts) - 1)] for j in range(n)]


def apply_to_clip(in_mp4, out_mp4, mits, look, frames_dir=None, seed=0, face_img=None):
    """Censor bars and reaction cuts on a finished scene clip. Returns a list of what was applied."""
    bars = [m for m in mits if m["kind"] == "censor_bar"]
    cuts = [m for m in mits if m["kind"] == "reaction_cut"]
    veils = [m for m in mits if m["kind"] in ("fade_to_fog", "dark_fog")]
    if not bars and not cuts and not veils:
        return []
    w, h, _ = _probe(in_mp4)
    dec = subprocess.run(["ffmpeg", "-v", "error", "-i", in_mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(dec, np.uint8).reshape(-1, h, w, 3)
    n = len(frames)
    focus = _focus_track(frames_dir, n) if frames_dir else [None] * n
    bar_from = int(min((m.get("at", [0.15, 1])[0] for m in bars), default=2) * n) if bars else n
    cut_from = int(min((m.get("at", [0.45, 1])[0] for m in cuts), default=2) * n) if cuts else n
    veil_from = int(min((m.get("at", [0.3, 1])[0] for m in veils), default=2) * n) if veils else n
    veil_tone = max((float(m.get("tone", 0.5)) for m in veils), default=0.5)
    react = []
    if cuts:
        tail = frames[max(cut_from - 1, 0)].astype(np.float32) / 255
        bg = cv2.resize(tail, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1, 3).mean(0)
        felt = next((m.get("emotion") for m in cuts if m.get("emotion")), None)        # dreamer.emotion from the spec
        emotion = {"fear": "fear", "sad": "sad"}.get(felt) or ("fear" if any("sexual" in m["reason"] or "gore" in m["reason"] for m in cuts) else "shock")
        if face_img is None:
            face_img = load_face(next((m.get("face_ref") for m in cuts if m.get("face_ref")), None))
        react = reaction_frames(n - cut_from, w, h, look, seed=seed, bg_rgb=tuple(bg), emotion=emotion, face_img=face_img)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", str(CONTAINER_FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4], stdin=subprocess.PIPE)
    xf = 5                                   # cross-dissolve frames into the reaction (a dream cut, not a hard edit)
    for j in range(n):
        fr = frames[j]
        if j >= bar_from and j < cut_from + xf:
            fr = draw_censor_bar(fr, focus[j], (j - bar_from) / CONTAINER_FPS, seed)
        if j >= cut_from:
            a = min(1.0, (j - cut_from + 1) / xf)
            fr = (react[j - cut_from] * a + fr * (1 - a)).astype(np.uint8)
        if j >= veil_from:                     # guarantee: the hidden part is unreadable whatever the fog stage kept
            fr = veil_frame(fr, min(1.0, (j - veil_from + 1) / (0.6 * CONTAINER_FPS)), veil_tone)
        enc.stdin.write(np.ascontiguousarray(fr).tobytes())
    enc.stdin.close(); enc.wait()
    done = []
    if veils:
        done.append(dict(kind="veil", from_frame=veil_from))
    if bars:
        done.append(dict(kind="censor_bar", from_frame=bar_from))
    if cuts:
        done.append(dict(kind="reaction_cut", from_frame=cut_from))
    return done


def apply_to_still(path, mits, look, focus=None, seed=0, face_img=None, u=0.5):
    """Posters and stills: a still at position u gets the bar or the reaction if that moment is hidden."""
    if not os.path.exists(path):
        return
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    cuts = [m for m in mits if m["kind"] == "reaction_cut" and m.get("at", [0.45])[0] <= u]
    bars = [m for m in mits if m["kind"] == "censor_bar"]
    veils = [m for m in mits if m["kind"] in ("fade_to_fog", "dark_fog") and m.get("at", [0.3])[0] <= u]
    if veils:
        im = veil_frame(im, 1.0, max(float(m.get("tone", 0.5)) for m in veils))
    elif cuts:
        bg = cv2.resize(im.astype(np.float32) / 255, (8, 8)).reshape(-1, 3).mean(0)
        im = reaction_frames(1, w, h, look, seed=seed, bg_rgb=tuple(bg), face_img=face_img)[0]
    elif bars:
        im = draw_censor_bar(im, focus, 1.0, seed)
    else:
        return
    cv2.imwrite(path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
