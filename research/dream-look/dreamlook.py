"""Reference "dream look" renderer.

Turns a sharp, realistic frame into a dream-fogged one, driven by a small set of
parameters that a dream-report analyser can fill in (clarity, style, mood, lucid,
nightmare). Research basis for each stage is in the report; short notes inline.

Stages (per frame):
  1. depth      monocular depth (MiDaS v2.1 small, ONNX) -> distance 0 near .. 1 far
  2. fog        atmospheric scattering I = J*t + A*(1-t), t = exp(-beta*d)  (Koschmieder)
  3. periphery  background + edge detail loss: blur & smooth by depth and radius
                (dream imagery lacks "colour saturation, background detail and focus")
  4. colour     desaturate with fog; explicit styles (bw, sepia, vivid, ...)
  5. glow       highlight bloom + warm halation (black-mist / old-lens look)
  6. tone       lifted blacks, lower contrast, mood tint
  7. vignette   darken + soften edges
  8. grain      luminance grain, re-rolled per held frame
Temporal (video): low output frame rate via holds ("on threes"), step-print smear,
gate weave, exposure breathing, drifting fog density.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
import onnxruntime as ort

# --------------------------------------------------------------------------- params


@dataclass
class DreamParams:
    clarity: float = 0.3          # 0 = barely remembered, 1 = "crystal clear, like real life"
    style: str = "natural"        # natural | bw | sepia | vivid | faded | noir | nightvision
    mood: str = "neutral"         # neutral | calm | warm | eerie | sad | dread
    lucid: bool = False           # lucid dreams are reported brighter, clearer, more saturated
    nightmare: bool = False
    indoor: bool = False          # relative depth is normalised per image, so rooms need less fog
    seed: int = 7
    focus: tuple = (0.5, 0.5)     # where the dreamer's attention sits (x, y in 0..1)
    focus_radius: float = 0.38    # size of the clearer "attention island"

    # Derived knobs (filled by resolve()). Exposed so a pipeline can override any.
    fog: float = field(default=0.0, init=False)
    fog_beta: float = field(default=0.0, init=False)
    base_haze: float = field(default=0.0, init=False)
    periph_blur: float = field(default=0.0, init=False)
    saturation: float = field(default=1.0, init=False)
    bloom: float = field(default=0.0, init=False)
    halation: float = field(default=0.0, init=False)
    contrast: float = field(default=1.0, init=False)
    black_lift: float = field(default=0.0, init=False)
    exposure: float = field(default=0.0, init=False)
    vignette: float = field(default=0.0, init=False)
    grain: float = field(default=0.0, init=False)
    out_fps: float = field(default=24.0, init=False)
    smear: float = field(default=0.0, init=False)
    weave_px: float = field(default=0.0, init=False)
    tint: tuple = field(default=(1.0, 1.0, 1.0), init=False)
    air: tuple = field(default=(0.86, 0.87, 0.88), init=False)

    def resolve(self) -> "DreamParams":
        c = float(np.clip(self.clarity, 0, 1))
        if self.lucid:
            c = max(c, 0.75)
        f = 1.0 - c                              # fog amount 0..1
        self.fog = f
        fe = f ** 1.4                            # perceptual easing: mid clarity stays readable
        self.fog_beta = 2.3 * fe                 # depth-scaled density
        self.base_haze = 0.14 * fe               # uniform veil, even on near objects
        if self.indoor:
            self.fog_beta *= 0.55
            self.base_haze *= 0.7
        self.periph_blur = f                     # 0..1 strength of edge/background loss
        self.saturation = 1.0 - 0.45 * f         # up to -45% chroma
        self.bloom = 0.10 + 0.45 * f
        self.halation = 0.05 + 0.20 * f
        self.contrast = 1.0 - 0.30 * f
        self.black_lift = 0.08 * f
        self.exposure = 0.0
        self.vignette = 0.15 + 0.30 * f
        self.grain = 0.008 + 0.020 * f
        # Temporal: clear dreams move closer to normal frame rate.
        self.out_fps = 24 / (1 + round(2 * f + 0.4))   # 24 -> 12 -> 8 fps
        self.smear = 0.15 + 0.35 * f
        self.weave_px = 0.4 + 1.6 * f
        self.tint = (1.0, 1.0, 1.0)
        self.air = (0.86, 0.87, 0.88)

        if self.lucid:
            self.exposure += 0.06
            self.saturation = min(1.25, self.saturation + 0.25)
            self.bloom += 0.1

        mood_tints = {
            "neutral": (1.00, 1.00, 1.00),
            "calm": (0.97, 1.00, 1.04),
            "warm": (1.06, 1.00, 0.90),
            "eerie": (0.94, 1.03, 0.98),
            "sad": (0.95, 0.98, 1.06),
            "dread": (0.98, 0.95, 1.02),
        }
        self.tint = mood_tints.get(self.mood, (1, 1, 1))
        if self.mood == "warm":
            self.air = (0.93, 0.87, 0.78)
        elif self.mood in ("sad", "calm"):
            self.air = (0.80, 0.85, 0.90)
        elif self.mood == "eerie":
            self.air = (0.78, 0.84, 0.80)

        if self.nightmare or self.mood == "dread":
            self.exposure -= 0.10
            self.contrast += 0.15
            self.vignette += 0.20
            self.air = (0.30, 0.32, 0.36)        # dark smoke instead of bright mist
            self.saturation *= 0.8
            self.weave_px += 0.8

        if self.style == "bw":
            self.saturation = 0.0
            self.grain *= 1.4
            self.contrast += 0.10
            self.halation = 0.0
        elif self.style == "sepia":
            self.saturation = 0.0
            self.grain *= 1.3
        elif self.style == "vivid":
            self.saturation = max(self.saturation, 1.15)
        elif self.style == "faded":
            self.saturation *= 0.7
            self.black_lift += 0.06
        elif self.style == "noir":
            self.saturation = 0.0
            self.contrast += 0.35
            self.black_lift = 0.0
            self.vignette += 0.2
            self.air = (0.55, 0.55, 0.55)
        return self

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- depth

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class Depth:
    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb float32 0..1 HxWx3 -> distance map 0 (near) .. 1 (far)."""
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        inv = self.sess.run(None, {self.inp: x})[0].reshape(256, 256)
        inv = cv2.resize(inv, (w, h), interpolation=cv2.INTER_CUBIC)
        lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
        near = np.clip((inv - lo) / (hi - lo + 1e-6), 0, 1)
        dist = 1.0 - near
        dist = cv2.GaussianBlur(dist, (0, 0), max(1.0, w / 300))
        return dist.astype(np.float32)


# --------------------------------------------------------------------------- helpers


def _lum(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _radial(h, w, cx=0.5, cy=0.45):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / w - cx) * (w / max(h, w)) * 2
    ny = (yy / h - cy) * (h / max(h, w)) * 2
    return np.sqrt(nx * nx + ny * ny) / math.sqrt(2) * 1.6  # ~0 centre .. ~1 corners


def _blur(img, sigma):
    if sigma < 0.3:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def _noise_field(h, w, scale, rng, t=0.0):
    """Cheap smooth noise for fog density variation (drifts with t)."""
    gh, gw = max(2, int(h / scale)) + 2, max(2, int(w / scale)) + 2
    base = rng.random((gh, gw)).astype(np.float32)
    big = cv2.resize(base, (w + int(scale * 2), h + int(scale * 2)), interpolation=cv2.INTER_CUBIC)
    ox = int((t * 0.6) % scale)
    return big[0:h, ox:ox + w]


# --------------------------------------------------------------------------- render


def render(rgb: np.ndarray, p: DreamParams, dist: np.ndarray | None = None, *,
           rng: np.random.Generator | None = None, t: float = 0.0,
           fog_field: np.ndarray | None = None) -> np.ndarray:
    """rgb float32 0..1. Returns float32 0..1."""
    rng = rng or np.random.default_rng(p.seed)
    h, w = rgb.shape[:2]
    img = rgb.astype(np.float32).copy()
    if dist is None:
        dist = np.full((h, w), 0.5, np.float32)
    rad = _radial(h, w)
    # attention island: what the dreamer looks at stays clearer (attention stabilises
    # dream imagery); fog and detail loss grow away from it.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy = p.focus
    rr = np.sqrt(((xx / w - fx) * (w / h)) ** 2 + (yy / h - fy) ** 2)
    att = np.exp(-(rr / p.focus_radius) ** 2).astype(np.float32)

    # 3. periphery: background & edge detail loss (before fog so fog stays smooth)
    if p.periph_blur > 0:
        s = p.periph_blur
        soft = cv2.bilateralFilter(cv2.resize(img, (w // 2, h // 2)), 9, 0.12, 6)
        soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
        soft = _blur(soft, w * 0.006 * s)
        far = _blur(img, w * 0.014 * s)
        m_soft = (np.clip(0.35 * s + 0.4 * dist * s, 0, 1) * (1 - 0.5 * att))[..., None]
        m_far = (np.clip((dist * 0.8 + np.clip(rr - 0.3, 0, 1) * 1.3) * s, 0, 1) ** 1.3) * (1 - 0.85 * att)
        img = img * (1 - m_soft) + soft * m_soft
        img = img * (1 - m_far[..., None]) + far * m_far[..., None]

    # 2. fog (Koschmieder): transmission falls with distance
    if p.fog > 0:
        d = dist if fog_field is None else np.clip(dist * (0.75 + 0.5 * fog_field), 0, 1.3)
        trans = np.exp(-p.fog_beta * d * (1 - 0.55 * att)) * (1 - p.base_haze * (1 - 0.5 * att))
        A = np.array(p.air, np.float32)
        # air light picks up a little of the scene's own colour, so fog isn't flat grey
        scene = _blur(img, w * 0.08).reshape(-1, 3).mean(0)
        A = 0.75 * A + 0.25 * scene
        # keep the veil from glowing brighter than the scene's own highlights
        A = np.minimum(A, np.percentile(img, 90, axis=(0, 1)) * 0.95 + 0.04)
        img = img * trans[..., None] + A * (1 - trans[..., None])

    # 4. colour
    L = _lum(img)[..., None]
    img = L + (img - L) * p.saturation
    if p.style == "sepia":
        img = np.concatenate([L * 1.07, L * 0.95, L * 0.78], axis=2)
    img = img * np.array(p.tint, np.float32)

    # 5. glow: bloom + halation
    if p.bloom > 0:
        Ln = _lum(img)
        hi = np.clip((Ln - 0.62) / 0.38, 0, 1)[..., None] * img
        bloom = _blur(hi, w * 0.025) * 0.6 + _blur(hi, w * 0.008) * 0.4
        img = 1 - (1 - img) * (1 - np.clip(bloom * p.bloom * 1.6, 0, 1))  # screen
    if p.halation > 0:
        Ln = _lum(img)
        hi = np.clip((Ln - 0.75) / 0.25, 0, 1)
        hal = _blur(hi, w * 0.006)
        img[..., 0] += hal * p.halation * 0.9
        img[..., 1] += hal * p.halation * 0.25

    # 6. tone
    img = img + p.exposure
    img = (img - 0.5) * p.contrast + 0.5
    img = p.black_lift + img * (1 - p.black_lift)
    if p.style == "noir":
        img = np.clip(img, 0, 1) ** 1.15

    # 7. vignette (darken + slight desat at edges)
    v = np.clip(rad - 0.25, 0, 1) ** 1.6
    img = img * (1 - p.vignette * v)[..., None]

    # 8. grain (luminance, slightly clumped)
    if p.grain > 0:
        g = rng.standard_normal((h, w)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.7) * 1.6
        Ln = np.clip(_lum(img), 0, 1)
        amp = p.grain * (0.6 + 0.8 * Ln * (1 - Ln) * 4) / 1.4   # strongest in mid-tones
        img = img + (g * amp)[..., None]

    return np.clip(img, 0, 1)


# --------------------------------------------------------------------------- io


def load(path, max_w=1280):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    if w > max_w:
        bgr = cv2.resize(bgr, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def save(path, rgb, quality=88):
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def render_video(src, dst, p: DreamParams, depth: Depth, max_w=960, max_seconds=6.0):
    """Low-framerate, step-printed dream clip. Output is 24 fps with held frames,
    which plays everywhere and keeps grain/weave changing only on new frames."""
    cap = cv2.VideoCapture(src)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = []
    while len(frames) < int(max_seconds * fps_in):
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        if w > max_w:
            f = cv2.resize(f, (max_w, int(h * max_w / w) // 2 * 2), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
    cap.release()
    h, w = frames[0].shape[:2]
    hold = max(1, round(24 / p.out_fps))
    step = fps_in / p.out_fps
    rng = np.random.default_rng(p.seed)
    tmp = dst + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    prev = None
    k = 0
    i = 0.0
    fog_rng = np.random.default_rng(p.seed + 1)
    fog_base = None
    while int(i) < len(frames):
        cur = frames[int(i)]
        # step-print smear: blend in the previous source frame(s)
        j = max(0, int(i) - max(1, int(step // 2)))
        mixed = cur * (1 - p.smear) + frames[j] * p.smear
        dist = depth(cur)
        if fog_base is None:
            fog_base = _noise_field(h, w, w / 4, fog_rng)
        ff = np.roll(fog_base, int(k * w * 0.01), axis=1)
        out = render(mixed, p, dist, rng=rng, fog_field=ff)
        # exposure breathing
        out = np.clip(out * (1 + 0.03 * math.sin(k * 0.9)), 0, 1)
        # gate weave
        dx, dy = rng.normal(0, p.weave_px, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # temporal persistence: ghost of previous output frame
        if prev is not None:
            out = out * 0.85 + prev * 0.15
        prev = out
        bgr = cv2.cvtColor((out * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for _ in range(hold):
            vw.write(bgr)
        k += 1
        i += step
    vw.release()
    os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p -crf 26 -preset slow -movflags +faststart "{dst}"')
    os.remove(tmp)
