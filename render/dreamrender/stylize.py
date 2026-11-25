"""Dream-fog stylisation: beauty frame + depth -> remembered frame.

All operations are plain numpy/OpenCV so they run anywhere and can be ported to
a GPU shader later. The look is controlled by a small dict of knobs that match
the app's "dream spec v1" `look` block (fog, fps, grain, echo, vignette, style,
saturation, tint), plus a few derived internals.

Ideas behind each stage (see README for references):
  * depth fog with light scattering: lights bloom into the fog instead of the fog being a flat grey card
  * the edge of the frame dissolves into fog rather than going black (memory fades at the edges)
  * soft focus that is sharper at the centre of attention than at the periphery
  * slow, low-frequency warping: shapes are not quite stable
  * desaturation + lifted blacks, because remembered colour is weaker than seen colour
  * film / mono / sepia / vhs / vivid overrides only when the dreamer names them
"""
from __future__ import annotations
import math
import numpy as np
import cv2

# --------------------------------------------------------------------------- look
STYLES = ("natural", "mono", "noir", "sepia", "vhs", "vivid", "pastel")


def clarity_to_fog(clarity: float) -> float:
    """Map parser clarity (0 barely remembered .. 1 crystal clear; typical 0.35) to fog (0..1, typical 0.55)."""
    c = float(np.clip(clarity, 0, 1))
    if c >= 0.35:
        return 0.55 - (c - 0.35) / 0.65 * 0.47      # 1.0 -> 0.08
    return 0.55 + (0.35 - c) / 0.35 * 0.35          # 0.0 -> 0.90


def fog_to_clarity(fog: float) -> float:
    f = float(np.clip(fog, 0, 1))
    if f <= 0.55:
        return 0.35 + (0.55 - f) / 0.47 * 0.65
    return 0.35 - (f - 0.55) / 0.35 * 0.35


def look_from_spec(spec: dict) -> dict:
    """Derive the look knobs from a parsed DreamSpec (dict)."""
    fog = clarity_to_fog(spec.get("clarity", 0.35))
    palette = spec.get("palette", "color")
    style = "natural"
    if palette == "bw":
        style = "mono"
    elif palette == "sepia":
        style = "sepia"
    elif spec.get("vhs"):
        style = "vhs"
    elif spec.get("saturation", 0.75) > 1.0:
        style = "vivid"
    film = bool(spec.get("film")) or style in ("mono", "sepia")
    tint = spec.get("tint")
    return dict(
        fog=fog,
        fps=float(np.interp(fog, [0.08, 0.55, 0.9], [12, 8, 6])),
        grain=float(np.interp(fog, [0.08, 0.9], [0.18, 0.45])) + (0.25 if film else 0),
        echo=float(np.interp(fog, [0.08, 0.9], [0.1, 0.55])),
        vignette=float(np.interp(fog, [0.08, 0.9], [0.25, 0.75])),
        style=style,
        saturation=0.85 if style != "vivid" else 1.35,
        tint=list(tint) if tint else None,
        film=film,
        mood=spec.get("mood", "eerie"),
        time_of_day=spec.get("time_of_day", "dusk"),
        fog_tone=float(spec.get("fog_tone", 0.5)),
        timeline=spec.get("timeline") or [],
    )


def look_at_time(look: dict, u: float) -> dict:
    """Fog density and tone at clip position u (0..1) from look['timeline'] keys.
    Changes happen as a turn between beats (smoothstep across the middle 60% of each gap), not a slow linear drift."""
    keys = look.get("timeline") or []
    if not keys:
        return look
    keys = sorted(keys, key=lambda k: k["at"])
    if u <= keys[0]["at"]:
        k0 = k1 = keys[0]; f = 0.0
    elif u >= keys[-1]["at"]:
        k0 = k1 = keys[-1]; f = 0.0
    else:
        i = max(i for i, k in enumerate(keys) if k["at"] <= u)
        k0, k1 = keys[i], keys[i + 1]
        x = (u - k0["at"]) / max(k1["at"] - k0["at"], 1e-6)
        x = float(np.clip((x - 0.2) / 0.6, 0, 1))
        f = x * x * (3 - 2 * x)
    out = dict(look)
    out["fog"] = k0["fog"] + (k1["fog"] - k0["fog"]) * f
    out["fog_tone"] = k0["tone"] + (k1["tone"] - k0["tone"]) * f
    return out


def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


# --------------------------------------------------------------------------- io
def load_beauty(png_path: str) -> np.ndarray:
    im = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB if im.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)


def load_depth(exr_path: str) -> np.ndarray:
    """Depth render: RGB = camera distance in metres, alpha 0 where only sky is visible."""
    import OpenEXR
    with OpenEXR.File(exr_path) as f:
        px = np.asarray(f.channels()["RGBA"].pixels, dtype=np.float32)
    z = px[..., 0] / np.maximum(px[..., 3], 1e-4)          # un-premultiply edge pixels
    z = np.where((px[..., 3] < 0.5) | ~np.isfinite(z), 1e9, z)
    return np.minimum(z, 1e9)


def to_lin(x):
    return np.power(np.clip(x, 0, None), 2.2)


def to_srgb(x):
    return np.power(np.clip(x, 0, 1), 1 / 2.2)


# --------------------------------------------------------------------------- noise
def smooth_noise(h, w, cells, seed, t=0.0, octaves=3):
    """Cheap animated value noise: bicubic-upscaled random grids, drifting with t."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = max(2, int(cells * (2 ** o)))
        a = rng.standard_normal((c + 3, int(c * w / h) + 3)).astype(np.float32)
        b = rng.standard_normal((c + 3, int(c * w / h) + 3)).astype(np.float32)
        ph = t * (0.6 + 0.3 * o)
        g = a * math.cos(ph) + b * math.sin(ph)               # rotate between two fields: loops smoothly
        up = cv2.resize(g, (w + w // c * 2, h + h // c * 2), interpolation=cv2.INTER_CUBIC)
        oy, ox = h // c, w // c
        out += amp * up[oy:oy + h, ox:ox + w]
        total += amp; amp *= 0.5
    return out / total


def gblur(x, sigma):
    if sigma <= 0.3:
        return x
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(x, (k, k), sigma, borderType=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------- core
def stylize(img: np.ndarray, depth: np.ndarray, look: dict, t: float = 0.0, seed: int = 0,
            frame_index: int = 0, fade: float = 0.0, focus=None) -> np.ndarray:
    """img: sRGB float HxWx3 (0..1). depth: HxW metres (sky ~1e9). t: seconds (animates fog/warp).
    fade: 0..1 extra dissolve into fog (used for clip in/out)."""
    h, w = img.shape[:2]
    S = w / 960.0                                   # resolution scale for pixel-sized effects
    fog = float(np.clip(look["fog"], 0, 1))
    style = look.get("style", "natural")
    rng = np.random.default_rng(seed * 1000 + frame_index)
    lin = to_lin(img)

    # ---- 1. warp: low-frequency displacement, larger when fog is high
    amp = (0.6 + 7.0 * fog ** 1.5) * S
    if amp > 0.3:
        dx = smooth_noise(h, w, 3, seed + 11, t * 0.35, 2) * amp
        dy = smooth_noise(h, w, 3, seed + 23, t * 0.35, 2) * amp
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        lin = cv2.remap(lin, mx + dx, my + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        depth = cv2.remap(depth, mx + dx, my + dy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    # ---- attention island: the thing the dream is about stays readable (research thread, "Dream Fog Atlas")
    fx, fy = focus if focus else (0.5, 0.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    island = np.exp(-(((xx / w - fx) / 0.16) ** 2 + ((yy / h - fy) / 0.24) ** 2))
    island *= 1.0 if focus else 0.5

    # ---- 2. depth fog with scattering
    finite = depth[depth < 1e8]
    dm = float(np.median(finite)) if finite.size else 50.0
    d50 = dm * float(np.interp(fog, [0.0, 0.08, 0.55, 0.9, 1.0], [60, 25, 3.5, 1.3, 0.9]))
    dens = math.log(2) / max(d50, 1e-3)
    tone = float(np.clip(look.get("fog_tone", 0.5), 0, 1))
    dark = max(0.0, (tone - 0.5) * 2)                 # 0 normal .. 1 black smoke
    pale = max(0.0, (0.5 - tone) * 2)                 # 0 normal .. 1 milky white
    pn = 0.5 + 0.5 * smooth_noise(h, w, 2, seed + 5, t * 0.25, 3)
    patch = (0.75 - 0.3 * dark) + (0.5 + 0.7 * dark) * pn        # drifting banks of mist; dark smoke billows more
    F = 1.0 - np.exp(-dens * (1 + 4.0 * dark) * np.minimum(depth, dm * 60) * patch)   # darkness closes in
    sky = depth >= 1e8
    F = np.where(sky, np.clip(F, 0, 1) * min(1.0, 0.25 + 0.75 * fog + 0.7 * dark), F)            # sky only partly veiled
    src_l = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    light_px = np.clip((src_l - (0.75 + 0.2 * dark)) / (0.25 - 0.2 * dark), 0, 1)
    F = F * (1 - 0.7 * light_px)                                          # lights punch through the fog
    F = F * (1 - 0.55 * island)
    F = np.clip(F, 0, 0.985)[..., None]

    far = lin[(depth > np.percentile(depth, 80))] if np.any(depth > 0) else lin.reshape(-1, 3)
    far_mean = far.mean(axis=0) if far.size else lin.reshape(-1, 3).mean(0)
    scatter = gblur(lin, 0.09 * w)                                        # lights glow through the fog
    lum_far = float(far_mean @ [0.2126, 0.7152, 0.0722])
    night = look.get("time_of_day") == "night"
    floor = np.array([0.02, 0.022, 0.028], np.float32) if night else far_mean * 0.6
    fog_col = (0.55 * far_mean + 0.45 * scatter + 0.25 * floor) * (np.array([0.62, 0.7, 0.9], np.float32) if night else 1.0)
    if look.get("tint"):
        tc = np.array(look["tint"], np.float32)
        fog_col = fog_col * (0.5 + 0.9 * tc / max(tc.max(), 1e-3))
    if pale > 0:      # milky, luminous fog
        m = float(fog_col.mean()) if fog_col.ndim == 1 else fog_col.mean(axis=(0, 1)).mean()
        fog_col = fog_col * (1 - pale) + (0.35 * fog_col + 0.6 * max(m, 0.18) + 0.12) * pale
    if dark > 0:      # black fog: swallows light, only strong lights still glow inside it
        ink = 0.02 * fog_col + 0.07 * scatter + np.array([0.002, 0.0025, 0.004], np.float32)
        fog_col = fog_col * (1 - dark) + ink * dark
    lin = lin * (1 - F) + fog_col * F
    if dark > 0:      # smoke between you and everything: near surfaces dim too, lights survive
        keep = light_px[..., None]
        lin = lin * (1 - (0.85 * dark * (0.45 + 0.55 * F)) * (1 - 0.85 * keep))

    # ---- 3. soft focus: sharp centre of attention, vague periphery, plus depth defocus
    r = np.sqrt(((xx - w * 0.5) / (w * 0.5)) ** 2 + ((yy - h * 0.48) / (h * 0.5)) ** 2)
    cy0, cx0 = int(np.clip(fy, 0.1, 0.9) * h), int(np.clip(fx, 0.1, 0.9) * w)
    focus_d = float(np.median(depth[max(cy0 - h // 10, 0):cy0 + h // 10, max(cx0 - w // 10, 0):cx0 + w // 10]))
    dof = np.clip(np.abs(np.log((depth + 0.1) / (focus_d + 0.1))) / 3.0, 0, 1)
    blur_mask = np.clip(0.55 * np.clip(r - 0.35, 0, 1) * 1.6 + 0.45 * dof, 0, 1) * (0.12 + 0.75 * fog) * (1 - 0.85 * island)
    b1, b2 = gblur(lin, 1.8 * S * (0.5 + fog)), gblur(lin, 6.0 * S * (0.4 + fog))
    m = blur_mask[..., None]
    lin = np.where(m < 0.5, lin * (1 - 2 * m) + b1 * (2 * m), b1 * (2 - 2 * m) + b2 * (2 * m - 1))

    # ---- 4. bloom / halation / orton glow
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    thr = max(float(np.percentile(lum, 97)) * 0.8, 0.6 * dark)     # in black fog only real lights bloom
    hi = lin * np.clip((lum - thr) / (thr + 1e-4), 0, 1)[..., None]
    bloom = 0.5 * gblur(hi, 4 * S) + 0.35 * gblur(hi, 16 * S) + 0.25 * gblur(hi, 48 * S)
    hal = gblur(hi, 10 * S) * np.array([1.0, 0.35, 0.15], np.float32)   # red-orange film halation
    lin = lin + bloom * (0.25 + 0.8 * fog) + hal * (0.15 + 0.4 * fog) * (1.5 if look.get("film") else 1)
    orton = gblur(lin, 14 * S)
    k = (0.05 + 0.2 * fog) * (1 - 0.8 * dark)
    lin = lin * (1 - k) + (1 - (1 - np.clip(lin, 0, 1)) * (1 - np.clip(orton, 0, 1))) * k   # screen blend

    # ---- 5. chromatic aberration (lens memory)
    ca = (0.0008 + 0.003 * fog) * (2.5 if style == "vhs" else 1)
    if ca > 0:
        def scale_ch(c, s_):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + s_)
            return cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        lin = np.stack([scale_ch(lin[..., 0], ca), lin[..., 1], scale_ch(lin[..., 2], -ca)], -1)

    # ---- 6. edges dissolve into fog, then gentle darkening
    vig = float(look.get("vignette", 0.5))
    edge = np.clip((r - (0.75 - 0.35 * fog)) / 0.6, 0, 1) ** 1.5
    edge_n = edge * (0.8 + 0.4 * (0.5 + 0.5 * smooth_noise(h, w, 3, seed + 41, t * 0.3)))
    edge_col = (gblur(lin, 0.06 * w) * 0.7 + fog_col.mean(axis=(0, 1)) * 0.3) * (1 - 0.85 * dark)
    ek = fog * 0.9 + 0.1 * dark
    lin = lin * (1 - edge_n[..., None] * ek) + edge_col * (edge_n[..., None] * ek)
    lin = lin * (1 - vig * 0.55 * np.clip(r - 0.45, 0, 1) ** 1.6)[..., None]

    # ---- 7. dissolve (clip in/out)
    if fade > 0:
        lin = lin * (1 - fade) + fog_col.mean(axis=(0, 1)) * 1.1 * fade

    # ---- 8. tone + colour of memory
    out = to_srgb(lin / (1 + 0.15 * lin))                 # soft shoulder
    lum = out @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = float(look.get("saturation", 0.85)) * float(np.interp(fog, [0, 1], [1.05, 0.85]))
    out = lum[..., None] + (out - lum[..., None]) * sat
    lift = (0.012 + 0.06 * fog) * (1 - 0.85 * dark) + 0.04 * pale                               # milky blacks
    out = lift + out * (1 - lift - 0.03 * fog)
    mood = look.get("mood", "eerie")
    shadows = {"eerie": (0.0, 0.01, 0.03), "scary": (0.01, 0.0, 0.03), "calm": (0.0, 0.015, 0.025),
               "joyful": (0.02, 0.01, 0.0)}.get(mood, (0, 0.01, 0.02))
    out = out + np.array(shadows, np.float32) * (1 - lum[..., None]) * (0.5 + fog)
    if look.get("tint"):
        tc = np.array(look["tint"], np.float32)
        tc = tc / max(tc.max(), 1e-3)
        mid = 4 * lum * (1 - lum)
        out = out * (1 - 0.45 * mid[..., None]) + (lum[..., None] * tc * 1.3) * (0.45 * mid[..., None])
        out = out * (0.8 + 0.2 * tc)

    # ---- 9. explicit styles
    lumf = out @ np.array([0.3, 0.6, 0.1], np.float32)   # orange-filter mono mix
    if style in ("mono", "noir", "sepia"):
        gamma = {"mono": 1.0, "noir": 0.8, "sepia": 0.95}[style]
        con = {"mono": 1.3, "noir": 1.45, "sepia": 1.1}[style]
        flick = 1 + (rng.standard_normal() * 0.025 if look.get("film") else 0)
        y = np.clip((lumf - 0.06) / 0.94, 0, 1)                                   # film black point
        y = np.clip((np.power(y, gamma) - 0.5) * con + 0.5, 0, 1) * flick
        if style == "sepia":
            out = np.stack([y * 1.07 + 0.03, y * 0.92 + 0.02, y * 0.72], -1)
        else:
            out = np.repeat(y[..., None], 3, -1)
    elif style == "vivid":
        lum2 = out @ np.array([0.299, 0.587, 0.114], np.float32)
        chroma = out - lum2[..., None]
        sat_now = np.abs(chroma).max(-1, keepdims=True)
        out = lum2[..., None] + chroma * (1.5 + 1.2 * np.clip(0.25 - sat_now, 0, 0.25) * 4)   # vibrance: lift dull colours most
        out = np.clip((out - 0.5) * 1.12 + 0.5, 0, 1)
    elif style == "pastel":
        out = 0.12 + out * 0.82
    elif style == "vhs":
        ycc = cv2.cvtColor(np.clip(out, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
        ycc[..., 1] = cv2.blur(ycc[..., 1], (int(9 * S) | 1, 1)); ycc[..., 2] = cv2.blur(ycc[..., 2], (int(9 * S) | 1, 1))
        ycc[..., 1:] = np.roll(ycc[..., 1:], int(3 * S), axis=1)
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
        out = out * (0.94 + 0.06 * (np.arange(h) % 2))[:, None, None]
        rows = rng.integers(0, h, 3)
        for r0 in rows:
            out[r0:r0 + int(3 * S) + 1] = np.roll(out[r0:r0 + int(3 * S) + 1], int(rng.integers(4, 14) * S), axis=1)

    # ---- 10. grain (luminance-weighted, slightly clumped), film dust
    g = float(look.get("grain", 0.3))
    if g > 0:
        n = rng.standard_normal((h, w)).astype(np.float32)
        n = 0.7 * n + 0.6 * gblur(n, 0.9 * S)
        yl = out.mean(-1)
        out = out + (n * g * 0.045 * (0.35 + 1.3 * yl * (1 - yl)))[..., None]
        if style not in ("mono", "noir", "sepia"):
            out = out + rng.standard_normal((h, w, 3)).astype(np.float32) * g * 0.012
    if look.get("film") and style in ("mono", "noir", "sepia"):
        for _ in range(rng.poisson(2.5)):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.circle(out, (cx, cy), int(rng.integers(1, 3) * S) + 1, (0.9, 0.9, 0.9) if rng.random() < 0.5 else (0.05,) * 3, -1)
        if rng.random() < 0.3:
            x0 = int(rng.integers(0, w))
            cv2.line(out, (x0, 0), (x0 + int(rng.integers(-8, 8)), h), (0.75,) * 3, 1)
    return np.clip(out, 0, 1)


def save(path, img):
    im = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
