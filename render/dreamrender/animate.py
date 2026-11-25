"""Rendered frames + depth -> low-framerate dream clip (mp4).

Motion language:
  * the picture updates at the dream's fps (6-12), not 24: each image is held for several
    container frames, with irregular holds so the rhythm feels remembered rather than filmed
  * held frames still breathe: fog banks, warp and grain keep moving underneath
  * echo: each new image is laid over a fading ghost of the previous ones
  * the clip surfaces out of fog and sinks back into it
"""
from __future__ import annotations
import glob, os, subprocess
import numpy as np
import cv2
from .stylize import stylize, load_beauty, load_depth, save, look_at_time

CONTAINER_FPS = 24


def load_sequence(render_dir):
    pngs = sorted(p for p in glob.glob(os.path.join(render_dir, "f*.png")))
    frames = [load_beauty(p) for p in pngs]
    depths = [load_depth(p[:-4] + "_depth.exr") for p in pngs]
    return frames, depths


def make_clip(render_dir: str, look: dict, out_mp4: str, seed: int = 0, duration: float = 6.0,
              size=None, poster: str | None = None):
    frames, depths = load_sequence(render_dir)
    n = len(frames)
    focus = [None] * n
    try:
        import json
        fm = json.load(open(os.path.join(render_dir, "render_meta.json")))["frames"]
        focus = [f.get("focus") for f in fm][:n] + [None] * max(0, n - len(fm))
    except Exception:
        pass
    h, w = frames[0].shape[:2]
    if size:
        w, h = size
    fps = float(np.clip(look.get("fps", 8), 4, 24))
    D = max(2, int(round(duration * fps)))
    rng = np.random.default_rng(seed)
    echo = float(look.get("echo", 0.3))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(CONTAINER_FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    prev = None
    fade_in, fade_out = 0.9, 1.1
    for j in range(D):
        p = j / (D - 1) * (n - 1)
        a, frac = int(p), p - int(p)
        b = min(a + 1, n - 1)
        img = frames[a] if frac < 0.5 else frames[b]
        dep = depths[a] if frac < 0.5 else depths[b]
        if echo > 0.25 and 0.3 < frac < 0.7:       # in-between: double exposure of neighbours
            img = frames[a] * 0.5 + frames[b] * 0.5
        if size:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            dep = cv2.resize(dep, (w, h), interpolation=cv2.INTER_NEAREST)
        t = j / fps
        fade = max(0.0, 1 - t / fade_in) ** 1.5 + max(0.0, 1 - (duration - t) / fade_out) ** 1.5
        fa, fb = focus[a], focus[b]
        fc = [fa[0] + (fb[0] - fa[0]) * frac, fa[1] + (fb[1] - fa[1]) * frac] if fa and fb else (fa or fb)
        lk = look_at_time(look, j / max(D - 1, 1))
        out = stylize(img, dep, lk, t=t, seed=seed, frame_index=j, fade=min(fade, 1.0), focus=fc)
        if prev is not None and echo > 0:
            k = 0.12 + 0.4 * echo
            out = out * (1 - k) + prev * k
        prev = out
        hold = CONTAINER_FPS / fps
        hold = int(max(1, round(hold + rng.choice([-1, 0, 0, 0, 1]))))
        if rng.random() < 0.05:
            hold *= 2                               # a beat where time sticks
        # within a hold, only grain changes (a few pixels of life), not the image
        base8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        for hh in range(hold):
            if hh == 0:
                ff.stdin.write(base8.tobytes())
            else:
                g = rng.standard_normal((h, w, 1)).astype(np.float32) * 255 * 0.012 * (0.5 + look.get("grain", 0.3))
                ff.stdin.write(np.clip(base8 + g, 0, 255).astype(np.uint8).tobytes())
        if poster and j == D // 2:
            save(poster, out)
    ff.stdin.close()
    ff.wait()
    return dict(display_frames=D, fps=fps, source_frames=n)
