"""End to end: dream text (or app spec) -> spec.json, raw + dream stills, fog dial, mp4 clip.

    python3 -m dreamrender.pipeline --text "I was in a hazy forest..." --out out/forest
    python3 -m dreamrender.pipeline --app-spec dream.json --out out/d_123
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from dataclasses import asdict
import numpy as np
import cv2
from .parse import parse
from .stylize import look_from_spec, stylize, load_beauty, load_depth, save
from .animate import make_clip

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render_frames(spec_dict, outdir, frames, res, samples):
    os.makedirs(outdir, exist_ok=True)
    sp = os.path.join(outdir, "spec.json")
    json.dump(spec_dict, open(sp, "w"), indent=2)
    env = {**os.environ, "PYTHONPATH": HERE}
    cmd = [sys.executable, "-m", "dreamrender.scenes", sp, os.path.join(outdir, "frames"),
           "--frames", str(frames), "--res", f"{res[0]}x{res[1]}", "--samples", str(samples)]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL if not os.environ.get("DREAM_VERBOSE") else None)


def run(spec, look, outdir, frames=24, res=(960, 540), samples=48, duration=6.0, skip_render=False, frames_dir=None):
    t0 = time.time()
    sd = asdict(spec) if not isinstance(spec, dict) else spec
    if not skip_render:
        render_frames(sd, outdir, frames, res, samples)
    fdir = frames_dir or os.path.join(outdir, "frames")   # frames_dir: re-style another dream's renders
    hero = len([f for f in os.listdir(fdir) if f.endswith(".png")]) // 2
    img = load_beauty(os.path.join(fdir, f"f{hero:03d}.png"))
    dep = load_depth(os.path.join(fdir, f"f{hero:03d}_depth.exr"))
    seed = int(sd.get("seed", 0))
    try:
        fc = json.load(open(os.path.join(fdir, "render_meta.json")))["frames"][hero].get("focus")
    except Exception:
        fc = None
    save(os.path.join(outdir, "raw.jpg"), img)
    save(os.path.join(outdir, "dream.jpg"), stylize(img, dep, look, t=3.0, seed=seed, frame_index=hero, focus=fc))
    # fog dial: the same memory at four levels of clarity
    dial = []
    for f in (0.1, 0.35, 0.6, 0.85):
        l2 = {**look, "fog": f}
        dial.append(cv2.resize(stylize(img, dep, l2, t=3.0, seed=seed, frame_index=hero, focus=fc), (img.shape[1] // 2, img.shape[0] // 2)))
    save(os.path.join(outdir, "fog_dial.jpg"), np.concatenate(dial, 1))
    # depth preview
    d = np.log1p(np.minimum(dep, 1e4)); d = 1 - (d - d.min()) / (np.ptp(d) + 1e-6)
    cv2.imwrite(os.path.join(outdir, "depth.jpg"), (d * 255).astype(np.uint8))
    info = make_clip(fdir, look, os.path.join(outdir, "clip.mp4"), seed=seed, duration=duration,
                     poster=os.path.join(outdir, "poster.jpg"))
    manifest = dict(spec=sd, look=look, prompt=spec.prompt() if hasattr(spec, "prompt") else None,
                    clip=info, seconds=round(time.time() - t0, 1),
                    files=["raw.jpg", "dream.jpg", "fog_dial.jpg", "depth.jpg", "clip.mp4", "poster.jpg"])
    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=2, default=str)
    return manifest


class ContentDeclined(Exception):
    """The content rating stage aborted this dream (see rating.py / CONTENT_POLICY.md). .rating has the details."""
    def __init__(self, rating):
        super().__init__(rating.get("dreamer_message") or "declined")
        self.rating = rating


def post_scene(sdir, mits, look, seed=0):
    """Censor bars / reaction cuts on a rendered scene folder (clip + stills)."""
    from .mitigate import apply_to_clip, apply_to_still
    if not any(m["kind"] in ("censor_bar", "reaction_cut", "fade_to_fog", "dark_fog") for m in mits):
        return []
    clip, tmp = os.path.join(sdir, "clip.mp4"), os.path.join(sdir, "clip.mitigated.mp4")
    done = apply_to_clip(clip, tmp, mits, look, frames_dir=os.path.join(sdir, "frames"), seed=seed)
    os.replace(tmp, clip)
    try:
        fm = json.load(open(os.path.join(sdir, "frames", "render_meta.json")))["frames"]
        fc = fm[len(fm) // 2].get("focus")
    except Exception:
        fc = None
    for f in ("poster.jpg", "dream.jpg", "fog_dial.jpg", "raw.jpg"):
        apply_to_still(os.path.join(sdir, f), mits, look, focus=fc if f != "fog_dial.jpg" else None, seed=seed)
    return done


def run_multi(parts, outdir, frames=16, res=(960, 540), samples=32, skip_render=False, content=None):
    """Render each scene, then join the clips: every scene surfaces from fog and sinks back, like a dream cut.
    content: a content_rating (rating.py); its censor bars / reaction cuts are applied to each scene clip."""
    from .rating import scene_mitigations
    os.makedirs(outdir, exist_ok=True)
    manifests = []
    for i, (spec, look, dur) in enumerate(parts):
        sdir = os.path.join(outdir, f"s{i}")
        manifests.append(run(spec, look, sdir, frames, res, samples, dur, skip_render))
        if content:
            manifests[-1]["mitigations_applied"] = post_scene(sdir, scene_mitigations(content, i), look,
                                                              seed=int(getattr(spec, "seed", 0) or 0))
    lst = os.path.join(outdir, "concat.txt")
    with open(lst, "w") as f:
        for i in range(len(parts)):
            f.write(f"file '{os.path.abspath(os.path.join(outdir, f's{i}', 'clip.mp4'))}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", os.path.join(outdir, "clip.mp4")], check=True)
    import shutil
    for f in ("poster.jpg", "dream.jpg", "raw.jpg", "fog_dial.jpg"):
        shutil.copy(os.path.join(outdir, "s0", f), os.path.join(outdir, f))
    m = dict(scenes=manifests, files=["clip.mp4", "poster.jpg", "dream.jpg"])
    if content:
        m["content_rating"] = {k: v for k, v in content.items() if k != "findings"}
    json.dump(m, open(os.path.join(outdir, "manifest.json"), "w"), indent=2, default=str)
    return m


def run_app_spec(app, outdir, frames=16, res=(960, 540), samples=32, skip_render=False, use_llm=None):
    """The gated entry point: rate -> abort, or veil + render + censor/cut. Raises ContentDeclined on abort."""
    from .rating import rate, apply_to_spec, scene_mitigations
    from .mitigate import veil_look
    from .app_spec import from_app_spec_multi
    os.makedirs(outdir, exist_ok=True)
    cr = rate(app, use_llm=use_llm)
    json.dump(cr, open(os.path.join(outdir, "content_rating.json"), "w"), indent=2)
    if cr["outcome"] == "abort":
        json.dump(dict(declined=True, content_rating={k: v for k, v in cr.items() if k != "findings"}),
                  open(os.path.join(outdir, "manifest.json"), "w"), indent=2)
        raise ContentDeclined(cr)
    safe = apply_to_spec(app, cr)
    json.dump(safe, open(os.path.join(outdir, "rated_spec.json"), "w"), indent=2)
    parts = [(sp, veil_look(lk, scene_mitigations(cr, i)), d) for i, (sp, lk, d) in enumerate(from_app_spec_multi(safe))]
    return run_multi(parts, outdir, frames, res, samples, skip_render, content=cr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text"); ap.add_argument("--app-spec"); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=24); ap.add_argument("--res", default="960x540")
    ap.add_argument("--samples", type=int, default=48); ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--seed", type=int); ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--rules-only", action="store_true", help="content rating without the Claude rater")
    a = ap.parse_args()
    if a.app_spec:
        try:
            m = run_app_spec(json.load(open(a.app_spec)), a.out, a.frames, tuple(int(v) for v in a.res.split("x")),
                             a.samples, a.skip_render, use_llm=False if a.rules_only else None)
        except ContentDeclined as e:
            print(json.dumps({"declined": True, "message": str(e), "reasons": e.rating["abort_reasons"]}))
            sys.exit(3)
        print(json.dumps({"scenes": len(m["scenes"]), "content": m.get("content_rating", {}).get("outcome")}))
        return
    from .rating import rate, scene_mitigations
    from .mitigate import veil_look
    cr = rate({"input": {"transcript": a.text}, "scenes": [{"text": a.text}]}, use_llm=False if a.rules_only else None)
    os.makedirs(a.out, exist_ok=True)
    json.dump(cr, open(os.path.join(a.out, "content_rating.json"), "w"), indent=2)
    if cr["outcome"] == "abort":
        print(json.dumps({"declined": True, "message": cr["dreamer_message"], "reasons": cr["abort_reasons"]}))
        sys.exit(3)
    spec = parse(a.text, seed=a.seed)
    look = veil_look(look_from_spec(asdict(spec)), scene_mitigations(cr, 0))
    m = run(spec, look, a.out, a.frames, tuple(int(v) for v in a.res.split("x")), a.samples, a.duration, a.skip_render)
    post_scene(a.out, scene_mitigations(cr, 0), look, seed=int(spec.seed or 0))
    print(json.dumps({k: m[k] for k in ("clip", "seconds")} | {"content": cr["outcome"]}))


if __name__ == "__main__":
    main()
