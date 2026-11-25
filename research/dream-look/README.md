# Dream look reference renderer

Report: https://claude.ai/artifact/RC5rDxfZ1FCWftMbTuckbX

`dreamlook.py` turns a realistic frame or clip into a dream-fogged one.
Inputs (from the dream analyser): clarity 0..1 (default 0.3), style
(natural|bw|sepia|faded|noir|vivid), mood (neutral|calm|warm|eerie|sad|dread),
lucid, nightmare, indoor, focus (x, y). All derived knobs are in
`DreamParams.resolve()` and can be overridden.

Needs: numpy, opencv-python-headless, onnxruntime, ffmpeg, and the MiDaS v2.1
small ONNX depth model:
https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx

    from dreamlook import Depth, DreamParams, load, save, render, render_video
    D = Depth("model-small.onnx")
    img = load("frame.jpg"); p = DreamParams(clarity=0.3, mood="calm").resolve()
    save("out.jpg", render(img, p, D(img)))
    render_video("clip.mp4", "dream.mp4", p, D)

Stills take ~1.5 s at 1200 px on one CPU. Values were calibrated by eye and
should be tuned with user tests.
