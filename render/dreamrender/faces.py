"""Dream faces: how a person looks in a dream, from one ordinary photo.

What this module can do on a CPU box with no generative model:
  * find the face (MediaPipe Face Landmarker, 478 points + 52 expression blendshapes)
  * cut the person out (MediaPipe selfie multiclass: hair / face skin / body skin / clothes)
  * change the expression by warping the photo along the face mesh (joy, sad, fear, anger, surprise, neutral)
  * apply the four dream identity states:
        known_clear      it's them and you saw their face clearly
        known_different  it's them, you just "knew", but the face was someone else's (hair, build, clothes kept)
        stranger         a person whose face never resolves (features dissolve, head shape kept)
        obscured         face hidden: in shadow / against the light
  * measure identity with ArcFace (insightface w600k_r50): cosine similarity of the result to the photo.
    That is the QA check a production pipeline needs: known_clear must stay >= ~0.45, known_different must fall < ~0.25.
  * composite the person into a rendered dream scene and hand both to stylize() with a face-aware fog.

What it cannot do: invent a new pose, angle, mouth interior or body. That needs an identity-conditioned image/video
model (see FACES.md). The photo warp is a demonstration and an offline fallback, not the production look.

Models (fetch with ../fetch_face_models.sh, or set DREAM_FACE_MODELS):
  face_landmarker.task, selfie_multiclass_256x256.tflite, w600k_r50.onnx
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

MODELS = os.environ.get("DREAM_FACE_MODELS", os.path.join(os.path.dirname(__file__), "..", "models"))

# ---- MediaPipe face mesh indices (image left/right = as seen in the photo)
MOUTH_L, MOUTH_R = 61, 291
UPPER_LIP, LOWER_LIP = [0, 37, 267, 13, 82, 312], [17, 84, 314, 14, 87, 317]
CHIN = [152, 148, 377, 176, 400]
BROW_L_INNER, BROW_L_MID, BROW_L_OUTER = [107, 55, 66, 65], [105, 52], [70, 46, 63, 53]
BROW_R_INNER, BROW_R_MID, BROW_R_OUTER = [336, 285, 296, 295], [334, 282], [300, 276, 293, 283]
LID_L_UP, LID_L_LO = [159, 158, 160, 157, 161], [145, 144, 153, 154, 163]
LID_R_UP, LID_R_LO = [386, 385, 387, 384, 388], [374, 373, 380, 381, 390]
EYE_L, EYE_R = [33, 133, 159, 145], [263, 362, 386, 374]
NOSE = [1, 4, 5, 195, 197, 2, 98, 327]
NOSE_WINGS = (98, 327)
CHEEK_L, CHEEK_R = [50, 101, 205], [280, 330, 425]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
JAW = [361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132]

MOUTH_ALL = [61, 291, 0, 37, 267, 13, 82, 312, 17, 84, 314, 14, 87, 317, 152, 148, 377, 176, 400,
             191, 80, 81, 310, 311, 415, 95, 88, 178, 402, 318, 324, 50, 101, 205, 280, 330, 425, 98, 327]
EMOTIONS = ("neutral", "joy", "sad", "fear", "anger", "surprise")
STATES = ("known_clear", "known_different", "stranger", "obscured")


@dataclass
class Face:
    pts: np.ndarray                       # 478x2 pixel coords
    blend: dict = field(default_factory=dict)   # blendshape name -> 0..1

    @property
    def eyes(self):
        a, b = self.pts[EYE_L].mean(0), self.pts[EYE_R].mean(0)
        return (a, b) if a[0] < b[0] else (b, a)

    @property
    def iod(self):
        a, b = self.eyes
        return float(np.linalg.norm(b - a))

    @property
    def center(self):
        return self.pts[FACE_OVAL].mean(0)

    def kps5(self):
        """ArcFace 5-point order: left eye, right eye, nose tip, left mouth, right mouth (image left/right)."""
        a, b = self.eyes
        ml, mr = self.pts[MOUTH_L], self.pts[MOUTH_R]
        if ml[0] > mr[0]:
            ml, mr = mr, ml
        return np.array([a, b, self.pts[4], ml, mr], np.float32)


class FaceKit:
    def __init__(self, models: str = MODELS):
        self.models = models
        self._lm = self._seg = self._arc = None

    # ---------------------------------------------------------------- models
    def _mp(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        return mp, BaseOptions, vision

    def landmarks(self, img: np.ndarray) -> Face | None:
        mp, BaseOptions, vision = self._mp()
        if self._lm is None:
            self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=os.path.join(self.models, "face_landmarker.task")),
                output_face_blendshapes=True, num_faces=1))
        u8 = np.ascontiguousarray(_u8(img))
        r = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=u8))
        if not r.face_landmarks:
            return None
        h, w = u8.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]], np.float32)
        blend = {b.category_name: float(b.score) for b in (r.face_blendshapes[0] if r.face_blendshapes else [])}
        return Face(pts, blend)

    def person_mask(self, img: np.ndarray) -> dict:
        """Soft masks (HxW float) for 'person', 'hair', 'face_skin', 'clothes'."""
        mp, BaseOptions, vision = self._mp()
        if self._seg is None:
            self._seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=os.path.join(self.models, "selfie_multiclass_256x256.tflite")),
                output_confidence_masks=True, output_category_mask=False))
        r = self._seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(_u8(img))))
        conf = [m.numpy_view().astype(np.float32) for m in r.confidence_masks]   # bg, hair, body, face, clothes, other
        return dict(person=1 - conf[0], hair=conf[1], face_skin=conf[3], clothes=conf[4])

    def identity(self, img: np.ndarray, face: Face | None = None) -> np.ndarray | None:
        """512-d ArcFace embedding (L2 normalised) or None when no face is found."""
        face = face or self.landmarks(img)
        if face is None:
            return None
        if self._arc is None:
            import onnxruntime as ort
            self._arc = ort.InferenceSession(os.path.join(self.models, "w600k_r50.onnx"),
                                             providers=["CPUExecutionProvider"])
        dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
        M, _ = cv2.estimateAffinePartial2D(face.kps5(), dst, method=cv2.LMEDS)
        crop = cv2.warpAffine(_u8(img), M, (112, 112), borderValue=0)
        x = ((crop.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        e = self._arc.run(None, {self._arc.get_inputs()[0].name: x})[0][0]
        return e / np.linalg.norm(e)


# -------------------------------------------------------------------- warping
def _u8(img):
    return img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def _f32(img):
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


def warp_points(img: np.ndarray, src: np.ndarray, disp: np.ndarray, sigma: float) -> np.ndarray:
    """Move control points src by disp (pixels) with a smooth, locally normalised Gaussian field."""
    h, w = img.shape[:2]
    s = 4                                          # solve on a coarse grid, upsample: fast and smooth
    gy, gx = np.mgrid[0:h:s, 0:w:s].astype(np.float32)
    tgt = src + disp                               # backward map: sample where the point came from
    num = np.zeros(gx.shape + (2,), np.float32)
    den = np.zeros(gx.shape, np.float32)
    for (tx, ty), d in zip(tgt, disp):
        wgt = np.exp(-((gx - tx) ** 2 + (gy - ty) ** 2) / (2 * sigma ** 2))
        num += wgt[..., None] * d
        den += wgt
    field_ = num / np.maximum(den, 1.0)[..., None]
    fx = cv2.resize(field_[..., 0], (w, h), interpolation=cv2.INTER_CUBIC)
    fy = cv2.resize(field_[..., 1], (w, h), interpolation=cv2.INTER_CUBIC)
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, mx - fx, my - fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _recipe(face: Face, emotion: str, strength: float):
    """Control-point displacements (in IOD units, y down) for an emotion, relative to the photo's own expression."""
    P, c = face.pts, face.center
    moves: dict[int, np.ndarray] = {}

    def add(idx, dx, dy, outward=False, toward_mid=False):
        for i in ([idx] if isinstance(idx, int) else idx):
            sx = 1.0
            if outward or toward_mid:
                sx = np.sign(P[i][0] - c[0]) * (1 if outward else -1)
            moves[i] = moves.get(i, np.zeros(2, np.float32)) + np.array([dx * sx, dy], np.float32)

    smile = 0.5 * (face.blend.get("mouthSmileLeft", 0) + face.blend.get("mouthSmileRight", 0))
    if emotion != "joy" and smile > 0.25:          # undo a photo smile before adding the new expression
        k = min(1.0, smile)
        add([MOUTH_L, MOUTH_R], 0.035 * k, 0.07 * k, toward_mid=True)
        add(UPPER_LIP[:3], 0, 0.015 * k)
        add(LID_L_LO + LID_R_LO, 0, 0.012 * k)
        add(CHEEK_L + CHEEK_R, 0, 0.03 * k)

    gap = float(np.linalg.norm(P[13] - P[14])) / face.iod
    if emotion in ("neutral", "sad", "anger") and gap > 0.06:
        add([0, 37, 267] + [191, 80, 81, 82, 13, 312, 311, 310, 415], 0, gap * 0.45 / max(strength, 1e-3))
        add([17, 84, 314] + [95, 88, 178, 87, 14, 317, 402, 318, 324], 0, -gap * 0.45 / max(strength, 1e-3))

    if emotion == "joy":
        add([MOUTH_L, MOUTH_R], 0.04, -0.08, outward=True)
        add(LID_L_LO + LID_R_LO, 0, -0.018)
        add(CHEEK_L + CHEEK_R, 0, -0.03)
    elif emotion == "sad":
        add([MOUTH_L, MOUTH_R], 0.0, 0.07)
        add(LOWER_LIP, 0, -0.01)
        add(BROW_L_INNER + BROW_R_INNER, 0.015, -0.07, toward_mid=True)
        add(BROW_L_OUTER + BROW_R_OUTER, 0, 0.03)
        add(LID_L_UP + LID_R_UP, 0, 0.018)
    elif emotion == "fear":
        add(BROW_L_INNER + BROW_R_INNER, 0.02, -0.08, toward_mid=True)
        add(BROW_L_MID + BROW_R_MID + BROW_L_OUTER + BROW_R_OUTER, 0, -0.06)
        add(LID_L_UP + LID_R_UP, 0, -0.03)
        add([MOUTH_L, MOUTH_R], 0.06, 0.02, outward=True)
        add(LOWER_LIP, 0, 0.05)
        add(CHIN, 0, 0.04)
    elif emotion == "anger":
        add(BROW_L_INNER + BROW_R_INNER, 0.035, 0.07, toward_mid=True)
        add(BROW_L_MID + BROW_R_MID, 0.0, 0.03)
        add(BROW_L_OUTER + BROW_R_OUTER, 0, -0.01)
        add(LID_L_UP + LID_R_UP, 0, 0.022)
        add(LID_L_LO + LID_R_LO, 0, -0.012)
        add([MOUTH_L, MOUTH_R], 0.02, 0.02, toward_mid=True)
        add(UPPER_LIP, 0, 0.012)
        add(NOSE_WINGS[0], 0.015, -0.01, outward=True); add(NOSE_WINGS[1], 0.015, -0.01, outward=True)
    elif emotion == "surprise":
        add(BROW_L_INNER + BROW_R_INNER + BROW_L_MID + BROW_R_MID + BROW_L_OUTER + BROW_R_OUTER, 0, -0.10)
        add(LID_L_UP + LID_R_UP, 0, -0.035)
        add([MOUTH_L, MOUTH_R], 0.025, 0.03, toward_mid=True)
        add(LOWER_LIP, 0, 0.12)
        add(CHIN, 0, 0.10)
    idx = np.array(list(moves), int)
    disp = np.array([moves[i] for i in idx], np.float32) * face.iod * strength
    return idx, disp


def anchors(face: Face, img_shape, scale=1.0):
    """Zero-displacement points that pin the forehead, ears and the image border."""
    h, w = img_shape[:2]
    c, iod = face.center, face.iod
    ring = [c + 1.9 * iod * scale * np.array([np.cos(a), np.sin(a) * 1.25]) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    border = [(x, y) for x in np.linspace(0, w - 1, 6) for y in (0, h - 1)] + [(x, y) for y in np.linspace(0, h - 1, 6) for x in (0, w - 1)]
    return np.array(ring + border, np.float32)


INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
INNER_UP, INNER_LO = [191, 80, 81, 82, 13, 312, 311, 310, 415], [95, 88, 178, 87, 14, 317, 402, 318, 324]


def close_mouth(img: np.ndarray, face: Face, cavity: bool = False) -> np.ndarray:
    """A warp cannot take teeth out of a smile, so first paint the mouth opening over: lip colour (closed mouth)
    or a soft dark cavity (open mouth for fear / surprise). Generative models do this properly; this is the fallback."""
    f = _f32(img).copy()
    h, w = f.shape[:2]
    poly = face.pts[INNER_LIPS].astype(np.int32)
    m = np.zeros((h, w), np.float32)
    cv2.fillPoly(m, [poly], 1.0)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=max(1, int(face.iod * 0.03)))
    m = cv2.GaussianBlur(m, (0, 0), face.iod * 0.025)[..., None]
    ring = np.zeros((h, w), np.uint8)
    cv2.polylines(ring, [face.pts[[61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91]].astype(np.int32)], True, 1,
                  max(2, int(face.iod * 0.06)))
    lip = np.median(f[ring > 0], axis=0)
    fill = lip * (0.25 if cavity else 0.85)
    if not cavity:                                   # a faint parting line where the lips meet
        line = np.zeros((h, w), np.float32)
        mid = (face.pts[INNER_UP] + face.pts[INNER_LO][::-1]) / 2
        cv2.polylines(line, [mid.astype(np.int32)], False, 1.0, max(1, int(face.iod * 0.02)))
        line = cv2.GaussianBlur(line, (0, 0), face.iod * 0.015)[..., None]
        base = np.broadcast_to(fill, f.shape) * (1 - 0.45 * line)
    else:
        base = np.broadcast_to(fill, f.shape)
    return f * (1 - m) + base * m


def mouth_open(face: Face) -> float:
    gap = np.linalg.norm(face.pts[13] - face.pts[14])
    return float(gap / face.iod)


def express(img: np.ndarray, face: Face, emotion: str, strength: float = 1.0, mouth: str = "auto") -> np.ndarray:
    """mouth: 'auto' moves the mouth only when the photo's mouth is closed (teeth can't be warped away, tested:
    painting the opening over reads as a smear), 'paint' forces close_mouth(), 'off' leaves it alone."""
    if emotion not in EMOTIONS:
        raise ValueError(emotion)
    teeth = mouth_open(face) > 0.06
    if mouth == "paint" and emotion != "joy" and teeth:
        img = close_mouth(img, face, cavity=emotion in ("fear", "surprise"))
    idx, disp = _recipe(face, emotion, strength)
    if mouth == "off" or (mouth == "auto" and teeth and emotion != "joy"):
        keep = ~np.isin(idx, MOUTH_ALL)
        idx, disp = idx[keep], disp[keep]
    if len(idx) == 0:
        return img.copy()
    A = anchors(face, img.shape)
    src = np.vstack([face.pts[idx], A])
    d = np.vstack([disp, np.zeros_like(A)])
    return warp_points(img, src, d, sigma=0.16 * face.iod)


# ------------------------------------------------------------- identity states
def face_mask(face: Face, shape, grow=1.0, feather=0.25):
    h, w = shape[:2]
    c = face.center
    poly = (c + (face.pts[FACE_OVAL] - c) * grow).astype(np.int32)
    m = np.zeros((h, w), np.float32)
    cv2.fillPoly(m, [poly], 1.0)
    k = max(3, int(face.iod * feather) | 1)
    return cv2.GaussianBlur(m, (0, 0), k / 2)


def different_face(img: np.ndarray, face: Face, amount: float = 1.0) -> np.ndarray:
    """Same person (hair, build, clothes, colouring) with a different face: re-proportion every feature.
    Eyes set wider and lower, longer nose, wider mouth lower down, broader jaw, heavier brows."""
    P, c, iod = face.pts, face.center, face.iod
    moves = {}

    def add(ids, dx, dy, outward=False):
        for i in ids:
            sx = np.sign(P[i][0] - c[0]) if outward else 1.0
            moves[i] = moves.get(i, np.zeros(2, np.float32)) + np.array([dx * sx, dy], np.float32)

    eyes = EYE_L + EYE_R + LID_L_UP + LID_L_LO + LID_R_UP + LID_R_LO
    add(list(set(eyes)), 0.09, 0.05, outward=True)
    add(BROW_L_INNER + BROW_L_MID + BROW_L_OUTER + BROW_R_INNER + BROW_R_MID + BROW_R_OUTER, 0.06, 0.08, outward=True)
    add([1, 4, 2], 0, 0.10)
    add(list(NOSE_WINGS), 0.07, 0.08, outward=True)
    add([MOUTH_L, MOUTH_R], 0.10, 0.11, outward=True)
    add(UPPER_LIP + LOWER_LIP, 0, 0.11)
    add(JAW, 0.14, 0.04, outward=True)
    idx = np.array(list(moves), int)
    disp = np.array([moves[i] for i in idx], np.float32) * iod * amount
    A = anchors(face, img.shape, scale=1.25)
    out = warp_points(img, np.vstack([P[idx], A]), np.vstack([disp, np.zeros_like(A)]), sigma=0.22 * iod)
    # a different face also reads as different skin: shift tone and soften the original's fine detail
    m = face_mask(face, img.shape, grow=0.9)[..., None]
    f = _f32(out)
    soft = cv2.bilateralFilter(f, 9, 0.08, 0.04 * iod)
    lab = cv2.cvtColor(soft, cv2.COLOR_RGB2LAB)
    lab[..., 0] *= 0.93
    lab[..., 1] += 3.0
    shifted = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(f * (1 - m * amount) + shifted * m * amount, 0, 1)


def stranger_face(img: np.ndarray, face: Face, amount: float = 1.0) -> np.ndarray:
    """The dream stranger: head, hair and light kept; eyes, nose and mouth dissolve into smooth skin."""
    f = _f32(img)
    m = face_mask(face, img.shape, grow=0.82, feather=0.45)[..., None] * amount
    iod = face.iod
    smooth = cv2.GaussianBlur(f, (0, 0), 0.32 * iod)                   # only the broad shading survives
    skin = np.median(f[face_mask(face, img.shape, 0.6, 0.01) > 0.5], axis=0)
    smooth = 0.55 * smooth + 0.45 * (skin + (smooth - smooth.mean((0, 1)))) # flatten toward skin colour
    noise = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1, f.shape[:2]).astype(np.float32), (0, 0), 0.08 * iod)
    smooth = smooth * (1 + 0.04 * noise[..., None])
    return np.clip(f * (1 - m) + smooth * m, 0, 1)


def obscured_face(img: np.ndarray, face: Face, amount: float = 1.0) -> np.ndarray:
    """Face turned into shadow, as if backlit: silhouette and hair rim stay, the features sink into dark."""
    f = _f32(img)
    m = face_mask(face, img.shape, grow=1.05, feather=0.6)[..., None] * amount
    dark = cv2.GaussianBlur(f, (0, 0), 0.12 * face.iod) * 0.12
    return np.clip(f * (1 - m) + dark * m, 0, 1)


def dream_face(img, face: Face, state: str = "known_clear", emotion: str = "neutral", strength: float = 1.0):
    out = express(img, face, emotion, strength) if state in ("known_clear", "known_different") else _f32(img)
    if state == "known_different":
        out = different_face(out, face)
    elif state == "stranger":
        out = stranger_face(out, face)
    elif state == "obscured":
        out = obscured_face(out, face)
    return _f32(out)


def cosine(a, b):
    return float(np.dot(a, b)) if a is not None and b is not None else float("nan")


# ------------------------------------------------------------- scene composite
def composite(bg: np.ndarray, bg_depth: np.ndarray, person: np.ndarray, mask: np.ndarray,
              x: float = 0.68, height: float = 0.92, dist: float = 2.2):
    """Place a cut-out person (bottom-aligned) in a rendered scene. Returns image, depth and the paste box."""
    H, W = bg.shape[:2]
    ph = int(H * height)
    pw = int(person.shape[1] * ph / person.shape[0])
    p = cv2.resize(person, (pw, ph), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    x0, y0 = int(W * x - pw / 2), H - ph
    out, dep = bg.copy(), bg_depth.copy()
    xs0, xs1 = max(0, x0), min(W, x0 + pw)
    sub = slice(xs0 - x0, xs1 - x0)
    mm = m[:, sub][..., None]
    # light wrap: tint the person toward the scene's ambient colour so they sit in the dream's light
    amb = cv2.GaussianBlur(bg, (0, 0), W * 0.05)[y0:H, xs0:xs1]
    pp = p[:, sub] * (0.78 + 0.22 * amb / max(float(amb.mean()), 1e-3))
    out[y0:H, xs0:xs1] = out[y0:H, xs0:xs1] * (1 - mm) + np.clip(pp, 0, 1) * mm
    dep[y0:H, xs0:xs1] = np.where(m[:, sub] > 0.5, dist, dep[y0:H, xs0:xs1])
    return out, dep, (x0, y0, pw, ph)
