#!/bin/sh
# Face models for dreamrender/faces.py (CPU). ~195 MB. Hugging Face is blocked in the cloud box; these hosts work.
# LICENCE: the InsightFace recognition model (w600k_r50, from buffalo_l) is for non-commercial research only.
# It is used here only as a QA metric; production needs a commercially licensed face-ID model.
set -e
mkdir -p "$(dirname "$0")/models" && cd "$(dirname "$0")/models"
G=https://storage.googleapis.com/mediapipe-models
[ -f face_landmarker.task ] || curl -fsSL -o face_landmarker.task $G/face_landmarker/face_landmarker/float16/1/face_landmarker.task
[ -f selfie_multiclass_256x256.tflite ] || curl -fsSL -o selfie_multiclass_256x256.tflite $G/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
if [ ! -f w600k_r50.onnx ]; then curl -fsSL -o /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && unzip -o -j /tmp/buffalo_l.zip w600k_r50.onnx -d . && rm /tmp/buffalo_l.zip; fi
echo ok
