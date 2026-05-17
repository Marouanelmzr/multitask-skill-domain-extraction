#!/bin/bash
set -e

cd "$(dirname "$0")/.."   # go to /app from /app/scripts/

MAIN_MODEL="models/model.onnx"
NORMALISER_MODEL="models/normaliser_onnx/model.onnx"

if [ -f "$MAIN_MODEL" ] && [ -f "$NORMALISER_MODEL" ]; then
    echo "Models already present, skipping download."
    exit 0
fi

echo "Downloading model files..."

[ -f "$MAIN_MODEL" ] || gdown "11RzVHpIHvuXV3G1dKEWHaRU-W6NCljCm" -O "$MAIN_MODEL"
[ -f "$NORMALISER_MODEL" ] || gdown "12cOJnaBG4jSzQMZpkwtRUShKR81HqnnO" -O "$NORMALISER_MODEL"

echo "Done."