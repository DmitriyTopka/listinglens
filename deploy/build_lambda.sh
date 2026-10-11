#!/usr/bin/env bash
# Build the three Lambda artefacts (arm64, Python 3.12) from pinned wheels. Works on any OS.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$HERE/build}
rm -rf "$OUT" && mkdir -p "$OUT"/{wheels,pkg,L1/python,L2/python,fn/listinglens,fn/app}
cd "$OUT"
pip download --only-binary=:all: --platform manylinux2014_aarch64 --python-version 3.12 --implementation cp \
  -d wheels -r "$HERE/requirements-lambda.txt"
for w in wheels/*.whl; do unzip -q -o "$w" -d pkg; done
find pkg -name tests -type d -prune -exec rm -rf {} +
find pkg -name '*.pyi' -delete
mv pkg/numpy pkg/numpy.libs pkg/numpy-*.dist-info L1/python/
mv pkg/cv2 pkg/opencv_python_headless.libs pkg/opencv_python_headless-*.dist-info L2/python/
(cd L1 && zip -q -9 -r ../layer-numpy.zip python)
(cd L2 && zip -q -9 -r ../layer-opencv.zip python)
cp "$HERE"/listinglens/*.py fn/listinglens/
cp "$HERE"/app/lambda_function.py "$HERE"/app/index.html fn/app/
cp -r "$HERE"/app/samples fn/app/
echo 'from app.lambda_function import handler  # noqa: F401' > fn/lambda_function.py
(cd fn && zip -q -9 -r ../function.zip .)
ls -la "$OUT"/*.zip
