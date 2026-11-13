#!/bin/sh
# Reproduce every number, figure and table of the paper (about 45 min on 4 CPU cores).
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[1/5] unit tests";           python3 tests/test_seasel.py
echo "[2/5] testbed + experiments"; python3 src/experiments.py
echo "[3/5] figures and tables";   python3 src/make_figures.py > /dev/null
echo "[4/5] diagrams";             sh scripts/build_diagrams.sh
echo "[5/5] manuscript";           python3 manuscript/build_manuscript.py
echo "Done."
