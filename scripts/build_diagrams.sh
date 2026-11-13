#!/bin/sh
# Render the three system-model views and the algorithm box to 300-dpi PNG (needs pdflatex + pdftoppm).
set -e
cd "$(dirname "$0")/../figures/diagrams"
for f in sys_physical sys_functional sys_temporal algorithm; do
  pdflatex -interaction=nonstopmode "$f.tex" > /dev/null
  pdftoppm -r 300 -png -singlefile "$f.pdf" "$f"
  rm -f "$f.aux" "$f.log" "$f.out" "$f.pdf"
done
echo "diagrams rendered in figures/diagrams/"
