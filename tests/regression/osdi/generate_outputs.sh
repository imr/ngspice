#!/bin/bash
OPENVA=~/bin/openva
SPICE=/usr/share/ngspice_VA2/bin/ngspice
FILTER="SPARSE|KLU|CPU|Dynamic|Note|Circuit|Trying|Reference|Date|Doing|---|v-sweep|time|est|Error|Warning|Data|Index|trans|acan|oise|nalysis|ole|Total|memory|urrent|Got|Added|BSIM|bsim|B4SOI|b4soi|codemodel|^binary raw file|^ngspice.*done|Operating"

for f in *.va; do
  name=$(basename "$f" .va)
  echo "Compiling $f..."
  $OPENVA -o "${name}.osdi" "$f"
done

for f in *.cir; do
  name=$(basename "$f" .cir)
  echo "Running $f..."
  $SPICE --batch -r "${name}.raw" "$f" > "${name}.test"
  egrep -v "$FILTER" "${name}.test" > "${name}.out"
  rm -f "${name}.test" "${name}.raw"
done
