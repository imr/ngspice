#!/bin/sh

# cmc model check specific driver for the automake `check' target

# arguments:
#   ngspice-executable subdirectory/qaSpec-file

# (compile "./check_cmc.sh thexec nmos/qaSpec")

executable="$1"
qaspec="$2"
subdir="$(dirname $2)"

echo "qaspec = $qaspec"
echo "subdir = $subdir"
echo "executable = $executable"

exec "$(dirname $0)/run_cmc_check" \
    --executable="${executable}" \
    --srcdir="${subdir}/" \
    -qa "${qaspec}" \
    ngspice
