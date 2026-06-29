#!/bin/bash

# Copyright 2024 The ngspice team
# Authors: Jim Monte, Holger Vogt, Dietmar Warning
# License: New BSD

if test "$1" = "?"; then
    echo "ngspice \"paranoia\" test suite"
    echo "Format: $0 [ngspice executable]"
    echo "The ngspice executable must be independent of the current directory."
    exit 1
fi

SECONDS=0

NGSPICE="$1"
NGSPICE="${NGSPICE:-ngspice}"
VALGRIND="valgrind --leak-check=full --suppressions=$(pwd)/ignore_shared_libs.supp"

## The following three take much time. They are started in the background and may be skipped after being run once.
cd examples/xspice/table
$NGSPICE -o ../../../table-generator-q-2d.log  table-generator-q-2d.sp
$NGSPICE -o ../../../table-generator-b4n-2d.log table-generator-b4n-2d.sp
$NGSPICE -o ../../../table-generator-b4p-2d.log table-generator-b4p-2d.sp
$NGSPICE -o ../../../table-generator-b4n-3d.log table-generator-b4n-3d.sp
$NGSPICE -o ../../../table-generator-b4p-3d.log table-generator-b4p-3d.sp
cd ../../..



# Check the results
# Find correct response: ngspice-<version> done
NGSPICE_OK="`$NGSPICE -v | awk '/level/ {print $2;}'` done"

echo "*******************************************"
echo "vlog files with errors found by valgrind:"
grep -L "ERROR SUMMARY: 0 errors from 0 context" ./*.vlog
echo "*******************************************"
echo "log files with ngspice errors:"
grep -L "$NGSPICE_OK" ./*.log
echo "*******************************************"
echo "log files with convergence issues:"
grep -l "Too many iterations without convergence" ./*.log
echo "*******************************************"
echo "log files with messages containing 'error':"
grep -i -l "error" ./*.log
echo "*******************************************"

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo
echo $ELAPSED

