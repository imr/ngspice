#!/bin/bash

if test -n "$1" && test -n "$2" ; then
  if test -d "$2" || test -e "$2" ; then
    echo "$2 already exists, remove it first"
    exit 1
  fi
python3 textract.py $1 $2
else
echo "arg 1 is the paranoia test script"
echo "arg 2 is the test script working directory"
exit 1
fi

SECONDS=0

time parallel -j4 bash ::: $2/*

wait
NGSPICE_OK="`ngspice -v | awk '/level/ {print $2;}'` done"

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

