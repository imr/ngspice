#!/bin/bash

if test -n "$1" && test -n "$2" ; then
rm -rf $2
python3 textract.py $1 $2
else
echo "arg 1 is the paranoia test script"
echo "arg 2 is the test script working directory"
exit 1
fi

SECONDS=0

time parallel -j8 bash ::: $2/*

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

