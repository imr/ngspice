#! /bin/sh

NGSPICE=$1
TEST=$2

DIFFPIPE="Added|Got|Reference|Analysis|CPU|memory|Date|Note|Sun|Mon|Tue|Wed|Thu|Fri|Sat|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

testname=`basename $TEST .cir`
testdir=`dirname $TEST`
$NGSPICE --batch $testdir/$testname.cir 2>&1 | egrep -v $DIFFPIPE > $testname.test
if diff -u $testdir/$testname.out $testname.test; then
    rm $testname.test
    exit 0
fi
exit 1
