#! /bin/sh
# 
# Ngspice test driver. 

NGSPICE=$(src_dir)ngspice
TEST=$1

DIFFPIPE="Analysis|CPU|memory|Date|Note|Mon|Tue|Wed|Thu|Fri|Sat|Sun"

testname=$(basename $TEST .cir)
testdir=$(dirname $TEST)
$NGSPICE < $testdir/$testname.cir 2>&1 | egrep -v $DIFFPIPE > $testname.test
exit 0 
