#! /bin/sh

NGSPICE=../src/ngspice
DIFFPIPE="Analysis|CPU|memory|Date|Note"

function spicetest () {
    $NGSPICE < $srcdir/$1.cir 2>&1 | egrep -v $DIFFPIPE > $1.test
    if diff -u $1.test $srcdir/$1.out; then
	rm $1.test
	exit 0
    fi
    exit 1
}
