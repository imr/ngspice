#! /bin/sh

NGSPICE=../src/ngspice
DIFFPIPE="Analysis|CPU|memory|Date|Note"

function spicetest () {
    $NGSPICE < $srcdir/$1.cir 2>&1 | egrep -v $DIFFPIPE > $1.test
    if diff -u $srcdir/$1.out $1.test; then
	rm $1.test
	exit 0
    fi
    exit 1
}
