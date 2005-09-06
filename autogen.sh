#!/bin/sh
# Run this to generate all the initial makefiles, etc.

PROJECT=ng-spice
TEST_TYPE=-f
FILE=DEVICES

DIE=0

(autoconf --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have autoconf installed to compile $PROJECT."
	echo "See http://www.gnu.org/software/automake/"
	echo "(newest stable release is recommended)"
	DIE=1
}

(libtoolize --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have libtool installed to compile $PROJECT."
	echo "See http://www.gnu.org/software/libtool/"
	echo "(newest stable release is recommended)"
	DIE=1
}

(automake --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have automake installed to compile $PROJECT."
	echo "See http://www.gnu.org/software/automake/"
	echo "(newest stable release is recommended)"
	DIE=1
}

if test "$DIE" -eq 1; then
	exit 1
fi

test $TEST_TYPE $FILE || {
	echo "You must run this script in the top-level $PROJECT directory"
	exit 1
}


libtoolize --copy --force
aclocal $ACLOCAL_FLAGS
# optionally feature autoheader
(autoheader --version)  < /dev/null > /dev/null 2>&1 && autoheader
automake -c -a $am_opt
autoconf
