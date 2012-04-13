#!/bin/sh
# Configuration script for ngspice.
#
# This script performs initial configuration of ngspice source
# package.

PROJECT=ngspice

# Exit variable
DIE=0


help()
{
    echo
    echo "$PROJECT autogen.sh help"
    echo
    echo "--help     -h: print this file"
    echo "--version  -v: print version"
    echo
}

version()
{
    echo
    echo "$PROJECT autogen.sh 1.0"
    echo
}

error_and_exit()
{
    echo "Error: $1"
    exit 1
}

check_autoconf()
{
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
}


case "$1" in
    "--help" | "-h")
        help
        exit 0
        ;;

    "--version" | "-v")
        version
        exit 0
        ;;

    *)
        ;;
esac


check_autoconf

if [ "$DIE" -eq 1 ]; then
    exit 1
fi

[ -f "DEVICES" ] || {
    echo "You must run this script in the top-level $PROJECT directory"
    exit 1
}


echo "Running aclocal $ACLOCAL_FLAGS"
aclocal $ACLOCAL_FLAGS \
    || error_and_exit "aclocal failed"

echo "Running libtoolize"
libtoolize --copy --force \
    || error_and_exit "libtoolize failed"

# optional feature: autoheader
(autoheader --version) < /dev/null > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "Running autoheader"
  autoheader \
      || error_and_exit "autoheader failed"
fi

echo "Running automake -Wall --copy --add-missing"
automake  -Wall --copy --add-missing \
    || error_and_exit "automake failed"

echo "Running autoconf"
autoconf \
    || error_and_exit "autoconf failed"

echo "Success."
exit 0
