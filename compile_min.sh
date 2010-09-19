#!/bin/sh
# ngspice build script for MINGW-w64
# compile_min.sh

#Procedure:
# Install MSYS, plus bison, flex, auto tools, perl, libiconv, libintl
# Install MINGW-w64, activate OpenMP support
#     See either http://mingw-w64.sourceforge.net/ or http://tdm-gcc.tdragon.net/
#     (allows to generate either 32 or 64 bit executables by setting flag -m32 or -m64)
# set path to compiler in msys/xx/etc/fstab (e.g. c:/MinGW64 /mingw)
# start compiling with
# './compile_min.sh' or './compile_min.sh 64'

# Options:
# --adms and --enable-adms will install extra HICUM, EKV and MEXTRAM models via the 
# adms interface.
# CIDER, XSPICE, and OpenMP may be selected at will.
# --disable-debug will give O2 optimization (versus O0 for debug) and removes all debugging info.


UNAME_ALL=`(uname -a) 2>/dev/null` || UNAME_ALL=unknown
tes="x"
tes=`echo $UNAME_ALL | sed 's/.*MINGW.*$//'`
if test -n "$tes"; then echo "Only for MINGW!"; exit 1; fi

./autogen.sh --adms
echo
if test "$1" = "64"; then
echo "configuring for 64 bit"
echo
./configure --with-windows --enable-xspice --enable-cider --enable-openmp --enable-adms --disable-debug prefix="C:/Spice64" CFLAGS="-m64" LDFLAGS="-m64"
else
echo "configuring for 32 bit"
echo
./configure --with-windows --enable-xspice --enable-cider --enable-openmp --enable-adms --disable-debug CFLAGS="-m32" LDFLAGS="-m32"
fi

echo
# make clean is required for properly making the code models
echo "cleaning (see make_clean.log)"
make clean > make_clean.log 2>&1
if [ $? -ne 0 ]; then  echo "make clean failed"; exit 1 ; fi
echo "compiling (see make.log)"
make > make.log 2>&1
if [ $? -ne 0 ]; then  echo "make failed"; exit 1 ; fi
# 32 bit: Install to C:\Spice
# 64 bit: Install to C:\Spice64
echo "installing (see make_install.log)"
make install > make_install.log 2>&1
if [ $? -ne 0 ]; then  echo "make install failed"; exit 1 ; fi

echo "success"
exit 0
