#!/bin/bash
# ngspice build script for Linux distributable, 64 bit
# compile_linux_dist.sh <d>

# Procedure:
# Install gcc, bison, flex, libtool, autoconf, automake, 
# libx11 and libx11-dev (headers), libXaw and libXaw-dev, libreadline and dev
# Declare 'compile_linux_dist.sh' executable and start compiling with
# './compile_linux_dist.sh' from the ngspice directory.

SECONDS=0


if [ ! -d "release" ]; then
   mkdir release
   if [ $? -ne 0 ]; then  echo "mkdir release failed"; exit 1 ; fi
fi


# If compiling sources from git, you may need to uncomment the following two lines:
./autogen.sh
if [ $? -ne 0 ]; then  echo "./autogen.sh failed"; exit 1 ; fi

echo
cd release
if [ $? -ne 0 ]; then  echo "cd release failed"; exit 1 ; fi
echo "configuring for 64 bit release"
echo

../configure --with-x --enable-xspice --enable-cider --with-readline=yes --enable-openmp --enable-osdi --disable-debug CFLAGS="-m64 -O2" LDFLAGS="-m64 -s"

if [ $? -ne 0 ]; then  echo "../configure failed"; exit 1 ; fi

echo
# make clean is required for properly making the code models
echo "cleaning (see make_clean.log)"
make clean 2>&1 -j8 | tee make_clean.log
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make clean failed"; exit 1 ; fi
echo "generate distribution (see make_dist.log)"
make dist 2>&1 -j8 | tee make_dist.log
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make dist failed"; exit 1 ; fi

ELAPSED="Elapsed compile time: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo
echo $ELAPSED
echo "success"
exit 0
