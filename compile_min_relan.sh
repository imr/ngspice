#!/bin/bash
# ngspice-26 from git branch Reliability_Analysis_New_Model.
# build script for MINGW MSYS2, release version, 64 bit
# compile_min_relan.sh

#Procedure:
# Install MSYS2, plus bison, flex, auto tools, perl, libiconv, libintl gsl
#     See https://www.msys2.org/
# activate OpenMP support

# Only generates 64 bit executables by setting flag -m64

# start compiling with
# './compile_min_relan.sh'

# Options:
# Not compatible with CIDER! 
# XSPICE, and OpenMP may be selected at will.
# --disable-debug will give O2 optimization (versus O0 for debug) and removes all debugging info.


if [ ! -d "release64" ]; then
   mkdir release64
   if [ $? -ne 0 ]; then  echo "mkdir release64 failed"; exit 1 ; fi
fi   


# If compiling sources from git, you may need to uncomment the following two lines:
./autogen.sh
if [ $? -ne 0 ]; then  echo "./autogen.sh failed"; exit 1 ; fi

echo
cd release64
if [ $? -ne 0 ]; then  echo "cd release64 failed"; exit 1 ; fi
echo "configuring for 64 bit"
echo
../configure --with-wingui --disable-debug --enable-xspice --enable-openmp --enable-relan prefix="C:/Spice64" CFLAGS="-m64 -O2" LDFLAGS="-m64 -s"

if [ $? -ne 0 ]; then  echo "../configure failed"; exit 1 ; fi

echo
# make clean is required for properly making the code models
echo "cleaning (see make_clean.log)"
make clean 2>&1 -j8 | tee make_clean.log
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make clean failed"; exit 1 ; fi
echo "compiling (see make.log)"
make 2>&1 -j8 | tee make.log
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make failed"; exit 1 ; fi
# 32 bit: Install to C:\Spice
# 64 bit: Install to C:\Spice64
echo "installing (see make_install.log)"
make install 2>&1 | tee make_install.log 
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make install failed"; exit 1 ; fi

echo "success"
exit 0
