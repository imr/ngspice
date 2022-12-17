#!/bin/bash
# ngspice build script for MINGW-w64, release or debug version, 64 bit
# compile_min.sh

#Procedure:
# Install MSYS2, plus gcc 64 bit, libtool, autoconf, automake, bison, git, and make
#     See either https://github.com/orlp/dev-on-windows/wiki/Installing-GCC--&-MSYS2
#     (allows to generate either 32 or 64 bit executables by setting flag -m32 or -m64)
# start compiling with
# './compile_min.sh' for release or './compile_min.sh d' for debug version.

# Options:
# --adms and --enable-adms will install extra HICUM, EKV and MEXTRAM models via the 
# adms interface.
# Please see http://ngspice.sourceforge.net/admshowto.html for more info on adms.
# CIDER, XSPICE, and OpenMP may be selected at will.
# --disable-debug will give O2 optimization (versus O0 for debug) and removes all debugging info.
# --enable-oldapps will make ngnutmeg ngsconvert ngproc2mod ngmultidec ngmakeidx in addition to ngspice

# ngspice as shared library:
# Replace --with-wingui by --with-ngshared in line ../configure ... .
# Add (optionally) --enable-relpath to avoid absolute paths when searching for code models.
# It might be necessary to uncomment and run ./autogen.sh .

SECONDS=0

if test "$1" = "d"; then
   if [ ! -d "debug" ]; then
      mkdir debug
      if [ $? -ne 0 ]; then  echo "mkdir debug failed"; exit 1 ; fi
   fi   
else
   if [ ! -d "release" ]; then
      mkdir release
      if [ $? -ne 0 ]; then  echo "mkdir release failed"; exit 1 ; fi
   fi
fi

# If compiling sources from git, you may need to uncomment the following two lines:
./autogen.sh
if [ $? -ne 0 ]; then  echo "./autogen.sh failed"; exit 1 ; fi

# Alternatively, if compiling sources from git, and want to add adms created devices,
# you may need to uncomment the following two lines (and don't forget to add adms option
# to the ../configure statement):
#./autogen.sh --adms
#if [ $? -ne 0 ]; then  echo "./autogen.sh failed"; exit 1 ; fi

echo
if test "$1" = "d"; then
   cd debug
   if [ $? -ne 0 ]; then  echo "cd debug failed"; exit 1 ; fi
  echo "configuring for 64 bit debug"
  echo
# You may add  --enable-adms to the following command for adding adms generated devices 
  ../configure --with-wingui --enable-xspice --enable-cider --enable-openmp --enable-osdi --enable-predictor prefix="C:/Spice64d" CFLAGS="-g -m64 -O0 -Wall -Wno-unused-but-set-variable" LDFLAGS="-g -m64"
else
   cd release
   if [ $? -ne 0 ]; then  echo "cd release failed"; exit 1 ; fi
  echo "configuring for 64 bit release"
  echo
# You may add  --enable-adms to the following command for adding adms generated devices 
  ../configure --with-wingui --enable-xspice --enable-cider --enable-openmp --enable-osdi --enable-predictor --disable-debug prefix="C:/Spice64" CFLAGS="-m64 -O2" LDFLAGS="-m64 -s"
fi
if [ $? -ne 0 ]; then  echo "../configure failed"; exit 1 ; fi

echo
# make clean is required for properly making the code models
echo "cleaning (see make_clean.log)"
make clean 2>&1 -j8 | tee make_clean.log
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make clean failed"; exit 1 ; fi
# echo "compiling the icon"
# windres ../src/ngicon.rc -O coff -o ./src/ngicon.o
# exitcode=${PIPESTATUS[0]}
# if [ $exitcode -ne 0 ]; then  echo "compiling the icon failed"; exit 1 ; fi
echo "compiling (see make.log)"
make 2>&1 -j8 | tee make.log
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make failed"; exit 1 ; fi
# 64 bit debug: Install to C:\Spice64d
# 64 bit: Install to C:\Spice64
echo "installing (see make_install.log)"
make install 2>&1 | tee make_install.log 
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make install failed"; exit 1 ; fi

ELAPSED="Elapsed compile time: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo
echo $ELAPSED
echo "success"
exit 0
