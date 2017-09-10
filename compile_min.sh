#!/bin/bash
# ngspice build script for MINGW-w64, release version, 32 or 64 bit
# compile_min.sh

#Procedure:
# Install MSYS, plus bison, flex, auto tools, perl, libiconv, libintl
# Install MINGW-w64, activate OpenMP support
#     See either http://mingw-w64.sourceforge.net/ or http://tdm-gcc.tdragon.net/
#     (allows to generate either 32 or 64 bit executables by setting flag -m32 or -m64)
# set path to compiler in msys/xx/etc/fstab (e.g. c:/MinGW64 /mingw)
# start compiling with
# './compile_min.sh' or './compile_min.sh 64'
# As an (more recent) alternative install MSYS2 and the tools cited above.

# Options:
# --adms and --enable-adms will install extra HICUM, EKV and MEXTRAM models via the 
# adms interface.
# Please see http://ngspice.sourceforge.net/admshowto.html for more info on adms.
# CIDER, XSPICE, and OpenMP may be selected at will.
# --disable-debug will give O2 optimization (versus O0 for debug) and removes all debugging info.

# ngspice as shared library:
# Replace --with-wingui by --with-ngshared in line ../configure ... .
# Add (optionally) --enable-relpath to avoid absolute paths when searching for code models.
# It might be necessary to uncomment and run ./autogen.sh .

if test "$1" = "64"; then
   if [ ! -d "release64" ]; then
      mkdir release64
      if [ $? -ne 0 ]; then  echo "mkdir release64 failed"; exit 1 ; fi
   fi   
else
   if [ ! -d "release" ]; then
      mkdir release
      if [ $? -ne 0 ]; then  echo "mkdir release failed"; exit 1 ; fi
   fi
fi

# If compiling sources from git, you may need to uncomment the following two lines:
#./autogen.sh
#if [ $? -ne 0 ]; then  echo "./autogen.sh failed"; exit 1 ; fi

# Alternatively, if compiling sources from git, and want to add adms created devices,
# you may need to uncomment the following two lines (and don't forget to add adms option
# to the ../configure statement):
#./autogen.sh --adms
#if [ $? -ne 0 ]; then  echo "./autogen.sh failed"; exit 1 ; fi

echo
if test "$1" = "64"; then
   cd release64
   if [ $? -ne 0 ]; then  echo "cd release64 failed"; exit 1 ; fi
  echo "configuring for 64 bit"
  echo
# You may add  --enable-adms to the following command for adding adms generated devices 
  ../configure --with-wingui --enable-xspice --enable-cider --enable-openmp --disable-debug prefix="C:/Spice64" CFLAGS="-m64 -O2" LDFLAGS="-m64 -s"
else
   cd release
   if [ $? -ne 0 ]; then  echo "cd release failed"; exit 1 ; fi
  echo "configuring for 32 bit"
  echo
# You may add  --enable-adms to the following command for adding adms generated devices 
  ../configure --with-wingui --enable-xspice --enable-cider --enable-openmp --disable-debug prefix="C:/Spice" CFLAGS="-m32 -O2" LDFLAGS="-m32 -s"
fi
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
