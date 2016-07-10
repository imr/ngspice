#!/bin/bash
# ngspice build script for LINUX, debug version, 32 or 64 bit
# compile_linux_d.sh

#Procedure:
# Install development packages (including headers) for Xext, Xaw, Xmu, readline
# Install gcc c++, libtool, bison, flex, auto tools
# cd into ngspice directory, start compiling with
# './compile_linux_d.sh' or './compile_linux_d.sh 64'

# Options:
# --adms and --enable-adms will install extra HICUM, EKV and MEXTRAM models via the
# adms interface.
# Please see http://ngspice.sourceforge.net/admshowto.html for more info on adms.
# CIDER, XSPICE, and OpenMP may be selected at will.


if test "$1" = "64"; then
   if [ ! -d "debug64" ]; then
      mkdir debug64
      if [ $? -ne 0 ]; then  echo "mkdir debug64 failed"; exit 1 ; fi
   fi
else
   if [ ! -d "debug" ]; then
      mkdir debug
      if [ $? -ne 0 ]; then  echo "mkdir debug failed"; exit 1 ; fi
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

# In the following ../configure commands you will find an additional entry to the CFLAGS
# '-fno-omit-frame-pointer'. This entry compensates for a compiler bug of actual mingw gcc 4.6.1.

echo
if test "$1" = "64"; then
   cd debug64
   if [ $? -ne 0 ]; then  echo "cd release64 failed"; exit 1 ; fi
  echo "configuring for 64 bit"
  echo
# You may add  --enable-adms to the following command for adding adms generated devices
  ../configure --with-x --with-readline=yes --enable-xspice --enable-cider --enable-openmp  CFLAGS="-m64 -g" LDFLAGS="-m64 -g"
else
   cd debug
   if [ $? -ne 0 ]; then  echo "cd release failed"; exit 1 ; fi
  echo "configuring for 32 bit"
  echo
# You may add  --enable-adms to the following command for adding adms generated devices
  ../configure --with-x --with-readline=yes --enable-xspice --enable-cider --enable-openmp  CFLAGS="-m32 -g" LDFLAGS="-m32 -g"
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
echo "installing (see make_install.log)"
make install 2>&1 | tee make_install.log
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make install failed"; exit 1 ; fi

echo "success"
exit 0
