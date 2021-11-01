#!/bin/bash
# ngspice build script for macOS, release or debug version, 64 bit
# compile_macos.sh <d>

# Procedure:
# Install gcc, bison, flex, libtool, autoconf, automake,
# libx11 and libx11-dev (headers), libXaw and libXaw-dev, libreadline and dev
# Declare 'compile_linux.sh' executable and start compiling with
# './compile_macos.sh' or './compile_macos.sh d' from the ngspice directory.
# Options:
# --adms and --enable-adms will install extra HICUM, EKV and MEXTRAM models via the
# adms interface. You need to download and install the *.va files via ng-adms-va.tgz
# Please see the ngspice manual, chapt. 13, for more info on adms.
# CIDER, XSPICE, and OpenMP may be selected at will.
# --disable-debug will give O2 optimization (versus O0 for debug) and removes all debugging info.

# ngspice as shared library:
# Replace --with-x by --with-ngshared in line ../configure ... .
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
# Builtin readline is not compatible (Big Sur), readline via Homebrew required (in /usr/local/opt)
# Use gcc-11 from Homebrew to support OpenMP
  ../configure --with-x --enable-xspice --enable-cider --with-readline=/usr/local/opt/readline CC="gcc-11" CXX="g++-11" CFLAGS="-m64 -O0 -g -Wall -I/opt/X11/include/freetype2 -I/usr/local/opt/readline/include" LDFLAGS="-m64 -g -L/usr/local/opt/readline/lib -L/opt/X11/lib"
else
   cd release
   if [ $? -ne 0 ]; then  echo "cd release failed"; exit 1 ; fi
  echo "configuring for 64 bit release"
  echo
# You may add  --enable-adms to the following command for adding adms generated devices
  ../configure --with-x --enable-xspice --enable-cider --with-readline=/usr/local/opt/readline --disable-debug --enable-openmp CC="gcc-11" CXX="g++-11" CFLAGS="-m64 -O2 -I/opt/X11/include/freetype2 -I/usr/local/opt/readline/include -I/usr/local/opt/ncurses/include -I/usr/local/include" LDFLAGS="-m64 -L/usr/local/opt/readline/lib -L/usr/local/opt/ncurses/lib -L/opt/X11/lib -L/usr/local/lib"
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
# Install to /usr/local
echo "installing (see make_install.log)"
make install 2>&1 | tee make_install.log
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make install failed"; exit 1 ; fi

ELAPSED="Elapsed compile time: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo
echo $ELAPSED
echo "success"
exit 0
