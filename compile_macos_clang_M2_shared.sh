#!/bin/bash
# ngspice build script for macOS, release or debug version, 64 bit
# compile_macos_clang_M2.sh <d>
# tested with Mac mini, Apple M2

# Procedure:
# Install gcc, bison, flex, libtool, autoconf, automake,
# libx11 and libx11-dev (headers), libXaw and libXaw-dev, libreadline and dev
# XCODE, commandline tools
# Declare 'compile_macos_clang_M2.sh' executable and start compiling with
# './compile_macoss_clang_M2.sh' or './compile_macoss_clang_M2.sh d' from the ngspice directory.
# Options:
# CIDER may be selected at will.
# OpenMP has been installed from https://mac.r-project.org/openmp/

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

echo
if test "$1" = "d"; then
   cd debug
   if [ $? -ne 0 ]; then  echo "cd debug failed"; exit 1 ; fi
  echo "configuring for 64 bit debug"
  echo

  ../configure --with-ngshared --enable-cider --with-readline=/opt/homebrew/opt/readline --enable-debug CFLAGS="-m64 -O0 -g -Wall -I/opt/X11/include/freetype2 -I/opt/homebrew/opt/readline/include" LDFLAGS="-m64 -g -L/opt/homebrew/opt/readline/lib -L/opt/X11/lib -L/usr/local/lib -lomp"
else
   cd release
   if [ $? -ne 0 ]; then  echo "cd release failed"; exit 1 ; fi
  echo "configuring for 64 bit release"
  echo
  ../configure --with-ngshared --enable-cider --with-readline=/opt/homebrew/opt/readline CFLAGS="-m64 -O2 -I/opt/X11/include/freetype2 -I/opt/homebrew/opt/readline/include -I/opt/homebrew/opt/ncurses/include" LDFLAGS="-m64 -L/opt/homebrew/opt/readline/lib -L/opt/homebrew/opt/ncurses/lib -L/opt/X11/lib -L/usr/local/lib -lomp"
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
