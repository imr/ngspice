#!/bin/bash
# ngspice build script for MINGW in MSYS2, release or debug version, 64 bit
# compile_min_shared.sh
# tested with macOS BigSur 11.7.9, MacBook Air i5

#Procedure:
# start compiling with
# './compile_macos_shared.sh' for release or './compile_macos_shared.sh d'
# for debug version of shared ngspice

# Options:
# KLU, OSDI, XSPICE, and OpenMP may be deselected at will.

# Add (optionally) --enable-relpath to avoid absolute paths when searching for code models.
# It might be necessary to uncomment and run ./autogen.sh .

SECONDS=0

if test "$1" = "d"; then
   if [ ! -d "debug-sh" ]; then
      mkdir debug-sh
      if [ $? -ne 0 ]; then  echo "mkdir debug-sh failed"; exit 1 ; fi
   fi   
else
   if [ ! -d "release-sh" ]; then
      mkdir release-sh
      if [ $? -ne 0 ]; then  echo "mkdir release-sh failed"; exit 1 ; fi
   fi
fi

# If compiling sources from git, you may need to uncomment the following two lines:
./autogen.sh
if [ $? -ne 0 ]; then  echo "./autogen.sh failed"; exit 1 ; fi

echo
if test "$1" = "d"; then
   cd debug-sh
   if [ $? -ne 0 ]; then  echo "cd debug-sh failed"; exit 1 ; fi
  echo "configuring for 64 bit debug"
  echo
  ../configure --with-ngshared --enable-cider CFLAGS="-O0" LDFLAGS=" -lomp"
else
   cd release-sh
   if [ $? -ne 0 ]; then  echo "cd release-sh failed"; exit 1 ; fi
  echo "configuring for 64 bit release"
  echo
  ../configure --with-ngshared  --enable-cider CC="gcc-11" CXX="g++-11" CFLAGS="-m64 -O2 -I/usr/local/include" LDFLAGS="-m64 -L/usr/local/lib"
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

ELAPSED="Elapsed compile time: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo
echo $ELAPSED
echo "success"
exit 0
