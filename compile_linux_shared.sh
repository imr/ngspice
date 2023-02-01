#!/bin/bash
# ngspice build script for Linux, release or debug version, 64 bit
# compile_linux_shared.sh <d>

# Procedure:
# Install gcc, bison, flex, libtool, autoconf, automake, 
# (not needed for shared library: libx11 and libx11-dev (headers), libXaw and libXaw-dev, libreadline and dev)
# Declare 'compile_linux_shared.sh' as being executable and start compiling with
# './compile_linux_shared.sh' or './compile_linux_shared.sh d' from the ngspice directory.
# Options:
# --enable-osdi will add the osdi interface which allows to dynamically load compiled Verilog-A 
# compact models. Compiling the VA code of the models is done by the OpenVAF compiler.
# Please see the ngspice manual, chapt. 13, for more info on OSDI/OpenVAF.
# CIDER, XSPICE, and OpenMP may be selected at will.
# --disable-debug will give O2 optimization (versus O0 for debug) and removes all debugging info.

# Add (optionally) --enable-relpath to avoid absolute paths when searching for code models.
# It might be necessary to uncomment and run ./autogen.sh especially if sources have been
# cloned from a git repository.

SECONDS=0

if test "$1" = "d"; then
   if [ ! -d "debugsh" ]; then
      mkdir debugsh
      if [ $? -ne 0 ]; then  echo "mkdir debugsh failed"; exit 1 ; fi
   fi   
else
   if [ ! -d "releasesh" ]; then
      mkdir releasesh
      if [ $? -ne 0 ]; then  echo "mkdir releasesh failed"; exit 1 ; fi
   fi
fi

# If compiling sources from git, you may need to uncomment the following two lines:
./autogen.sh
if [ $? -ne 0 ]; then  echo "./autogen.sh failed"; exit 1 ; fi

echo
if test "$1" = "d"; then
   cd debugsh
   if [ $? -ne 0 ]; then  echo "cd debugsh failed"; exit 1 ; fi
  echo "configuring shared lib for 64 bit, debug enabled"
  echo
# The --prefix (and perhaps --libdir) may be used to determine a different install location
# (depending on the Linux distribution, and on the calling programs search path).
  ../configure --with-ngshared --enable-xspice --enable-cider --enable-openmp --enable-osdi --prefix=/usr CFLAGS="-g -m64 -O0 -Wall" LDFLAGS="-m64 -g"
else
   cd releasesh
   if [ $? -ne 0 ]; then  echo "cd releasesh failed"; exit 1 ; fi
  echo "configuring shared lib for 64 bit release"
  echo
# The --prefix (and perhaps --libdir) may be used to determine a different install location
# (depending on the Linux distribution, and on the calling programs search path).
  ../configure --with-ngshared --enable-xspice --enable-cider --enable-openmp --disable-debug --enable-osdi --prefix=/usr CFLAGS="-m64 -O2" LDFLAGS="-m64 -s"
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
