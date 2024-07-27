#!/bin/bash
# ngspice build script for CYGWIN console (X11), release version, 64 bit
# compile_cyg_make_short_check.sh

# short version, skipping several of the device checks

#Procedure:
# Install CYGWIN, plus bison, flex, auto tools, perl, libiconv, libintl
# Install gcc, activate OpenMP support
# start compiling with
# './compile_cyg_auto.sh'

# Options:
# CIDER may be selected at will.
# XSPICE, OSDI, KLU, and OpenMP may be deselected if not require
# --enable-oldapps will make ngnutmeg ngsconvert ngproc2mod ngmultidec ngmakeidx in addition to ngspice
# --enable-shortcheck will provide a fast 'make check' by checking only BSIM3 and BSIM4

if [ ! -d "release64_cyg" ]; then
   mkdir release64_cyg
   if [ $? -ne 0 ]; then  echo "mkdir release64_cyg failed"; exit 1 ; fi
fi

# If compiling sources from CVS, you may need to uncomment the following two lines:
./autogen.sh
if [ $? -ne 0 ]; then  echo "./autogen.sh failed"; exit 1 ; fi

echo
cd release64_cyg
if [ $? -ne 0 ]; then  echo "cd release64_cyg failed"; exit 1 ; fi
echo
../configure --with-x=yes --enable-cider --enable-shortcheck CFLAGS="-O2 -m64" LDFLAGS="-s -m64"

if [ $? -ne 0 ]; then  echo "../configure failed"; exit 1 ; fi

echo
# make clean is required for properly making the code models
#echo "cleaning (see make_clean.log)"
#make clean 2>&1 -j8 | tee make_clean.log 
#exitcode=${PIPESTATUS[0]}
#if [ $exitcode -ne 0 ]; then  echo "make clean failed"; exit 1 ; fi
echo "compiling (see make.log)"
make 2>&1 -j8 | tee make.log
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make failed"; exit 1 ; fi
echo "installing (see make_install.log)"
make install 2>&1 -j8 | tee make_install.log 
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then  echo "make install failed"; exit 1 ; fi
echo "run make check"
make check 2>&1 -j8 | tee make_check.log 
exitcode=${PIPESTATUS[0]}
if [ $exitcode -ne 0 ]; then
    echo "make check failed";
    echo "Did you consider setting 'set ngbehavior=mc' in .spiceinit?";
    exit 1 ;
fi

echo "success"
exit 0
