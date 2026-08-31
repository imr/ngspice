# NgspiceOptions.cmake -- the CMake equivalents of the ./configure switches.
#
#   configure                        CMake
#   ------------------------------   -----------------------------------------
#   --enable-xspice                  -DNGSPICE_ENABLE_XSPICE=ON
#   --enable-cider                   -DNGSPICE_ENABLE_CIDER=ON
#   --enable-osdi                    -DNGSPICE_ENABLE_OSDI=ON
#   --disable-klu                    -DNGSPICE_ENABLE_KLU=OFF
#   --with-fftw3=no                  -DNGSPICE_WITH_FFTW3=OFF
#   --with-readline=yes              -DNGSPICE_WITH_READLINE=readline
#   --with-editline=yes              -DNGSPICE_WITH_READLINE=editline
#   --with-ngshared                  -DNGSPICE_BUILD_SHARED=ON
#   --with-wingui                    -DNGSPICE_WINGUI=ON
#   --disable-openmp                 -DNGSPICE_ENABLE_OPENMP=OFF (default ON)
#   --without-x                      -DNGSPICE_ENABLE_X11=OFF
#   --enable-pss / --enable-sp       -DNGSPICE_ENABLE_PSS / _RFSPICE=ON
#   --enable-relpath                 -DNGSPICE_ENABLE_RELPATH=ON
#   --enable-debug                   -DCMAKE_BUILD_TYPE=Debug

# --- stage 1 features ------------------------------------------------------
option(NGSPICE_ENABLE_XSPICE      "XSPICE mixed-signal extensions"        ON)
option(NGSPICE_ENABLE_CODEMODELS  "Build the XSPICE .cm code models"      ON)
option(NGSPICE_ENABLE_CIDER       "CIDER numerical device models"         OFF)
option(NGSPICE_ENABLE_OSDI        "OSDI / OpenVAF model loading"          ON)
option(NGSPICE_ENABLE_NDEV        "NDEV external device interface"        OFF)
option(NGSPICE_ENABLE_PSS         "Periodic steady state analysis"        OFF)
# configure.ac: "if test x$enable_sp = xno ... else AC_DEFINE(RFSPICE)",
# i.e. on unless --disable-sp is given.
option(NGSPICE_ENABLE_RFSPICE     "S-parameter (RF) analysis"             ON)
option(NGSPICE_ENABLE_SENSE2      "Old sense2 sensitivity analysis"       OFF)
option(NGSPICE_ENABLE_CMATHTESTS  "Build the cmaths self tests"           OFF)
option(NGSPICE_ENABLE_OLDAPPS     "Build ngnutmeg/ngsconvert/... "        OFF)
option(NGSPICE_ENABLE_HELP        "Build the X11 help browser"            OFF)
option(NGSPICE_ENABLE_ADMS        "ADMS Verilog-A models (unsupported)"   OFF)

option(NGSPICE_ENABLE_KLU         "KLU sparse solver"                     ON)
set(NGSPICE_KLU_PROVIDER "bundled" CACHE STRING
    "Where KLU comes from: bundled (src/maths/KLU) or system (SuiteSparse)")
set_property(CACHE NGSPICE_KLU_PROVIDER PROPERTY STRINGS bundled system)

option(NGSPICE_WITH_FFTW3         "Use external FFTW3 for Fourier transforms" ON)

# configure.ac has ": ${enable_openmp:=yes}", i.e. OpenMP is on by default.
option(NGSPICE_ENABLE_OPENMP      "OpenMP parallel device loading"        ON)
option(NGSPICE_ENABLE_X11         "X11 plotting (Unix)"                   ON)
# --enable-relpath makes ivars.c use the literal string "../share/ngspice"
# for Spice_Lib_Dir, which is resolved against the *current directory*.
#
# On Windows that is the wrong tool: without HAS_RELPATH, ivars.c already
# takes ngdirname(argv0) and appends "../share/ngspice", and dlopen() in
# src/spicelib/devices/dev.c resolves a relative code model path against
# GetModuleFileName(NULL). So a Windows build is relocatable anyway, and
# independent of the working directory. visualc/src/include/ngspice/config.h
# does exactly this: NGSPICEDATADIR = "../share/ngspice", no HAS_RELPATH.
#
# Default it off everywhere; see NGSPICE_WINDOWS_RELATIVE_PATHS below.
option(NGSPICE_ENABLE_RELPATH "Look up spinit relative to the current directory" OFF)
# --enable-debug maps to the Debug configuration:
#   single config : -DCMAKE_BUILD_TYPE=Debug
#   multi config  : cmake --build <dir> --config Debug
# That configuration also defines NGDEBUG, as configure.ac does.

# --disable-utf8 -> defines EXT_ASC, and src/winmain.c then provides
# WinMain() instead of wWinMain(), which changes the MinGW/MSVC entry point.
option(NGSPICE_DISABLE_UTF8       "Disable UNICODE/utf-8 (defines EXT_ASC)" OFF)

set(NGSPICE_WITH_READLINE "auto" CACHE STRING
    "Command line editing: auto, readline, editline or none")
set_property(CACHE NGSPICE_WITH_READLINE PROPERTY STRINGS auto readline editline none)

# --- build products --------------------------------------------------------
# SHARED_MODULE changes the code globally (it redirects fprintf to sh_fprintf,
# among other things), so a configuration produces either the binary or the
# library. src/Makefile.am expresses the same with "if !SHARED_MODULE" around
# bin_PROGRAMS. One switch is therefore enough; NGSPICE_BUILD_EXECUTABLE
# follows from it and is not meant to be set by hand.
option(NGSPICE_BUILD_SHARED "Build libngspice instead of the ngspice binary" OFF)

if(NGSPICE_BUILD_SHARED)
    set(NGSPICE_BUILD_EXECUTABLE OFF)
else()
    set(NGSPICE_BUILD_EXECUTABLE ON)
endif()

cmake_dependent_option(NGSPICE_WINGUI "Native Windows GUI frontend" ON
                       "WIN32" OFF)

# --- stage 2 ---------------------------------------------------------------
option(NGSPICE_ENABLE_TESTS       "Register the tests/ suite with CTest"  ON)

# --- consistency -----------------------------------------------------------
if(WIN32 OR APPLE)
    set(NGSPICE_ENABLE_X11 OFF CACHE BOOL "" FORCE)
endif()

if(NOT NGSPICE_ENABLE_XSPICE)
    set(NGSPICE_ENABLE_CODEMODELS OFF CACHE BOOL "" FORCE)
endif()

if(NGSPICE_ENABLE_CIDER)
    set(NGSPICE_ENABLE_NUMDEV ON)
else()
    set(NGSPICE_ENABLE_NUMDEV OFF)
endif()
