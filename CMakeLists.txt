#
# CMake build system for ngspice -- runs in parallel to the autotools build.
#
# The autotools build remains the reference for source lists; they are
# extracted into cmake/NgspiceSourceLists.cmake by cmake/am2cmake.py.
#
#   cmake -S . -B build -DNGSPICE_ENABLE_XSPICE=ON
#   cmake --build build -j
#   ctest --test-dir build
#

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# The overlay consists of five CMakeLists.txt plus the cmake/ directory.
# Copying only the *.txt files leaves cmake/ behind, which otherwise shows up
# as four confusing "include could not find requested file" errors followed
# by a misleading hint to re-run am2cmake.py.
foreach(_needed NgspiceOptions NgspiceDependencies NgspiceConfigHeader
                NgspiceCodeModels NgspiceSourceLists RunNgspiceTest config.h.in)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${_needed}"
       AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${_needed}.cmake")
        message(FATAL_ERROR
            "cmake/${_needed}.cmake is missing.\n"
            "The CMake overlay is more than the CMakeLists.txt files -- the "
            "whole cmake/ directory belongs to it. Unpack the tarball again "
            "with\n"
            "    tar xzf ngspice-cmake-overlay.tar.gz --strip-components=1 "
            "-C <ngspice-root>\n"
            "and check that ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ exists.")
    endif()
endforeach()

# --- version, taken from configure.ac so there is a single source of truth ---
# Development branches carry a suffix, e.g. ngspice_major_version = [47+].
# Keep that string for PACKAGE_VERSION, but project() accepts digits and dots
# only, so strip everything else for the numeric version.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/configure.ac" _cfg_ac)
string(REGEX MATCH "m4_define\\(\\[ngspice_major_version\\], *\\[([^]]+)\\]" _m "${_cfg_ac}")
set(NGSPICE_VERSION_MAJOR_FULL "${CMAKE_MATCH_1}")
string(REGEX MATCH "m4_define\\(\\[ngspice_minor_version\\], *\\[([^]]+)\\]" _m "${_cfg_ac}")
set(NGSPICE_VERSION_MINOR_FULL "${CMAKE_MATCH_1}")
unset(_cfg_ac)

string(REGEX REPLACE "[^0-9].*$" "" NGSPICE_VERSION_MAJOR "${NGSPICE_VERSION_MAJOR_FULL}")
string(REGEX REPLACE "[^0-9].*$" "" NGSPICE_VERSION_MINOR "${NGSPICE_VERSION_MINOR_FULL}")
# note: a plain if(NOT NGSPICE_VERSION_MINOR) would fire on the literal "0"
if(NGSPICE_VERSION_MAJOR STREQUAL "" OR NGSPICE_VERSION_MINOR STREQUAL "")
    message(FATAL_ERROR
        "could not read the version from configure.ac "
        "(major='${NGSPICE_VERSION_MAJOR_FULL}', "
        "minor='${NGSPICE_VERSION_MINOR_FULL}')")
endif()

# what configure.ac assembles as ngspice_version
set(NGSPICE_VERSION_STRING "${NGSPICE_VERSION_MAJOR_FULL}.${NGSPICE_VERSION_MINOR_FULL}")

project(ngspice
        VERSION ${NGSPICE_VERSION_MAJOR}.${NGSPICE_VERSION_MINOR}
        LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakeDependentOption)
include(NgspiceOptions)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
# hicum2 (hicumL2.cpp) and cppduals need C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# configure.ac builds optimised unless --enable-debug is given (-O2 -s for
# GCC), so Release is the default for both generator kinds. For multi-config
# generators the first entry of CMAKE_CONFIGURATION_TYPES is what MSBuild and
# Ninja Multi-Config pick when --config is omitted, so Release goes first.
get_property(_ngspice_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_ngspice_multi)
    if(NOT NGSPICE_CONFIGURATION_TYPES_SET)
        set(CMAKE_CONFIGURATION_TYPES "Release;Debug;RelWithDebInfo;MinSizeRel"
            CACHE STRING "" FORCE)
        set(NGSPICE_CONFIGURATION_TYPES_SET ON CACHE INTERNAL "")
    endif()
elseif(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --enable-debug also defines NGDEBUG, so tie it to the Debug configuration.
# It cannot live in config.h, which is generated once for all configurations.
set(NGSPICE_NGDEBUG_GENEX $<$<CONFIG:Debug>:NGDEBUG>)

# ---------------------------------------------------------------------------
# Where generated headers go
# ---------------------------------------------------------------------------
set(NGSPICE_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated/include")

# Windows installs go to the locations the visualc batch files use:
# C:\Spice / C:\Spice64 for release, with a trailing "d" for debug.
# configure.ac does the same with dprefix="C:/Spice".
if(WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_ngspice_prefix "C:/Spice64")
    else()
        set(_ngspice_prefix "C:/Spice")
    endif()
    # Multi-config generators do not know the configuration yet, so a debug
    # install needs -DCMAKE_INSTALL_PREFIX=C:/Spice64d by hand.
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        string(APPEND _ngspice_prefix "d")
    endif()
    set(CMAKE_INSTALL_PREFIX "${_ngspice_prefix}" CACHE PATH "" FORCE)
endif()

# Run-time locations baked into config.h.
#
# Windows uses relative strings without HAS_RELPATH: ivars.c prepends the
# directory of the executable, and dlopen() in dev.c resolves a relative code
# model path against GetModuleFileName(NULL). That is what
# visualc/src/include/ngspice/config.h does, and it makes the installed tree
# movable without depending on the working directory.
if(NGSPICE_ENABLE_RELPATH OR WIN32)
    set(NGSPICEBINDIR "../bin")
    set(NGSPICEDATADIR "../share/ngspice")
    set(NGSPICELIBDIR "../lib/ngspice")
    set(NGSPICE_PKGLIBDIR "../lib/ngspice")
    set(NGSPICE_PKGDATADIR "../share/ngspice")
else()
    set(NGSPICEBINDIR "${CMAKE_INSTALL_FULL_BINDIR}")
    set(NGSPICEDATADIR "${CMAKE_INSTALL_FULL_DATADIR}/ngspice")
    set(NGSPICELIBDIR "${CMAKE_INSTALL_FULL_LIBDIR}/ngspice")
    set(NGSPICE_PKGLIBDIR "${CMAKE_INSTALL_FULL_LIBDIR}/ngspice")
    set(NGSPICE_PKGDATADIR "${CMAKE_INSTALL_FULL_DATADIR}/ngspice")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
include(NgspiceDependencies)

# ---------------------------------------------------------------------------
# config.h
# ---------------------------------------------------------------------------
include(NgspiceConfigHeader)

# ---------------------------------------------------------------------------
# Compiler warnings / platform quirks, kept in one place
# ---------------------------------------------------------------------------
add_library(ngspice_settings INTERFACE)

# NGDEBUG for Debug builds, matching configure's --enable-debug
target_compile_definitions(ngspice_settings INTERFACE ${NGSPICE_NGDEBUG_GENEX})

target_include_directories(ngspice_settings INTERFACE
    "${NGSPICE_GENERATED_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include/cppduals"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spicelib/devices")

if(MSVC)
    # Mirror the PreprocessorDefinitions of visualc/vngspice.vcxproj.
    #
    # Do NOT define WIN32_LEAN_AND_MEAN globally: it removes commdlg.h from
    # windows.h, and src/frontend/wdisp/winprint.c needs PRINTDLG / PD_*
    # from there. The tree sets that macro per file, in winmain.c,
    # frontend/resource.c and misc/misc_time.c, which do not need commdlg.
    target_compile_definitions(ngspice_settings INTERFACE
        _CRT_SECURE_NO_WARNINGS _CRT_SECURE_NO_DEPRECATE
        _CRT_NONSTDC_NO_DEPRECATE)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        # selects the "r64"/"d64" pg_config variable in frontend/cpitf.c
        target_compile_definitions(ngspice_settings INTERFACE CONFIG64)
    endif()
    target_compile_options(ngspice_settings INTERFACE
        /wd4996 /wd4244 /wd4267 /wd4018 /utf-8)
    # NOTE: do *not* put visualc/src/include on the include path. Its only
    # content is a hand maintained ngspice/config.h for the .vcxproj build,
    # which would shadow the one CMake generates.
    # visualc/src/msvc-compat.c supplies trunc/nearbyint/asinh/... for
    # _MSC_VER < 1800 (see src/include/ngspice/ngspice.h); harmless to
    # compile unconditionally, since the whole file is guarded there.
else()
    target_compile_options(ngspice_settings INTERFACE
        -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
        -Wno-missing-field-initializers)
endif()

if(APPLE)
    # code models are loaded with dlopen(); undefined symbols are resolved
    # against the host binary
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
endif()

# ---------------------------------------------------------------------------
# The source lists extracted from the Makefile.am files
# ---------------------------------------------------------------------------
include(NgspiceSourceLists)

add_subdirectory(src)

set(NGSPICE_TESTS_SUMMARY "OFF")
if(NGSPICE_ENABLE_TESTS)
    if(NOT NGSPICE_BUILD_EXECUTABLE)
        # ctest drives the ngspice binary; without it there is nothing to run
        set(NGSPICE_TESTS_SUMMARY "OFF (needs the ngspice executable)")
    else()
        enable_testing()
        add_subdirectory(tests)
        get_property(_n_tests GLOBAL PROPERTY NGSPICE_TEST_COUNT)
        set(NGSPICE_TESTS_SUMMARY "ON (${_n_tests} registered)")
    endif()
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "ngspice ${NGSPICE_VERSION_STRING} -- CMake configuration")
get_property(_ngspice_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_ngspice_multi)
    list(GET CMAKE_CONFIGURATION_TYPES 0 _ngspice_first_cfg)
    message(STATUS "  configurations ..... ${CMAKE_CONFIGURATION_TYPES}")
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        # cmake --build defaults to Debug here, while cmake --install
        # defaults to Release -- always pass --config to both.
        message(STATUS "                       --config is required for build, "
                       "test and install:")
        message(STATUS "                       cmake --build <dir> --config Release")
        message(STATUS "                       cmake --install <dir> --config Release")
    else()
        message(STATUS "                       select at build time: "
                       "cmake --build <dir> --config Release")
        message(STATUS "                       without --config it defaults to ${_ngspice_first_cfg}")
    endif()
else()
    message(STATUS "  build type ......... ${CMAKE_BUILD_TYPE}")
endif()
math(EXPR _ngspice_bits "${CMAKE_SIZEOF_VOID_P} * 8")
message(STATUS "  target ............. ${CMAKE_SYSTEM_NAME} ${_ngspice_bits} bit (${CMAKE_C_COMPILER_ID})")
message(STATUS "  XSPICE ............. ${NGSPICE_ENABLE_XSPICE}")
message(STATUS "  code models ........ ${NGSPICE_ENABLE_CODEMODELS}")
message(STATUS "  CIDER .............. ${NGSPICE_ENABLE_CIDER}")
message(STATUS "  OSDI ............... ${NGSPICE_ENABLE_OSDI}")
message(STATUS "  RFSPICE (S-param) .. ${NGSPICE_ENABLE_RFSPICE}")
message(STATUS "  PSS ................ ${NGSPICE_ENABLE_PSS}")
message(STATUS "  KLU ................ ${NGSPICE_ENABLE_KLU} (${NGSPICE_KLU_PROVIDER})")
message(STATUS "  FFTW3 .............. ${NGSPICE_HAVE_FFTW3}")
message(STATUS "  OpenMP ............. ${NGSPICE_ENABLE_OPENMP}")
message(STATUS "  X11 ................ ${NGSPICE_ENABLE_X11}")
message(STATUS "  readline/editline .. ${NGSPICE_READLINE_PROVIDER}")
message(STATUS "  ngspice binary ..... ${NGSPICE_BUILD_EXECUTABLE}")
message(STATUS "  libngspice ......... ${NGSPICE_BUILD_SHARED}")
message(STATUS "  tests .............. ${NGSPICE_TESTS_SUMMARY}")
message(STATUS "")
