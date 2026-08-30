# NgspiceDependencies.cmake -- find the optional external libraries.
#
# Everything found here is exposed as an imported target that src/CMakeLists.txt
# links against, so no global variables leak into the build.

find_package(PkgConfig QUIET)

# ---------------------------------------------------------------------------
# FFTW3  (--with-fftw3)
#
# On Windows the visualc projects expect the official FFTW DLL package
# unpacked next to the source tree:
#
#     visualc/vngspice-fftw.vcxproj   ..\..\fftw-3.3-dll64\libfftw3-3.def
#
# i.e. <ngspice>/../fftw-3.3-dll64 (or -dll32), see visualc/how-to-fftw.txt.
# That package contains fftw3.h, libfftw3-3.dll and libfftw3-3.def but no
# import library, so the .vcxproj generates one in a PreBuildEvent with
#     lib /machine:x64 /def:...\libfftw3-3.def /out:...\libfftw3-3.lib
# The same is done here at configure time.
#
# Lookup order:
#   1. -DFFTW3_INCLUDE_DIR= / -DFFTW3_LIBRARY= if given
#   2. NGSPICE_FFTW3_DIR (default <ngspice>/../fftw-3.3-dll{64,32})
#   3. pkg-config / FFTW3_ROOT / the usual prefixes
# ---------------------------------------------------------------------------
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_fftw_default "${CMAKE_SOURCE_DIR}/../fftw-3.3-dll64")
    set(_fftw_other   "${CMAKE_SOURCE_DIR}/../fftw-3.3-dll32")
    set(_fftw_machine x64)
    set(_fftw_bits 64)
else()
    set(_fftw_default "${CMAKE_SOURCE_DIR}/../fftw-3.3-dll32")
    set(_fftw_other   "${CMAKE_SOURCE_DIR}/../fftw-3.3-dll64")
    set(_fftw_machine x86)
    set(_fftw_bits 32)
endif()
set(NGSPICE_FFTW3_DIR "${_fftw_default}" CACHE PATH
    "Directory holding fftw3.h / libfftw3-3.dll (official FFTW Windows package)")

set(NGSPICE_FFTW3_DLL "" CACHE FILEPATH "" FORCE)
mark_as_advanced(NGSPICE_FFTW3_DLL)

set(NGSPICE_HAVE_FFTW3 OFF)
if(NGSPICE_WITH_FFTW3)
    if((WIN32 OR NGSPICE_FORCE_FFTW3_DIR) AND NOT FFTW3_INCLUDE_DIR AND EXISTS "${NGSPICE_FFTW3_DIR}")
        find_path(FFTW3_INCLUDE_DIR fftw3.h
                  HINTS "${NGSPICE_FFTW3_DIR}" NO_DEFAULT_PATH)
        find_library(FFTW3_LIBRARY
                     NAMES libfftw3-3 fftw3-3 fftw3
                     HINTS "${NGSPICE_FFTW3_DIR}" NO_DEFAULT_PATH)

        # the DLL package ships no .lib -- build the import library from the
        # .def, exactly like the PreBuildEvent in vngspice-fftw.vcxproj
        if(MSVC AND NOT FFTW3_LIBRARY
           AND EXISTS "${NGSPICE_FFTW3_DIR}/libfftw3-3.def")
            set(_implib "${CMAKE_BINARY_DIR}/fftw3/libfftw3-3.lib")
            if(NOT EXISTS "${_implib}")
                file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/fftw3")
                execute_process(
                    COMMAND "${CMAKE_AR}" /nologo "/machine:${_fftw_machine}"
                            "/def:${NGSPICE_FFTW3_DIR}/libfftw3-3.def"
                            "/out:${_implib}"
                    RESULT_VARIABLE _lib_rc
                    OUTPUT_VARIABLE _lib_out
                    ERROR_VARIABLE  _lib_out)
                if(NOT _lib_rc EQUAL 0)
                    message(WARNING
                        "could not create the FFTW3 import library:\n${_lib_out}")
                endif()
            endif()
            if(EXISTS "${_implib}")
                set(FFTW3_LIBRARY "${_implib}" CACHE FILEPATH "" FORCE)
                message(STATUS "Generated FFTW3 import library ${_implib}")
            endif()
        endif()

        if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
            message(STATUS "Using FFTW3 from ${NGSPICE_FFTW3_DIR}")
        endif()

        find_file(_fftw_dll
                  NAMES libfftw3-3.dll fftw3.dll
                  HINTS "${NGSPICE_FFTW3_DIR}" NO_DEFAULT_PATH)
        if(_fftw_dll)
            set(NGSPICE_FFTW3_DLL "${_fftw_dll}" CACHE FILEPATH "" FORCE)
        endif()
    endif()

    if(NOT TARGET FFTW3::fftw3)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(PC_FFTW3 QUIET fftw3)
        endif()

        find_path(FFTW3_INCLUDE_DIR fftw3.h
                  HINTS ${PC_FFTW3_INCLUDE_DIRS} ${FFTW3_ROOT}
                  PATH_SUFFIXES include)
        # libfftw3 on Unix, libfftw3-3 in the official Windows DLL package
        find_library(FFTW3_LIBRARY
                     NAMES fftw3 libfftw3-3 fftw3-3
                     HINTS ${PC_FFTW3_LIBRARY_DIRS} ${FFTW3_ROOT}
                     PATH_SUFFIXES lib)

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(FFTW3
            REQUIRED_VARS FFTW3_LIBRARY FFTW3_INCLUDE_DIR)

        if(FFTW3_FOUND)
            add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
            set_target_properties(FFTW3::fftw3 PROPERTIES
                IMPORTED_LOCATION "${FFTW3_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
        endif()
    endif()

    if(TARGET FFTW3::fftw3)
        set(NGSPICE_HAVE_FFTW3 ON)
        set(HAVE_LIBFFTW3 1)
        set(HAVE_FFTW3_H 1)
    else()
        set(_msg
            "FFTW3 requested but not found -- falling back to the internal FFT.")
        if(WIN32)
            string(APPEND _msg
                "\nThis is a ${_fftw_bits} bit build, so the directory looked "
                "for was\n    ${NGSPICE_FFTW3_DIR}")
            if(EXISTS "${_fftw_other}")
                string(APPEND _msg
                    "\n\n*** ${_fftw_other} exists, but the current toolchain "
                    "targets ${_fftw_bits} bit. ***\n"
                    "You are most likely in a developer shell for the wrong "
                    "architecture. Use the x64 Native Tools prompt (or "
                    "Launch-VsDevShell.ps1 -Arch amd64), or the MSYS2 UCRT64 / "
                    "MINGW64 shell, and configure into a fresh build "
                    "directory.")
            else()
                string(APPEND _msg
                    "\nUnpack the FFTW ${_fftw_bits} bit DLL package there, or "
                    "set -DNGSPICE_FFTW3_DIR=<dir> / -DFFTW3_ROOT=<dir>.")
            endif()
        else()
            string(APPEND _msg
                "\nSet FFTW3_ROOT or install the fftw3 development package.")
        endif()
        string(APPEND _msg "\nUse -DNGSPICE_WITH_FFTW3=OFF to silence this.")
        message(WARNING "${_msg}")
    endif()
endif()

# ---------------------------------------------------------------------------
# KLU  (--enable-klu)
#
# bundled : compile src/maths/KLU, exactly like the autotools build
# system  : link against SuiteSparse's KLU (klu, amd, btf, colamd,
#           suitesparseconfig).  The bundled headers under src/include/ngspice
#           are then bypassed in favour of SuiteSparse's own klu.h.
# ---------------------------------------------------------------------------
if(NGSPICE_ENABLE_KLU AND NGSPICE_KLU_PROVIDER STREQUAL "system")
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PC_KLU QUIET klu)
    endif()

    find_path(KLU_INCLUDE_DIR klu.h
              HINTS ${PC_KLU_INCLUDE_DIRS} ${SuiteSparse_ROOT}
              PATH_SUFFIXES include include/suitesparse suitesparse)

    set(_klu_libs "")
    foreach(_l klu amd btf colamd suitesparseconfig)
        find_library(KLU_${_l}_LIBRARY NAMES ${_l}
                     HINTS ${PC_KLU_LIBRARY_DIRS} ${SuiteSparse_ROOT}
                     PATH_SUFFIXES lib)
        if(KLU_${_l}_LIBRARY)
            list(APPEND _klu_libs "${KLU_${_l}_LIBRARY}")
        elseif(NOT _l STREQUAL "suitesparseconfig")
            message(FATAL_ERROR
                "NGSPICE_KLU_PROVIDER=system, but lib${_l} was not found.")
        endif()
    endforeach()

    if(NOT KLU_INCLUDE_DIR)
        message(FATAL_ERROR
            "NGSPICE_KLU_PROVIDER=system, but klu.h was not found. "
            "Install SuiteSparse or set SuiteSparse_ROOT.")
    endif()

    add_library(ngspice_klu INTERFACE IMPORTED)
    set_target_properties(ngspice_klu PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${KLU_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${_klu_libs}"
        INTERFACE_COMPILE_DEFINITIONS "NGSPICE_SYSTEM_KLU=1")

    message(STATUS "Using system SuiteSparse KLU: ${KLU_INCLUDE_DIR}")
endif()

# ---------------------------------------------------------------------------
# Command line editing  (--with-readline / --with-editline)
# ---------------------------------------------------------------------------
set(NGSPICE_READLINE_PROVIDER "none")
if(NOT NGSPICE_WITH_READLINE STREQUAL "none" AND NOT WIN32)
    if(NGSPICE_WITH_READLINE MATCHES "auto|readline")
        find_path(READLINE_INCLUDE_DIR readline/readline.h)
        find_library(READLINE_LIBRARY NAMES readline)
        find_library(NGSPICE_TERM_LIBRARY NAMES ncurses termcap tinfo curses)
        if(READLINE_INCLUDE_DIR AND READLINE_LIBRARY)
            add_library(ngspice_lineedit INTERFACE IMPORTED)
            set_target_properties(ngspice_lineedit PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${READLINE_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${READLINE_LIBRARY}")
            if(NGSPICE_TERM_LIBRARY)
                set_property(TARGET ngspice_lineedit APPEND PROPERTY
                    INTERFACE_LINK_LIBRARIES "${NGSPICE_TERM_LIBRARY}")
                set(HAVE_TERMCAP 1)
            endif()
            set(HAVE_GNUREADLINE 1)
            set(NGSPICE_READLINE_PROVIDER "readline")
        endif()
    endif()

    if(NGSPICE_READLINE_PROVIDER STREQUAL "none"
       AND NGSPICE_WITH_READLINE MATCHES "auto|editline")
        find_path(EDITLINE_INCLUDE_DIR editline/readline.h)
        find_library(EDITLINE_LIBRARY NAMES edit)
        if(EDITLINE_INCLUDE_DIR AND EDITLINE_LIBRARY)
            add_library(ngspice_lineedit INTERFACE IMPORTED)
            set_target_properties(ngspice_lineedit PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${EDITLINE_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${EDITLINE_LIBRARY}")
            set(HAVE_BSDEDITLINE 1)
            set(NGSPICE_READLINE_PROVIDER "editline")
        endif()
    endif()

    if(NGSPICE_READLINE_PROVIDER STREQUAL "none"
       AND NOT NGSPICE_WITH_READLINE STREQUAL "auto")
        message(FATAL_ERROR
            "NGSPICE_WITH_READLINE=${NGSPICE_WITH_READLINE} but the library was not found.")
    endif()
endif()

# ---------------------------------------------------------------------------
# X11  (--with-x)
# ---------------------------------------------------------------------------
if(NGSPICE_ENABLE_X11)
    find_package(X11)
    if(NOT X11_FOUND)
        message(STATUS "X11 not found -- building without the X11 plot window")
        set(NGSPICE_ENABLE_X11 OFF CACHE BOOL "" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP  (--enable-openmp)
# ---------------------------------------------------------------------------
# AC_OPENMP just adds the flag when the compiler supports it, so a missing
# OpenMP is not fatal here either.
if(NGSPICE_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS C)
    if(NOT OpenMP_C_FOUND)
        message(WARNING
            "OpenMP requested but not supported by ${CMAKE_C_COMPILER_ID} "
            "-- building without it. Use -DNGSPICE_ENABLE_OPENMP=OFF to "
            "silence this.")
        set(NGSPICE_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# pthread
#
# configure runs AC_CHECK_LIB([pthread], [pthread_create]) when --with-ngshared
# is used on anything but MinGW. Without the resulting HAVE_LIBPTHREAD,
# src/misc/alloc.c, src/frontend/dvec.c and src/sharedspice.c fall into their
# Windows branch and try to use CRITICAL_SECTION.
# ---------------------------------------------------------------------------
if(NOT WIN32)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads)
    if(CMAKE_USE_PTHREADS_INIT)
        set(HAVE_LIBPTHREAD 1)
    elseif(NGSPICE_BUILD_SHARED)
        message(FATAL_ERROR
            "NGSPICE_BUILD_SHARED needs pthreads, which were not found.")
    endif()
endif()

# ---------------------------------------------------------------------------
# dlopen, needed by XSPICE code model loading and OSDI
# ---------------------------------------------------------------------------
if(UNIX)
    include(CheckLibraryExists)
    check_library_exists(dl dlopen "" NGSPICE_NEED_LIBDL)
endif()

# ---------------------------------------------------------------------------
# bison, needed for parse-bison.y and inpptree-parser.y
# flex, needed for cmpp's ifs_lex.l / mod_lex.l
#
# CMake's FindBISON/FindFLEX already know the winflexbison executable names
# (win_bison / win_flex), but they only search PATH and the usual prefixes.
# The visualc projects instead expect winflexbison unpacked next to the
# source tree:
#
#     visualc/vngspice.vcxproj      ..\..\flex-bison\win_bison.exe
#     visualc/xspice/cmpp/*.vcxproj ..\..\..\..\flex-bison\win_flex.exe
#
# both of which resolve to <ngspice>/../flex-bison.  That convention is kept
# working here: the directory is probed first, then PATH.  Override with
#     -DNGSPICE_FLEX_BISON_DIR=D:/myspice/flex-bison
# or point at the binaries directly with -DBISON_EXECUTABLE / -DFLEX_EXECUTABLE.
# ---------------------------------------------------------------------------
set(NGSPICE_FLEX_BISON_DIR "${CMAKE_SOURCE_DIR}/../flex-bison" CACHE PATH
    "Directory holding win_bison.exe / win_flex.exe (winflexbison)")

if(WIN32 OR NGSPICE_FORCE_FLEX_BISON_DIR)
    # find_program() is a no-op when the cache entry is already set, so this
    # runs before find_package() and simply pre-seeds the result.
    find_program(BISON_EXECUTABLE
                 NAMES win_bison win-bison bison
                 HINTS "${NGSPICE_FLEX_BISON_DIR}"
                 NO_DEFAULT_PATH)
    find_program(FLEX_EXECUTABLE
                 NAMES win_flex win-flex flex
                 HINTS "${NGSPICE_FLEX_BISON_DIR}"
                 NO_DEFAULT_PATH)
    if(BISON_EXECUTABLE)
        message(STATUS "Using winflexbison from ${NGSPICE_FLEX_BISON_DIR}")
    endif()
endif()

find_package(BISON 3.0)
if(NOT BISON_FOUND)
    message(FATAL_ERROR
        "bison >= 3.0 is required (src/frontend/parse-bison.y).\n"
        "On Windows, unpack win_flex_bison-latest.zip into\n"
        "    ${NGSPICE_FLEX_BISON_DIR}\n"
        "(the location visualc/*.vcxproj uses), put it on PATH, or pass\n"
        "-DBISON_EXECUTABLE=<path>/win_bison.exe")
endif()

if(NGSPICE_ENABLE_CODEMODELS)
    find_package(FLEX)
    if(NOT FLEX_FOUND)
        message(FATAL_ERROR
            "flex is required to build cmpp (the XSPICE code model "
            "preprocessor). See the bison note above; the same winflexbison "
            "package provides win_flex.exe.")
    endif()
endif()
