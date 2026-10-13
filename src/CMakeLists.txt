# src/CMakeLists.txt
#
# Everything the simulator needs goes into one static library, ngspice_core;
# the executable and the shared library link against it.
#
# Source files are not listed here. Each source directory has its own
# CMakeLists.txt that adds its files to ngspice_core with target_sources().

set(NGSPICE_GEN "${CMAKE_BINARY_DIR}/generated/src")

# ---------------------------------------------------------------------------
# The core library
# ---------------------------------------------------------------------------
add_library(ngspice_core STATIC)
target_link_libraries(ngspice_core PUBLIC ngspice_settings)

# The per-directory AM_CPPFLAGS in the Makefile.am files add these; since
# everything lands in one target here, they are added once, globally.
#   grep -rh AM_CPPFLAGS src --include=Makefile.am | grep -o '\-I\$(top_srcdir)/[^ ]*'
target_include_directories(ngspice_core PUBLIC
    "${NGSPICE_GEN}/frontend"
    "${NGSPICE_GEN}/spicelib/parser"
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend"
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/wdisp"
    "${CMAKE_CURRENT_SOURCE_DIR}/maths/poly"
    "${CMAKE_CURRENT_SOURCE_DIR}/maths/dense"
    "${CMAKE_CURRENT_SOURCE_DIR}/spicelib/parser")

# Bundled KLU: a second copy of the kernel, compiled with -DCOMPLEX. Its
# sources are added in maths/KLU/CMakeLists.txt. System SuiteSparse ships both.
if(NGSPICE_KLU_BUNDLED)
    add_library(ngspice_klu_complex OBJECT)
    target_compile_definitions(ngspice_klu_complex PRIVATE COMPLEX)
    target_link_libraries(ngspice_klu_complex PRIVATE ngspice_settings)
    target_sources(ngspice_core PRIVATE $<TARGET_OBJECTS:ngspice_klu_complex>)
elseif(NGSPICE_ENABLE_KLU)
    target_link_libraries(ngspice_core PUBLIC ngspice_klu)
endif()

# While the autotools build exists, the CMakeLists.txt below are generated
# from the Makefile.am files. Stop if someone changed a Makefile.am and did
# not run the generator, instead of silently leaving files out.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Makefile.am")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/am2cmake.py" --check
            RESULT_VARIABLE _am_rc OUTPUT_VARIABLE _am_out ERROR_VARIABLE _am_out)
        if(NOT _am_rc EQUAL 0)
            message(FATAL_ERROR "${_am_out}"
                "A Makefile.am was changed. Run: python3 cmake/am2cmake.py")
        endif()
    endif()
endif()

# ===========================================================================
# Source directories.  A new top-level directory below src/ is added here;
# deeper directories are added in the CMakeLists.txt of their parent.
# ===========================================================================
add_subdirectory(misc)
add_subdirectory(maths)
add_subdirectory(frontend)
add_subdirectory(spicelib)
if(NGSPICE_ENABLE_XSPICE)
    add_subdirectory(xspice)
endif()
if(NGSPICE_ENABLE_OSDI)
    add_subdirectory(osdi)
endif()
if(NGSPICE_ENABLE_CIDER)
    add_subdirectory(ciderlib)
endif()

# ---------------------------------------------------------------------------
# Generated parsers
# ---------------------------------------------------------------------------
file(MAKE_DIRECTORY "${NGSPICE_GEN}/frontend" "${NGSPICE_GEN}/spicelib/parser")

bison_target(NgspiceFrontendParser
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/parse-bison.y"
    "${NGSPICE_GEN}/frontend/parse-bison.c"
    DEFINES_FILE "${NGSPICE_GEN}/frontend/parse-bison.h")

bison_target(NgspiceInpPtreeParser
    "${CMAKE_CURRENT_SOURCE_DIR}/spicelib/parser/inpptree-parser.y"
    "${NGSPICE_GEN}/spicelib/parser/inpptree-parser.c"
    DEFINES_FILE "${NGSPICE_GEN}/spicelib/parser/inpptree-parser.h")

target_sources(ngspice_core PRIVATE
    ${BISON_NgspiceFrontendParser_OUTPUTS}
    ${BISON_NgspiceInpPtreeParser_OUTPUTS})

# ---------------------------------------------------------------------------
# MSVC specifics
# ---------------------------------------------------------------------------
if(MSVC)
    # trunc/nearbyint/asinh/... for VC++ before 2013, see ngspice.h
    target_sources(ngspice_core PRIVATE
        "${CMAKE_SOURCE_DIR}/visualc/src/msvc-compat.c")
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/frontend/snp2va.c"
        PROPERTIES LANGUAGE CXX)
endif()

# Visual Studio: show the sources in their directory tree
get_target_property(_srcs ngspice_core SOURCES)
set(_tree "")
foreach(_f IN LISTS _srcs)
    if(NOT _f MATCHES "^\\$<")
        cmake_path(IS_PREFIX CMAKE_CURRENT_SOURCE_DIR "${_f}" NORMALIZE _in_src)
        if(_in_src)
            list(APPEND _tree "${_f}")
        endif()
    endif()
endforeach()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${_tree})

if(NGSPICE_HAVE_FFTW3)
    target_link_libraries(ngspice_core PUBLIC FFTW3::fftw3)
endif()
if(NGSPICE_HAVE_SNDFILE)
    target_link_libraries(ngspice_core PUBLIC ngspice_sndfile ngspice_samplerate)
endif()
if(NGSPICE_ENABLE_OPENMP)
    target_link_libraries(ngspice_core PUBLIC OpenMP::OpenMP_C)
endif()
if(NGSPICE_ENABLE_X11)
    target_link_libraries(ngspice_core PUBLIC X11::X11)
    if(TARGET X11::Xaw)
        target_link_libraries(ngspice_core PUBLIC X11::Xaw X11::Xmu X11::Xext)
    endif()
endif()
if(TARGET ngspice_lineedit)
    target_link_libraries(ngspice_core PUBLIC ngspice_lineedit)
endif()
if(UNIX)
    target_link_libraries(ngspice_core PUBLIC m)
    if(NGSPICE_NEED_LIBDL)
        target_link_libraries(ngspice_core PUBLIC ${CMAKE_DL_LIBS})
    endif()
    find_package(Threads)
    if(Threads_FOUND)
        target_link_libraries(ngspice_core PUBLIC Threads::Threads)
    endif()
endif()
if(WIN32)
    target_link_libraries(ngspice_core PUBLIC psapi shlwapi ws2_32)
endif()

# ---------------------------------------------------------------------------
# spinit (loaded at run time; tells ngspice where the .cm files live)
# ---------------------------------------------------------------------------
if(NGSPICE_ENABLE_XSPICE AND NGSPICE_ENABLE_CODEMODELS)
    set(XSPICEINIT "")
else()
    set(XSPICEINIT "*")
endif()

# spinit.in uses @pkglibdir@ / @XSPICEINIT@, i.e. autoconf's @VAR@ syntax,
# which configure_file() understands as-is.
#
# Neither copy may sit at the top of the build tree: on Windows, cpitf.c
# falls back to "./spinit" in the current directory when $Lib_Path/spinit is
# missing, so a stray spinit there would silently win -- with code model
# paths that point at the install tree.
set(pkglibdir "${NGSPICE_PKGLIBDIR}")
configure_file(spinit.in "${CMAKE_BINARY_DIR}/install/spinit" @ONLY)

# A second copy pointing into the build tree, so ngspice can be run from the
# build directory without installing. Used by the regression tests through
# SPICE_SCRIPTS.
set(pkglibdir "${CMAKE_BINARY_DIR}/codemodels")
configure_file(spinit.in "${CMAKE_BINARY_DIR}/scripts/spinit" @ONLY)
file(APPEND "${CMAKE_BINARY_DIR}/scripts/spinit"
     "\n* resolve .include against the directory of the test netlist\nset sourcepath = ( $ngspice_vpath . )\n")

# ---------------------------------------------------------------------------
# Executables and libraries
# ---------------------------------------------------------------------------
if(NGSPICE_BUILD_EXECUTABLE)
    set(_ngspice_main_sources main.c conf.c ngspice.c)
    if(NGSPICE_WINGUI)
        list(APPEND _ngspice_main_sources winmain.c hist_info.c)
    endif()
    if(WIN32)
        # src/Makefile.am adds ngicon.rc under WINRESOURCE, which only depends
        # on windres being available -- console builds get the icon as well.
        list(APPEND _ngspice_main_sources ngicon.rc)
    endif()

    add_executable(ngspice ${_ngspice_main_sources})
    target_compile_definitions(ngspice PRIVATE SIMULATOR)
    target_link_libraries(ngspice PRIVATE ngspice_core)
    if(NGSPICE_WINGUI)
        set_target_properties(ngspice PROPERTIES WIN32_EXECUTABLE ON)
        # Unless EXT_ASC is defined, src/winmain.c provides wWinMain(), not
        # WinMain().  autotools handles this with
        #     ngspice_LDFLAGS = -municode
        # in src/Makefile.am; without it the MinGW startup code looks for
        # WinMain and the link fails with
        #     crtexewin.c: undefined reference to `WinMain'
        if(NOT NGSPICE_DISABLE_UTF8)
            if(MSVC)
                target_link_options(ngspice PRIVATE /ENTRY:wWinMainCRTStartup)
            elseif(MINGW OR CYGWIN)
                target_link_options(ngspice PRIVATE -municode)
            endif()
        endif()
    endif()

    # code models must resolve MIF_* etc. against the executable
    if(NGSPICE_ENABLE_CODEMODELS AND NOT WIN32)
        set_target_properties(ngspice PROPERTIES ENABLE_EXPORTS ON)
    endif()

    # cmake --build defaults to Debug on the Visual Studio generator while
    # cmake --install defaults to Release, so installing without --config
    # after building without --config looks for a binary that was never
    # built. Say so, instead of leaving "file INSTALL cannot find ...".
    install(CODE "
if(NOT EXISTS \"$<TARGET_FILE:ngspice>\")
    message(FATAL_ERROR
        \"\${CMAKE_INSTALL_CONFIG_NAME} was not built: $<TARGET_FILE:ngspice> \"
        \"does not exist.\\n\"
        \"With a multi-config generator, build and install need the same \"
        \"--config:\\n\"
        \"    cmake --build <dir> --config \${CMAKE_INSTALL_CONFIG_NAME}\\n\"
        \"    cmake --install <dir> --config \${CMAKE_INSTALL_CONFIG_NAME}\")
endif()")

    install(TARGETS ngspice RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

    # The Windows packages of FFTW3, libsndfile and libsamplerate live
    # outside the build tree; put their DLLs next to ngspice.exe, otherwise
    # neither ctest nor the installed binary can start.
    if(WIN32 AND NGSPICE_RUNTIME_DLLS)
        add_custom_command(TARGET ngspice POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${NGSPICE_RUNTIME_DLLS} "$<TARGET_FILE_DIR:ngspice>"
            COMMENT "copying runtime DLLs next to ngspice.exe"
            VERBATIM)
        install(FILES ${NGSPICE_RUNTIME_DLLS}
                DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()
endif()

if(NGSPICE_BUILD_SHARED)
    # libngspice_la_SOURCES in src/Makefile.am: conf.c, ngspice.c and
    # sharedspice.c -- no main.c. The library has no main(), and
    # sharedspice.c defines ft_sim, SPfrontEnd, nutmeginfo and the other
    # globals itself, so adding main.c gives "multiple definition" errors.
    add_library(ngspice_shared SHARED conf.c ngspice.c sharedspice.c)
    set_target_properties(ngspice_shared PROPERTIES
        OUTPUT_NAME ngspice
        C_VISIBILITY_PRESET default
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})
    target_compile_definitions(ngspice_shared PRIVATE
        SIMULATOR
        SHAREDSPICE_version="${PROJECT_VERSION}")
    target_link_libraries(ngspice_shared PRIVATE ngspice_core)
    if(WIN32 AND NGSPICE_RUNTIME_DLLS)
        add_custom_command(TARGET ngspice_shared POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${NGSPICE_RUNTIME_DLLS} "$<TARGET_FILE_DIR:ngspice_shared>"
            COMMENT "copying runtime DLLs next to ngspice.dll"
            VERBATIM)
        install(FILES ${NGSPICE_RUNTIME_DLLS}
                DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()
    install(TARGETS ngspice_shared
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

# ---------------------------------------------------------------------------
# XSPICE code models
# ---------------------------------------------------------------------------
if(NGSPICE_ENABLE_CODEMODELS)
    add_subdirectory(xspice/cmpp)
    include(NgspiceCodeModels)
    ngspice_add_all_code_models()
endif()

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
install(FILES "${CMAKE_BINARY_DIR}/install/spinit"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/ngspice/scripts")
install(FILES setplot spectrum
        DESTINATION "${CMAKE_INSTALL_DATADIR}/ngspice/scripts")

# NGSPICEDATADIR / NGSPICELIBDIR are compiled into config.h and spinit is
# generated from CMAKE_INSTALL_PREFIX, both at configure time -- exactly as
# ./configure --prefix does. "cmake --install --prefix" therefore produces a
# tree the binary cannot find its way around in. Use DESTDIR for staging
# instead, which keeps the baked paths intact.
if(NOT NGSPICE_ENABLE_RELPATH)
install(CODE "
if(NOT \"\${CMAKE_INSTALL_PREFIX}\" STREQUAL \"${CMAKE_INSTALL_PREFIX}\")
    message(WARNING
        \"installing to \${CMAKE_INSTALL_PREFIX}, but the paths compiled into \"
        \"config.h and spinit point at ${CMAKE_INSTALL_PREFIX}. The installed \"
        \"ngspice will not find spinit or the code models.\\n\"
        \"Configure with -DCMAKE_INSTALL_PREFIX=\${CMAKE_INSTALL_PREFIX} \"
        \"instead, use DESTDIR for staging, or build with \"
        \"-DNGSPICE_ENABLE_RELPATH=ON.\")
endif()")
endif()
if(NGSPICE_ENABLE_CIDER)
    install(FILES devload devaxis ciderinit
            DESTINATION "${CMAKE_INSTALL_DATADIR}/ngspice/scripts")
endif()
