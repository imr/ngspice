# src/CMakeLists.txt
#
# The autotools build produces ~50 libtool convenience libraries whose only
# purpose is directory structure; the link order between them is fragile.
# CMake does not need that: everything goes into one static library
# ngspice_core, and the executables/shared library link against it.

set(NGSPICE_GEN "${CMAKE_BINARY_DIR}/generated/src")

# ---------------------------------------------------------------------------
# Collect the modules that make up the simulator core
# ---------------------------------------------------------------------------
set(_modules
    misc_libmisc_la
    maths_cmaths_libcmaths_la
    maths_deriv_libderiv_la
    maths_fft_libmathfft_la
    maths_misc_libmathmisc_la
    maths_ni_libni_la
    maths_poly_libpoly_la
    maths_sparse_libsparse_la
    frontend_libfte_la
    frontend_numparam_libnumparam_la
    frontend_parser_libparser_la
    frontend_plotting_libplotting_la
    frontend_trannoise_libtrannoise_la
    spicelib_analysis_libckt_la
    spicelib_parser_libinp_la
    spicelib_devices_libdev_la)

if(NGSPICE_ENABLE_KLU AND NGSPICE_KLU_PROVIDER STREQUAL "bundled")
    list(APPEND _modules maths_KLU_libKLU_la maths_KLU_libKLU_real_la)
endif()
if(NGSPICE_ENABLE_RFSPICE)
    list(APPEND _modules maths_dense_libdense_la)
endif()
if(NGSPICE_ENABLE_XSPICE)
    list(APPEND _modules
        xspice_cm_libcmxsp_la
        xspice_mif_libmifxsp_la
        xspice_evt_libevtxsp_la
        xspice_enh_libenhxsp_la
        xspice_idn_libidnxsp_la)
endif()
if(NGSPICE_ENABLE_OSDI)
    list(APPEND _modules osdi_libosdi_la)
endif()
if(NGSPICE_ENABLE_CIDER)
    list(APPEND _modules
        ciderlib_input_libciderinput_la
        ciderlib_oned_libcideroned_la
        ciderlib_support_libcidersuprt_la
        ciderlib_twod_libcidertwod_la)
endif()
if(NGSPICE_ENABLE_X11 AND NOT NGSPICE_WINGUI)
    list(APPEND _modules frontend_help_libhlp_la)
endif()
if(NGSPICE_WINGUI)
    list(APPEND _modules frontend_wdisp_libwindisp_la)
endif()

# Device model directories are picked up automatically, so adding a new
# device only means adding it to dev.c and its own Makefile.am.
set(_numdev_modules
    spicelib_devices_nbjt_libnbjt_la
    spicelib_devices_nbjt2_libnbjt2_la
    spicelib_devices_numd_libnumd_la
    spicelib_devices_numd2_libnumd2_la
    spicelib_devices_numos_libnumos_la)

get_cmake_property(_all_vars VARIABLES)
foreach(_v IN LISTS _all_vars)
    if(_v MATCHES "^NGSPICE_SRC_(spicelib_devices_[a-z0-9_]+_la)$")
        set(_dev "${CMAKE_MATCH_1}")
        if(_dev IN_LIST _numdev_modules)
            if(NGSPICE_ENABLE_CIDER)
                list(APPEND _modules "${_dev}")
            endif()
        elseif(_dev STREQUAL "spicelib_devices_ndev_libndev_la")
            if(NGSPICE_ENABLE_NDEV)
                list(APPEND _modules "${_dev}")
            endif()
        else()
            list(APPEND _modules "${_dev}")
        endif()
    endif()
endforeach()
list(REMOVE_DUPLICATES _modules)

# ---------------------------------------------------------------------------
# Turn the module names into an absolute source list
# ---------------------------------------------------------------------------
set(_core_sources "")
foreach(_m IN LISTS _modules)
    if(NOT DEFINED NGSPICE_SRC_${_m})
        message(FATAL_ERROR
            "source list NGSPICE_SRC_${_m} is missing -- "
            "re-run 'python3 cmake/am2cmake.py'")
    endif()
    foreach(_f IN LISTS NGSPICE_SRC_${_m})
        if(NOT _f MATCHES "\\.(y|l)$")
            list(APPEND _core_sources "${CMAKE_CURRENT_SOURCE_DIR}/${_f}")
        endif()
    endforeach()
    # keep the Visual Studio solution tidy
    string(REGEX REPLACE "_lib.*$" "" _grp "${_m}")
    string(REPLACE "_" "/" _grp "${_grp}")
    source_group("${_grp}" FILES ${NGSPICE_SRC_${_m}})
endforeach()

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

list(APPEND _core_sources
    ${BISON_NgspiceFrontendParser_OUTPUTS}
    ${BISON_NgspiceInpPtreeParser_OUTPUTS})

# MSVC before VC++ 2013 lacks trunc/nearbyint/asinh/acosh/atanh; ngspice.h
# redirects them to x_*() and visualc/vngspice.vcxproj compiles this file.
if(MSVC)
    list(APPEND _core_sources "${CMAKE_SOURCE_DIR}/visualc/src/msvc-compat.c")
endif()

# ---------------------------------------------------------------------------
# The core library
# ---------------------------------------------------------------------------
add_library(ngspice_core STATIC ${_core_sources})

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

# KLU: the bundled solver needs a second copy of the kernel compiled
# with -DCOMPLEX; the system SuiteSparse already ships both.
if(NGSPICE_ENABLE_KLU)
    if(NGSPICE_KLU_PROVIDER STREQUAL "bundled")
        set(_klu_complex_sources "")
        foreach(_f IN LISTS NGSPICE_SRC_maths_KLU_libKLU_complex_la)
            list(APPEND _klu_complex_sources "${CMAKE_CURRENT_SOURCE_DIR}/${_f}")
        endforeach()
        add_library(ngspice_klu_complex OBJECT ${_klu_complex_sources})
        target_compile_definitions(ngspice_klu_complex PRIVATE COMPLEX)
        target_link_libraries(ngspice_klu_complex PRIVATE ngspice_settings)
        target_sources(ngspice_core PRIVATE
            $<TARGET_OBJECTS:ngspice_klu_complex>)
    else()
        target_link_libraries(ngspice_core PUBLIC ngspice_klu)
    endif()
endif()

if(NGSPICE_HAVE_FFTW3)
    target_link_libraries(ngspice_core PUBLIC FFTW3::fftw3)
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

    # The official FFTW Windows package is DLL-only and lives outside the
    # build tree, so put it next to ngspice.exe -- otherwise neither ctest
    # nor the installed binary can start.
    if(WIN32 AND NGSPICE_FFTW3_DLL)
        add_custom_command(TARGET ngspice POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${NGSPICE_FFTW3_DLL}" "$<TARGET_FILE_DIR:ngspice>"
            COMMENT "copying $<PATH:GET_FILENAME,${NGSPICE_FFTW3_DLL}> next to ngspice.exe"
            VERBATIM)
        install(FILES "${NGSPICE_FFTW3_DLL}"
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
