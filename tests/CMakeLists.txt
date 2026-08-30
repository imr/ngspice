# tests/CMakeLists.txt
#
# Stage 2: the existing regression suite, driven by CTest.
#
# The test names come from the TESTS = ... lines of the leaf Makefile.am
# files, so nothing has to be maintained twice: adding a test to the
# autotools suite adds it here as well (after a re-configure).
#
#   ctest --test-dir build -j8
#   ctest --test-dir build -R xspice          # only the XSPICE tests
#   ctest --test-dir build --output-on-failure

if(NOT TARGET ngspice)
    # the caller checks this too; kept as a guard
    set_property(GLOBAL PROPERTY NGSPICE_TEST_COUNT 0)
    return()
endif()

# ---------------------------------------------------------------------------
# A spinit that points at the code models in the build tree, so the tests
# run without installing anything.
# ---------------------------------------------------------------------------
# written by src/CMakeLists.txt, points at build/codemodels
set(NGSPICE_TEST_SCRIPTS "${CMAKE_BINARY_DIR}/scripts")

# ---------------------------------------------------------------------------
# Which test directories are active
#
# tests/Makefile.am distinguishes two sets:
#
#   SUBDIRS       what "make check" actually runs
#   DIST_SUBDIRS  everything that is shipped
#
# Most directories are dist-only and are never executed by autotools; several
# of them have stale reference output and at least one (tests/vbic/FG.cir)
# does not terminate.  The default therefore mirrors SUBDIRS.
#
#   -DNGSPICE_TEST_LEVEL=check   regression + xspice          (default, seconds)
#   -DNGSPICE_TEST_LEVEL=all     plus every dist-only directory
# ---------------------------------------------------------------------------
set(NGSPICE_TEST_LEVEL "check" CACHE STRING
    "Which part of tests/ to register: check (like 'make check') or all")
set_property(CACHE NGSPICE_TEST_LEVEL PROPERTY STRINGS check all)

set(NGSPICE_TEST_TIMEOUT 120 CACHE STRING
    "Per test timeout in seconds; keeps a non-converging netlist from \
stalling the run")

# SUBDIRS in tests/Makefile.am
set(_check_dirs regression)
if(NGSPICE_ENABLE_XSPICE)
    list(APPEND _check_dirs xspice)
endif()
# bsim3, bsim4, bsimsoi, hisim, hicum2, hisimhv1 and hisimhv2 are also in
# SUBDIRS, but they are driven by tests/bin/check_cmc.sh over nmos/qaSpec
# rather than by "TESTS = *.cir", so nothing is registered for them here.
# Running the CMC model QA suites from CTest is a separate task.

set(_skip_dirs "")
if(NOT NGSPICE_ENABLE_XSPICE)
    list(APPEND _skip_dirs "xspice")
endif()
if(NOT NGSPICE_ENABLE_CIDER)
    list(APPEND _skip_dirs "cider" "numdev")
endif()

# ---------------------------------------------------------------------------
# Harvest TESTS = ... from every tests/**/Makefile.am
# ---------------------------------------------------------------------------
file(GLOB_RECURSE _makefiles
     LIST_DIRECTORIES false
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*/Makefile.am")

set(_n_tests 0)

foreach(_mk IN LISTS _makefiles)
    get_filename_component(_dir "${_mk}" DIRECTORY)
    file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_dir}")

    set(_skipped FALSE)
    foreach(_s IN LISTS _skip_dirs)
        if(_rel MATCHES "(^|/)${_s}(/|$)")
            set(_skipped TRUE)
        endif()
    endforeach()
    if(_skipped)
        continue()
    endif()

    # restrict to what "make check" would run
    set(_in_check FALSE)
    foreach(_c IN LISTS _check_dirs)
        if(_rel MATCHES "^${_c}(/|$)")
            set(_in_check TRUE)
        endif()
    endforeach()
    if(NOT _in_check AND NOT NGSPICE_TEST_LEVEL STREQUAL "all")
        continue()
    endif()

    file(READ "${_mk}" _content)
    string(REGEX REPLACE "\\\\[ \t]*\n" " " _content "${_content}")

    # TESTS = a.cir b.cir ...
    set(_cirs "")
    if(_content MATCHES "\n[ \t]*TESTS[ \t]*=([^\n]*)")
        string(REGEX MATCHALL "[A-Za-z0-9_.+-]+\\.cir" _cirs "${CMAKE_MATCH_1}")
    endif()
    if(NOT _cirs)
        continue()
    endif()

    # Some directories pass extra flags, e.g. tests/xspice/digital uses
    #   $(top_builddir)/src/ngspice -r foobaz
    set(_extra "")
    if(_content MATCHES "src/ngspice([^\"\n]*)\"")
        string(STRIP "${CMAKE_MATCH_1}" _extra)
    endif()

    foreach(_cir IN LISTS _cirs)
        if(NOT EXISTS "${_dir}/${_cir}")
            message(WARNING "TESTS lists ${_rel}/${_cir}, but it does not exist")
            continue()
        endif()

        get_filename_component(_base "${_cir}" NAME_WE)
        string(REPLACE "/" "." _prefix "${_rel}")
        set(_test_name "${_prefix}.${_base}")
        set(_work "${CMAKE_BINARY_DIR}/tests-run/${_test_name}")

        add_test(NAME "${_test_name}"
                 COMMAND ${CMAKE_COMMAND}
                         "-DNGSPICE=$<TARGET_FILE:ngspice>"
                         "-DTEST_CIR=${_dir}/${_cir}"
                         "-DWORKDIR=${_work}"
                         "-DEXTRA_ARGS=${_extra}"
                         "-DSPICE_SCRIPTS=${NGSPICE_TEST_SCRIPTS}"
                         -P "${CMAKE_SOURCE_DIR}/cmake/RunNgspiceTest.cmake")

        set(_labels "${_prefix}")
        string(REGEX REPLACE "\\..*" "" _top "${_prefix}")
        list(APPEND _labels "${_top}")
        if(_in_check)
            list(APPEND _labels "check")
        else()
            list(APPEND _labels "extra")
        endif()

        set_tests_properties("${_test_name}" PROPERTIES
            LABELS "${_labels}"
            TIMEOUT ${NGSPICE_TEST_TIMEOUT})

        math(EXPR _n_tests "${_n_tests} + 1")
    endforeach()
endforeach()

if(NGSPICE_ENABLE_CODEMODELS)
    # make sure the .cm files exist before any test runs
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "")
endif()

set_property(GLOBAL PROPERTY NGSPICE_TEST_COUNT ${_n_tests})
message(STATUS
    "registered ${_n_tests} tests with CTest "
    "(NGSPICE_TEST_LEVEL=${NGSPICE_TEST_LEVEL}, timeout ${NGSPICE_TEST_TIMEOUT}s)")
