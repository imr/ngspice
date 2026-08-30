# RunNgspiceTest.cmake -- portable replacement for tests/bin/check.sh
#
# Invoked by CTest as
#   cmake -DNGSPICE=... -DTEST_CIR=... -DWORKDIR=... -DEXTRA_ARGS=...
#         -DSPICE_SCRIPTS=... -P RunNgspiceTest.cmake
#
# The logic follows tests/bin/check.sh: run the netlist in batch mode,
# normalise three-digit exponents, drop the volatile lines and compare
# what remains against the reference .out file.  Doing this in CMake
# rather than /bin/sh means the same test suite runs under MSVC.

if(NOT NGSPICE OR NOT TEST_CIR)
    message(FATAL_ERROR "NGSPICE and TEST_CIR must be set")
endif()

get_filename_component(_name "${TEST_CIR}" NAME_WE)
get_filename_component(_dir  "${TEST_CIR}" DIRECTORY)
set(_ref "${_dir}/${_name}.out")

if(NOT EXISTS "${_ref}")
    message(FATAL_ERROR "reference output ${_ref} is missing")
endif()

file(MAKE_DIRECTORY "${WORKDIR}")

# Lines that legitimately differ between runs and platforms.
set(NGSPICE_TEST_FILTER
    "SPARSE|KLU|CPU|Dynamic|Note|Circuit|Trying|Reference|Date|Doing|---|v-sweep|time|est|Error|Warning|Data|Index|trans|acan|oise|nalysis|ole|Total|memory|urrent|Got|Added|BSIM|bsim|B4SOI|b4soi|codemodel|^binary raw file|^ngspice.*done|Operating")

separate_arguments(_extra NATIVE_COMMAND "${EXTRA_ARGS}")

set(ENV{SPICE_SCRIPTS} "${SPICE_SCRIPTS}")
set(ENV{ngspice_vpath} "${_dir}")

execute_process(
    COMMAND "${NGSPICE}" ${_extra} --batch "${TEST_CIR}"
    WORKING_DIRECTORY "${WORKDIR}"
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
    TIMEOUT 600)

if(NOT _rc EQUAL 0)
    message(FATAL_ERROR
        "ngspice exited with ${_rc}\n--- stdout ---\n${_out}\n--- stderr ---\n${_err}")
endif()

file(READ "${_ref}" _refdata)


# Reduce a blob of text to the list of significant lines.
function(_normalise raw out_var)
    # Windows CRTs print three exponent digits where C99 prints two
    string(REGEX REPLACE "([.0-9][eE][+-]?)0([0-9][0-9])" "\\1\\2" raw "${raw}")
    string(REPLACE "\r\n" "\n" raw "${raw}")
    string(REPLACE ";" "\\;" raw "${raw}")
    string(REPLACE "\n" ";" _lines "${raw}")

    set(_keep "")
    foreach(_l IN LISTS _lines)
        if(_l MATCHES "${NGSPICE_TEST_FILTER}")
            continue()
        endif()
        # diff -w: whitespace differences are not significant
        string(REGEX REPLACE "[ \t]+" " " _l "${_l}")
        string(STRIP "${_l}" _l)
        # diff -B: blank lines are not significant
        if(NOT _l STREQUAL "")
            list(APPEND _keep "${_l}")
        endif()
    endforeach()
    set(${out_var} "${_keep}" PARENT_SCOPE)
endfunction()

_normalise("${_out}"     _got)
_normalise("${_refdata}" _want)

if(_got STREQUAL _want)
    return()
endif()

# Produce something readable in the CTest log
list(LENGTH _got  _ngot)
list(LENGTH _want _nwant)
set(_report "output differs from ${_ref} (${_ngot} vs ${_nwant} significant lines)\n")

set(_max ${_nwant})
if(_ngot GREATER _nwant)
    set(_max ${_ngot})
endif()
if(_max GREATER 0)
    math(EXPR _max "${_max} - 1")
endif()
set(_shown 0)
foreach(_i RANGE 0 ${_max})
    if(_shown GREATER_EQUAL 20)
        string(APPEND _report "  ...\n")
        break()
    endif()
    set(_a "")
    set(_b "")
    if(_i LESS _nwant)
        list(GET _want ${_i} _a)
    endif()
    if(_i LESS _ngot)
        list(GET _got ${_i} _b)
    endif()
    if(NOT _a STREQUAL _b)
        string(APPEND _report "  line ${_i}:\n    expected: ${_a}\n    actual  : ${_b}\n")
        math(EXPR _shown "${_shown} + 1")
    endif()
endforeach()

message(FATAL_ERROR "${_report}")
