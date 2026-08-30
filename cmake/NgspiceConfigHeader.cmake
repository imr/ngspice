# NgspiceConfigHeader.cmake
#
# Performs the system introspection that configure does and writes
# ${NGSPICE_GENERATED_INCLUDE_DIR}/ngspice/config.h from cmake/config.h.in.

include(CheckIncludeFile)
include(CheckFunctionExists)
include(CheckSymbolExists)
include(CheckTypeSize)
include(CheckStructHasMember)
include(CheckCSourceCompiles)

# ---------------------------------------------------------------------------
# Progress output
#
# These checks are ~110 separate compiler invocations. That is quick with gcc
# but takes over a minute with MSVC, so report each one. The label is printed
# before the check runs, which also shows where configuration is stalling;
# message() would append a newline, hence "cmake -E echo_append".
#
# -DNGSPICE_QUIET_CONFIG=ON reduces this to one summary line per group.
# ---------------------------------------------------------------------------
option(NGSPICE_QUIET_CONFIG "Only summarise the system checks" OFF)

set(_ngspice_probe_pad "........................................")

function(_ngspice_probe_start kind name)
    if(NGSPICE_QUIET_CONFIG)
        return()
    endif()
    string(LENGTH "${name}" _l)
    math(EXPR _n "34 - ${_l}")
    if(_n LESS 1)
        set(_n 1)
    endif()
    string(SUBSTRING "${_ngspice_probe_pad}" 0 ${_n} _dots)
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append
                    "-- ${kind} ${name} ${_dots} ")
endfunction()

function(_ngspice_probe_result var)
    if(NGSPICE_QUIET_CONFIG)
        return()
    endif()
    if(${var})
        execute_process(COMMAND ${CMAKE_COMMAND} -E echo "found")
    else()
        execute_process(COMMAND ${CMAKE_COMMAND} -E echo "missing")
    endif()
endfunction()

set(CMAKE_REQUIRED_QUIET TRUE)

# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
set(_ngspice_headers
    alloca.h arpa/inet.h ctype.h dirent.h dlfcn.h editline/readline.h
    fcntl.h float.h getopt.h ieeefp.h inttypes.h libintl.h limits.h
    malloc.h memory.h ncurses/termcap.h ndir.h netdb.h netinet/in.h pwd.h
    readline/history.h readline/readline.h samplerate.h sgtty.h sndfile.h
    stdbool.h stddef.h stdint.h stdlib.h string.h strings.h stropts.h
    sys/dir.h sys/file.h sys/io.h sys/ioctl.h sys/ndir.h sys/param.h
    sys/resource.h sys/select.h sys/socket.h sys/stat.h sys/sysctl.h
    sys/time.h sys/timeb.h sys/types.h sys/wait.h term.h termcap.h
    termio.h termios.h unistd.h values.h vfork.h)

set(_n_found 0)
foreach(_h IN LISTS _ngspice_headers)
    string(TOUPPER "${_h}" _v)
    string(REGEX REPLACE "[/.]" "_" _v "${_v}")
    _ngspice_probe_start("header  " "${_h}")
    check_include_file("${_h}" HAVE_${_v})
    _ngspice_probe_result(HAVE_${_v})
    if(HAVE_${_v})
        math(EXPR _n_found "${_n_found} + 1")
    endif()
endforeach()
list(LENGTH _ngspice_headers _n)
message(STATUS "headers: ${_n_found} of ${_n} available")

if(EXISTS "/proc/meminfo")
    set(HAVE__PROC_MEMINFO 1)
endif()

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
if(UNIX)
    list(APPEND CMAKE_REQUIRED_LIBRARIES m)
    set(HAVE_LIBM 1)
endif()

set(_ngspice_funcs
    access acosh alloca asinh atanh clock_gettime dup2 endpwent erfc finite
    ftime getcwd gethostbyname getopt_long getrlimit gettimeofday getwd index
    isatty isnan localtime logb memmove memset modf popen qsort random
    rindex scalb scalbn select snprintf socket strchr strdup strerror
    strncasecmp strrchr strstr strtol tcgetattr tcsetattr tdestroy time
    tsearch ulimit utimes vprintf fork vfork dirname)

# configure.ac: "Do not use times or getrusage function for CPU time
# measurement under OpenMP" -- both report the summed CPU time of all
# threads. Leaving them undefined makes the #elif chain in
# src/misc/misc_time.c fall through to a wall clock source.
if(NOT NGSPICE_ENABLE_OPENMP)
    list(APPEND _ngspice_funcs times getrusage)
endif()

set(_n_found 0)
foreach(_f IN LISTS _ngspice_funcs)
    string(TOUPPER "${_f}" _v)
    _ngspice_probe_start("function" "${_f}")
    check_function_exists("${_f}" HAVE_${_v})
    _ngspice_probe_result(HAVE_${_v})
    if(HAVE_${_v})
        math(EXPR _n_found "${_n_found} + 1")
    endif()
endforeach()
list(LENGTH _ngspice_funcs _n)
message(STATUS "functions: ${_n_found} of ${_n} available")

if(HAVE_FORK)
    set(HAVE_WORKING_FORK 1)
endif()
if(HAVE_VFORK)
    set(HAVE_WORKING_VFORK 1)
endif()

# sigsetjmp is a macro on glibc, so check_function_exists would fail
_ngspice_probe_start("symbol  " "sigsetjmp")
check_symbol_exists(sigsetjmp "setjmp.h" HAVE_SIGSETJMP)
_ngspice_probe_result(HAVE_SIGSETJMP)

_ngspice_probe_start("symbol  " "isinf")
check_symbol_exists(isinf "math.h" _ngspice_decl_isinf)
_ngspice_probe_result(_ngspice_decl_isinf)
_ngspice_probe_start("symbol  " "isnan")
check_symbol_exists(isnan "math.h" _ngspice_decl_isnan)
_ngspice_probe_result(_ngspice_decl_isnan)
_ngspice_probe_start("symbol  " "tzname")
check_symbol_exists(tzname "time.h" _ngspice_decl_tzname)
_ngspice_probe_result(_ngspice_decl_tzname)
set(HAVE_DECL_ISINF ${_ngspice_decl_isinf})
set(HAVE_DECL_ISNAN ${_ngspice_decl_isnan})
set(HAVE_DECL_TZNAME ${_ngspice_decl_tzname})

_ngspice_probe_start("type    " "_Bool")
check_type_size(_Bool HAVE__BOOL)
_ngspice_probe_result(HAVE__BOOL)
_ngspice_probe_start("member  " "struct tm.tm_zone")
check_struct_has_member("struct tm" tm_zone "time.h" HAVE_STRUCT_TM_TM_ZONE)
_ngspice_probe_result(HAVE_STRUCT_TM_TM_ZONE)
if(HAVE_STRUCT_TM_TM_ZONE)
    set(HAVE_TM_ZONE 1)
endif()
if(HAVE_DECL_TZNAME)
    set(HAVE_TZNAME 1)
endif()

if(HAVE_SYS_TIME_H)
    _ngspice_probe_start("feature " "time.h with sys/time.h")
    check_c_source_compiles("
        #include <sys/types.h>
        #include <sys/time.h>
        #include <time.h>
        int main(void) { struct tm t; struct timeval tv; (void)t; (void)tv; return 0; }"
        TIME_WITH_SYS_TIME)
    _ngspice_probe_result(TIME_WITH_SYS_TIME)
endif()

set(STDC_HEADERS 1)

# select() prototype
if(WIN32)
    set(SELECT_TYPE_ARG1 int)
    set(SELECT_TYPE_ARG234 "(fd_set *)")
    set(SELECT_TYPE_ARG5 "(struct timeval *)")
    set(HAVE_QUERYPERFORMANCECOUNTER 1)
else()
    set(SELECT_TYPE_ARG1 int)
    set(SELECT_TYPE_ARG234 "(fd_set *)")
    set(SELECT_TYPE_ARG5 "(struct timeval *)")
endif()

# ---------------------------------------------------------------------------
# Derived from the feature options
# ---------------------------------------------------------------------------
set(XSPICE          ${NGSPICE_ENABLE_XSPICE})
set(CIDER           ${NGSPICE_ENABLE_CIDER})
set(OSDI            ${NGSPICE_ENABLE_OSDI})
set(KLU             ${NGSPICE_ENABLE_KLU})
set(RFSPICE         ${NGSPICE_ENABLE_RFSPICE})
set(WITH_PSS        ${NGSPICE_ENABLE_PSS})
set(NDEV            ${NGSPICE_ENABLE_NDEV})
set(WANT_SENSE2     ${NGSPICE_ENABLE_SENSE2})
set(USE_OMP         ${NGSPICE_ENABLE_OPENMP})
set(SHARED_MODULE   ${NGSPICE_BUILD_SHARED})
set(HAS_WINGUI      ${NGSPICE_WINGUI})
set(HAS_RELPATH     ${NGSPICE_ENABLE_RELPATH})
set(EXT_ASC         ${NGSPICE_DISABLE_UTF8})
if(NOT NGSPICE_ENABLE_X11)
    set(X_DISPLAY_MISSING 1)
endif()

# Build date -- reproducible builds honour SOURCE_DATE_EPOCH
if(DEFINED ENV{SOURCE_DATE_EPOCH})
    string(TIMESTAMP NGSPICE_BUILD_DATE "%Y-%m-%d" UTC)
else()
    string(TIMESTAMP NGSPICE_BUILD_DATE "%Y-%m-%d" UTC)
endif()

# OS id, see the case $host_os block in configure.ac
if(MINGW OR MSYS)
    set(NGSPICE_OS_COMPILED 1)
elseif(CYGWIN)
    set(NGSPICE_OS_COMPILED 2)
elseif(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
    set(NGSPICE_OS_COMPILED 3)
elseif(CMAKE_SYSTEM_NAME STREQUAL "OpenBSD")
    set(NGSPICE_OS_COMPILED 4)
elseif(CMAKE_SYSTEM_NAME STREQUAL "SunOS")
    set(NGSPICE_OS_COMPILED 5)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(NGSPICE_OS_COMPILED 6)
elseif(APPLE)
    set(NGSPICE_OS_COMPILED 7)
else()
    set(NGSPICE_OS_COMPILED 0)
endif()

# ---------------------------------------------------------------------------
# Write it
# ---------------------------------------------------------------------------
file(MAKE_DIRECTORY "${NGSPICE_GENERATED_INCLUDE_DIR}/ngspice")
configure_file("${CMAKE_CURRENT_LIST_DIR}/config.h.in"
               "${NGSPICE_GENERATED_INCLUDE_DIR}/ngspice/config.h")

unset(CMAKE_REQUIRED_QUIET)
