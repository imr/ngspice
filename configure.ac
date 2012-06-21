dnl configure.ac script for ngspice
dnl
dnl This file is part of ngspice.
dnl
dnl (Process this file with autoconf to produce a configure script.)
dnl
dnl This configure file has been updated to mostly follow the guidence in the
dnl GNU Autoconf manual for version 2.59


dnl Initialisation
dnl ---------------

dnl Initialisation of configure
AC_INIT([ngspice], [24], [http://ngspice.sourceforge.net/bugrep.html])

dnl Unique file in the source directory
AC_CONFIG_SRCDIR([src/ngspice.c])

dnl Ensure a recent version of Autoconf is in use
dnl Older versions may not work with this script and this will report the
dnl problem to the user.  (2.52 is a guess and might need some adjustment)
AC_PREREQ([2.59])

dnl Revison stamp the generated ./configure script
AC_REVISION([$Revision$])

dnl Create a configuration header
AC_CONFIG_HEADER([src/include/ngspice/config.h])

dnl Initialise automake stuff
AM_INIT_AUTOMAKE([-Wall -Werror nostdinc])

AC_SUBST([AM_CPPFLAGS], ['-I. -I$(srcdir) -I$(top_builddir)/src/include'])

AC_CONFIG_MACRO_DIR([m4])

dnl Package Options
dnl ---------------

dnl --enable-ansi: try to force --ansi option to the compiler
AC_ARG_ENABLE([ansi],
    [AS_HELP_STRING([--enable-ansi], [Force --ansi option for compilation])])

dnl --disable-debug: remove -g and -Wall  option to the compiler
AC_ARG_ENABLE([debug],
    [AS_HELP_STRING([--enable-debug], [Add -g option for compilation (default)])])

dnl --enable-gprof: add -pg option to the compiler (GCC)
AC_ARG_ENABLE([gprof],
   [AS_HELP_STRING([--enable-gprof], [Produce gprof profiling data in 'gmon.out' (GCC only).])])

dnl --enable-checker: add --with-checker-debug  option to the compiler
AC_ARG_ENABLE([checker],
    [AS_HELP_STRING([--enable-checkergcc], [Option for compilation with checkergcc])])

dnl --enable-gc: Enable Boehm-Weiser Conservative Garbage Collector
AC_ARG_ENABLE([checker],
    [AS_HELP_STRING([--enable-gc], [Use Boehm-Weiser Conservative Garbage Collector. Disabled by default.])])

dnl --enable-nobypass: define NOBYPASS for the code
AC_ARG_ENABLE([nobypass],
    [AS_HELP_STRING([--enable-nobypass], [Don't bypass recalculations of slowly changing variables])])

dnl --enable-capbypass: define CAPBYPASS for the code
AC_ARG_ENABLE([capbypass],
    [AS_HELP_STRING([--enable-capbypass], [Bypass calculation of cbd/cbs in the mosfets if the vbs/vbd voltages are unchanged])])

dnl --enable-nodelimiting: define NODELIMITING for the code
AC_ARG_ENABLE([nodelimiting],
    [AS_HELP_STRING([--enable-nodelimiting], [Experimental damping scheme.])])

dnl --enable-predictor: define PREDICTOR for the code
AC_ARG_ENABLE([predictor],
    [AS_HELP_STRING([--enable-predictor], [Enable a predictor method for convergence.])])

dnl --enable-newpred: define NEWPRED for the code
AC_ARG_ENABLE([newpred],
    [AS_HELP_STRING([--enable-newpred], [Enable NEWPRED whatever it is(?)])])

dnl --enable-newtrunc: define NEWTRUNC for the code
AC_ARG_ENABLE([newtrunc],
    [AS_HELP_STRING([--enable-newtrunc], [Enable, how we want extrapolate capacitances.])])

dnl --enable-sense2: define WANT_SENSE2 for the code
AC_ARG_ENABLE([sense2],
    [AS_HELP_STRING([--enable-sense2], [Use spice2 sensitivity analysis.])])

dnl --enable-ftedebug: enable frontend debug macros
AC_ARG_ENABLE([ftedebug],
    [AS_HELP_STRING([--enable-ftedebug], [Enable ngspice frontend debug])])

dnl --enable-cpdebug: enable frontend debug macros
AC_ARG_ENABLE([cpdebug],
    [AS_HELP_STRING([--enable-cpdebug], [Enable ngspice shell debug])])

dnl --enable-sensdebug: define SENSDEBUG for the code
AC_ARG_ENABLE([sensdebug],
    [AS_HELP_STRING([--enable-sensdebug], [Debug sensitivity code *SENSDEBUG*])])

dnl --enable-asdebug: define ASDEBUG for the code
AC_ARG_ENABLE([asdebug],
    [AS_HELP_STRING([--enable-asdebug], [Debug sensitivity code *ASDEBUG*])])

dnl --enable-stepdebug: define STEPDEBUG for the code
AC_ARG_ENABLE([stepdebug],
    [AS_HELP_STRING([--enable-stepdebug], [Unknown debug option])])

dnl --enable-pzdebug: define PZDEBUG for the code
AC_ARG_ENABLE([pzdebug],
    [AS_HELP_STRING([--enable-pzdebug], [Debug pole/zero code])])

dnl --enable-blktmsdebug: define D_DBG_BLOCKTIMES for the code
AC_ARG_ENABLE([blktmsdebug],
    [AS_HELP_STRING([--enable-blktmsdebug], [Debug distortion code *BLOCKTIMES*])])

dnl --enable-smltmsdebug: define D_DBG_SMALLTIMES for the code
AC_ARG_ENABLE([smltmsdebug],
    [AS_HELP_STRING([--enable-smltmsdebug], [Debug distortion code *SMALLTIMES*])])

dnl --enable-smoketest: a smoketest
AC_ARG_ENABLE([smoketest],
    [AS_HELP_STRING([--enable-smoketest], [Enable smoketest compile])])

dnl --enable-experimental: define EXPERIMENTAL_CODE for the code
AC_ARG_ENABLE([experimental],
    [AS_HELP_STRING([--enable-experimental], [Enable some experimental code])])

dnl --enable-expdevices: Enable the compilation of experimental devices
AC_ARG_ENABLE([expdevices],
    [AS_HELP_STRING([--enable-expdevices], [Enable experimental devices (they do not compile)])])

dnl --enable-xspice: define XSPICE in the code. This is for xspice support
AC_ARG_ENABLE([xspice],
    [AS_HELP_STRING([--enable-xspice], [Enable XSpice enhancements, (experimental)])])

dnl --enable-cider: define CIDER in the code. This is for CIDER support
AC_ARG_ENABLE([cider],
    [AS_HELP_STRING([--enable-cider], [Enable CIDER enhancements, (experimental)])])

dnl --enable-adms: define ADMS in the code. This is for the adms Verilog-A compiler support
AC_ARG_ENABLE([adms],
    [AS_HELP_STRING([--enable-adms], [Enable ADMS code models, (experimental)])])

dnl --enable-adms3: define ADMS in the code. This is for the adms Verilog-A compiler support
AC_ARG_ENABLE([adms3],
    [AS_HELP_STRING([--enable-adms3], [Enable ADMS code models, (experimental) (adms3)])])

dnl --enable-xspice: define XSPICE in the code. This is for xspice support
AC_ARG_ENABLE([pss],
    [AS_HELP_STRING([--enable-pss], [Enable PSS analysis, (very experimental)])])

dnl --enable-ndev: define NDEV in the code. An interface for external device i.e. numerical device
AC_ARG_ENABLE([ndev],
    [AS_HELP_STRING([--enable-ndev], [Enable NDEV interface, (experimental)])])

dnl --enable-cluster: define CLUSTER in the code. This is for cluster support
AC_ARG_ENABLE([cluster],
    [AS_HELP_STRING([--enable-cluster], [Enable cluster support, (experimental)])])

dnl --enable-help: try to force --ansi option to the compiler
AC_ARG_ENABLE([help],
    [AS_HELP_STRING([--enable-help], [Force building nghelp (deprecated)])])

dnl --enable-xgraph: Compile the xgraph plotting program. Default is "no".
dnl Xgraph is a plotting package for X11 once very popular.
AC_ARG_ENABLE([xgraph],
    [AS_HELP_STRING([--enable-xgraph], [Enable xgraph compilation.])])

dnl --with-readline:  Includes GNU readline support into CLI.  Default is "no".
AC_ARG_WITH([readline],
    [AS_HELP_STRING([--with-readline[=yes/no]], [Enable GNU readline support for CLI. Default=no.])])

dnl --with-editline:  Includes BSD Editline support into CLI.  Default is "no".
AC_ARG_WITH([editline],
    [AS_HELP_STRING([--with-editline[=yes/no]], [Enable BSD editline support for CLI. Default=no.])])

dnl --with-tcl: define TCL_MODULE in the code. This is for tcl support
AC_ARG_WITH([tcl],
    [  --with-tcl[=tcldir] Compiles the tcl module instead, [experimental], see README.Tcl])

dnl readline and editline cannot both be enabled
if test "$with_editline" = "yes"; then
    if test "$with_readline" = "yes"; then
        AC_MSG_ERROR([Readline and editline cannot both be enabled])
    fi
fi


dnl Enable maintainer commands only if requested
AM_MAINTAINER_MODE


dnl Compiler checks
dnl ---------------

ext_CFLAGS="$CFLAGS"

AC_PROG_CC

dnl Work on compiler options according to system:
dnl Set default CFLAGS - only use -Wall if we have gcc

dnl the above AC_PROG_CC may set CFLAGS to "-O2 -g"
if test "$enable_debug" = "no"; then
    if test "x$GCC" = "xyes"; then
        AC_MSG_WARN([Removing debugging option!])
        CFLAGS="-O2 -Wall -Wextra -Wmissing-prototypes -Wstrict-prototypes -Wnested-externs -Wold-style-definition -Wredundant-decls -Wconversion -s $ext_CFLAGS"
    else
        AC_MSG_WARN([Removing debugging option!])
        CFLAGS="-O2 $ext_CFLAGS"
    fi
else
    AC_DEFINE([NGDEBUG], [1], [Compile with debug info])
    if test "x$GCC" = "xyes"; then
        CFLAGS="-g -O1 -Wall -Wextra -Wmissing-prototypes -Wstrict-prototypes -Wnested-externs -Wold-style-definition -Wredundant-decls -Wconversion $ext_CFLAGS"
    else
        CFLAGS="-g $ext_CFLAGS"
    fi
fi

dnl the profiling option
if test "$enable_gprof" = "yes"; then
    dnl Probably other compilers support -pg or something similar;
    dnl feel free to extend this to include them.
    if test "$GCC" = "yes"; then
        if test "$enable_shared" = "yes" ; then
            AC_MSG_ERROR([Can't have --enable-gprof without --disable-shared (we
                         recommend also using --enable-all-static).])
        fi
        if test ! "$enable_static" = "yes" ; then
            AC_MSG_WARN([We recommend --enable-static with --enable-gprof.])
        fi
        AC_MSG_NOTICE([Enabling gprof profiling data (to gmon.out).])
        CFLAGS="$CFLAGS -pg"
        LDFLAGS="$LDFLAGS -pg"
    else
        AC_MSG_ERROR([We only support --enable-gprof with GCC right now.])
    fi
fi

dnl Not sure that this will work....
if test "$with_checkergcc" = "yes"; then
    CC="checkergcc"
fi


dnl Checks for ANSI-C header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([strings.h])

if test "$enable_ansi" = "yes"; then
    AC_PROG_CC()
    AC_DIAGNOSE([obsolete], [AM_PROG_CC_STDC:
	your code should no longer depend upon `am_cv_prog_cc_stdc', but upon
	`ac_cv_prog_cc_stdc'.  Remove this warning and the assignment when
	you adjust the code.  You can also remove the above call to
	AC_PROG_CC if you already called it elsewhere.])
    am_cv_prog_cc_stdc=$ac_cv_prog_cc_stdc

    if test ! "$am_cv_prog_cc_stdc"="yes"; then
        AC_MSG_WARN([Failed to find Ansi flag!])
    fi
fi

if test "$enable_smoketest" = "yes"; then
dnl CFLAGS="$CFLAGS -Werror"
    CFLAGS="$CFLAGS -pedantic -W -Wmissing-prototypes"
    CFLAGS="$CFLAGS -Wstrict-prototypes -Wtraditional"
    CFLAGS="$CFLAGS -Wconversion -Wshadow -Wpointer-arith"
    CFLAGS="$CFLAGS -Wcast-qual -Wcast-align -Wwrite-strings"
    CFLAGS="$CFLAGS -Waggregate-return -fshort-enums -fno-common"
    CFLAGS="$CFLAGS -Wnested-externs -Dinline= -g -O4"
fi


dnl Check system we're on , and tune accordingly
AC_CANONICAL_HOST

dnl Expand the prefix variable
if eval "test x$prefix = xNONE"; then
dnl  with the MS WINDOWS executable we suggest C:/Spice/bin as *.exe-path
    if test "$with_windows" = "yes"; then
        dprefix="C:/Spice"
        prefix=$dprefix
    else
        dprefix=$ac_default_prefix
    fi
else
    dprefix=$prefix
fi

##########################################################################
#
# tcl libraries test
#
##########################################################################

dnl
dnl The tclSpice options
dnl
if test "x$with_tcl" != "x" -a "$with_tcl" != "no" ; then
    has_tcl_module=true
    AC_DEFINE([TCL_MODULE], [1], [Tcl Module])
    case $host_os in
        *mingw* )
            has_tclwin=true
            has_no_help=true
            AC_DEFINE([HAS_TCLWIN], [1], [Tcl Windows]);;
        *cygwin* )
            has_tclcyg=true ;;
        *)
            ;;
    esac
 
    with_x=no
    enable_shared=yes
    enable_static=no
 
    AC_MSG_CHECKING([for tclConfig.sh])
    tcl_config_sh=""
    if test "x$with_tcl" != "xyes" ; then
        for dir in \
            $with_tcl
        do
            if test -r "$dir/tclConfig.sh" ; then
                tcl_config_sh="$dir/tclConfig.sh"
                break
            elif test -r "$dir/lib/tclConfig.sh" ; then
                tcl_config_sh="$dir/lib/tclConfig.sh"
                break
            elif test -r "$dir/unix/tclConfig.sh" ; then
                tcl_config_sh="$dir/unix/tclConfig.sh"
                break
            fi
        done
    else
        for dir in \
            $prefix \
            $exec_prefix
        do
            if test -r "$dir/tclConfig.sh" ; then
                tcl_config_sh="$dir/tclConfig.sh"
                break
            elif test -r "$dir/lib/tclConfig.sh" ; then
                tcl_config_sh="$dir/lib/tclConfig.sh"
                break
            elif test -r "$dir/unix/tclConfig.sh" ; then
                tcl_config_sh="$dir/unix/tclConfig.sh"
                break
            fi
        done
 
 
       if test "x$tcl_config_sh" = "x" ; then
           for dir in \
               `ls -dr /usr/local/tcltk/tcl[[7-9]].[[0-9]]* 2>/dev/null` \
               `ls -dr /usr/local/tcl/tcl[[7-9]].[[0-9]]* 2>/dev/null` \
               `ls -dr /usr/local/tcl[[7-9]].[[0-9]]* 2>/dev/null` \
               `ls -dr /usr/share/tcltk/tcl[[7-9]].[[0-9]]* 2>/dev/null` \
               `ls -dr /usr/share/tcl/tcl[[7-9]].[[0-9]]* 2>/dev/null` \
               `ls -dr /usr/share/tcl[[7-9]].[[0-9]]* 2>/dev/null` \
               `ls -dr /usr/lib/tcl/tcl[[7-9]].[[0-9]]* 2>/dev/null` \
               `ls -dr /usr/lib/tcl[[7-9]].[[0-9]]* 2>/dev/null` \
               /usr/local/tcl \
               /usr/local \
               /usr \
               /mingw
          do
              if test -r "$dir/tclConfig.sh" ; then
                  echo "Automatic research of tclConfig.sh Found one, accepting it and stopping research."
                  tcl_config_sh="$dir/tclConfig.sh"
                  break
              elif test -r "$dir/lib/tclConfig.sh" ; then
                  echo "Automatic research of tclConfig.sh Found one, accepting it and stopping research."
                  tcl_config_sh="$dir/lib/tclConfig.sh"
                  break
              fi
          done
       fi
    fi
 
    AC_MSG_RESULT([${tcl_config_sh}])
 
    if test "x$tcl_config_sh" = "x" ; then
        echo "can't find Tcl configuration script \"tclConfig.sh\""
        find /usr/ -name 'tclConfig.sh' -exec echo "Should you add --with-tcl={} to ./configure arguments?" \; | sed -re "s/\/tclConfig.sh/ /"
        exit 1
    fi
 
 .  $tcl_config_sh
 
    CFLAGS="$CFLAGS $TCL_INCLUDE_SPEC"
    CPPFLAGS="$CPPFLAGS $TCL_INCLUDE_SPEC"
 
    AC_CHECK_HEADERS([tcl.h blt.h], [], [AC_MSG_ERROR([Couldn't find Tcl/BLT headers])])
 
    if test ! -x "$TCL_EXEC_PREFIX/bin/tclsh$TCL_VERSION" ; then
        AC_MSG_ERROR([Couldn't find $TCL_EXEC_PREFIX/bin/tclsh$TCL_VERSION])
    fi

    AC_MSG_CHECKING([for TCL module BLT])
    rm -f conftest.tcl
    cat > conftest.tcl << EOF
package require BLT;
exit;
EOF

    if ($TCL_EXEC_PREFIX/bin/tclsh$TCL_VERSION conftest.tcl; exit) 2>/dev/null
    then
        AC_MSG_RESULT([Found])
    else
        AC_MSG_ERROR([Couldn't find BLT])
    fi
    rm -f conftest.tcl
    
    AC_CHECK_LIB([pthread], [pthread_create])
    
    if test -n $libdir ; then
        AC_SUBST([libdir])
        echo "libraries in "$libdir
    fi
    
    if test -z $libdir ; then
        for TCL_PACKAGE_PATH_ELEMENT in $TCL_PACKAGE_PATH ; do
            if test -a $TCL_PACKAGE_PATH_ELEMENT ; then
                libdir=$TCL_PACKAGE_PATH_ELEMENT
                AC_SUBST([libdir])
                echo "library in "$libdir
                break
            fi
        done
    fi

else
    TCL_PACKAGE_PATH=""
    TCL_LIB_SPEC=""
    enable_shared=no
    enable_static=yes
fi

AM_CONDITIONAL([TCL_MODULE], [test x$has_tcl_module = xtrue])
AM_CONDITIONAL([TCLWIN], [test x$has_tclwin = xtrue])
AM_CONDITIONAL([TCLCYG], [test x$has_tclcyg = xtrue])

AC_SUBST([TCL_PACKAGE_PATH])
AC_SUBST([TCL_LIB_SPEC])
##################################################################
#
#
#End of tcl libraries test
#
#################################################################



dnl Checks for programs
AC_LIBTOOL_WIN32_DLL
AC_LIBTOOL_DLOPEN
AM_PROG_CC_C_O
## AM_PROG_LIBTOOL
AC_PROG_LIBTOOL


dnl --with-windows : the user wants to use generate the MS WINDOWS executable
AC_MSG_CHECKING([whether windows code is enabled])
AC_ARG_WITH([windows],
    [AS_HELP_STRING([--with-windows], [MS WINDOWS executable])],
    [if test "$with_windows" = "yes"; then
        AC_DEFINE([HAS_WINDOWS], [1], [define to specify MS Windows executable])
    fi],
    [with_windows="no"]
)
AC_MSG_RESULT([$with_windows])
AM_CONDITIONAL([WINDOWS], [test x$with_windows = xyes])

case $with_windows in
    yes )
        AC_DEFINE([X_DISPLAY_MISSING])
        AC_MSG_RESULT([No X display!])
        has_no_x=true
        if test "x$enable_help" = "xyes"; then
            has_no_help=false
        else
            has_no_help=true
        fi
        CFLAGS="$CFLAGS -mwindows";;
    * )

dnl Check for /proc (virtual process information file system)
AC_CHECK_HEADERS([/proc/meminfo])

dnl Checks for X11 header files and libraries - X11 support can be disabled
dnl by passing the '--without-x' option to configure:

dnl Try to locate the X-Windows include files and libraries
dnl   will set no_x="yes" or no_x=""
AC_PATH_X

dnl will set X_CFLAGS, X_LIBS, and might define X_DISPLAY_MISSING
AC_PATH_XTRA

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_ALLOCA
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_CLOSEDIR_VOID
AC_FUNC_SELECT_ARGTYPES
AC_FUNC_SETVBUF_REVERSED
AC_FUNC_VPRINTF

dnl  Check if the user wants to use the appropriate rpath commands to compile in
dnl  the shared library path for the X libraries.  This is off by default since
dnl  that is how it has been historically.  Using --enable-rpath will turn this on

_use_rpath=no
AC_MSG_CHECKING([if hardcoding of the X11 runtime library path is desired])
AC_ARG_ENABLE([rpath],
    [  --enable-rpath         Enable hardcoding the X11 runtime library path [default=disabled]],
    [
        if test "X$enable_rpath" = "Xno" ; then
            AC_MSG_RESULT([no])
            _use_rpath=no
        else
            AC_MSG_RESULT([yes])
            _use_rpath=yes
        fi
    ],
    [
        AC_MSG_RESULT([no])
        _use_rpath=no
    ]
)

if test "X$_use_rpath" = "Xyes" ; then
    # Try to figure out if we need -Rpath for finding X11 libs
    # at runtime.  Why autoconf doesn't already do this, I don't
    # know...
    xlib_path=""
    for p in $X_LDFLAGS ; do
        case $p in
            -L*)
                xlib_path="$xlib_path $p"
                ;;
        esac
    done

    _save_LIBS=$LIBS
    LIBS="$LIBS $X_LDFLAGS"
    rpath=""
    for fl in "-Wl,-rpath " "-Wl,--rpath " "-rpath " "--rpath " "-Wl,-R" "-R" "-R " ; do
        xlib_rpath=`echo $xlib_path | sed "s/-L/$fl/g"`
        LIBS="$_save_LIBS $X_LIBS $xlib_rpath"
        AC_MSG_CHECKING([if the compiler accepts ${fl}path for runtime libraries])
        AC_LINK_IFELSE([AC_LANG_PROGRAM()],
            [AC_MSG_RESULT([yes])
            rpath=$fl],
            [AC_MSG_RESULT([no])])
        test -n "$rpath" && break
    done
    if test -n "$rpath"; then
        X_LDFLAGS="$X_LDFLAGS $xlib_rpath"
    fi
    LIBS=$_save_LIBS
fi

dnl Checks for X libraries - if X11 wasn't found then don't make following
dnl tests and compile without X11 support - otherwise, check if the following
dnl libraries are present (error if they are not)
dnl In CYGWIN library ordering has to be changed. Is this compatible to LINUX?
dnl XShmAttach is a struct in CYGWIN, not a function
dnl SJB: 13th march 2005
dnl Library order is giving linker warnings on MacOSX
dnl It's not clear to me which order is required for Cygwin (see comment above)
dnl and neither can I find any authoritative answer for the correct link order
dnl for MacOSX or Linux, but
dnl            -lXaw -lXmu -lXt -lXext -lX11
dnl seems to be the popular choice.
dnl (The previous order was -lX11 -lXt -lXext -lXmu -lXaw)

if test "x$no_x" != "xyes" ; then
    AC_CHECK_LIB([Xaw], [main], [X_LIBS="$X_LIBS -lXaw"], [AC_MSG_ERROR([Couldn't find Xaw library])], [$X_LIBS  $X_EXTRA_LIBS])
    AC_CHECK_LIB([Xmu], [main], [X_LIBS="$X_LIBS -lXmu"], [AC_MSG_ERROR([Couldn't find Xmu library])], [$X_LIBS $X_EXTRA_LIBS])
    X_LIBS="$X_LIBS -lXt"
    AC_CHECK_LIB([Xext], [XShmAttach], [X_LIBS="$X_LIBS -lXext"], [AC_MSG_ERROR([Couldn't find Xext library])], [$X_LIBS $X_EXTRA_LIBS])
    X_LIBS="$X_LIBS -lX11"
    has_no_x=false
else
    AC_DEFINE([X_DISPLAY_MISSING])
    AC_MSG_RESULT([No X display!])
    has_no_x=true
fi

if ( test "x$enable_help" = "xyes" ) && ! ( test "$no_x" = "yes" ) ; then
    has_no_help=false
else
    has_no_help=true
fi ;;

esac
# end of case --with-windows

AM_CONDITIONAL([NO_X], [test x$has_no_x = xtrue])
AM_CONDITIONAL([NO_HELP], [test x$has_no_help = xtrue])

dnl Check for a few typdefs:
AC_TYPE_PID_T
AC_TYPE_SIGNAL

dnl sighandler_t is the type of the signal handler on GNU variants,
dnl sig_t is the type of a signal handler on 4.4BSD's,
dnl other systems use __sighandler_t.
AC_CHECK_TYPES([sighandler_t, sig_t, __sighandler_t], [], [],
    [#include <sys/types.h>
     #include <signal.h>])

dnl Check for a few libraries and headers:
AC_HEADER_DIRENT
AC_CHECK_HEADERS([unistd.h ctype.h pwd.h fcntl.h sys/ioctl.h stropts.h])
AC_HEADER_SYS_WAIT
AC_HEADER_STAT

AC_CHECK_HEADERS([arpa/inet.h netdb.h netinet/in.h stddef.h sys/file.h sys/param.h sys/socket.h sys/time.h sys/timeb.h sys/io.h])


dnl Check time and resources headers and functions:
AC_HEADER_TIME
AC_STRUCT_TM
AC_STRUCT_TIMEZONE
AC_CHECK_FUNCS([localtime])

AC_CHECK_FUNCS([ftime gettimeofday])
dnl Do not use time or getrusage function for CPU time measurement under OpenMP
if test "$enable_openmp" != "yes"; then
    AC_CHECK_FUNCS([time getrusage])
fi
AC_CHECK_FUNCS([utimes])
AC_CHECK_FUNCS([getrlimit ulimit], [break])

AC_CHECK_FUNCS([endpwent gethostbyname memset select socket strdup strerror strncasecmp strstr strtol])

dnl Look for termios first (posix)
AC_CHECK_HEADERS([termios.h termio.h sgtty.h], [break])
AC_CHECK_FUNCS([isatty tcgetattr tcsetattr])

dnl Check for a few functions:
AC_FUNC_FORK([])
AC_CHECK_FUNCS([access bcopy bzero qsort dup2 popen])
AC_CHECK_FUNCS([strchr index], [break])
AC_CHECK_FUNCS([strrchr rindex], [break])
AC_CHECK_FUNCS([getcwd getwd], [break])


AC_MSG_RESULT([Checking mathematical features of the system:])
dnl Look for math library:
AC_CHECK_LIB([m], [sqrt])
AC_CHECK_HEADERS([float.h limits.h values.h ieeefp.h])

dnl Check for a few mathematical functions:
AC_CHECK_FUNCS([erfc logb scalb scalbn asinh acosh atanh finite])
dnl According POSIX we should look for macros first
AC_CHECK_DECLS([isinf], [], [], [[#include <math.h>]])
AC_CHECK_DECLS([isnan], [], [], [[#include <math.h>]])
dnl But may be we have still functions
if test "$ac_cv_have_decl_isinf" != yes; then
    AC_CHECK_FUNC([isinf])
fi
if test "$ac_cv_have_decl_isnan" != yes; then
    AC_CHECK_FUNC([isnan])
fi

dnl If user enables garbage collection, look for garbage collector
if test "$TCL_PACKAGE_PATH" = ""; then
    if test "$enable_gc" = "yes"; then
        AC_MSG_RESULT([Checking for the presence of the Garbage Collector:])
        AC_CHECK_LIB([gc], [GC_malloc],
            [AC_DEFINE([HAVE_LIBGC], [], [Define if we want garbage collection enabled])
             LIBS="$LIBS -lgc"])
    fi
fi
dnl check, if we have sigsetjmp and siglongjmp.
dnl A trivial AC_CHECK_FUNCS(sigsetjmp) won't do because sigsetjmp() might be a
dnl macro declared in <setjmp.h>. (joze)
AC_MSG_CHECKING([for sigsetjmp])
AC_TRY_LINK([#include <setjmp.h>],[jmp_buf env; sigsetjmp(env, 1);],
    [AC_MSG_RESULT([yes])
        AC_DEFINE([HAVE_SIGSETJMP], [], [Define if we have sigsetjmp().])],
    [AC_MSG_RESULT([no])])

dnl Check for some headers (asprintf, dirname, etc.)
AC_CHECK_HEADERS([libiberty.h libgen.h])

dnl Check for the asprintf function:
AC_CHECK_FUNCS([asprintf], [], [AC_CHECK_LIB([iberty], [asprintf], [AC_DEFINE([HAVE_ASPRINTF], [1], [Have asprintf in libiberty]) LIBS="$LIBS -liberty"])])

dnl Check for the snprintf function:
if test "$ac_cv_func_asprintf" = no ; then
    AC_CHECK_FUNCS([snprintf])
fi

dnl Check for the dirname function:
AC_CHECK_FUNCS([dirname], [], [AC_CHECK_LIB([gen], [dirname], [AC_DEFINE([HAVE_DIRNAME], [1], [Have dirname in libgen]) LIBS="$LIBS -lgen"])])

AC_CHECK_HEADERS([getopt.h])
AC_CHECK_FUNC([getopt_long], [getopt_long=true])
if test "$getopt_long" = "true"; then
    AC_DEFINE([HAVE_GETOPT_LONG], [1], [Have fcn getopt_long()])
fi

AC_DEFINE_UNQUOTED([NGSPICEBINDIR], ["`echo $dprefix/bin`"], [Define the directory for executables])
AC_DEFINE_UNQUOTED([NGSPICEDATADIR], ["`echo $dprefix/share/ngspice`"], [Define the directory for architecture independent data files])
AC_DEFINE_UNQUOTED([NGSPICEBUILDDATE], ["`date`"], [Define the build date])

if test "$with_windows" = "yes"; then
    AC_MSG_RESULT([WINDOWS code enabled])
    AC_CHECK_FUNCS([memmove])
fi

# Recapitulate settings:
AC_MSG_RESULT([Settings which were chosen:])

if test "$enable_sense2" = "yes"; then
    AC_DEFINE([WANT_SENSE2], [], [Define if we want spice2 sensitivity analysis])
    AC_MSG_RESULT([Spice2 sensitivity analysis enabled])
fi
if test "$enable_nobypass" = "yes"; then
    AC_DEFINE([NOBYPASS], [], [Define if we want NOBYPASS])
    AC_MSG_RESULT([NOBYPASS option enabled])
fi
if test "$enable_capbypass" = "yes"; then
    AC_DEFINE([CAPBYPASS], [], [Define if we want to bypass cbd/cbs calculation for non varying vbs/vbd])
    AC_MSG_RESULT([CAPBYPASS option enabled])
fi
if test "$enable_nodelimiting" = "yes"; then
    AC_DEFINE([NODELIMITING], [], [Experimental code never implemented to damp Newton iterations])
    AC_MSG_RESULT([NODELIMITING option enabled])
fi
if test "$enable_predictor" = "yes"; then
    AC_DEFINE([PREDICTOR], [], [Define if we want predictor algorithm])
    AC_MSG_RESULT([PREDICTOR algorithm enabled])
fi
if test "$enable_newpred" = "yes"; then
    AC_DEFINE([NEWPRED], [], [Define if you want to discover :)])
    AC_MSG_RESULT([NEWPRED enabled])
fi
if test "$enable_newtrunc" = "yes"; then
    AC_DEFINE([NEWTRUNC], [], [Do not trigger unwanted traps by default])
    AC_MSG_RESULT([New truncation error calculation enabled])
fi
if test "$enable_experimental" = "yes"; then
    AC_DEFINE([EXPERIMENTAL_CODE], [], [Define if we want some experimental code])
    AC_MSG_RESULT([EXPERIMENTAL_CODE enabled])
fi
if test "$enable_cpdebug" = "yes"; then
    AC_DEFINE([CPDEBUG], [], [Define if you want to debug ngspice shell])
    AC_MSG_RESULT([WARNING: Shell debug is enabled])
fi
if test "$enable_ftedebug" = "yes"; then
    AC_DEFINE([FTEDEBUG], [], [Define if you want to debug frontend])
    AC_MSG_RESULT([WARNING: Frontend debug is enabled])
fi
if test "$enable_sensdebug" = "yes"; then
    AC_DEFINE([SENSDEBUG], [], [Define if we want debug sensititvity analysis])
    AC_MSG_RESULT([WARNING: Sensitivity code debug *SENSDEBUG* is enabled])
fi
if test "$enable_asdebug" = "yes"; then
    AC_DEFINE([ASDEBUG], [], [Define if we want debug sensitivity analysis])
    AC_MSG_RESULT([WARNING: Sensitivity code debug *ASDEBUG* is enabled])
fi
if test "$enable_stepdebug" = "yes"; then
    AC_DEFINE([STEPDEBUG], [], [Define if we want stepdebug])
    AC_MSG_RESULT([WARNING: STEPDEBUG debug is enabled])
fi
if test "$enable_pzdebug" = "yes"; then
    AC_DEFINE([PZDEBUG], [], [Define if you want to debug pole-zero analysis])
    AC_MSG_RESULT([WARNING: Pole/Zero analysis debug is enabled])
fi
if test "$enable_pss" = "yes"; then
    AC_DEFINE([WITH_PSS], [], [Define if you want PSS analysis])
fi
if test "$enable_blktmsdebug" = "yes"; then
    AC_DEFINE([D_DBG_BLOCKTIMES], [], [Define if we want debug distortion analysis (BLOCKTIMES)])
    AC_MSG_RESULT([WARNING: Distortion analysis debug *D_DBG_BLOCKTIMES* is enabled])
fi
if test "$enable_smltmsdebug" = "yes"; then
    AC_DEFINE([D_DBG_SMALLTIMES], [], [Define if we want debug distortion analysis (SMALLTIMES)])
    AC_MSG_RESULT([WARNING: Distortion analysis debug *D_DBG_SMALLTIMES* is enabled])
fi
if test "$enable_xgraph" = "yes"; then
    AC_MSG_RESULT([Xgraph compilation enabled.])
    AC_CONFIG_SUBDIRS([xgraph])
    XGRAPHDIR="xgraph"
    NOTXGRAPH=""
else
    XGRAPHDIR=""
    NOTXGRAPH="xgraph"
fi

AC_SUBST([XGRAPHDIR])
AC_SUBST([NOTXGRAPH])

AC_CHECK_PROGS([YACC], ['bison -y' byacc yacc])

#################  XSPICE ##################################################
dnl Add new code models to the build by pointing to them here.

if test "$enable_xspice" = "yes"; then
    if test -z "${YACC}" ; then
        AC_MSG_ERROR([XSPICE build requires bison, byacc or yacc parser generator])
    fi
    AC_MSG_RESULT([X-Spice features included])
    AC_DEFINE([XSPICE], [1], [The xspice enhancements])
dnl Define variables for LEX
    AC_CHECK_PROGS([LEX], [flex lex])
    if test -z $LEX ; then
        AC_MSG_ERROR([Flex is required for building XSPICE])
    fi
    case $host_os in
        *mingw* )
            AC_DEFINE([IPC_DEBUG_VIA_STDIO], [1], [Client-Server only via stdio.])
            XSPICEDLLIBS="";;
        *cygwin* )
            AC_DEFINE([IPC_UNIX_SOCKETS], [1], [Client-Server via socket.])
            XSPICEDLLIBS="" ;;
        *freebsd* )
            AC_DEFINE([IPC_UNIX_SOCKETS], [1], [Client-Server via socket.])
            XSPICEDLLIBS="";;
        *openbsd* )
            AC_DEFINE([IPC_UNIX_SOCKETS], [1], [Client-Server via socket.])
            XSPICEDLLIBS="";;
        * )
            AC_DEFINE([IPC_UNIX_SOCKETS], [1], [Client-Server via socket.])
            XSPICEDLLIBS="-ldl";;
    esac
    XSPICEINIT=""
    AC_CHECK_FUNCS([modf])
    AC_CHECK_HEADERS([libintl.h malloc.h])
else
    XSPICEINIT="*"

    if test -z "${YACC}" ; then
        AC_MSG_WARN([No bison, byacc, yacc found: Uses prebuilt default parsers])
    fi
fi
AC_SUBST([XSPICEINIT])
AC_SUBST([XSPICEDLLIBS])

AM_CONDITIONAL([XSPICE_WANTED], [test "$enable_xspice" = "yes"])


dnl Add CIDER enhancements to ngspice.
if test "$enable_cider" = "yes"; then
    AC_MSG_RESULT([CIDER features enabled])
    AC_DEFINE([CIDER], [1], [The CIDER enhancements])
fi

AM_CONDITIONAL([CIDER_WANTED], [test "$enable_cider" = "yes"])
AM_CONDITIONAL([NUMDEV_WANTED], [test "$enable_cider" = "yes"])


AM_CONDITIONAL([PSS_WANTED], [test "$enable_pss" = "yes"])

dnl adms option
if test "$enable_adms" = "yes" -o "$enable_adms3" = "yes" ; then
    AC_MSG_RESULT([**********************************
* ADMS  support is  experimental *
**********************************])
    AC_CHECK_PROGS([ADMSXML], [admsXml admsXml.exe], [no])

    if test "$ADMSXML" = "no"; then
        AC_MSG_ERROR([If you want Verilog-A models you should install admsXml])
    fi
    if test "$enable_adms3" = "yes" ; then
        AC_DEFINE([ADMS], [3], [Support for Verilog-A models (adms3)])
    else
        AC_DEFINE([ADMS], [1], [Support for Verilog-A models])
    fi

    VLADEVDIR=" adms/ekv \
                adms/hicum0 \
                adms/hicum2 \
                adms/mextram \
                adms/psp102 "

dnl The makefiles for adms (to be added to AC_CONFIG_FILES by ./autogen.sh --adms)
#VLAMKF    src/spicelib/devices/adms/ekv/Makefile
#VLAMKF    src/spicelib/devices/adms/hicum0/Makefile
#VLAMKF    src/spicelib/devices/adms/hicum2/Makefile
#VLAMKF    src/spicelib/devices/adms/mextram/Makefile
#VLAMKF    src/spicelib/devices/adms/psp102/Makefile

    NOTVLADEVDIR=""

    VLADEV=" spicelib/devices/adms/ekv/libekv.la \
             spicelib/devices/adms/hicum0/libhicum0.la \
             spicelib/devices/adms/hicum2/libhicum2.la \
             spicelib/devices/adms/mextram/libbjt504t.la \
             spicelib/devices/adms/psp102/libpsp102.la "

else

    VLADEVDIR=""
    NOTVLADEVDIR="adms"

fi

AC_SUBST([ADMSXML])
AC_SUBST([VLADEVDIR])
AC_SUBST([VLADEV])
AC_SUBST([NOTVLADEVDIR])

dnl NDEV option
if test "$enable_ndev" = "yes"; then
    AC_MSG_RESULT([NDEV features enabled])
    AC_DEFINE([NDEV], [], [The NDEV interface])
fi
AM_CONDITIONAL([NDEV_WANTED], [test "$enable_ndev" = "yes"])


dnl Cluster option
if test "$enable_cluster" = "yes"; then
    AC_MSG_RESULT([Cluster version is being compiled])
    AC_DEFINE([CLUSTER], [], [Spice cluster support])
    LIBS="$LIBS -lpthread"
fi

if test "$enable_expdevices" = "yes"; then
    AC_DEFINE([EXP_DEV], [], [Define if we want to enable experimental devices])
    AC_MSG_RESULT([WARNING: Experimental devices enabled])
fi

dnl  ----  Option to include GNU readline support in ngspice CLI  ----
dnl  ----  Default: disabled.                                     ----
dnl  ----  Hope to see in the future readline replacement.        ----

if test "$with_readline" != "yes"; then
    AC_MSG_RESULT([GNU readline disabled.])
else
    if test "x$with_tcl" = "x" -o "$with_tcl" = "no" ; then
        AC_MSG_RESULT([Checking for readline:])
        AC_CHECK_HEADERS([readline/readline.h readline/history.h],
            [AC_DEFINE([HAVE_GNUREADLINE], [], [Define if we have GNU readline])],
            [AC_MSG_ERROR([Couldn't find GNU readline headers.])])
        AC_SEARCH_LIBS([tputs], [ncurses termcap],
            [AC_DEFINE([HAVE_TERMCAP], [], [Define if we have ncurses or termcap])],
            [AC_MSG_ERROR([Found neither ncurses or termcap])])
        AC_CHECK_LIB([readline], [readline],
            [LIBS="$LIBS -lreadline"],
            [AC_MSG_ERROR([Couldn't find readline libraries.])])
    fi
fi


dnl  ----  Option to include BSD editline support in ngspice CLI  ----
dnl  ----  Default: disabled.                                     ----

if test "$with_editline" != "yes"; then
    AC_MSG_RESULT([BSD editline disabled.])
else
    AC_MSG_RESULT([Checking for editline:])
    AC_CHECK_HEADERS([editline/readline.h],
        [AC_DEFINE([HAVE_BSDEDITLINE], [1], [Define to enable BSD editline])],
        [AC_MSG_ERROR([Couldn't find BSD editline headers.])])
    AC_SEARCH_LIBS([tputs], [ncurses termcap],
        [AC_DEFINE([HAVE_TERMCAP], [], [Define if we have ncurses or termcap])],
        [AC_MSG_ERROR([Found neither ncurses or termcap])])
    AC_CHECK_LIB([edit], [readline],
        [LIBS="$LIBS -ledit"],
        [AC_MSG_ERROR([Couldn't find editline libraries.])],
        [-lncurses])
fi

dnl Use AC_CHECK_HEADERS so the HAVE_*_H symbol gets defined
AC_CHECK_HEADERS([ncurses/termcap.h termcap.h])

dnl --enable-openmp: Use OpenMP on multi-core processors
AC_ARG_ENABLE([openmp],
    [AS_HELP_STRING([--enable-openmp], [Use OpenMP parallel processing])])

dnl Add OpenMP to ngspice.
: ${enable_openmp:=no}
AC_OPENMP

if test "$enable_openmp" = "yes"; then
    AC_DEFINE([USE_OMP], [1], [OpenMP parallel processing])
    CFLAGS="$CFLAGS $OPENMP_CFLAGS"
    AC_MSG_RESULT([OpenMP feature enabled])
fi

dnl Output Files
dnl ------------

AC_CONFIG_FILES([Makefile
                 man/Makefile
                 man/man1/Makefile
                 src/Makefile
                 src/spicelib/Makefile
                 src/spicelib/analysis/Makefile
                 src/spicelib/devices/Makefile
                 src/spicelib/devices/asrc/Makefile
                 src/spicelib/devices/bjt/Makefile
                 src/spicelib/devices/bsim1/Makefile
                 src/spicelib/devices/bsim2/Makefile
                 src/spicelib/devices/bsim3/Makefile
                 src/spicelib/devices/bsim3v0/Makefile
                 src/spicelib/devices/bsim3v1/Makefile
                 src/spicelib/devices/bsim3v32/Makefile
                 src/spicelib/devices/bsim4/Makefile
                 src/spicelib/devices/bsim4v4/Makefile
                 src/spicelib/devices/bsim4v5/Makefile
                 src/spicelib/devices/bsim4v6/Makefile
                 src/spicelib/devices/bsim3soi/Makefile
                 src/spicelib/devices/bsim3soi_pd/Makefile
                 src/spicelib/devices/bsim3soi_fd/Makefile
                 src/spicelib/devices/bsim3soi_dd/Makefile
                 src/spicelib/devices/cap/Makefile
                 src/spicelib/devices/cccs/Makefile
                 src/spicelib/devices/ccvs/Makefile
                 src/spicelib/devices/csw/Makefile
                 src/spicelib/devices/cpl/Makefile
                 src/spicelib/devices/dio/Makefile
                 src/spicelib/devices/ind/Makefile
                 src/spicelib/devices/isrc/Makefile
                 src/spicelib/devices/hfet1/Makefile
                 src/spicelib/devices/hfet2/Makefile
                 src/spicelib/devices/hisim2/Makefile
                 src/spicelib/devices/hisimhv/Makefile
                 src/spicelib/devices/jfet/Makefile
                 src/spicelib/devices/jfet2/Makefile
                 src/spicelib/devices/ltra/Makefile
                 src/spicelib/devices/mes/Makefile
                 src/spicelib/devices/mesa/Makefile
                 src/spicelib/devices/mos1/Makefile
                 src/spicelib/devices/mos2/Makefile
                 src/spicelib/devices/mos3/Makefile
                 src/spicelib/devices/mos6/Makefile
                 src/spicelib/devices/mos9/Makefile
                 src/spicelib/devices/ndev/Makefile
                 src/spicelib/devices/res/Makefile
                 src/spicelib/devices/soi3/Makefile
                 src/spicelib/devices/sw/Makefile
                 src/spicelib/devices/tra/Makefile
                 src/spicelib/devices/txl/Makefile
                 src/spicelib/devices/urc/Makefile
                 src/spicelib/devices/vbic/Makefile
                 src/spicelib/devices/vccs/Makefile
                 src/spicelib/devices/vcvs/Makefile
                 src/spicelib/devices/vsrc/Makefile
                 src/spicelib/devices/nbjt/Makefile
                 src/spicelib/devices/nbjt2/Makefile
                 src/spicelib/devices/numd/Makefile
                 src/spicelib/devices/numd2/Makefile
                 src/spicelib/devices/numos/Makefile
                 src/spicelib/parser/Makefile
                 src/ciderlib/Makefile
                 src/ciderlib/input/Makefile
                 src/ciderlib/support/Makefile
                 src/ciderlib/oned/Makefile
                 src/ciderlib/twod/Makefile
                 src/frontend/Makefile
                 src/frontend/numparam/Makefile
                 src/frontend/help/Makefile
                 src/frontend/parser/Makefile
                 src/frontend/plotting/Makefile
                 src/frontend/trannoise/Makefile
                 src/frontend/wdisp/Makefile
                 src/include/ngspice/Makefile
                 src/maths/Makefile
                 src/maths/cmaths/Makefile
                 src/maths/fft/Makefile
                 src/maths/misc/Makefile
                 src/maths/ni/Makefile
                 src/maths/deriv/Makefile
                 src/maths/poly/Makefile
                 src/maths/sparse/Makefile
                 src/misc/Makefile
                 src/xspice/Makefile
                 src/xspice/cm/Makefile
                 src/xspice/cmpp/Makefile
                 src/xspice/icm/makedefs
                 src/xspice/icm/GNUmakefile
                 src/xspice/mif/Makefile
                 src/xspice/evt/Makefile
                 src/xspice/enh/Makefile
                 src/xspice/ipc/Makefile
                 src/xspice/idn/Makefile
                 tests/Makefile
                 tests/bsim1/Makefile
                 tests/bsim2/Makefile
                 tests/bsim3/Makefile
                 tests/bsim3/ac_sim/Makefile
                 tests/bsim3/dc_sim/Makefile
                 tests/bsim3/tran_sim/Makefile
                 tests/bsim3soi/Makefile
                 tests/bsim3soidd/Makefile
                 tests/bsim3soifd/Makefile
                 tests/bsim3soipd/Makefile
                 tests/bsim4/Makefile
                 tests/filters/Makefile
                 tests/general/Makefile
                 tests/hfet/Makefile
                 tests/hisim/Makefile
                 tests/hisimhv/Makefile
                 tests/jfet/Makefile
                 tests/mes/Makefile
                 tests/mesa/Makefile
                 tests/mos6/Makefile
                 tests/polezero/Makefile
                 tests/sensitivity/Makefile
                 tests/transient/Makefile
                 tests/transmission/Makefile
                 tests/xspice/Makefile
                 tests/xspice/digital/Makefile
                 tests/xspice/digital/spinit
                 tests/resistance/Makefile
                 tests/vbic/Makefile])
AC_OUTPUT
