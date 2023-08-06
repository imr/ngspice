/*************
 * Main header file for ngspice
 * 1999 E. Rouat
 ************/
#ifndef ngspice_NGSPICE_H
#define ngspice_NGSPICE_H

/* #include "memwatch.h"
 #define MEMWATCH */


/* 
 * This file will eventually replace spice.h and lots of other 
 * files in src/include
 */
#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include "ngspice/config.h"
#include <stddef.h>

#ifdef HAVE_LIMITS_H
#  include <limits.h>
#endif
#ifdef HAVE_FLOAT_H
#  include <float.h>
#endif

#include "ngspice/memory.h"
#include "ngspice/defines.h"
#include "ngspice/macros.h"
#include "ngspice/bool.h"
#include "ngspice/complex.h"
#include "ngspice/typedefs.h"

#include <math.h>
#include <stdio.h>

#ifdef HAVE_IEEEFP_H
#include <ieeefp.h>
#endif

#include "ngspice/missing_math.h"

#ifdef STDC_HEADERS
#  include <stdlib.h>
#  include <string.h>
#endif

#ifdef HAVE_STRINGS_H
#  include <strings.h>
#endif

#ifdef HAVE_CTYPE_H
#  include <ctype.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_TERMIOS_H
#include <termios.h>
#else
#  ifdef HAVE_SGTTY_H
#  include <sgtty.h>
#    else
#    ifdef HAVE_TERMIO_H
#      include <termio.h>
#    endif
#  endif
#endif

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif

#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif

#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_DIR_H
#include <sys/dir.h>
#else
#  ifdef HAVE_DIRENT_H
#  include <sys/types.h>
#  include <dirent.h>
#  ifndef direct
#  define direct dirent
#  endif
#  endif
#endif

#ifdef HAVE_GETRLIMIT
#  include <sys/time.h>
#  include <sys/resource.h>
#endif
#ifdef HAVE_GETRUSAGE
#  ifndef HAVE_GETRLIMIT
#    include <sys/time.h>
#    include <sys/resource.h>
#  endif
#else
#  ifdef HAVE_TIMES
#    include <sys/times.h>
#    include <sys/param.h>
#  else
#    ifdef HAVE_FTIME
#      include <sys/timeb.h>
#    endif
#  endif
#endif

#ifdef HAVE_UNISTD_H
#include <sys/types.h>
#include <unistd.h>
#endif

#ifdef HAS_TIME_H
#include <time.h>
#endif

#ifdef HAS_WINGUI
#include "ngspice/wstdio.h"
#define HAS_PROGREP
extern void SetAnalyse(const char *Analyse, int Percent);
#endif

#if defined (__MINGW32__) || defined (__CYGWIN__) || defined (_MSC_VER)
#include <io.h>
#else
#  ifdef HAVE_SYS_IO_H
#    include <sys/io.h>
#  endif
#endif

#if defined (__MINGW32__)
#include <process.h> /* getpid() */
#endif

#if defined (_MSC_VER)
#include <direct.h>
#include <process.h>
/* C99 not available before VC++ 2013) */
#if (_MSC_VER < 1800)
#define trunc x_trunc
extern double x_trunc(double);
#define nearbyint x_nearbyint
extern double x_nearbyint(double);
#define asinh x_asinh
extern double x_asinh(double);
#define acosh x_acosh
extern double x_acosh(double);
#define atanh x_atanh
extern double x_atanh(double);
#define hypot _hypot
#endif
#define strdup _strdup
#define unlink _unlink
#define fileno _fileno
#define getcwd _getcwd
#define chdir _chdir
#if (_MSC_VER < 1800)
#define isnan _isnan
#endif
#define finite _finite
#define scalb _scalb
#define logb _logb
#define getpid _getpid
#define access _access
#define dup2 _dup2
#define open _open
#define write _write
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define isatty _isatty
#define inline __inline
#define popen _popen
#define pclose _pclose

// undo a #define bool _Bool in MS Visual Studio 2015
#if defined(bool)
#undef bool
#endif

// warning C4127: Bedingter Ausdruck ist konstant
#pragma warning(disable: 4127)
#endif

#if defined(__APPLE__) && defined(__MACH__)
#define finite isfinite
#endif

#if !defined(NAN)
#if defined(_MSC_VER)
    /* NAN is not defined in VS 2012 or older */
    static const __int64 global_nan = 0x7ff8000000000000i64;
    #define NAN (*(const double *) &global_nan)
#else
    #define NAN (0.0/0.0)
#endif
#endif

#ifndef EXT_ASC
#if defined(__MINGW32__) || defined(_MSC_VER)
#define fopen newfopen
extern FILE *newfopen(const char *fn, const char* md);
#endif
#endif

#if defined(__GNUC__)
#define ATTRIBUTE_NORETURN __attribute__ ((noreturn))
#elif defined(_MSC_VER)
#define ATTRIBUTE_NORETURN __declspec (noreturn)
#else
#define ATTRIBUTE_NORETURN
#endif

/* Fast random number generator */
//#define FastRand
#define WaGauss
#define RR_MAX RAND_MAX

#if !defined(HAVE_STRCHR) && defined(HAVE_INDEX)
#   define strchr index
#   define strrchr rindex
#endif

/* added for CYGWIN */
#ifndef HUGE
#define HUGE HUGE_VAL
#endif

void findtok_noparen(char **p_str, char **p_token, char **p_token_end);
extern char *gettok_noparens(char **s);
extern char *gettok_node(char **s);
extern char *gettok_iv(char **s);
extern char *nexttok(const char *s);
extern char *nexttok_noparens(const char *s);
extern char *gettok_model(char **s);
extern int get_l_paren(char **s);
extern int get_r_paren(char **s);

/* Some external variables */

extern char *Spice_Exec_Dir;
extern char *Spice_Lib_Dir;
extern char *Def_Editor;
extern char *Bug_Addr;
extern int AsciiRawFile;
extern char *Spice_Host;
extern char *Spiced_Log;

extern char Spice_Version[];
extern char Spice_Notice[];
extern char Spice_Build_Date[];
extern char Spice_Manual[];

extern char *News_File;
extern char *Spice_Path;
extern char *Help_Path;
extern char *Lib_Path;
extern char *Inp_Path;
extern char *Infile_Path;
extern char *Spice_Exec_Path;

#ifdef TCL_MODULE

#include <errno.h>

extern int tcl_printf(const char *format, ...);
extern int tcl_fprintf(FILE *f, const char *format, ...);

#undef printf
#define printf tcl_printf

#undef fprintf
#define fprintf tcl_fprintf

#undef perror
#define perror(string) fprintf(stderr, "%s: %s\n", string, strerror(errno))

#elif defined SHARED_MODULE

#include <errno.h>
#include <stdarg.h>

extern int sh_printf(const char *format, ...);
extern int sh_fprintf(FILE *fd, const char *format, ...);
extern int sh_vfprintf(FILE *fd, const char *format, va_list args);
extern int sh_fputs(const char *input, FILE *fd);
extern int sh_fputc(int input, FILE *fd);
extern int sh_putc(int input, FILE *fd);
extern void SetAnalyse(const char *analyse, int percent);

#define HAS_PROGREP

#undef vfprintf
#define vfprintf sh_vfprintf

#undef printf
#define printf sh_printf

#undef fprintf
#define fprintf sh_fprintf

#undef perror
#define perror(string) fprintf(stderr, "%s: %s\n", string, strerror(errno))

#undef fputs
#define fputs sh_fputs

#undef fputc
#define fputc sh_fputc

#undef putc
#define putc sh_putc

#endif


void soa_printf(CKTcircuit *ckt, GENinstance *instance, const char *fmt, ...);

ATTRIBUTE_NORETURN void controlled_exit(int status);


/* macro to ignore unused variables and parameters */
#define NG_IGNORE(x)  (void)x
#define NG_IGNOREABLE(x)  (void)x


#if !defined(va_copy) && defined(_MSC_VER)
#define va_copy(dst, src) ((dst) = (src))
#endif


/*
 * type safe variants of the <ctype.h> functions for char arguments
 */

#if !defined(isalpha_c)

inline static int char_to_int(char c) { return (unsigned char) c; }

#define isalpha_c(x) isalpha(char_to_int(x))
#define islower_c(x) islower(char_to_int(x))
#define isdigit_c(x) isdigit(char_to_int(x))
#define isalnum_c(x) isalnum(char_to_int(x))
#define isprint_c(x) isprint(char_to_int(x))
#define isblank_c(x) isblank(char_to_int(x))
#define isspace_c(x) isspace(char_to_int(x))
#define isupper_c(x) isupper(char_to_int(x))

#define tolower_c(x) ((char) tolower(char_to_int(x)))
#define toupper_c(x) ((char) toupper(char_to_int(x)))

#endif


#endif
