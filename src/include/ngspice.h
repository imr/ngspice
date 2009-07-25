/*************
 * Main header file for ngspice
 * 1999 E. Rouat
 ************/
#ifndef NGSPICE_H_INCLUDED /* va */
#define NGSPICE_H_INCLUDED

/* #include "memwatch.h"
 #define MEMWATCH */


/* 
 * This file will eventually replace spice.h and lots of other 
 * files in src/include
 */
#define _GNU_SOURCE

#include "config.h"
#include <stddef.h>

#ifdef HAVE_LIMITS_H
#  include <limits.h>
#endif
#ifdef HAVE_FLOAT_H
#  include <float.h>
#endif

#include "memory.h"
#include "defines.h"
#include "macros.h"
#include "bool.h"
#include "complex.h"

#include <math.h>
#include <stdio.h>

#ifdef HAVE_IEEEFP_H
#include <ieeefp.h>
#endif

#include "missing_math.h"

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
struct timeb timebegin;
#    endif
#  endif
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAS_TIME_H
#include <time.h>
#endif

#ifdef HAS_WINDOWS
#include "wstdio.h"
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
#define strdup _strdup
#define unlink _unlink
#define fileno _fileno
#define getcwd _getcwd
#define chdir _chdir
#define isnan _isnan
#define finite _finite
#define scalb _scalb
#define logb _logb
#define getpid _getpid
#define access _access
#define dup2 _dup2
#define open _open
#define write _write
#endif

#ifndef HAVE_RANDOM
#define srandom(a) srand(a)
#define random rand
#define RR_MAX RAND_MAX
#else
#define RR_MAX LONG_MAX
#endif

#ifdef HAVE_INDEX
#   define strchr index
#   define strrchr rindex
#else /* va: no index, but strchr */
#    ifdef HAVE_STRCHR
#        define index  strchr
#        define rindex strrchr
#    endif /* va: no index, but strchr */
#endif

/* added for CYGWIN */
#ifndef HUGE
#define HUGE HUGE_VAL
#endif

extern char *gettok(char **s);
extern char *gettok_noparens(char **s);
extern char *gettok_node(char **s);
extern int get_l_paren(char **s);
extern int get_r_paren(char **s);
extern void appendc(char *s, char c);
extern int scannum(char *str);
extern int ciprefix(register char *p, register char *s);
extern int cieq(register char *p, register char *s);
extern void strtolower(char *str);
extern char *tildexpand(char *str);

extern char *canonicalize_pathname(char *path);
extern char *absolute_pathname(char *str, char *dot_path);

extern char *smktemp(char *id);

extern char *copy(char *str);
extern int prefix(char *p, char *str);
extern int substring(char *sub, char *str);
extern void cp_printword(char *str, FILE *fp);

extern char *datestring(void);
extern double seconds(void);

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

extern char *News_File;
extern char *Default_MFB_Cap;
extern char *Spice_Path;
extern char *Help_Path;
extern char *Lib_Path;

extern int ARCHme;	/* My logical process number */
extern int ARCHsize;	/* Total number of processes */

#ifdef TCL_MODULE

#include <errno.h>

extern int tcl_printf(const char *format, ...);
extern int tcl_fprintf(FILE *f, const char *format, ...);

#undef printf
#define printf tcl_printf

#undef perror
#define perror(string) fprintf(stderr,"%s: %s\n",string,sys_errlist[errno])

#endif

#ifdef CIDER
/* Definitions of globals for Machine Accuracy Limits 
 * Imported from cider
*/

extern double BMin;          /* lower limit for B(x) */
extern double BMax;          /* upper limit for B(x) */
extern double ExpLim;        /* limit for exponential */
extern double Accuracy;      /* accuracy of the machine */
extern double Acc, MuLim, MutLim;
#endif /* CIDER */ 


#endif /* NGSPICE_H_INCLUDED */
