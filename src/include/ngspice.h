/*************
 * Main header file for ngspice
 * 1999 E. Rouat
 ************/

/* 
 * This file will eventually replace spice.h and lots of other 
 * files in src/include
 */
#define _GNU_SOURCE

#include <config.h>
#include <stddef.h>

#ifdef HAVE_LIMITS_H
#  include <limits.h>
#endif

#include "memory.h"
#include "defines.h"
#include "macros.h"

#include <math.h>
#include <stdio.h>

#ifndef HUGE
#define HUGE HUGE_VAL
#endif

#ifdef STDC_HEADERS
#  include <stdlib.h>
#  include <string.h>
#else
#  include <strings.h>
#endif

#ifdef HAVE_CTYPE_H
#  include <ctype.h>
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

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif


#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

#ifdef HAVE_SYS_DIR_H
#include <sys/types.h>
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

#ifdef HAVE_INDEX
#   define strchr index
#   define strrchr rindex
#endif

#ifdef HAS_TIME_
#include <time.h>
#endif

#ifdef __MINGW32__
#define srandom srand
#define random rand
#define index strchr
#endif

extern char *gettok(char **s);
extern void appendc(char *s, char c);
extern int scannum(char *str);
extern int ciprefix(register char *p, register char *s);
extern int cieq(register char *p, register char *s);
extern void strtolower(char *str);
extern char *tilde_expand(char *string);

extern char *smktemp(char *id);

extern char *copy(char *str);
extern int prefix(char *p, char *str);
extern int substring(char *sub, char *str);
extern void cp_printword(char *string, FILE *fp);

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

extern char Spice_Notice[ ];
extern char Spice_Version[ ];
extern char Spice_Build_Date[ ];

extern char *News_File;
extern char *Default_MFB_Cap;
extern char *Spice_Path;
extern char *Help_Path;
extern char *Lib_Path;

extern int ARCHme;	/* My logical process number */
extern int ARCHsize;	/* Total number of processes */

#ifdef TCL_MODULE

extern int tcl_printf(const char *format, ...);
extern int tcl_fprintf(FILE *f, const char *format, ...);

#undef printf
#define printf tcl_printf

#undef fprintf
#define fprintf tcl_fprintf

#endif
