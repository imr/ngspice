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

#include <config.h>
#include <stddef.h>

#ifdef HAVE_LIMITS_H
#  include <limits.h>
#endif

#include "memory.h"
#include "defines.h"
#include "macros.h"
#include "bool.h"
#include "complex.h"

#include <math.h>
#include <stdio.h>


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
#else /* va: no index, but strchr */
#    ifdef HAVE_STRCHR
#        define index  strchr
#        define rindex strrchr
#    endif /* va: no index, but strchr */
#endif

#ifdef HAS_TIME_H
#include <time.h>
#endif

/* added for CYGWIN */
#ifndef HUGE
#define HUGE HUGE_VAL
#endif

#ifdef HAS_WINDOWS
#include "wstdio.h"
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
extern char *tildexpand(char *string);

extern char *canonicalize_pathname(char *path);
extern char *absolute_pathname(char *string, char *dot_path);

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
