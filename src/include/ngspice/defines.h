/*************
 * Definitions header file
 * 1999 E. Rouat
 ************/

/* 
 * This file will contain all #defines needed
 * by ngspice code (in construction)
 * It should only #define numeric constants, not macros.
 */


#ifndef ngspice_DEFINES_H
#define ngspice_DEFINES_H

/* Floating point and integral limits */
#include <float.h>
#include <limits.h>

/* 
 * Physical constants (const.h)
 */
 /* For definitions of CHARGE, CONSTCtoK CONSTboltz, CONSTepsZero,
  * CONSTepsSi02, CONSTmuZero, REFTEMP */
#include "ngspice/const.h"

/* These constants are defined by GCC in math.h, but they are not part of
 * the ANSI C standard. The #ifndefs will prevent warnings regarding macro
 * redefinitions if this file is included AFTER math.h. However, if the
 * order is reversed, the warnings will occur. Thus, they introduce a header
 * order dependency. A better solution would be to rename the macros to
 * something like NGM_* (ngspice math) throughout the source code. Then there
 * would be no header ordering dependency. */
#ifndef M_PI
#define M_PI       CONSTpi
#endif
#ifndef M_E
#define M_E        CONSTnap
#endif
#ifndef M_LOG2E
#define M_LOG2E    CONSTlog2e
#endif
#ifndef M_LOG10E
#define M_LOG10E   CONSTlog10e
#endif

/*
 *  IEEE Floating point
 */

/* Largest exponent such that exp(MAX_EXP_ARG) <= DBL_MAX
 * Actual max is ln(DBL_MAX) = 709.78. Unsure if there was any reason
 * for setting the value lower */
#define MAX_EXP_ARG 709.0



/* Standard initialisation file name */
#define INITSTR            ".spiceinit"

/* Alternate initialisation file name */
#define ALT_INITSTR         "spice.rc"

#if defined(__MINGW32__) || defined(_MSC_VER) || defined (HAS_WINGUI)
#define DIR_PATHSEP         "\\"
#define DIR_TERM            '\\'
#define DIR_PATHSEP_LINUX   "/"
#define DIR_TERM_LINUX      '/'
#define DIR_CWD             "."

#define TEMPFORMAT          "%s%d.tmp"
#define TEMPFORMAT2         "%s%d_%d.tmp"

/*
#define SYSTEM_PLOT5LPR     "lpr -P%s -g %s"
#define SYSTEM_PSLPR        "lpr -P%s %s"
#define SYSTEM_MAIL         "Mail -s \"%s (%s) Bug Report\" %s"
*/
#else

#define DIR_PATHSEP         "/"
#define DIR_TERM            '/'
#define DIR_CWD             "."

#define TEMPFORMAT          "/tmp/%s%d"
#define TEMPFORMAT2         "/tmp/%s%d_%d"
#define SYSTEM_PLOT5LPR     "lpr -P%s -g %s"
#define SYSTEM_PSLPR        "lpr -P%s %s"
#define SYSTEM_MAIL         "Mail -s \"%s (%s) Bug Report\" %s"

#endif

/*
 *  #define-s that are always on
 */

/* On Unix the following should always be true, so they should jump out */

#define HAS_ASCII
#define HAS_TTY_
#define HAS_TIME_H
#define HAS_RLIMIT_

#define void void

#ifndef SIGNAL_FUNCTION
# ifdef HAVE_SIGHANDLER_T
#  define SIGNAL_FUNCTION sighandler_t
# elif HAVE_SIG_T
#  define SIGNAL_FUNCTION sig_t
# elif HAVE___SIGHANDLER_T
#  define SIGNAL_FUNCTION __sighandler_t
# else
#  define SIGNAL_FUNCTION void (*)(int)
# endif
#endif

#define BSIZE_SP      512
#define LBSIZE_SP    4096


#define EXIT_NORMAL 0
#define EXIT_BAD    1
#define EXIT_INFO   2
#define EXIT_SEGV   3

#define TRUE 1
#define FALSE 0

/*

#define DIR_PATHSEP	"/"
#define DIR_TERM	'/'
#define DIR_CWD		"."

#define TEMPFORMAT	"/tmp/%s%d"
#define SYSTEM_PLOT5LPR	"lpr -P%s -g %s"
#define SYSTEM_PSLPR	"lpr -P%s %s"
#define SYSTEM_MAIL	"Mail -s \"%s (%s) Bug Report\" %s"

*/



#endif

