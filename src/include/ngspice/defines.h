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


#ifndef	M_PI
#  define M_PI       3.14159265358979323846
#endif
#ifndef	M_E
#  define M_E  	     2.7182818284590452354
#endif
#ifndef	M_LOG2E
#  define M_LOG2E    1.4426950408889634074
#endif
#ifndef	M_LOG10E
#  define M_LOG10E   0.43429448190325182765
#endif


/*
 *  IEEE Floating point
 */

#define MAX_EXP_ARG	709.0

#ifndef DBL_EPSILON
# define DBL_EPSILON	2.2204460492503131e-16
#endif
#ifndef DBL_MAX
# define DBL_MAX	1.79769313486231e+308
#endif
#ifndef DBL_MIN
# define DBL_MIN	2.22507385850721e-308
#endif
#ifndef SHRT_MAX
# define SHRT_MAX	32766
#endif
#ifndef INT_MAX
# define INT_MAX	2147483646
#endif
#ifndef LONG_MAX
# define LONG_MAX	2147483646
#endif

#define MAXPOSINT	INT_MAX

/* 
 * Physical constants (const.h)
 */

#define CHARGE (1.6021918e-19)
#define CONSTCtoK (273.15)
#define CONSTboltz (1.3806226e-23)
#define CONSTepsZero (8.854214871e-12)   /* epsilon zero F/m */
#define CONSTepsSiO2 (3.4531479969e-11)  /* epsilon SiO2 F/m */
#define CONSTmuZero  (1.25663706143592e-6)         /* MuZero H/m       */

#define REFTEMP 300.15 /* 27 degrees C */

/* Standard initialisation file name */
#define INITSTR		".spiceinit"

/* Alternate initialisation file name */
#define ALT_INITSTR	"spice.rc"	

#if defined(__MINGW32__) || defined(_MSC_VER) || defined (HAS_WINGUI)
#define DIR_PATHSEP	"\\"
#define DIR_TERM	'\\'
#define DIR_PATHSEP_LINUX	"/"
#define DIR_TERM_LINUX	'/'
#define DIR_CWD		"."

#define TEMPFORMAT	"%s%d.tmp"
/*
#define SYSTEM_PLOT5LPR	"lpr -P%s -g %s"
#define SYSTEM_PSLPR	"lpr -P%s %s"
#define SYSTEM_MAIL	"Mail -s \"%s (%s) Bug Report\" %s"
*/
#else

#define DIR_PATHSEP	"/"
#define DIR_TERM	'/'
#define DIR_CWD		"."

#define TEMPFORMAT	"/tmp/%s%d"
#define SYSTEM_PLOT5LPR	"lpr -P%s -g %s"
#define SYSTEM_PSLPR	"lpr -P%s %s"
#define SYSTEM_MAIL	"Mail -s \"%s (%s) Bug Report\" %s"

#endif

/*
 *  #define-s that are always on
 */

/* On Unix the following should always be true, so they should jump out */

#define HAS_ASCII
#define HAS_TTY_
#define HAS_TIME_H
#define HAS_RLIMIT_



#ifndef SIGNAL_FUNCTION
# ifdef HAVE_SIGHANDLER_T
#  define SIGNAL_FUNCTION sighandler_t
# elif HAVE_SIG_T
#  define SIGNAL_FUNCTION sig_t
# elif HAVE___SIGHANDLER_T
#  define SIGNAL_FUNCTION __sighandler_t
# else
#  define SIGNAL_FUNCTION RETSIGTYPE (*)(int)
# endif
#endif

#define BSIZE_SP      512


#define EXIT_NORMAL 0
#define EXIT_BAD    1
#define EXIT_INFO   2

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

