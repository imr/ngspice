/*
 * Globally accessible information from xgraph
 */

#ifndef _XGRAPH_H_
#define _XGRAPH_H_

#include "autoconf.h"

#include <X11/Xos.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>

/*
 * Get definitions from headers.
 */
#include <stdio.h>   /* sprintf */

#ifdef HAVE_STRING_H
#include <string.h>  /* str* */
#else
#ifdef HAVE_STRINGS_H
#include <strings.h>  /* str* */
#else
extern char *strcpy();
extern char *strcat();
extern char *rindex();
extern char *index();
#endif /* HAVE_STRINGS_H */
#endif /* HAVE_STRING_H */

#ifdef HAVE_UNISTD_H
#include <unistd.h>  /* exit, abort */
#endif /* HAVE_UNISTD_H */
#ifdef HAVE_STDLIB_H
#include <stdlib.h>  /* atof */
#endif /* HAVE_STDLIB_H */

#include "xgout.h"

#define VERSION_STRING	"12.1 December 1999 "

#define MAXKEYS		50
#define MAXATTR 	8
#define MAXSETS		112
#define MAXBUFSIZE 	120
#define MAXLS		50

#define STRDUP(xx)	(strcpy(Malloc((unsigned) (strlen(xx)+1)), (xx)))
#define SCREENX(ws, userX) \
    (((int) (((userX) - ws->UsrOrgX)/ws->XUnitsPerPixel + 0.5)) + ws->XOrgX)
#define SCREENY(ws, userY) \
    (ws->XOppY - ((int) (((userY) - ws->UsrOrgY)/ws->YUnitsPerPixel + 0.5)))
#define HARDCOPY_IN_PROGRESS	0x01

/* Portability */
/* try to get those constants */
#include <math.h>
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif /* HAVE_LIMITS */
#ifdef HAVE_FLOAT_H
#include <float.h>
#endif /* HAVE_FLOAT_H */

#ifdef  CRAY
#undef  MAXFLOAT
#define MAXFLOAT 10.e300
#endif				/* CRAY */

#ifndef MAXFLOAT
#if defined(FLT_MAX)
#define MAXFLOAT	FLT_MAX
#elif defined(HUGE)
#define MAXFLOAT	HUGE
#endif
#endif

#ifndef BIGINT
#if defined(INT_MAX)
#define BIGINT		INT_MAX
#elif defined(MAXINT)
#define BIGINT MAXINT
#else
#define BIGINT 0xffffffff
#endif
#endif

#define GRIDPOWER 	10
#define INITSIZE 	128

#define CONTROL_D	'\004'
#define CONTROL_C	'\003'
#define TILDE		'~'

#define BTNPAD		1
#define BTNINTER	3

#ifndef MAX
#define MAX(a,b)	((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b)	((a) < (b) ? (a) : (b))
#endif
#ifndef ABS
#define ABS(x)		((x) < 0 ? -(x) : (x))
#endif
#define ZERO_THRES	1.0E-07

/* To get around an inaccurate log */
#define nlog10(x)	(x == 0.0 ? 0.0 : log10(x) + 1e-15)

#define ISCOLOR		(wi->dev_info.dev_flags & D_COLOR)

#define PIXVALUE(set) 	((set) % MAXATTR)

#define LINESTYLE(set) \
(ISCOLOR ?  ((set)/MAXATTR) : ((set) % MAXATTR))

#define MARKSTYLE(set) \
(colorMark ? COLMARK(set) : BWMARK(set))

#define COLMARK(set) \
((set) / MAXATTR)

#define BWMARK(set) \
((set) % MAXATTR)

#define LOG_X	0x01
#define LOG_Y	0x02
#define STK	0x04
#define FITX	0x08
#define FITY	0x10
typedef unsigned long Pixel;

/* Globally accessible values */
extern Display *disp;		/* Open display            */
extern Visual *vis;		/* Standard visual         */
extern Colormap cmap;		/* Standard colormap       */
extern int screen;		/* Screen number           */
extern int depth;		/* Depth of screen         */

extern void do_hardcopy();	/* Carries out hardcopy    */
extern void ho_dialog();	/* Hardcopy dialog         */
extern void set_X();		/* Initializes X device    */

typedef struct point_list {
    int     numPoints;		/* Number of points in group */
    int     allocSize;		/* Allocated size            */
    double *xvec;		/* X values                  */
    double *yvec;		/* Y values                  */
    struct point_list *next;	/* Next set of points        */
}       PointList;

typedef struct new_data_set {
    char   *setName;		/* Name of data set     */
    PointList *list;		/* List of point arrays */
}       NewDataSet;

typedef struct local_win {
    double  loX,
            loY,
            hiX,
            hiY;		/* Local bounding box of window         */
    int     XOrgX,
            XOrgY;		/* Origin of bounding box on screen     */
    int     XOppX,
            XOppY;		/* Other point defining bounding box    */
    double  UsrOrgX,
            UsrOrgY;		/* Origin of bounding box in user space */
    double  UsrOppX,
            UsrOppY;		/* Other point of bounding box          */
    double  XUnitsPerPixel;	/* X Axis scale factor                  */
    double  YUnitsPerPixel;	/* Y Axis scale factor                  */
    struct xg_out   dev_info;	/* Device information                   */
    Window  close,
            hardcopy;		/* Buttons for closing and hardcopy     */
    Window  about;		/* Version information                  */
    Window  rewind;             /* PW Added this, for animation.        */
    Window  replot;             /* PW Added this, for animation.        */
    Window  deriv;             /* PW Added this, for animation.        */
    int     flags;		/* Window flags                         */
    int     DOrder;             /* Which order of Derivative is being set? */
}       LocalWin;

extern NewDataSet PlotData[MAXSETS], DataD1[MAXSETS], DataD2[MAXSETS];
extern  XSegment *Xsegs[2];		/* Point space for X */
extern double llx, lly, urx, ury;	/* Bounding box of all data */
extern int numFiles;			/* Number of input files   */
extern char *inFileNames[MAXSETS];	/* File names              */

/* Total number of active windows */
extern int Num_Windows;
extern char *Prog_Name;
extern char *disp_name;

/* To make lint happy */
extern char *Malloc();
extern char *Realloc();
extern void Free();

#ifndef _POSIX_SOURCE
/* extern int sprintf(); ---conflicts with sunos */
extern void exit();
extern double atof();
extern void abort();
#endif /* _POSIX_SOURCE */

#endif				/* _XGRAPH_H_ */
