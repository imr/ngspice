/*
 * xtb - a mini-toolbox for X11
 *
 * David Harrison
 * University of California, Berkeley
 * 1988
 */

#ifndef _XTB_
#define _XTB_

#include "copyright.h"

/* Handler function return codes */
typedef enum xtb_hret_defn {
    XTB_NOTDEF, XTB_HANDLED, XTB_STOP
}       xtb_hret;

/* If you have an ANSI compiler,  some checking will be done */
#ifdef __STDC__
#define DECLARE(func, rtn, args)	extern rtn func args
typedef void *xtb_data;

#else
#define DECLARE(func, rtn, args)	extern rtn func ()
typedef char *xtb_data;

#endif

/* Basic return value */
typedef struct xtb_frame_defn {
    Window  win;
    int     x_loc,
            y_loc;
    unsigned int width,
            height;
}       xtb_frame;

DECLARE(xtb_init, void, (Display * disp, int scrn,
			 unsigned long foreground,
			 unsigned long background,
			 XFontStruct * font));
 /* Initializes mini-toolbox */

/*
 * Basic event handling
 */

DECLARE(xtb_register, void, (Window win,
			     xtb_hret(*func) (XEvent * evt, xtb_data info),
			     xtb_data info));
 /* Registers call-back function */

DECLARE(xtb_lookup, xtb_data, (Window win));
 /* Returns data associated with window */

DECLARE(xtb_dispatch, xtb_hret, (XEvent * evt));
 /* Dispatches events for mini-toolbox */

DECLARE(xtb_unregister, int, (Window win, xtb_data * info));
 /* Unregisters a call-back function */

/*
 * Command button frame
 */

DECLARE(xtb_bt_new, void, (Window win, char *text,
			   xtb_hret(*func) (Window win, int state,
					    xtb_data val),
			   xtb_data val,
			   xtb_frame * frame));
 /* Creates new button  */

DECLARE(xtb_bt_get, int, (Window win, xtb_data * stuff, int *na));
 /* Returns state of button */
DECLARE(xtb_bt_set, int, (Window win, int val, xtb_data stuff, int na));
 /* Sets state of button */
DECLARE(xtb_bt_del, void, (Window win, xtb_data * info));
 /* Deletes a button */

/*
 * Button row frame - built on top of buttons
 */

DECLARE(xtb_br_new, void, (Window win, int cnt, char *lbls[], int init,
			   xtb_hret(*func) (Window win, int prev,
					    int this, xtb_data val),
			   xtb_data val,
			   xtb_frame * frame));
 /* Creates a new button row frame */

DECLARE(xtb_br_get, int, (Window win));
 /* Returns currently selected button */
DECLARE(xtb_br_del, void, (Window win));
 /* Deletes a button row */

/*
 * Text output (label) frames
 */

DECLARE(xtb_to_new, void, (Window win, char *text,
			   XFontStruct * ft, xtb_frame * frame));
 /* Create new text output frame */
DECLARE(xtb_to_del, void, (Window win));

/*
 * Text input (editable text) frames
 */

#define MAXCHBUF	1024

DECLARE(xtb_ti_new, void, (Window win, char *text, int maxchar,
			   xtb_hret(*func) (Window win, int ch,
					    char *textcopy, xtb_data * val),
			   xtb_data val, xtb_frame * frame));
 /* Creates a new text input frame */

DECLARE(xtb_ti_get, void, (Window win, char text[MAXCHBUF], xtb_data * val));
 /* Returns state of text input frame */
DECLARE(xtb_ti_set, int, (Window win, char *text, xtb_data val));
 /* Sets the state of text input frame */
DECLARE(xtb_ti_ins, int, (Window win, int ch));
 /* Inserts character onto end of text input frame */
DECLARE(xtb_ti_dch, int, (Window win));
 /* Deletes character from end of text input frame */
DECLARE(xtb_ti_del, void, (Window win, xtb_data * info));
 /* Deletes an text input frame */

/*
 * Block frame
 */

DECLARE(xtb_bk_new, void, (Window win, unsigned width, unsigned height,
			   xtb_frame * frame));
 /* Makes a new block frame */
DECLARE(xtb_bk_del, void, (Window win));
 /* Deletes a block frame */


/*
 * Formatting support
 */

#define MAX_BRANCH	50

typedef enum xtb_fmt_types_defn {
    W_TYPE, A_TYPE
}       xtb_fmt_types;
typedef enum xtb_fmt_dir_defn {
    HORIZONTAL, VERTICAL
}       xtb_fmt_dir;
typedef enum xtb_just_defn {
    XTB_CENTER = 0, XTB_LEFT, XTB_RIGHT, XTB_TOP, XTB_BOTTOM
}       xtb_just;

typedef struct xtb_fmt_widget_defn {
    xtb_fmt_types type;		/* W_TYPE */
    xtb_frame *w;
}       xtb_fmt_widget;

typedef struct xtb_fmt_align_defn {
    xtb_fmt_types type;		/* A_TYPE */
    xtb_fmt_dir dir;		/* HORIZONTAL or VERTICAL */
    int     padding;		/* Outside padding        */
    int     interspace;		/* Internal padding       */
    xtb_just just;		/* Justification          */
    int     ni;			/* Number of items */
    union xtb_fmt_defn *items[MAX_BRANCH];	/* Branches themselves */
}       xtb_fmt_align;

typedef union xtb_fmt_defn {
    xtb_fmt_types type;		/* W_TYPE or A_TYPE */
    xtb_fmt_widget wid;
    xtb_fmt_align align;
}       xtb_fmt;

#define NE	0

DECLARE(xtb_w, xtb_fmt *, (xtb_frame * w));
 /* Returns formatting structure for frame */
DECLARE(xtb_hort, xtb_fmt *, (xtb_just just, int padding, int interspace,...));
 /* Varargs routine for horizontal formatting */
DECLARE(xtb_vert, xtb_fmt *, (xtb_just just, int padding, int interspace,...));
 /* Varargs routine for vertical formatting */
DECLARE(xtb_fmt_do, xtb_fmt *, (xtb_fmt * def, unsigned *w, unsigned *h));
 /* Carries out formatting */
DECLARE(xtb_mv_frames, void, (int nf, xtb_frame frames[]));
 /* Actually moves widgets */
DECLARE(xtb_fmt_free, void, (xtb_fmt * def));
 /* Frees resources claimed by xtb_w, xtb_hort, and xtb_vert */

#endif				/* _XTB_ */
