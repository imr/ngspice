/*
 * Tgif Output
 *
 * Christos Zoulas
 */

#include <stdio.h>
#include <X11/Xlib.h>
#include "hard_devices.h"
#include "xgout.h"

#define COLOR	"DarkSlateGray"

typedef struct {
    char   *title_font;
    char   *axis_font;
    int     title_size;
    int     axis_size;
    FILE   *strm;
}       Info;

char   *tgif_prologue[] =
{
    "state(0,13,0,0,0,16,1,5,1,1,0,0,1,0,1,0,1,0,4,0,0,0,10,0).\n",
    "%\n",
    "% Tgif xgraph output.\n",
    "%\n",
    0
};

/*
 * Hardcopy Interface for Xgraph
 *
 * Major differences from first version:
 *   Four new parameters are passed to the device initialization routine:
 *   title_family, title_size, axis_family, and axis_size.  See the
 *   description of xg_init() for details.
 *
 *   Clipping is done automatically by xgraph.  The xg_clip() routine
 *   is obsolete.
 *
 *   The xg_line() routine has become the xg_seg() routine.  It now
 *   draws segments rather than a series of lines.
 *
 *   A new field (max_segs) in the device structure now specifies
 *   the maximum number of segments the device can handle in a group.
 */




void    tgifText();
void    tgifDot();
void    tgifSeg();
void    tgifEnd();

int 
tgifInit(strm, width, height, title_family, title_size,
	  axis_family, axis_size, flags, out_info, errmsg)
FILE   *strm;			/* Output stream              */
int     width,
        height;			/* Size of space (microns)    */
char   *title_family;		/* Name of title font family  */
double  title_size;		/* Title font height (points) */
char   *axis_family;		/* Name of axis font family   */
double  axis_size;		/* Axis font height (points)  */
int     flags;			/* Flags                      */
xgOut  *out_info;		/* Device info (RETURN)       */
char    errmsg[ERRBUFSIZE];	/* Error message area         */

{
    Info   *tgif_info;
    char  **l;
    double  scx,
            scy;

    tgif_info = (Info *) Malloc(sizeof(*tgif_info));

    for (l = tgif_prologue; *l; l++)
	fprintf(strm, "%s\n", *l);

    out_info->dev_flags = 0;
    scx = width /  512.0;
    scy = height / 512.0;
    if (scx > scy) {
	scy /= scx;
	scx = 1;
    }
    else {
	scx /= scy;
	scy = 1;
    }
    out_info->bdr_pad = title_size / 4;
    out_info->axis_pad = 2.0 * axis_size;
    out_info->legend_pad = 0;

    out_info->area_w = width * 0.00283 * scx;	/* pts per micron */
    out_info->area_h = height * 0.00283 * scy;

    out_info->tick_len = axis_size;
    out_info->axis_height = axis_size;
    out_info->title_height = title_size;
    out_info->axis_width = (axis_size * 5.0) / 12.0;
    out_info->title_width = (title_size * 5.0) / 12.0;
    out_info->max_segs = 100;
    out_info->xg_text = tgifText;
    out_info->xg_seg = tgifSeg;
    out_info->xg_dot = tgifDot;
    out_info->xg_end = tgifEnd;
    out_info->user_state = (char *) tgif_info;

    tgif_info->title_font = title_family;
    tgif_info->axis_font = axis_family;
    tgif_info->title_size = title_size;
    tgif_info->axis_size = axis_size;
    tgif_info->strm = strm;
    return 1;
}

/* Text justifications */
#define T_CENTER	0
#define T_LEFT		1
#define T_UPPERLEFT	2
#define T_TOP		3
#define T_UPPERRIGHT	4
#define T_RIGHT		5
#define T_LOWERRIGHT	6
#define T_BOTTOM	7
#define T_LOWERLEFT	8

/* Text styles */
#define T_AXIS		0
#define T_TITLE		1

static void 
tgif_just(x, y, just, size, len)
int    *x,
       *y;			/* Given location (lower left) */
int     just;			/* Justification */
int     size;			/* Size in points */
int     len;			/* Number of chars */

/*
 * Unfortunately, tgif really can't display text with a justification.
 * This is a horrible hack to try to get around the problem.  It tries
 * to compute a rough bounding box for the text based on the text height
 * and the string length and offset `x,y' appropriately for the justification.
 * This is only a hack...
 */
{
    int     t_width,
            t_height;

    t_height = size;
    t_width = (size * len * 5) / 12;	/* Horrible estimate */

    switch (just) {
    case T_CENTER:
	*x -= t_width / 2;
	*y -= t_height / 2;
	break;
    case T_LEFT:
	*y -= t_height / 2;
	break;
    case T_UPPERLEFT:
	/* nothing */
	break;
    case T_TOP:
	*x -= t_width / 2;
	break;
    case T_UPPERRIGHT:
	*x -= t_width;
	break;
    case T_RIGHT:
	*x -= t_width;
	*y -= t_height / 2;
	break;
    case T_LOWERRIGHT:
	*x -= t_width;
	*y -= t_height;
	break;
    case T_BOTTOM:
	*x -= t_width / 2;
	*y -= t_height;
	break;
    case T_LOWERLEFT:
	*y -= t_height;
	break;
    }

    /*
     * Also, tgif seems to put a space above all text it draws. The
     * computation below compensates for this.
     */
    *y += (size / 3);
}

void 
tgifText(user_state, x, y, text, just, style)
char   *user_state;		/* Value set in xg_init   */
int     x,
        y;			/* Text position (pixels) */
char   *text;			/* Null terminated text   */
int     just;			/* Justification (above)  */
int     style;			/* Text style (above)     */

/*
 * This routine should draw text at the indicated position using
 * the indicated justification and style.  The justification refers
 * to the location of the point in reference to the text.  For example,
 * if just is T_LOWERLEFT,  (x,y) should be located at the lower left
 * edge of the text string.
 */
{
    char   *font;
    int     size;
    Info   *tgif = (Info *) user_state;

    /*
     * Obj = text(_Color,_X,_Y,_Font,_TextStyle,_TextSize,_NumLines,_TextJust,
     *		  _TextRotate,_PenPat,_BBoxW,_BBoxH,_Id,_TextDPI,_Asc,_Des,
     *		  _ObjFill,_Vspace,StrList),
     */
    /* font ok too */
    style == T_AXIS ? tgif->axis_font :
	tgif->title_font;
    /* ok 0, 1 as in tgif */
    size = style == T_AXIS ? tgif->axis_size :
	tgif->title_size;
    tgif_just(&x, &y, just, size, strlen(text));

    if (size <= 8)
	size = 0;
    else if (size <= 10)
	size = 1;
    else if (size <= 12)
	size = 2;
    else if (size <= 14)
	size = 3;
    else if (size <= 18)
	size = 4;
    else
	size = 5;

    fprintf(tgif->strm, 
	"text('%s',%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,[\n\t",
	    COLOR, x, y, 0, style, size, 1, 0, 0, 1, 0, 0, 0, 0, 18, 4, 0, 0);
    fprintf(tgif->strm,
	    "\"%s\"]).\n", text);
}

/* Line Styles */
#define L_AXIS		0
#define L_ZERO		1
#define L_VAR		2

void 
tgifSeg(user_state, ns, seglist, width, style, lappr, color)
char   *user_state;		/* Value set in xg_init */
int     ns;			/* Number of segments   */
XSegment *seglist;		/* X array of segments  */
int     width;			/* Width of lines       */
int     style;			/* See above            */
int     lappr;			/* Line appearence      */
int     color;			/* Line color (if any)  */
{
    Info   *tgif = (Info *) user_state;
    int     i,
            j,
            k;

    /*
     * poly(_Color,_NumVs,_Vs,_LineStyle,_LineWidth,_PenPat,_Id,_Spline,
     *		_ObjFill,_Dash,AttrList),
     */
    static int style_list[] =
    {
	1, 10, 7, 6, 5, 4, 3, 2
    };

    for (i = 0; i < ns; i++) {
	fprintf(tgif->strm, "poly('%s',2,[%d,%d,%d,%d],", COLOR, 
		seglist[i].x1, seglist[i].y1,
		seglist[i].x2, seglist[i].y2);
	fprintf(tgif->strm, "%d,%d,%d,%d,%d,%d,%d,[\n]).\n", 0, width, 
		style_list[lappr], 0, 0, style_list[lappr], 0);
    }
}

/* Marker styles */
#define P_PIXEL		0
#define P_DOT		1
#define P_MARK		2

void 
tgifDot(user_state, x, y, style, type, color)
char   *user_state;		/* Value set in xg_init    */
int     x,
        y;			/* Location in pixel units */
int     style;			/* Dot style               */
int     type;			/* Type of marker          */
int     color;			/* Marker color (if any)   */

/*
 * This routine should draw a marker at location `x,y'.  If the
 * style is P_PIXEL,  the dot should be a single pixel.  If
 * the style is P_DOT,  the dot should be a reasonably large
 * dot.  If the style is P_MARK,  it should be a distinguished
 * mark which is specified by `type' (0-7).  If the output
 * device is capable of color,  the marker should be drawn in
 * `color' (0-7) which corresponds with the color for xg_line.
 */
{
}

void 
tgifEnd(user_state)
char   *user_state;

/*
 * This routine is called after a drawing sequence is complete.
 * It can be used to clean up the user state and set the device
 * state appropriately.  This routine is optional in the structure.
 */
{
    Info   *tgif = (Info *) user_state;

    fclose(tgif->strm);
}
