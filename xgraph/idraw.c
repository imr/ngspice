/*
 * Idraw Output
 *
 * Beorn Johnson
 * Alan Kramer
 * David Harrison
 */

#include <stdio.h>
#include <X11/Xlib.h>
#include "hard_devices.h"
#include "xgout.h"

#define HEIGHT	792
#define FIX(X)	X = HEIGHT - X;

typedef struct {
    char   *title_font;
    char   *axis_font;
    int     title_size;
    int     axis_size;
    FILE   *strm;
}       Info;

char   *idraw_prologue[] =
{
    "%I Idraw 4",
    "Begin",
    "%I b u",
    "%I cfg u",
    "%I cbg u",
    "%I f u",
    "%I p u",
    "%I t",
    "[ 1 0 0 1 0 0 ] concat",
    "/originalCTM matrix currentmatrix def",
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


/*
 * Adding an output device to xgraph
 *
 * Step 1
 *   Write versions of the following routines for your device:
 *   xg_init(), xg_text(), xg_seg(), xg_dot(), and xg_end().
 *   The interface and function of these routines are described
 *   in detail below.  These routines should be named according
 *   to your device.  For example,  the initialization routine
 *   for the Postscript output device is psInit().  Also,  name
 *   your source file after your device (e.g. the postscript
 *   routines are in the file ps.c).  Instructions continue
 *   after the description of the interface routines.
 */


void    idrawText();
void    idrawDot();
void    idrawSeg();
void    idrawEnd();

int 
idrawInit(strm, width, height, title_family, title_size,
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

/*
 * This routine is called by xgraph just before drawing is to
 * begin.  The desired size of the plot is given by `width'
 * and `height'.  The parameters `title_family', `title_size',
 * `axis_family', and `axis_size' specify the names of the
 * title and axis fonts and their vertical sizes (in points).
 * These parameters can be ignored if your device does not
 * support multiple fonts.  Binary flags are specified in
 * the `flags' field.  These include:
 *  D_DOCU:
 *      If this flag is set,  it indicates the user has specified that
 *	the output will be included in some larger document.  Devices
 *	may choose to use this information to produce output that
 *	can be integrated into documents with less effort.  For example,
 *	the Postscript output routines produce bounding box information
 *	when this flag is set.
 * The routine should fill in all of the fields of `out_info' with
 * appropriate values.  The values are described below:
 *  area_w, area_h:
 * 	Size of the drawing space in device coordinates.
 *	This should take in account the requested area
 *	given by `width', and `height'.
 *  bdr_pad:
 * 	Xgraph will leave this number of device coordinates around
 *	all of the outer edges of the graph.
 *  axis_pad:
 *	Additional space around axis labels (in devcoords)
 *	so that the labels do not appear crowded.
 *  legend_pad:
 *	Space (in devcoords) from the top of legend text to
 *	the representative line drawn above the legend text.
 *  tick_len:
 *	Size of a tick mark placed on axis (in devcoords)
 *  axis_width:
 *	An estimate of the width of a large character in
 *      the axis font (in devcoords).  This can be an overestimate.  An
 *      underestimate may produce bad results.
 *  axis_height:
 *	An estimate of the height of a large character in
 *      the axis labeling font (in devcoords).
 *  title_width, title_height:
 *	Same as above except for the title font.
 *  max_segs:
 *	Due to buffering constraints,  some devices may not be able to
 *	handle massive segment lists.  This parameter tells xgraph not
 *	to send more than `max_segs' segments in one request.
 * Output to the device should be written to the stream `strm'.
 * The functions are described individually below.  After filling
 * in the parameters and setting the function pointers,  the routine
 * should initialize its drawing state and store any extra needed
 * information in `user_state'.  This value will be passed to all
 * other routines during the drawing sequence.  If the device
 * cannot initialize,  it should return a zero status and fill
 * `errmsg' with an informative error message.
 */
{
    Info   *idraw_info;
    char  **l;
    double  scx,
            scy;

    idraw_info = (Info *) Malloc(sizeof(*idraw_info));

    for (l = idraw_prologue; *l; l++)
	fprintf(strm, "%s\n", *l);

    out_info->dev_flags = 0;
    scx = width / 612;
    scy = height / 792.0;
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

    out_info->area_w = width * 0.00283;	/* pts per micron */
    out_info->area_h = height * 0.00283;

    out_info->tick_len = axis_size;
    out_info->axis_height = axis_size;
    out_info->title_height = title_size;
    out_info->axis_width = (axis_size * 5.0) / 12.0;
    out_info->title_width = (title_size * 5.0) / 12.0;
    out_info->max_segs = 100;
    out_info->xg_text = idrawText;
    out_info->xg_seg = idrawSeg;
    out_info->xg_dot = idrawDot;
    out_info->xg_end = idrawEnd;
    out_info->user_state = (char *) idraw_info;

    idraw_info->title_font = title_family;
    idraw_info->axis_font = axis_family;
    idraw_info->title_size = title_size;
    idraw_info->axis_size = axis_size;
    idraw_info->strm = strm;
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
idraw_just(x, y, just, size, len)
int    *x,
       *y;			/* Given location (lower left) */
int     just;			/* Justification */
int     size;			/* Size in points */
int     len;			/* Number of chars */

/*
 * Unfortunately, idraw really can't display text with a justification.
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
	*y += t_height / 2;
	break;
    case T_LEFT:
	*y += t_height / 2;
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
	*y += t_height / 2;
	break;
    case T_LOWERRIGHT:
	*x -= t_width;
	*y += t_height;
	break;
    case T_BOTTOM:
	*x -= t_width / 2;
	*y += t_height;
	break;
    case T_LOWERLEFT:
	*y += t_height;
	break;
    }

    /*
     * Also, idraw seems to put a space above all text it draws. The
     * computation below compensates for this.
     */
    *y += (size / 3);
}

void 
idrawText(user_state, x, y, text, just, style)
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
    Info   *idraw = (Info *) user_state;

    FIX(y);
    font = style == T_AXIS ? idraw->axis_font :
	idraw->title_font;
    size = style == T_AXIS ? idraw->axis_size :
	idraw->title_size;

    idraw_just(&x, &y, just, size, strlen(text));

    fprintf(idraw->strm, "Begin %%I Text\n");
    fprintf(idraw->strm, "%%I cfg Black\n");
    fprintf(idraw->strm, "0 0 0 SetCFg\n");
    fprintf(idraw->strm, "%%I f *%s*-%d-*\n", font, size);
    fprintf(idraw->strm, "/%s %d SetF\n", font, size);
    fprintf(idraw->strm, "%%I t\n");
    fprintf(idraw->strm, "[ 1 0 0 1 %d %d ] concat\n", x, y);
    fprintf(idraw->strm, "%%I\n");
    fprintf(idraw->strm, "[\n");
    fprintf(idraw->strm, "(%s)\n", text);
    fprintf(idraw->strm, "] Text\n");
    fprintf(idraw->strm, "End\n");

}

/* Line Styles */
#define L_AXIS		0
#define L_ZERO		1
#define L_VAR		2

void 
idrawSeg(user_state, ns, seglist, width, style, lappr, color)
char   *user_state;		/* Value set in xg_init */
int     ns;			/* Number of segments   */
XSegment *seglist;		/* X array of segments  */
int     width;			/* Width of lines       */
int     style;			/* See above            */
int     lappr;			/* Line appearence      */
int     color;			/* Line color (if any)  */

/*
 * This routine draws a number of line segments at the points
 * given in `seglist'.  Note that contiguous segments need not share
 * endpoints but often do.  All segments should be `width' devcoords wide
 * and drawn in style `style'.  If `style' is L_VAR,  the parameters
 * `color' and `lappr' should be used to draw the line.  Both
 * parameters vary from 0 to 7.  If the device is capable of
 * color,  `color' varies faster than `style'.  If the device
 * has no color,  `style' will vary faster than `color' and
 * `color' can be safely ignored.  However,  if the
 * the device has more than 8 line appearences,  the two can
 * be combined to specify 64 line style variations.
 * Xgraph promises not to send more than the `max_segs' in the
 * xgOut structure passed back from xg_init().
 */
{
    Info   *idraw = (Info *) user_state;
    short   to_style;
    int     i,
            j,
            k;

    static unsigned short style_list[] =
    {
	0xffff, 0xf0f0, 0xcccc, 0xaaaa,
	0xf060, 0xf198, 0x7f55, 0x0000,
    };

    to_style = style == L_AXIS ? 65535
	: style == L_ZERO ? 65535
	: style_list[lappr];

    for (i = 0; i < ns; i++) {
	FIX(seglist[i].y1);
	FIX(seglist[i].y2);
    }

    for (i = 0; i < ns; i = j) {

	for (j = i + 1; j < ns
	     && seglist[j - 1].x2 == seglist[j].x1
	     && seglist[j - 1].y2 == seglist[j].y1;
	     j++);

	fprintf(idraw->strm, "Begin %%I MLine\n");
	fprintf(idraw->strm, "%%I b %d\n", to_style);
	fprintf(idraw->strm, "%d 0 0 [", width);
	/* fprintf(idraw -> strm, "%d"); */
	fprintf(idraw->strm, "] 0 SetB\n");
	fprintf(idraw->strm, "%%I cfg Black\n");
	fprintf(idraw->strm, "0 0 0 SetCFg\n");
	fprintf(idraw->strm, "%%I cbg White\n");
	fprintf(idraw->strm, "1 1 1 SetCBg\n");
	fprintf(idraw->strm, "none SetP %%I p n\n");
	fprintf(idraw->strm, "%%I t u\n");

	fprintf(idraw->strm, "%%I %d\n", j - i + 1);

	for (k = i; k < j; k++)
	    fprintf(idraw->strm, "%d %d\n",
		    seglist[k].x1, seglist[k].y1);

	fprintf(idraw->strm, "%d %d\n",
		seglist[k - 1].x2, seglist[k - 1].y2);

	fprintf(idraw->strm, "%d MLine\n", j - i + 1);
	fprintf(idraw->strm, "End\n");
    }


}

/* Marker styles */
#define P_PIXEL		0
#define P_DOT		1
#define P_MARK		2

void 
idrawDot(user_state, x, y, style, type, color)
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
idrawEnd(user_state)
char   *user_state;

/*
 * This routine is called after a drawing sequence is complete.
 * It can be used to clean up the user state and set the device
 * state appropriately.  This routine is optional in the structure.
 */
{
    Info   *idraw = (Info *) user_state;

    fprintf(idraw->strm, "End %%I eop\n");
    fclose(idraw->strm);
}

/*
 * Adding an output device to xgraph
 *
 * Step 2
 *   Edit the file hard_devices.c.  Declare your initialization
 *   function and add your device to the list of devices,
 *   hard_devices[].  The structure hard_dev is described below:
 */

#ifdef notdef
extern int idrawInit();

struct hard_dev idraw =
{
    "idraw format", idrawInit,
    0, ".clipboard", 0,
    25, "Times-Bold", 18, "Times", 12
};

#endif

/*
 * dev_spec:
 *    The dev_spec field should be a command that directly outputs to
 *    your device.  The command should contain one %s directive that
 *    will be filled in with the name of the device from the hardcopy
 *    dialog.
 * dev_file:
 *    The default file to write output to if the user selects `To File'.
 * dev_printer:
 *    The default printer to write output to if the user selects
 *    `To Device'.
 * dev_max_dim:
 *    The default maximum dimension for the device in centimeters.
 * dev_title_font, dev_title_size:
 *    The default title font and size.  Sizes are specified in
 *    points (1/72 inch).
 * dev_axis_font, dev_axis_size:
 *    The default axis font and size.
 */

/*
 * Adding an output device to xgraph
 *
 * Step 3
 *   Edit the file Makefile.  Add your source file to the SRC variable
 *   and the corresponding object file to the OBJ variable.  Finally,
 *   remake xgraph.  Your device should now be available in the
 *   hardcopy dialog.
 */
