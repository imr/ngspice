/*
 * Generic Output Driver for X
 * X version 11
 *
 * This is the primary output driver used by the new X graph
 * to display output to the X server.  It has been factored
 * out of the original xgraph to allow mulitple hardcopy
 * output devices to share xgraph's capabilities.  Note:
 * xgraph is still heavily X oriented.  This is not intended
 * for porting to other window systems.
 */

#include "copyright.h"
#include "xgout.h"
#include "params.h"

#define PADDING 	2
#define SPACE 		10
#define TICKLENGTH	5
#define MAXSEGS		1000

struct x_state {
    Window  win;		/* Primary window           */
};

void    text_X();
void    seg_X();
void    dot_X();


typedef struct attr_set {
    char    lineStyle[MAXLS];
    int     lineStyleLen;
    Pixel   pixelValue;
    Pixmap  markStyle;
}       AttrSet;

static AttrSet AllAttrs[MAXATTR];

static Pixmap dotMap = (Pixmap) 0;

/*
 * Marker bitmaps
 */

#include "bitmaps/dot.11"

#include "bitmaps/mark1.11"
#include "bitmaps/mark2.11"
#include "bitmaps/mark3.11"
#include "bitmaps/mark4.11"
#include "bitmaps/mark5.11"
#include "bitmaps/mark6.11"
#include "bitmaps/mark7.11"
#include "bitmaps/mark8.11"

/* Sizes exported for marker drawing */
static unsigned int dot_w = dot_width;
static unsigned int dot_h = dot_height;
static unsigned int mark_w = mark1_width;
static unsigned int mark_h = mark1_height;
static int mark_cx = mark1_x_hot;
static int mark_cy = mark1_y_hot;



void 
set_X(new_win, out_info)
Window  new_win;		/* Newly created window */
xgOut  *out_info;		/* Information to set   */

/*
 * Sets some of the common parameters for the X output device.
 */
{
    struct x_state *new_state;
    XFontStruct *font;

    out_info->dev_flags = ((depth > 3) ? D_COLOR : 0);
    out_info->area_w = out_info->area_h = 0;	/* Set later */
    out_info->bdr_pad = PADDING;
    out_info->axis_pad = SPACE;
    out_info->legend_pad = 0;
    out_info->tick_len = TICKLENGTH;

    font = PM_FONT("LabelFont");
#ifdef OLD
    out_info->axis_width =
	font->max_bounds.rbearing - font->max_bounds.lbearing;
#endif
    out_info->axis_width = XTextWidth(font, "8", 1);
    out_info->axis_height =
	font->max_bounds.ascent + font->max_bounds.descent;

    font = PM_FONT("TitleFont");
#ifdef OLD
    out_info->title_width =
	font->max_bounds.rbearing - font->max_bounds.lbearing;
#endif
    out_info->title_width = XTextWidth(font, "8", 1);
    out_info->title_height =
	font->max_bounds.ascent + font->max_bounds.descent;
    out_info->max_segs = MAXSEGS;

    out_info->xg_text = text_X;
    out_info->xg_seg = seg_X;
    out_info->xg_dot = dot_X;
    out_info->xg_end = (void (*) ()) 0;
    new_state = (struct x_state *) Malloc(sizeof(struct x_state));
    new_state->win = new_win;
    out_info->user_state = (char *) new_state;
}



static void 
init_once()
/*
 * Initializes AllAttrs.
 */
{
    Window  temp_win;
    XSetWindowAttributes wattr;
    char    name[1024];
    int     idx;
    params  style_val;

    /* Get attributes out parameters database */
    for (idx = 0; idx < MAXATTR; idx++) {
	(void) sprintf(name, "%d.Style", idx);
	(void) param_get(name, &style_val);
	AllAttrs[idx].lineStyleLen = style_val.stylev.len;
	(void) strncpy(AllAttrs[idx].lineStyle, style_val.stylev.dash_list,
		       style_val.stylev.len);
	(void) sprintf(name, "%d.Color", idx);
	AllAttrs[idx].pixelValue = PM_PIXEL(name);
    }

    /* Create a temporary window for representing depth */
    temp_win = XCreateWindow(disp, RootWindow(disp, screen),
			     0, 0, 10, 10, 0, depth, InputOutput,
			     vis, (unsigned long) 0, &wattr);

    /* Store bitmaps for dots and markers */
    dotMap = XCreateBitmapFromData(disp, temp_win, dot_bits, dot_w, dot_h);

    AllAttrs[0].markStyle = XCreateBitmapFromData(disp, temp_win,
						  mark1_bits, mark_w, mark_h);
    AllAttrs[1].markStyle = XCreateBitmapFromData(disp, temp_win,
						  mark2_bits, mark_w, mark_h);
    AllAttrs[2].markStyle = XCreateBitmapFromData(disp, temp_win,
						  mark3_bits, mark_w, mark_h);
    AllAttrs[3].markStyle = XCreateBitmapFromData(disp, temp_win,
						  mark4_bits, mark_w, mark_h);
    AllAttrs[4].markStyle = XCreateBitmapFromData(disp, temp_win,
						  mark5_bits, mark_w, mark_h);
    AllAttrs[5].markStyle = XCreateBitmapFromData(disp, temp_win,
						  mark6_bits, mark_w, mark_h);
    AllAttrs[6].markStyle = XCreateBitmapFromData(disp, temp_win,
						  mark7_bits, mark_w, mark_h);
    AllAttrs[7].markStyle = XCreateBitmapFromData(disp, temp_win,
						  mark8_bits, mark_w, mark_h);
    XDestroyWindow(disp, temp_win);
}

/*ARGSUSED*/
void 
init_X(user_state)
char   *user_state;

/*
 * Initializes for an X drawing sequence.  Sets up drawing attributes
 * by reading values from the parameter database.
 */
{
    static int initialized = 0;

    if (!initialized) {
	init_once();
	initialized = 1;
    }
}

static GC 
textGC(t_win, t_font)
Window  t_win;			/* Window for making GC */
XFontStruct *t_font;		/* Text font            */

/*
 * Sets the fields above in a global graphics context.  If
 * the graphics context does not exist,  it is created.
 */
{
    static GC text_gc = (GC) 0;
    XGCValues gcvals;
    unsigned long gcmask;

    gcvals.font = t_font->fid;
    gcmask = GCFont;
    if (text_gc == (GC) 0) {
	gcvals.foreground = PM_PIXEL("Foreground");
	gcmask |= GCForeground;
	text_gc = XCreateGC(disp, t_win, gcmask, &gcvals);
    }
    else {
	XChangeGC(disp, text_gc, gcmask, &gcvals);
    }
    return text_gc;
}

static GC 
segGC(l_win, l_fg, l_style, l_width, l_chars, l_len)
Window  l_win;			/* Window for making GC */
Pixel   l_fg;			/* Foreground color */
int     l_style;		/* Line style       */
int     l_width;		/* Line width       */
char   *l_chars;		/* Character spec   */
int     l_len;			/* Length of spec   */

/*
 * Sets the fields above in a global graphics context.  If the
 * graphics context does not exist, it is created.
 */
{
    static GC segment_gc = (GC) 0;
    XGCValues gcvals;
    unsigned long gcmask;

    gcvals.foreground = l_fg;
    gcvals.line_style = l_style;
    gcvals.line_width = l_width;
    gcmask = GCForeground | GCLineStyle | GCLineWidth;
    if (segment_gc == (GC) 0) {
	segment_gc = XCreateGC(disp, l_win, gcmask, &gcvals);
    }
    else {
	XChangeGC(disp, segment_gc, gcmask, &gcvals);
    }
    if (l_len > 0) {
	XSetDashes(disp, segment_gc, 0, l_chars, l_len);
    }
    return segment_gc;
}

static GC 
dotGC(d_win, d_fg, d_clipmask, d_xorg, d_yorg)
Window  d_win;			/* Window for making GC */
Pixel   d_fg;			/* Foreground color */
Pixmap  d_clipmask;		/* Clipmask         */
int     d_xorg,
        d_yorg;			/* Clipmask origin  */

/*
 * Sets the fields above in a global graphics context.  If the
 * graphics context does not exist, it is created.
 */
{
    static GC dot_gc = (GC) 0;
    XGCValues gcvals;
    unsigned long gcmask;

    gcvals.foreground = d_fg;
    gcvals.clip_mask = d_clipmask;
    gcvals.clip_x_origin = d_xorg;
    gcvals.clip_y_origin = d_yorg;
    gcmask = GCForeground | GCClipMask | GCClipXOrigin | GCClipYOrigin;
    if (dot_gc == (GC) 0) {
	dot_gc = XCreateGC(disp, d_win, gcmask, &gcvals);
    }
    else {
	XChangeGC(disp, dot_gc, gcmask, &gcvals);
    }
    return dot_gc;
}



void 
text_X(user_state, x, y, text, just, style)
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
    struct x_state *st = (struct x_state *) user_state;
    XCharStruct bb;
    int     rx = 0,
            ry = 0,
            len,
            height,
            width,
            dir;
    int     ascent,
            descent;
    XFontStruct *font;

    len = strlen(text);
    font = ((style == T_TITLE) ? PM_FONT("TitleFont") : PM_FONT("LabelFont"));
    XTextExtents(font, text, len, &dir, &ascent, &descent, &bb);
    width = bb.rbearing - bb.lbearing;
    height = bb.ascent + bb.descent;

    switch (just) {
    case T_CENTER:
	rx = x - (width / 2);
	ry = y - (height / 2);
	break;
    case T_LEFT:
	rx = x;
	ry = y - (height / 2);
	break;
    case T_UPPERLEFT:
	rx = x;
	ry = y;
	break;
    case T_TOP:
	rx = x - (width / 2);
	ry = y;
	break;
    case T_UPPERRIGHT:
	rx = x - width;
	ry = y;
	break;
    case T_RIGHT:
	rx = x - width;
	ry = y - (height / 2);
	break;
    case T_LOWERRIGHT:
	rx = x - width;
	ry = y - height;
	break;
    case T_BOTTOM:
	rx = x - (width / 2);
	ry = y - height;
	break;
    case T_LOWERLEFT:
	rx = x;
	ry = y - height;
	break;
    }
    XDrawString(disp, st->win,
		textGC(st->win, font),
		rx, ry + bb.ascent, text, len);
}



void 
seg_X(user_state, ns, segs, width, style, lappr, color)
char   *user_state;		/* Value set in xg_init */
int     ns;			/* Number of segments   */
XSegment *segs;			/* X array of segments  */
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
    struct x_state *st = (struct x_state *) user_state;
    param_style ps;
    GC      gc;

    if (style == L_AXIS) {
	ps = PM_STYLE("GridStyle");
	if (ps.len < 2) {
	    gc = segGC(st->win, PM_PIXEL("Foreground"),
		       LineSolid, PM_INT("GridSize"), (char *) 0, 0);
	}
	else {
	    gc = segGC(st->win, PM_PIXEL("Foreground"),
		       LineOnOffDash, PM_INT("GridSize"),
		       ps.dash_list, ps.len);
	}
    }
    else if (style == L_ZERO) {
	/* Set the color and line style */
	ps = PM_STYLE("ZeroStyle");
	if (ps.len < 2) {
	    gc = segGC(st->win, PM_PIXEL("ZeroColor"),
		       LineSolid, PM_INT("ZeroWidth"), (char *) 0, 0);
	}
	else {
	    gc = segGC(st->win, PM_PIXEL("ZeroColor"),
		       LineOnOffDash, PM_INT("ZeroWidth"),
		       ps.dash_list, ps.len);
	}
    }
    else {
	/* Color and line style vary */
	if (lappr == 0) {
	    gc = segGC(st->win, AllAttrs[color].pixelValue, LineSolid,
		       width, (char *) 0, 0);
	}
	else {
	    gc = segGC(st->win, AllAttrs[color].pixelValue, LineOnOffDash,
	      width, AllAttrs[lappr].lineStyle, AllAttrs[lappr].lineStyleLen);
	}
        /* PW */
	if (lappr == 16) {
	    gc = segGC(st->win, PM_PIXEL("BackGround"), LineSolid,
		       width, (char *) 0, 0);
	}
    }
    XDrawSegments(disp, st->win, gc, segs, ns);
}


#define LAST_CHECK

void 
dot_X(user_state, x, y, style, type, color)
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
    struct x_state *st = (struct x_state *) user_state;

    switch (style) {
    case P_PIXEL:
	XDrawPoint(disp, st->win,
		 dotGC(st->win, AllAttrs[color].pixelValue, (Pixmap) 0, 0, 0),
		   x, y);
	break;
    case P_DOT:
	XFillRectangle(disp, st->win,
		       dotGC(st->win, AllAttrs[color].pixelValue, dotMap,
			     (int) (x - (dot_w >> 1)),
			     (int) (y - (dot_h >> 1))),
		       (int) (x - (dot_w >> 1)), (int) (y - (dot_h >> 1)),
		       dot_w, dot_h);
	break;
    case P_MARK:
	XFillRectangle(disp, st->win,
		       dotGC(st->win, AllAttrs[color].pixelValue,
			     AllAttrs[type].markStyle,
			     (int) (x - mark_cx),
			     (int) (y - mark_cy)),
		       (int) (x - mark_cx), (int) (y - mark_cy),
		       mark_w, mark_h);
	break;
    }
}
