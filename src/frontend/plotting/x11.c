/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jeffrey M. Hsu
**********/

/*
	X11 drivers.
*/


#include <ngspice.h>

#ifndef X_DISPLAY_MISSING

#  include <sys/time.h>
#  include <sys/types.h>  /* PN */
#  include <unistd.h>     /* PN */

#  include <graph.h>
#  include <ftedbgra.h>
#  include <ftedev.h>
#  include <fteinput.h>
#  include <cpdefs.h>
#  include <ftedefs.h>
#  include <variable.h>

/* Added X11/ prefix to the next includes - ER */

#  include <X11/IntrinsicP.h>
#  include <X11/Xatom.h>
#  include <X11/StringDefs.h>
#  include <X11/Xutil.h>
#  include <X11/cursorfont.h>
#  include <X11/Xaw/Box.h>
#  include <X11/Xaw/Command.h>
#  include <X11/Xaw/Form.h>
#  include <X11/Shell.h>

#include "x11.h"

static void linear_arc(int x0, int y0, int radius, double theta1, double theta2);


#  ifdef DEBUG
extern int _Xdebug;
#  endif


#define RAD_TO_DEG	(180.0 / M_PI)

/* X dependent default parameters */
#define DEF_FONT "10x14"
#define NUMLINESTYLES 8
#define MW_LINEWIDTH 2  /* MW. I want larger lines */
#define NXPLANES 5      /* note: What is this used for? */
#define BOXSIZE 30      /* initial size of bounding box for zoomin */

typedef struct x11info {
	Window window;
	int	isopen;
	Widget shell, form, view, buttonbox, buttons[2];
	XFontStruct *font;
	GC gc;
	int lastx, lasty;   /* used in X_DrawLine */
	int lastlinestyle;  /* used in X_DrawLine */
} X11devdep;

#define DEVDEP(g) (*((X11devdep *) (g)->devdep))

static void linear_arc(int x0, int y0, int radius, double theta1, double theta2);
static Display *display;
static GC xorgc;
static char *xlinestyles[NUMLINESTYLES] = {	/* test patterns XXX */
	"\001\001\001\001",	/* solid */
	"\001\002\001\002",	/* dots */
	"\007\007\007\007",	/* longdash */
	"\003\003\003\003",	/* shortdash */
	"\007\002\002\002",	/* dots longdash */
	"\003\002\001\002",	/* dots shortdash */
	"\003\003\007\003",	/* short/longdash */
};

static Widget toplevel;
static Bool noclear = False;
static GRAPH *lasthardcopy; /* graph user selected */
static int X11_Open = 0;
static int numdispplanes;


extern void internalerror (char *message);
extern void externalerror (char *message);
static void initlinestyles (void);
static void initcolors (GRAPH *graph);
extern void PushGraphContext (GRAPH *graph);
extern void SetColor (int colorid);
extern void Text (char *text, int x, int y);
extern void SaveText (GRAPH *graph, char *text, int x, int y);
extern void PopGraphContext (void);
void slopelocation (GRAPH *graph, int x0, int y0);
void zoomin (GRAPH *graph);
static void X_ScreentoData (GRAPH *graph, int x, int y, double *fx, double *fy);
extern int DestroyGraph (int id);
extern void gr_redraw (GRAPH *graph);
extern void gr_resize (GRAPH *graph);


int
errorhandler(Display *display, XErrorEvent *errorev)
{
	XGetErrorText(display, errorev->error_code, ErrorMessage, 1024);
	externalerror(ErrorMessage);
	return 0;
}



int
X11_Init(void)
{

	char buf[512];
	char *displayname;

	XGCValues gcvalues;

	/* grrr, Xtk forced contortions */
	char *argv[2];
	int argc = 2;

	if (cp_getvar("display", VT_STRING, buf)) {
	  displayname = buf;
	} else if (!(displayname = getenv("DISPLAY"))) {
	  internalerror("Can't open X display.");
	  return (1);
	}

#  ifdef DEBUG
	_Xdebug = 1;
#  endif

	argv[0] = "ngspice";
	argv[1] = displayname;
/*
	argv[2] = "-geometry";
	argv[3] = "=1x1+2+2";
*/

	/* initialize X toolkit */
	toplevel = XtInitialize("ngspice", "Nutmeg", NULL, 0, &argc, argv);

	display = XtDisplay(toplevel);

	X11_Open = 1;

	/* "invert" works better than "xor" for B&W */

	/* xor gc should be a function of the pixels that are written on */
	/* gcvalues.function = GXxor; */
	/* this patch makes lines visible on true color displays
        Guenther Roehrich 22-Jan-99 */
 	gcvalues.function = GXinvert;
	gcvalues.line_width = 1;
	gcvalues.foreground = 1;
	gcvalues.background = 0;

	xorgc = XCreateGC(display, DefaultRootWindow(display),
	        GCLineWidth | GCFunction | GCForeground | GCBackground,
	        &gcvalues);

	/* set correct information */
	dispdev->numlinestyles = NUMLINESTYLES;
	dispdev->numcolors = NUMCOLORS;

	dispdev->width = DisplayWidth(display, DefaultScreen(display));
	dispdev->height = DisplayHeight(display, DefaultScreen(display));

	/* we don't want non-fatal X errors to call exit */
	XSetErrorHandler(errorhandler);

	numdispplanes = DisplayPlanes(display, DefaultScreen(display));

	return (0);

}

static void
initlinestyles(void)
{

	int i;

	if (numdispplanes > 1) {
	  /* Dotted lines are a distraction when we have colors. */
	  for (i = 2; i < NUMLINESTYLES; i++) {
	    xlinestyles[i] = xlinestyles[0];
	  }
	}

}

static void
initcolors(GRAPH *graph)
{
    int i;
    static char *colornames[] = {   "black",    /* white */
	"white", "red", "blue",
	"orange", "green", "pink",
	"brown", "khaki", "plum",
	"orchid", "violet", "maroon",
	"turquoise", "sienna", "coral",
	"cyan", "magenta", "gold",
	"yellow", ""
	};

    XColor visualcolor, exactcolor;
    char buf[BSIZE_SP], colorstring[BSIZE_SP];
    int xmaxcolors = NUMCOLORS; /* note: can we get rid of this? */

    if (numdispplanes == 1) {
	/* black and white */
	xmaxcolors = 2;
	graph->colors[0] = DEVDEP(graph).view->core.background_pixel;
	if (graph->colors[0] == WhitePixel(display, DefaultScreen(display)))
	    graph->colors[1] = BlackPixel(display, DefaultScreen(display));
	else
	    graph->colors[1] = WhitePixel(display, DefaultScreen(display));

    } else {
	if (numdispplanes < NXPLANES)
	    xmaxcolors = 1 << numdispplanes;

	for (i = 0; i < xmaxcolors; i++) {
	    (void) sprintf(buf, "color%d", i);
	    if (!cp_getvar(buf, VT_STRING, colorstring))
	    (void) strcpy(colorstring, colornames[i]);
	    if (!XAllocNamedColor(display,
		    DefaultColormap(display, DefaultScreen(display)),
		    colorstring, &visualcolor, &exactcolor)) {
		(void) sprintf(ErrorMessage,
		    "can't get color %s\n", colorstring);
		externalerror(ErrorMessage);
		graph->colors[i] = i ? BlackPixel(display,
		    DefaultScreen(display))
		    : WhitePixel(display, DefaultScreen(display));
		continue;
	    }
	    graph->colors[i] = visualcolor.pixel;
	    
	    
	/* MW. I don't need this, everyone must know what he is doing
	    if (i > 0 &&
		graph->colors[i] == DEVDEP(graph).view->core.background_pixel) {
		graph->colors[i] = graph->colors[0];
	    } */
	    
	}
		/* MW. Set Beackgroound here */
		XSetWindowBackground(display, DEVDEP(graph).window, graph->colors[0]);
		
/*	if (graph->colors[0] != DEVDEP(graph).view->core.background_pixel) {
	    graph->colors[0] = DEVDEP(graph).view->core.background_pixel;
	 } */
    }

    for (i = xmaxcolors; i < NUMCOLORS; i++) {
	graph->colors[i] = graph->colors[i + 1 - xmaxcolors];
    }
}


void
handlekeypressed(Widget w, caddr_t clientdata, caddr_t calldata)
{

	XKeyEvent *keyev = (XKeyPressedEvent *) calldata;
	GRAPH *graph = (GRAPH *) clientdata;
	char text[4];
	int nbytes;

	nbytes = XLookupString(keyev, text, 4, NULL, NULL);
	if (!nbytes) return;
	/* write it */
	PushGraphContext(graph);
	text[nbytes] = '\0';
	SetColor(1);
	Text(text, keyev->x, graph->absolute.height - keyev->y);
	/* save it */
	SaveText(graph, text, keyev->x, graph->absolute.height - keyev->y);
	/* warp mouse so user can type in sequence */
	XWarpPointer(display, None, DEVDEP(graph).window, 0, 0, 0, 0,
	    keyev->x + XTextWidth(DEVDEP(graph).font, text, nbytes),
	    keyev->y);
	PopGraphContext();

}


void
handlebuttonev(Widget w, caddr_t clientdata, caddr_t calldata)
{

	XButtonEvent *buttonev = (XButtonEvent *) calldata;

	switch (buttonev->button) {
	  case Button1:
	    slopelocation((GRAPH *) clientdata, buttonev->x, buttonev->y);
	    break;
	  case Button3:
	    zoomin((GRAPH *) clientdata);
	    break;
	}

}


/* Recover from bad NewViewPort call. */
#define RECOVERNEWVIEWPORT()    tfree((char *) graph);\
	            graph = (GRAPH *) NULL; 
	    /* need to do this or else DestroyGraph will free it again */

/* NewViewport is responsible for filling in graph->viewport */
int
X11_NewViewport(GRAPH *graph)
{

	char fontname[513]; /* who knows . . . */
	char *p, *q;
	Cursor cursor;
	XSetWindowAttributes	w_attrs;
	XGCValues gcvalues;
	static Arg formargs[ ] = {
	    { XtNleft, (XtArgVal) XtChainLeft },
	    { XtNresizable, (XtArgVal) TRUE }
	};
	static Arg bboxargs[ ] = {
	    { XtNfromHoriz, (XtArgVal) NULL },
	    { XtNbottom, (XtArgVal) XtChainTop },
	    { XtNtop, (XtArgVal) XtChainTop },
	    { XtNleft, (XtArgVal) XtChainRight },
	    { XtNright, (XtArgVal) XtChainRight }
	};
	static Arg buttonargs[ ] = {
	    { XtNlabel, (XtArgVal) NULL },
	    { XtNfromVert, (XtArgVal) NULL },
	    { XtNbottom, (XtArgVal) XtChainTop },
	    { XtNtop, (XtArgVal) XtChainTop },
	    { XtNleft, (XtArgVal) XtRubber },
	    { XtNright, (XtArgVal) XtRubber },
	    { XtNresizable, (XtArgVal) TRUE }
	};
	static Arg viewargs[] = {
	    { XtNresizable, (XtArgVal) TRUE },
	    { XtNwidth, (XtArgVal) 300 },
	    { XtNheight, (XtArgVal) 300 },
	    { XtNright, (XtArgVal) XtChainRight }
	};
	int	trys;

	graph->devdep = tmalloc(sizeof(X11devdep));

	/* set up new shell */
	DEVDEP(graph).shell = XtCreateApplicationShell("shell",
	        topLevelShellWidgetClass, NULL, 0);

	/* set up form widget */
	DEVDEP(graph).form = XtCreateManagedWidget("form",
	    formWidgetClass, DEVDEP(graph).shell, formargs, XtNumber(formargs));

	/* set up viewport */
	DEVDEP(graph).view = XtCreateManagedWidget("viewport", widgetClass,
						   DEVDEP(graph).form,
						   viewargs,
						   XtNumber(viewargs));
	XtAddEventHandler(DEVDEP(graph).view, ButtonPressMask, FALSE,
			  (XtEventHandler) handlebuttonev, graph);
	XtAddEventHandler(DEVDEP(graph).view, KeyPressMask, FALSE,
			 (XtEventHandler) handlekeypressed, graph);
	XtAddEventHandler(DEVDEP(graph).view, StructureNotifyMask, FALSE,
			 (XtEventHandler) resize, graph);
	XtAddEventHandler(DEVDEP(graph).view, ExposureMask, FALSE,
	        (XtEventHandler) redraw, graph);

	/* set up button box */
	XtSetArg(bboxargs[1], XtNfromHoriz, DEVDEP(graph).view);
	DEVDEP(graph).buttonbox = XtCreateManagedWidget("buttonbox",
	    boxWidgetClass, DEVDEP(graph).form, bboxargs, XtNumber(bboxargs));

	/* set up buttons */
	XtSetArg(buttonargs[0], XtNlabel, "quit");
	XtSetArg(bboxargs[1], XtNfromVert, NULL);
	DEVDEP(graph).buttons[0] = XtCreateManagedWidget("quit",
	    commandWidgetClass, DEVDEP(graph).buttonbox,
	    buttonargs, 1);
	XtAddCallback(DEVDEP(graph).buttons[0], XtNcallback, (XtCallbackProc) killwin, graph);

	XtSetArg(buttonargs[0], XtNlabel, "hardcopy");
	XtSetArg(bboxargs[1], XtNfromVert, DEVDEP(graph).buttons[0]);
	DEVDEP(graph).buttons[1] = XtCreateManagedWidget("hardcopy",
	    commandWidgetClass, DEVDEP(graph).buttonbox,
	    buttonargs, 1);
	XtAddCallback(DEVDEP(graph).buttons[1], XtNcallback, (XtCallbackProc) hardcopy, graph);

	/* set up fonts */
	if (!cp_getvar("font", VT_STRING, fontname)) {
	  (void) strcpy(fontname, DEF_FONT);
	}

	for (p = fontname; *p && *p <= ' '; p++)
		;
	if (p != fontname) {
		for (q = fontname; *p; *q++ = *p++)
			;
		*q = 0;
	}

	trys = 1;
	while (!(DEVDEP(graph).font = XLoadQueryFont(display, fontname))) {
	  sprintf(ErrorMessage, "can't open font %s", fontname);
	  strcpy(fontname, "fixed");
	  if (trys > 1) {
	      internalerror(ErrorMessage);
	      RECOVERNEWVIEWPORT();
	      return(1);
	  }
	  trys += 1;
	}

	graph->fontwidth = DEVDEP(graph).font->max_bounds.rbearing -
	        DEVDEP(graph).font->min_bounds.lbearing + 1;
	graph->fontheight = DEVDEP(graph).font->max_bounds.ascent +
	        DEVDEP(graph).font->max_bounds.descent + 1;

	XtRealizeWidget(DEVDEP(graph).shell);

	DEVDEP(graph).window = XtWindow(DEVDEP(graph).view);
	DEVDEP(graph).isopen = 0;
	w_attrs.bit_gravity = ForgetGravity;
	XChangeWindowAttributes(display, DEVDEP(graph).window, CWBitGravity,
		&w_attrs);
	/* have to note font and set mask GCFont in XCreateGC, p.w.h. */
	gcvalues.font = DEVDEP(graph).font->fid;
	gcvalues.line_width = MW_LINEWIDTH;
	gcvalues.cap_style = CapNotLast;
	gcvalues.function = GXcopy;
	DEVDEP(graph).gc = XCreateGC(display, DEVDEP(graph).window,
	        GCFont | GCLineWidth | GCCapStyle | GCFunction, &gcvalues);

	/* should absolute.positions really be shell.pos? */
	graph->absolute.xpos = DEVDEP(graph).view->core.x;
	graph->absolute.ypos = DEVDEP(graph).view->core.y;
	graph->absolute.width = DEVDEP(graph).view->core.width;
	graph->absolute.height = DEVDEP(graph).view->core.height;

	initlinestyles();
	initcolors(graph);

	/* set up cursor */
	cursor = XCreateFontCursor(display, XC_left_ptr);
	XDefineCursor(display, DEVDEP(graph).window, cursor);

	return (0);
}

/* This routine closes the X connection.
	It is not to be called for finishing a graph. */
void
X11_Close(void)
{
	XCloseDisplay(display);
}

void
X11_DrawLine(int x1, int y1, int x2, int y2)
{

	if (DEVDEP(currentgraph).isopen)
		XDrawLine(display, DEVDEP(currentgraph).window,
			DEVDEP(currentgraph).gc,
			x1, currentgraph->absolute.height - y1,
			x2, currentgraph->absolute.height - y2);


}


void
X11_Arc(int x0, int y0, int radius, double theta1, double theta2)
{

    int	t1, t2;

    if (!cp_getvar("x11lineararcs", VT_BOOL, (char *) &t1)) {
	linear_arc(x0, y0, radius, theta1, theta2);
    }

    if (DEVDEP(currentgraph).isopen) {
	if (theta1 >= theta2)
	    theta2 = 2 * M_PI + theta2;
	t1 = 64 * (180.0 / M_PI) * theta1;
	t2 = 64 * (180.0 / M_PI) * theta2 - t1;
	if (t2 == 0)
		return;
	XDrawArc(display, DEVDEP(currentgraph).window, DEVDEP(currentgraph).gc,
		x0 - radius,
		currentgraph->absolute.height - radius - y0,
		2 * radius, 2 * radius, t1, t2);		
    }
}

/* note: x and y are the LOWER left corner of text */
void
X11_Text(char *text, int x, int y)
{

/* We specify text position by lower left corner, so have to adjust for
	X11's font nonsense. */

	if (DEVDEP(currentgraph).isopen)
		XDrawString(display, DEVDEP(currentgraph).window,
		    DEVDEP(currentgraph).gc, x,
		    currentgraph->absolute.height
			- (y + DEVDEP(currentgraph).font->max_bounds.descent),
		    text, strlen(text));

	/* note: unlike before, we do not save any text here */

}


int
X11_DefineColor(int colorid, double red, double green, double blue)
{
	internalerror("X11_DefineColor not implemented.");
	return(0);
}


void
X11_DefineLinestyle(int linestyleid, int mask)
{
	internalerror("X11_DefineLinestyle not implemented.");
}

void
X11_SetLinestyle(int linestyleid)
{
	XGCValues values;

	if (currentgraph->linestyle != linestyleid) {

	  if ((linestyleid == 0 || numdispplanes > 1) && linestyleid != 1) {
	    /* solid if linestyle 0 or if has color, allow only one
	     * dashed linestyle */
	    values.line_style = LineSolid;
	  } else {
	    values.line_style = LineOnOffDash;
	  }
	  XChangeGC(display, DEVDEP(currentgraph).gc, GCLineStyle, &values);

	  currentgraph->linestyle = linestyleid;
	  XSetDashes(display, DEVDEP(currentgraph).gc, 0,
		xlinestyles[linestyleid], 4);
	}
}

void
X11_SetColor(int colorid)
{

	currentgraph->currentcolor = colorid;
	XSetForeground(display, DEVDEP(currentgraph).gc,
	        currentgraph->colors[colorid]);

}

void
X11_Update(void)
{

	if (X11_Open)
		XSync(display, 0);

}

void
X11_Clear(void)
{

	if (!noclear) /* hack so exposures look like they're handled nicely */
	  XClearWindow(display, DEVDEP(currentgraph).window);

}

static void
X_ScreentoData(GRAPH *graph, int x, int y, double *fx, double *fy)
{
	double	lmin, lmax;

	if (graph->grid.gridtype == GRID_XLOG
		|| graph->grid.gridtype == GRID_LOGLOG)
	{
		lmin = log10(graph->datawindow.xmin);
		lmax = log10(graph->datawindow.xmax);
		*fx = exp(((x - graph->viewportxoff)
			* (lmax - lmin) / graph->viewport.width + lmin)
			* M_LN10);
	} else {
		*fx = (x - graph->viewportxoff) * graph->aspectratiox +
			graph->datawindow.xmin;
	}

	if (graph->grid.gridtype == GRID_YLOG
		|| graph->grid.gridtype == GRID_LOGLOG)
	{
		lmin = log10(graph->datawindow.ymin);
		lmax = log10(graph->datawindow.ymax);
		*fy = exp(((graph->absolute.height - y - graph->viewportxoff)
			* (lmax - lmin) / graph->viewport.height + lmin)
			* M_LN10);
	} else {
		*fy = ((graph->absolute.height - y) - graph->viewportyoff)
			* graph->aspectratioy + graph->datawindow.ymin;
	}

}



void
slopelocation(GRAPH *graph, int x0, int y0)
             
                    /* initial position of mouse */
{

	int x1, y1;
	int x, y;
	Window rootwindow, childwindow;
	int rootx, rooty;
	unsigned int state;
	double fx0, fx1, fy0, fy1;
	double angle;

	x1 = x0;
	y1 = y0;
	XQueryPointer(display, DEVDEP(graph).window, &rootwindow, &childwindow,
	        &rootx, &rooty, &x, &y, &state);
	XDrawLine(display, DEVDEP(graph).window, xorgc, x0, y0, x0, y1-1);
	XDrawLine(display, DEVDEP(graph).window, xorgc, x0, y1, x1, y1);
	while (state & Button1Mask) {
	  if (x != x1 || y != y1) {
	    XDrawLine(display, DEVDEP(graph).window, xorgc,
	        x0, y0, x0, y1-1);
	    XDrawLine(display, DEVDEP(graph).window, xorgc,
	        x0, y1, x1, y1);
	    x1 = x;
	    y1 = y;
	    XDrawLine(display, DEVDEP(graph).window, xorgc, x0, y0, x0, y1-1);
	    XDrawLine(display, DEVDEP(graph).window, xorgc, x0, y1, x1, y1);
	  }
	  XQueryPointer(display, DEVDEP(graph).window, &rootwindow,
	        &childwindow, &rootx, &rooty, &x, &y, &state);
	}
	XDrawLine(display, DEVDEP(graph).window, xorgc, x0, y0, x0, y1-1);
	XDrawLine(display, DEVDEP(graph).window, xorgc, x0, y1, x1, y1);

	X_ScreentoData(graph, x0, y0, &fx0, &fy0);
	X_ScreentoData(graph, x1, y1, &fx1, &fy1);

	/* print it out */
	if (x1 == x0 && y1 == y0) {     /* only one location */
	    fprintf(stdout, "\nx0 = %g, y0 = %g\n", fx0, fy0);
	    if (graph->grid.gridtype == GRID_POLAR
	        || graph->grid.gridtype == GRID_SMITH
		|| graph->grid.gridtype == GRID_SMITHGRID)
	    {
		angle = RAD_TO_DEG * atan2( fy0, fx0 );
		fprintf(stdout, "r0 = %g, a0 = %g\n",
		    sqrt( fx0*fx0 + fy0*fy0 ),
		    (angle>0)?angle:(double) 360+angle);
	    }


	} else {    /* need to print info about two points */
	  fprintf(stdout, "\nx0 = %g, y0 = %g    x1 = %g, y1 = %g\n",
	      fx0, fy0, fx1, fy1);
	  fprintf(stdout, "dx = %g, dy = %g\n", fx1-fx0, fy1 - fy0);
	  if (x1 != x0 && y1 != y0) {
	/* add slope info if both dx and dy are zero,
	     because otherwise either dy/dx or dx/dy is zero,
	     which is uninteresting
	*/
	    fprintf(stdout, "dy/dx = %g    dx/dy = %g\n",
	    (fy1-fy0)/(fx1-fx0), (fx1-fx0)/(fy1-fy0));
	  }
	}

	return;

}

/* should be able to do this by sleight of hand on graph parameters */
void
zoomin(GRAPH *graph)
{
/* note: need to add circular boxes XXX */

	int x0, y0, x1, y1;
	double fx0, fx1, fy0, fy1, ftemp;
	char buf[BSIZE_SP];
	char buf2[128];
	char *t;
	wordlist *wl;
	int dummy;

	Window rootwindow, childwindow;
	int rootx, rooty;
	unsigned int state;
	int x, y, upperx, uppery, lowerx, lowery;

	/* open box and get area to zoom in on */

	XQueryPointer(display, DEVDEP(graph).window, &rootwindow,
	        &childwindow, &rootx, &rooty, &x0, &y0, &state);

	x = lowerx = x1 = x0 + BOXSIZE;
	y = lowery = y1 = y0 + BOXSIZE;
	upperx = x0;
	uppery = y0;

	XDrawRectangle(display, DEVDEP(graph).window, xorgc,
	        upperx, uppery, lowerx - upperx, lowery - uppery);

/* note: what are src_x, src_y, src_width, and src_height for? XXX */
	XWarpPointer(display, None, DEVDEP(graph).window, 0, 0, 0, 0, x1, y1);

	while (state & Button3Mask) {
	  if (x != x1 || y != y1) {
	    XDrawRectangle(display, DEVDEP(graph).window, xorgc,
	        upperx, uppery, lowerx - upperx, lowery - uppery);
	    x1 = x;
	    y1 = y;
	    /* figure out upper left corner */
	    /* remember X11's (and X10's) demented coordinate system */
	    if (y0 < y1) {
	      uppery = y0;
	      upperx = x0;
	      lowery = y1;
	      lowerx = x1;
	    } else {
	      uppery = y1;
	      upperx = x1;
	      lowery = y0;
	      lowerx = x0;
	    }
	    XDrawRectangle(display, DEVDEP(graph).window, xorgc,
	        upperx, uppery, lowerx - upperx, lowery - uppery);
	  }
	  XQueryPointer(display, DEVDEP(graph).window, &rootwindow,
	           &childwindow, &rootx, &rooty, &x, &y, &state);
	}
	XDrawRectangle(display, DEVDEP(graph).window, xorgc,
	        upperx, uppery, lowerx - upperx, lowery - uppery);

	X_ScreentoData(graph, x0, y0, &fx0, &fy0);
	X_ScreentoData(graph, x1, y1, &fx1, &fy1);

	if (fx0 > fx1) {
	  ftemp = fx0;
	  fx0 = fx1;
	  fx1 = ftemp;
	}
	if (fy0 > fy1) {
	  ftemp = fy0;
	  fy0 = fy1;
	  fy1 = ftemp;
	}

	strncpy(buf2, graph->plotname, sizeof(buf2));
	if ((t =strchr(buf2, ':')))
		*t = 0;

	if (!eq(plot_cur->pl_typename, buf2)) {
	  (void) sprintf(buf,
"setplot %s; %s xlimit %.20e %.20e ylimit %.20e %.20e; setplot $curplot\n",
	   buf2, graph->commandline, fx0, fx1, fy0, fy1);
	} else {
	  (void) sprintf(buf, "%s xlimit %e %e ylimit %e %e\n",
	        graph->commandline, fx0, fx1, fy0, fy1);
	}

/* don't use the following if using GNU Readline - AV */
#ifndef HAVE_GNUREADLINE
	/* hack for Gordon Jacobs */
	/* add to history list if plothistory is set */
	if (cp_getvar("plothistory", VT_BOOL, (char *) &dummy)) {
	  wl = cp_parse(buf);
	  (void) cp_addhistent(cp_event++, wl);
	}

#endif /* HAVE_GNUREADLINE */

	(void) cp_evloop(buf);

}

void
hardcopy(Widget w, caddr_t client_data, caddr_t call_data)
{

	lasthardcopy = (GRAPH *) client_data;
	com_hardcopy(NULL);

}

void
killwin(Widget w, caddr_t client_data, caddr_t call_data)
{

	GRAPH *graph = (GRAPH *) client_data;

	/* Iplots are done asynchronously */
	DEVDEP(graph).isopen = 0;
/* MW. Not sure but DestroyGraph might free() to much - try Xt...() first */	
	XtDestroyWidget(DEVDEP(graph).shell);
	DestroyGraph(graph->graphid);
	

}

/* call higher gr_redraw routine */
void
redraw(Widget w, caddr_t client_data, caddr_t call_data)
{

	GRAPH *graph = (GRAPH *) client_data;
	XExposeEvent *pev = (XExposeEvent *) call_data;
	XEvent ev;
	XRectangle rects[30];
	int n = 1;

	DEVDEP(graph).isopen = 1;


	rects[0].x = pev->x;
	rects[0].y = pev->y;
	rects[0].width = pev->width;
	rects[0].height = pev->height;

	/* XXX */
	/* pull out all other expose regions that need to be redrawn */
	while (n < 30 && XCheckWindowEvent(display, DEVDEP(graph).window,
	        (long) ExposureMask, &ev)) {
	  pev = (XExposeEvent *) &ev;
	  rects[n].x = pev->x;
	  rects[n].y = pev->y;
	  rects[n].width = pev->width;
	  rects[n].height = pev->height;
	  n++;
	}
	XSetClipRectangles(display, DEVDEP(graph).gc, 0, 0,
	        rects, n, Unsorted);

	noclear = True;
	gr_redraw(graph);
	noclear = False;

	XSetClipMask(display, DEVDEP(graph).gc, None);

}

void
resize(Widget w, caddr_t client_data, caddr_t call_data)
{

	GRAPH *graph = (GRAPH *) client_data;
	XEvent ev;

	/* pull out all other exposure events
	   Also, get rid of other StructureNotify events on this window. */

	while (XCheckWindowEvent(display, DEVDEP(graph).window,
		(long) /* ExposureMask | */ StructureNotifyMask, &ev))
		;

	XClearWindow(display, DEVDEP(graph).window);
	graph->absolute.width = w->core.width;
	graph->absolute.height = w->core.height;
	gr_resize(graph);

}



void
X11_Input(REQUEST *request, RESPONSE *response)
{

	XEvent ev;
	int nfds;
        fd_set rfds;

	switch (request->option) {
	  case char_option:

	    nfds = ConnectionNumber(display) > fileno(request->fp) ?
	        ConnectionNumber(display) :
	        fileno(request->fp);

	    while (1) {

	      /* first read off the queue before doing the select */
	      while (XtPending()) {
	        XtNextEvent(&ev);
	        XtDispatchEvent(&ev);
	      }

	      /* block on ConnectionNumber and request->fp */
	      /* PN: added fd_set * casting */
              FD_ZERO(&rfds);
              FD_SET(fileno(request->fp), &rfds);
              FD_SET(ConnectionNumber(display), &rfds);
	      select (nfds + 1,
                      &rfds,
                      (fd_set *)NULL,
                      (fd_set *)NULL,
                      NULL);
              
	      /* handle X events first */
              if (FD_ISSET (ConnectionNumber(display), &rfds)) {
		    /* handle ALL X events */
		    while (XtPending()) {
			  XtNextEvent(&ev);
			  XtDispatchEvent(&ev);
		    }
	      }

	      if (FD_ISSET (fileno(request->fp), &rfds)) {
		    response->reply.ch = inchar(request->fp);
		    goto out;
	      }

	    }
	    break;

	  case click_option:
	    /* let's fake this */
	    response->reply.graph = lasthardcopy;
	    break;

	  case button_option:
	    /* sit and handle events until get a button selection */
	    internalerror("button_option not implemented");
	    response->option = error_option;
	    return;
	    break;

	  case checkup_option:
	    /* first read off the queue before doing the select */
	    while (XtPending()) {
	        XtNextEvent(&ev);
	        XtDispatchEvent(&ev);
	    }
	    break;

	  default:
	    internalerror("unrecognized input type");
	    response->option = error_option;
	    return;
	    break;
	}

out:
	if (response)
	    response->option = request->option;
	return;

}

static void
linear_arc(int x0, int y0, int radius, double theta1, double theta2)
              /* x coordinate of center */
              /* y coordinate of center */
                      /* radius of arc */
                      /* initial angle ( +x axis = 0 rad ) */
                      /* final angle ( +x axis = 0 rad ) */
    /*
     * Notes:
     *    Draws an arc of radius and center at (x0,y0) beginning at
     *    angle theta1 (in rad) and ending at theta2
     */
{
    int x1, y1, x2, y2;
    int s = 60;
    double dphi, phi;

    x2 = x0 + (int) (radius * cos(theta1));
    y2 = y0 + (int) (radius * sin(theta1));

    while(theta1 >= theta2)
	    theta2 += 2 * M_PI;
    dphi = (theta2 - theta1) / s;

    if ((theta1 + dphi) == theta1) {
	    theta2 += 2 * M_PI;
	    dphi = (theta2 - theta1) / s;
    }


    for(phi = theta1 + dphi; phi < theta2; phi += dphi) {
	    x1 = x2;
	    y1 = y2;
	    x2 = x0 + (int)(radius * cos(phi));
	    y2 = y0 + (int)(radius * sin(phi));
	    X11_DrawLine(x1,y1,x2,y2);
    }

    x1 = x2;
    y1 = y2;
    x2 = x0 + (int)(radius * cos(theta2));
    y2 = y0 + (int)(radius * sin(theta2));
    X11_DrawLine(x1,y1,x2,y2);
}

#else 
int x11_dummy_symbol;
/* otherwise, some linkers get upset */
#endif /* X_DISPLAY_MISSING */
