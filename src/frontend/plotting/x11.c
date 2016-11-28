/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jeffrey M. Hsu
**********/

/*
  X11 drivers.
*/

#include "ngspice/ngspice.h"

#ifndef X_DISPLAY_MISSING

#  include <sys/time.h>
#  include <sys/types.h>  /* PN */
#  include <unistd.h>     /* PN */

#  include "ngspice/graph.h"
#  include "ngspice/ftedbgra.h"
#  include "ngspice/ftedev.h"
#  include "ngspice/fteinput.h"
#  include "ngspice/cpdefs.h"
#  include "ngspice/ftedefs.h"
#  include <variable.h>
#  include "../com_hardcopy.h"

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
#  include <X11/Intrinsic.h>

#  ifdef DEBUG
#     include <X11/Xlib.h> /* for _Xdebug */
#  endif

#include "x11.h"
#include "graphdb.h"
#include "display.h"
#include "graf.h"

#define RAD_TO_DEG      (180.0 / M_PI)

/* X dependent default parameters */
#define DEF_FONT "10x14"
#define NUMLINESTYLES 8
#define MW_LINEWIDTH 2  /* MW. I want larger lines */
#define NXPLANES 5      /* note: What is this used for? */
#define BOXSIZE 30      /* initial size of bounding box for zoomin */

#define NUMCOLORS 20

typedef struct x11info {
    Window window;
    int isopen;
    Widget shell, form, view, buttonbox, buttons[2];
    XFontStruct *font;
    GC gc;
    int lastx, lasty;   /* used in X_DrawLine */
    int lastlinestyle;  /* used in X_DrawLine */
    Pixel colors[NUMCOLORS];
} X11devdep;

#define DEVDEP(g) (*((X11devdep *) (g)->devdep))

static Display *display;
static GC xorgc;
static char *xlinestyles[NUMLINESTYLES] = {     /* test patterns XXX */
    "\001\001\001\001", /* solid */
    "\001\002\001\002", /* dots */
    "\007\007\007\007", /* longdash */
    "\003\003\003\003", /* shortdash */
    "\007\002\002\002", /* dots longdash */
    "\003\002\001\002", /* dots shortdash */
    "\003\003\007\003", /* short/longdash */
};

/* atoms for catching window delet by WM x-button */
static Atom atom_wm_delete_window;
static Atom atom_wm_protocols;

static Widget toplevel;
static Bool noclear = False;
static GRAPH *lasthardcopy; /* graph user selected */
static int X11_Open = 0;
static int numdispplanes;

/* static functions */
static void initlinestyles(void);
static void initcolors(GRAPH *graph);
static void X_ScreentoData(GRAPH *graph, int x, int y, double *fx, double *fy);
static void linear_arc(int x0, int y0, int radius, double theta, double delta_theta);
static void slopelocation(GRAPH *graph, int x0, int y0);
static void zoomin(GRAPH *graph);

//XtEventHandler
static void handlekeypressed(Widget w, XtPointer clientdata, XEvent *ev, Boolean *continue_dispatch);
static void handlebuttonev(Widget w, XtPointer graph, XEvent *ev, Boolean *continue_dispatch);
static void redraw(Widget w, XtPointer client_data, XEvent *ev, Boolean *continue_dispatch);
static void resize(Widget w, XtPointer client_data, XEvent *ev, Boolean *continue_dispatch);

//XtCallbackProc
static void hardcopy(Widget w, XtPointer client_data, XtPointer call_data);
static void killwin(Widget w, XtPointer client_data, XtPointer call_data);


static int
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

    if (cp_getvar("display", CP_STRING, buf)) {
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

    if (numdispplanes > 1)
        /* Dotted lines are a distraction when we have colors. */
        for (i = 2; i < NUMLINESTYLES; i++)
            xlinestyles[i] = xlinestyles[0];
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
        DEVDEP(graph).colors[0] = DEVDEP(graph).view->core.background_pixel;
        if (DEVDEP(graph).colors[0] == WhitePixel(display, DefaultScreen(display)))
            DEVDEP(graph).colors[1] = BlackPixel(display, DefaultScreen(display));
        else
            DEVDEP(graph).colors[1] = WhitePixel(display, DefaultScreen(display));

    } else {
        if (numdispplanes < NXPLANES)
            xmaxcolors = 1 << numdispplanes;

        for (i = 0; i < xmaxcolors; i++) {
            (void) sprintf(buf, "color%d", i);
            if (!cp_getvar(buf, CP_STRING, colorstring))
                (void) strcpy(colorstring, colornames[i]);
            if (!XAllocNamedColor(display,
                                  DefaultColormap(display, DefaultScreen(display)),
                                  colorstring, &visualcolor, &exactcolor)) {
                (void) sprintf(ErrorMessage,
                               "can't get color %s\n", colorstring);
                externalerror(ErrorMessage);
                DEVDEP(graph).colors[i] = i ? BlackPixel(display,
                                                         DefaultScreen(display))
                    : WhitePixel(display, DefaultScreen(display));
                continue;
            }
            DEVDEP(graph).colors[i] = visualcolor.pixel;

            /* MW. I don't need this, everyone must know what he is doing
               if (i > 0 &&
               DEVDEP(graph).colors[i] == DEVDEP(graph).view->core.background_pixel) {
               DEVDEP(graph).colors[i] = DEVDEP(graph).colors[0];
               } */

        }
        /* MW. Set Beackgroound here */
        XSetWindowBackground(display, DEVDEP(graph).window, DEVDEP(graph).colors[0]);

        /* if (DEVDEP(graph).colors[0] != DEVDEP(graph).view->core.background_pixel) {
        DEVDEP(graph).colors[0] = DEVDEP(graph).view->core.background_pixel;
        } */
    }

    for (i = xmaxcolors; i < NUMCOLORS; i++) {
        DEVDEP(graph).colors[i] = DEVDEP(graph).colors[i + 1 - xmaxcolors];
    }
}


static void
handlekeypressed(Widget w, XtPointer client_data, XEvent *ev, Boolean *continue_dispatch)
{
    XKeyEvent *keyev = & ev->xkey;
    GRAPH *graph = (GRAPH *) client_data;
    char text[4];
    int nbytes;

    NG_IGNORE(w);
    NG_IGNORE(continue_dispatch);

    nbytes = XLookupString(keyev, text, 4, NULL, NULL);
    if (!nbytes)
        return;
    /* write it */
    PushGraphContext(graph);
    text[nbytes] = '\0';
    SetColor(1);
    DevDrawText(text, keyev->x, graph->absolute.height - keyev->y, 0);
    /* save it */
    SaveText(graph, text, keyev->x, graph->absolute.height - keyev->y);
    /* warp mouse so user can type in sequence */
    XWarpPointer(display, None, DEVDEP(graph).window, 0, 0, 0, 0,
                 keyev->x + XTextWidth(DEVDEP(graph).font, text, nbytes),
                 keyev->y);
    PopGraphContext();
}


static void
handlebuttonev(Widget w, XtPointer client_data, XEvent *ev, Boolean *continue_dispatch)
{
    GRAPH *graph = (GRAPH *) client_data;

    NG_IGNORE(w);
    NG_IGNORE(continue_dispatch);

    switch (ev->xbutton.button) {
    case Button1:
        slopelocation(graph, ev->xbutton.x, ev->xbutton.y);
        break;
    case Button3:
        zoomin(graph);
        break;
    }
}


/* callback function for catching window deletion by WM x-button */
static void
handle_wm_messages(Widget w, XtPointer client_data, XEvent *ev, Boolean *cont)
{
    GRAPH *graph = (GRAPH *) client_data;

    NG_IGNORE(w);
    NG_IGNORE(cont);

    if (ev->type == ClientMessage &&
        ev->xclient.message_type == atom_wm_protocols &&
        (Atom) ev->xclient.data.l[0] == atom_wm_delete_window)
    {
        RemoveWindow(graph);
    }
}


/* Recover from bad NewViewPort call. */
#define RECOVERNEWVIEWPORT()                    \
    do {                                        \
        tfree(graph);                           \
        graph = NULL;                           \
    } while(0)
/* need to do this or else DestroyGraph will free it again */


/* NewViewport is responsible for filling in graph->viewport */
int
X11_NewViewport(GRAPH *graph)
{
    char fontname[513]; /* who knows . . . */
    char *p, *q;
    Cursor cursor;
    XSetWindowAttributes w_attrs;
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
    int trys;

    graph->devdep = TMALLOC(X11devdep, 1);

    /* set up new shell */
    DEVDEP(graph).shell = XtCreateApplicationShell
        ("shell", topLevelShellWidgetClass, NULL, 0);

    XtVaSetValues(DEVDEP(graph).shell, XtNtitle, graph->plotname, NULL);

    /* set up form widget */
    DEVDEP(graph).form = XtCreateManagedWidget
        ("form", formWidgetClass, DEVDEP(graph).shell, formargs, XtNumber(formargs));

    /* set up viewport */
    DEVDEP(graph).view = XtCreateManagedWidget
        ("viewport", widgetClass, DEVDEP(graph).form, viewargs, XtNumber(viewargs));
    XtAddEventHandler(DEVDEP(graph).view, ButtonPressMask, FALSE,
                      handlebuttonev, graph);
    XtAddEventHandler(DEVDEP(graph).view, KeyPressMask, FALSE,
                      handlekeypressed, graph);
    XtAddEventHandler(DEVDEP(graph).view, StructureNotifyMask, FALSE,
                      resize, graph);
    XtAddEventHandler(DEVDEP(graph).view, ExposureMask, FALSE,
                      redraw, graph);

    /* set up button box */
    XtSetArg(bboxargs[1], XtNfromHoriz, DEVDEP(graph).view);
    DEVDEP(graph).buttonbox = XtCreateManagedWidget
        ("buttonbox", boxWidgetClass, DEVDEP(graph).form, bboxargs, XtNumber(bboxargs));

    /* set up buttons */
    XtSetArg(buttonargs[0], XtNlabel, "quit");
    XtSetArg(bboxargs[1], XtNfromVert, NULL);
    DEVDEP(graph).buttons[0] = XtCreateManagedWidget
        ("quit", commandWidgetClass, DEVDEP(graph).buttonbox, buttonargs, 1);
    XtAddCallback(DEVDEP(graph).buttons[0], XtNcallback, killwin, graph);

    XtSetArg(buttonargs[0], XtNlabel, "hardcopy");
    XtSetArg(bboxargs[1], XtNfromVert, DEVDEP(graph).buttons[0]);
    DEVDEP(graph).buttons[1] = XtCreateManagedWidget
        ("hardcopy", commandWidgetClass, DEVDEP(graph).buttonbox, buttonargs, 1);
    XtAddCallback(DEVDEP(graph).buttons[1], XtNcallback, hardcopy, graph);

    /* set up fonts */
    if (!cp_getvar("xfont", CP_STRING, fontname))
        (void) strcpy(fontname, DEF_FONT);

    for (p = fontname; *p && *p <= ' '; p++)
        ;

    if (p != fontname) {
        for (q = fontname; *p; *q++ = *p++)
            ;
        *q = '\0';
    }

    trys = 1;
    while (!(DEVDEP(graph).font = XLoadQueryFont(display, fontname))) {
        sprintf(ErrorMessage, "can't open font %s", fontname);
        strcpy(fontname, "fixed");
        if (trys > 1) {
            internalerror(ErrorMessage);
            RECOVERNEWVIEWPORT();
            return (1);
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

    /* WM_DELETE_WINDOW protocol */
    atom_wm_protocols = XInternAtom(display, "WM_PROTOCOLS", False);
    atom_wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XtAddEventHandler(DEVDEP(graph).shell, NoEventMask, True, handle_wm_messages, graph);
    XSetWMProtocols(display, XtWindow(DEVDEP(graph).shell), &atom_wm_delete_window, 1);

    return (0);
}


/* This routine closes the X connection.
   It is not to be called for finishing a graph. */
int
X11_Close(void)
{
    // don't, this has never been mapped, there is no window ...
    // XtUnmapWidget(toplevel);
    XtDestroyWidget(toplevel);

    XtAppContext app = XtDisplayToApplicationContext(display);
    XtDestroyApplicationContext(app);

    // don't, XtDestroyApplicationContext(app) seems to have done that
    // XCloseDisplay(display);

    return 0;
}


int
X11_DrawLine(int x1, int y1, int x2, int y2)
{
    if (DEVDEP(currentgraph).isopen)
        XDrawLine(display, DEVDEP(currentgraph).window,
                  DEVDEP(currentgraph).gc,
                  x1, currentgraph->absolute.height - y1,
                  x2, currentgraph->absolute.height - y2);

    return 0;
}


int
X11_Arc(int x0, int y0, int radius, double theta, double delta_theta)
{
    int t1, t2;

    if (0 && !cp_getvar("x11lineararcs", CP_BOOL, NULL))
        linear_arc(x0, y0, radius, theta, delta_theta);

    if (DEVDEP(currentgraph).isopen) {
        t1 = (int) (64 * (180.0 / M_PI) * theta);
        t2 = (int) (64 * (180.0 / M_PI) * delta_theta);
        if (t2 == 0)
            return 0;
        XDrawArc(display, DEVDEP(currentgraph).window, DEVDEP(currentgraph).gc,
                 x0 - radius,
                 currentgraph->absolute.height - radius - y0,
                 (Dimension) (2 * radius), (Dimension) (2 * radius), t1, t2);
    }

    return 0;
}


/* note: x and y are the LOWER left corner of text */
int
X11_Text(char *text, int x, int y, int angle)
{
    /* We specify text position by lower left corner, so have to adjust for
       X11's font nonsense. */
    NG_IGNORE(angle);

    if (DEVDEP(currentgraph).isopen)
        XDrawString(display, DEVDEP(currentgraph).window,
                    DEVDEP(currentgraph).gc, x,
                    currentgraph->absolute.height
                    - (y + DEVDEP(currentgraph).font->max_bounds.descent),
                    text, (int) strlen(text));

    /* note: unlike before, we do not save any text here */
    return 0;
}


int
X11_DefineColor(int colorid, double red, double green, double blue)
{
    NG_IGNORE(blue);
    NG_IGNORE(green);
    NG_IGNORE(red);
    NG_IGNORE(colorid);

    internalerror("X11_DefineColor not implemented.");
    return 0;
}


int
X11_DefineLinestyle(int linestyleid, int mask)
{
    NG_IGNORE(mask);
    NG_IGNORE(linestyleid);

    internalerror("X11_DefineLinestyle not implemented.");
    return 0;
}


int
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

    return 0;
}


int
X11_SetColor(int colorid)
{
    currentgraph->currentcolor = colorid;
    XSetForeground(display, DEVDEP(currentgraph).gc,
                   DEVDEP(currentgraph).colors[colorid]);
    return 0;
}


int
X11_Update(void)
{
    if (X11_Open)
        XSync(display, 0);
    return 0;
}


int
X11_Clear(void)
{
    if (!noclear) /* hack so exposures look like they're handled nicely */
        XClearWindow(display, DEVDEP(currentgraph).window);
    return 0;
}


static void
X_ScreentoData(GRAPH *graph, int x, int y, double *fx, double *fy)
{
    double      lmin, lmax;

    if (graph->grid.gridtype == GRID_XLOG ||
        graph->grid.gridtype == GRID_LOGLOG)
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

    if (graph->grid.gridtype == GRID_YLOG ||
        graph->grid.gridtype == GRID_LOGLOG)
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


static void
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
        if (graph->grid.gridtype == GRID_POLAR ||
            graph->grid.gridtype == GRID_SMITH ||
            graph->grid.gridtype == GRID_SMITHGRID)
        {
            angle = RAD_TO_DEG * atan2(fy0, fx0);
            fprintf(stdout, "r0 = %g, a0 = %g\n",
                    hypot(fx0, fy0),
                    (angle>0)?angle:360.0+angle);
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
}


/* should be able to do this by sleight of hand on graph parameters */
static void
zoomin(GRAPH *graph)
{
/* note: need to add circular boxes XXX */

    int x0, y0, x1, y1;
    double fx0, fx1, fy0, fy1;
    char buf[BSIZE_SP];
    char buf2[128];
    char *t;

    Window rootwindow, childwindow;
    int rootx, rooty;
    unsigned int state;
    int x, y, upperx, uppery;
    unsigned width, height;

    /* open box and get area to zoom in on */

    XQueryPointer(display, DEVDEP(graph).window, &rootwindow,
                  &childwindow, &rootx, &rooty, &x0, &y0, &state);

    x = x1 = x0 + BOXSIZE;
    y = y1 = y0 + BOXSIZE;

    upperx = x0;
    uppery = y0;

    width  = BOXSIZE;
    height = BOXSIZE;

    XDrawRectangle(display, DEVDEP(graph).window, xorgc,
                   upperx, uppery, width, height);

    XWarpPointer(display, None, DEVDEP(graph).window, 0, 0, 0, 0, x1, y1);

    while (state & Button3Mask) {
        if (x != x1 || y != y1) {

            x1 = x;
            y1 = y;

            XDrawRectangle(display, DEVDEP(graph).window, xorgc,
                           upperx, uppery, width, height);

            upperx = MIN(x1, x0);
            uppery = MIN(y1, y0);

            width  = (unsigned) ABS(x1 - x0);
            height = (unsigned) ABS(y1 - y0);

            XDrawRectangle(display, DEVDEP(graph).window, xorgc,
                           upperx, uppery, width, height);
        }
        XQueryPointer(display, DEVDEP(graph).window, &rootwindow,
                      &childwindow, &rootx, &rooty, &x, &y, &state);
    }
    XDrawRectangle(display, DEVDEP(graph).window, xorgc,
                   upperx, uppery, width, height);

    X_ScreentoData(graph, x0, y0, &fx0, &fy0);
    X_ScreentoData(graph, x1, y1, &fx1, &fy1);

    if (fx0 > fx1) {
        SWAP(double, fx0, fx1);
    }
    if (fy0 > fy1) {
        SWAP(double, fy0, fy1);
    }

    strncpy(buf2, graph->plotname, sizeof(buf2));
    if ((t = strchr(buf2, ':')) != NULL)
        *t = '\0';

    if (!eq(plot_cur->pl_typename, buf2)) {
        (void) sprintf(buf,
                       "setplot %s; %s xlimit %.20e %.20e ylimit %.20e %.20e; setplot $curplot\n",
                       buf2, graph->commandline, fx0, fx1, fy0, fy1);
    } else {
        (void) sprintf(buf, "%s xlimit %e %e ylimit %e %e\n",
                       graph->commandline, fx0, fx1, fy0, fy1);
    }

/* don't use the following if using GNU Readline or BSD EditLine */
#if !defined(HAVE_GNUREADLINE) && !defined(HAVE_BSDEDITLINE)
    {
        wordlist *wl;

        /* hack for Gordon Jacobs */
        /* add to history list if plothistory is set */
        if (cp_getvar("plothistory", CP_BOOL, NULL)) {
            wl = cp_parse(buf);
            (void) cp_addhistent(cp_event++, wl);
        }
    }
#endif /* !defined(HAVE_GNUREADLINE) && !defined(HAVE_BSDEDITLINE) */

    (void) cp_evloop(buf);
}


static void
hardcopy(Widget w, XtPointer client_data, XtPointer call_data)
{
    NG_IGNORE(call_data);
    NG_IGNORE(w);

    /* com_hardcopy() -> gr_resize() -> setcolor() dirung postscript
       printing will act on currentgraph with a DEVDEP inherited from PSdevdep.
       But currentgraph had not changed its devdep, which was derived from
       incompatible X11devdep, thus overwriting some variables. Here you find a
       temporary remedy, until there will be a cleanup of graph handling. E.g.
       CopyGraph() does not make a copy of its devdep, but just copies the pointer,
       so keeping the old devdep. */

    lasthardcopy = (GRAPH *) client_data;

    /* FIXME #1: this should print currentgraph with
     *            currentgraph dynamically bound to client_data
     *  FIXME #2: the !currentgraphs case,
     *            don't bother do call com_hardcopy
     */

    if (currentgraph) {
        void *devdep = currentgraph->devdep;
        com_hardcopy(NULL);
        currentgraph->devdep = devdep;
    } else {
        com_hardcopy(NULL);
    }
}


static void
killwin(Widget w, XtPointer client_data, XtPointer call_data)
{
    GRAPH *graph = (GRAPH *) client_data;

    NG_IGNORE(call_data);
    NG_IGNORE(w);

    RemoveWindow(graph);
}


/* called from postcoms.c
   In the command 'destroy ac2' Will remove window associated with
   the plot (e.g. ac2) just before data of the plot are deleted.*/
void
RemoveWindow(GRAPH *graph)
{
    if (graph->devdep) {
        /* Iplots are done asynchronously */
        DEVDEP(graph).isopen = 0;
        /* MW. Not sure but DestroyGraph might free() to much - try Xt...() first */
        XtUnmapWidget(DEVDEP(graph).shell);
        XtDestroyWidget(DEVDEP(graph).shell);
        XFreeFont(display, DEVDEP(graph).font);
        XFreeGC(display, DEVDEP(graph).gc);
    }

    if (graph == currentgraph)
        currentgraph = NULL;

    DestroyGraph(graph->graphid);
}


/* call higher gr_redraw routine */
static void
redraw(Widget w, XtPointer client_data, XEvent *event, Boolean *continue_dispatch)
{
    GRAPH *graph = (GRAPH *) client_data;
    XExposeEvent *pev = & event->xexpose;
    XEvent ev;
    XRectangle rects[30];
    int n = 1;

    NG_IGNORE(w);
    NG_IGNORE(continue_dispatch);

    DEVDEP(graph).isopen = 1;

    rects[0].x = (Position) pev->x;
    rects[0].y = (Position) pev->y;
    rects[0].width  = (Dimension) pev->width;
    rects[0].height = (Dimension) pev->height;

    /* XXX */
    /* pull out all other expose regions that need to be redrawn */
    while (n < 30 && XCheckWindowEvent(display, DEVDEP(graph).window,
                                       ExposureMask, &ev)) {
        pev = (XExposeEvent *) &ev;
        rects[n].x = (Position) pev->x;
        rects[n].y = (Position) pev->y;
        rects[n].width  = (Dimension) pev->width;
        rects[n].height = (Dimension) pev->height;
        n++;
    }
    XSetClipRectangles(display, DEVDEP(graph).gc, 0, 0, rects, n, Unsorted);

    noclear = True;
    {
        GRAPH *tmp = currentgraph;
        currentgraph = graph;
        gr_redraw(graph);
        currentgraph = tmp;
    }
    noclear = False;

    XSetClipMask(display, DEVDEP(graph).gc, None);
}


static void
resize(Widget w, XtPointer client_data, XEvent *call_data, Boolean *continue_dispatch)
{
    GRAPH *graph = (GRAPH *) client_data;
    XEvent ev;

    NG_IGNORE(call_data);
    NG_IGNORE(continue_dispatch);

    /* pull out all other exposure events
       Also, get rid of other StructureNotify events on this window. */

    while (XCheckWindowEvent(display, DEVDEP(graph).window,
                             /* ExposureMask | */ StructureNotifyMask, &ev))
        ;

    XClearWindow(display, DEVDEP(graph).window);
    graph->absolute.width = w->core.width;
    graph->absolute.height = w->core.height;
    {
        GRAPH *tmp = currentgraph;
        currentgraph = graph;
        gr_resize(graph);
        currentgraph = tmp;
    }
}


int
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

        for (;;) {

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
                    NULL,
                    NULL,
                    NULL);

            /* handle X events first */
            if (FD_ISSET (ConnectionNumber(display), &rfds))
                /* handle ALL X events */
                while (XtPending()) {
                    XtNextEvent(&ev);
                    XtDispatchEvent(&ev);
                }

            if (FD_ISSET (fileno(request->fp), &rfds))
                goto out;

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
        return 1;
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
        return 1;
        break;
    }

out:
    if (response)
        response->option = request->option;
    return 0;
}


static void
linear_arc(int x0, int y0, int radius, double theta, double delta_theta)
/* x coordinate of center */
/* y coordinate of center */
/* radius of arc */
/* initial angle ( +x axis = 0 rad ) */
/* delta angle */
/*
 * Notes:
 *    Draws an arc of radius and center at (x0,y0) beginning at
 *    angle theta (in rad) and ending at theta + delta_theta
 */
{
    int x1, y1, x2, y2;
    int i, s = 60;
    double dphi;

    x2 = x0 + (int) (radius * cos(theta));
    y2 = y0 + (int) (radius * sin(theta));

    dphi = delta_theta / s;

    for (i = 1; i <= s; i++) {
        x1 = x2;
        y1 = y2;
        x2 = x0 + (int)(radius * cos(theta + i*dphi));
        y2 = y0 + (int)(radius * sin(theta + i*dphi));
        X11_DrawLine(x1, y1, x2, y2);
    }
}


#else

int x11_dummy_symbol;
/* otherwise, some linkers get upset */

#endif /* X_DISPLAY_MISSING */
