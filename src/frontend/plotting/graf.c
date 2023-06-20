/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jeffrey M. Hsu
**********/

/*
 *  Most of the gr_ module resides here, in particular, gr_init
 *      and gr_point, expect for the gr_ grid routines.
 *
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"             /* for CP_ */
#include "ngspice/cpextern.h"
#include "ngspice/plot.h"
#include "ngspice/ftedebug.h"           /* for iplot */
#include "ngspice/dvec.h"               /* for struct dvec */
#include "ngspice/ftedefs.h"            /* for FTEextern.h and IPOINT{MIN,MAX} */
#include "ngspice/fteinput.h"
#include "ngspice/ftedbgra.h"
#include "ngspice/ftedev.h"
#include "ngspice/graph.h"
#include "ngspice/grid.h"
#include "ngspice/sim.h"
#include "ngspice/stringskip.h"
#include "breakp2.h"
#include "display.h"
#include "graf.h"
#include "graphdb.h"
#include "runcoms.h"
#include "terminal.h"


static void gr_start_internal(struct dvec *dv, bool copyvec);
static void set(struct plot *plot, struct dbcomm *db, bool value, short mode);
static char *getitright(char *buf, double num);

/* for legends, set in gr_start, reset in gr_iplot and gr_init */
static struct {
    int plotno;
    int color;                  /* for assigning unique colors */
    int linestyle;              /* for assigning line styles */
} cur;

/* invariant:  currentgraph contains the current graph */

/* These are what gets plotted as points when you specify point plots */
static char pointchars[128];
#define DEFPOINTCHARS   "ox+#*abcdefhgijklmnpqrstuvwyz"

/* Buffer for ticmarks if given a list */
static char ticbuf[1024];
static char *ticlist = ticbuf;
#define MAXTICS 100

#define XFACTOR 1       /* How much to expand the X scale during iplot. */
#define YFACTOR 0.2     /* How much to expand the Y scale during iplot. */


/*
 *  Start of a new graph.
 *  Fill in the data that gets displayed.
 *  Difference from old gr_init
 *    we don't try to determine the look of the screen from here
 *    leave to lower level routines
 *
 */

int gr_init(double *xlims, double *ylims, /* The size of the screen. */
        const char *xname,
        const char *plotname, /* What to label things. */
        const char *hcopy, /* The raster file. */
        int nplots, /* How many plots there will be. */
        double xdelta, double ydelta, /* Line increments for the scale. */
        GRIDTYPE gridtype, /* The grid type */
        PLOTTYPE plottype, /*  and the plot type. */
        const char *xlabel,
        const char *ylabel, /* Labels for axes. */
        int xtype, int ytype, /* The types of the data graphed. */
        const char *pname,
        const char *commandline, /* For xi_zoomdata() */
        int prevgraph)                /* plot id, if started from a previous plot*/
{
    GRAPH *graph;
    wordlist *wl;

    NG_IGNORE(nplots);

    if ((graph = NewGraph()) == (GRAPH *) NULL) {
        return FALSE;
    }

    /*
      The global currentgraph will always be the current graph.
    */
    SetGraphContext(graph->graphid);

    graph->onevalue = (xname ? FALSE : TRUE);

    /* communicate filename to plot 5 driver */
    if (hcopy) {
        graph->devdep = copy(hcopy);
        graph->n_byte_devdep = strlen(hcopy) + 1;
    }

    cur.plotno = 0;

    /* note: should do only once, maybe in gr_init_once */
    if (!cp_getvar("pointchars", CP_STRING,
            pointchars, sizeof(pointchars))) {
        (void) strcpy(pointchars, DEFPOINTCHARS);
    }

    if (!cp_getvar("ticmarks", CP_NUM, &graph->ticmarks, 0)) {
        if (cp_getvar("ticmarks", CP_BOOL, NULL, 0)) {
            graph->ticmarks = 10;
        }
        else {
            graph->ticmarks = 0;
        }
    }

    if (!cp_getvar("ticchar", CP_STRING, graph->ticchar, 1)) {
        strcpy(graph->ticchar, "X");
    }

    if (cp_getvar("ticlist", CP_LIST, ticlist, 0)) {
        wl = vareval("ticlist");
        ticlist = wl_flatten(wl);
        graph->ticdata = readtics(ticlist);
    }
    else {
        graph->ticdata = NULL;
    }

    cp_getvar("nolegend", CP_BOOL, &(graph->nolegend), 0);
    cp_getvar("nounits", CP_BOOL, &(graph->nounits), 0);

    if (!xlims || !ylims) {
        internalerror("gr_init:  no range specified");
        return FALSE;
    }

    /* save upper and lower limits */
    graph->data.xmin = xlims[0];
    graph->data.xmax = xlims[1];
    graph->data.ymin = ylims[0];
    graph->data.ymax = ylims[1];

    /* get title into plot window */
    if (!pname) {
        pname = "(unknown)";
    }
    if (!plotname) {
        plotname = "(unknown)";
    }

    graph->plotname = tprintf("%s: %s", pname, plotname);


    /* restore background color from previous graph, e.g. for zooming,
       it will be used in NewViewport(graph) */
    if (prevgraph > 0) {
        graph->mgraphid = prevgraph;
    }
    else {
        graph->mgraphid = 0;
    }

    /* note: have enum here or some better convention */
    if (NewViewport(graph) == 1) {
        /* note: where is the error message generated? */
        /* note: undo tmallocs */
        fprintf(cp_err, "Can't open viewport for graphics.\n");
        return (FALSE);
    }

    /* restore data from previous graph, e.g. for zooming */
    if (prevgraph > 0) {
        int i;
        GRAPH* pgraph = FindGraph(prevgraph);
        /* transmit colors */
        for (i = 0; i < 25; i++) {
            graph->colorarray[i] = pgraph->colorarray[i];
        }
        strcpy(graph->ticchar, pgraph->ticchar);
        graph->ticdata = pgraph->ticdata;
        graph->ticmarks = pgraph->ticmarks;
        graph->nolegend = pgraph->nolegend;
    }

    /* layout decisions */
    /* note: have to do before gr_fixgrid and after NewViewport */
    graph->viewportxoff = graph->fontwidth * 8;  /* 8 lines on left */
    graph->viewportyoff = graph->fontheight * 4; /* 4 on bottom */

    DevClear();

    graph->grid.gridtype = gridtype;
    graph->plottype = plottype;
    graph->grid.xdatatype = xtype;
    graph->grid.ydatatype = ytype;
    graph->grid.xdelta = xdelta;
    graph->grid.ydelta = ydelta;
    graph->grid.ysized = 0;
    graph->grid.xsized = 0;

    if (!graph->onevalue) {
        if (xlabel) {
            graph->grid.xlabel = copy(xlabel);
        }
        else {
            graph->grid.xlabel = copy(xname);
        }

        if (ylabel) {
            graph->grid.ylabel = copy(ylabel);
        }
        else {
            graph->grid.ylabel = (char *) NULL;
        }
    }
    else {
        if (xlabel) {
            graph->grid.xlabel = copy(xlabel);
        }
        else {
            graph->grid.xlabel = copy("real");
        }

        if (ylabel) {
            graph->grid.ylabel = copy(ylabel);
        }
        else {
            graph->grid.ylabel = copy("imag");
        }
    }

    gr_resize_internal(graph);
    gr_redrawgrid(graph);

    /* Set up colors and line styles. */
    if (dispdev->numlinestyles == 1) {
        cur.linestyle = 0; /* Use the same one all the time. */
    }
    else {
        cur.linestyle = 1;
    }

    /* XXX Special exception for SMITH */
    if (dispdev->numcolors > 2 &&
            (graph->grid.gridtype == GRID_SMITH ||
            graph->grid.gridtype == GRID_SMITHGRID)) {
        cur.color = 3;
    }
    else {
        cur.color = 1;
    }

    graph->commandline = copy(commandline);

    return TRUE;
}


/* Once the line compression code is thorougly tested, checking code can
 * be removed.  But for now ...
 */
#define LINE_COMPRESSION_CHECKS

/* Data and functions for line compression:
 * try to not keep drawing the same pixels by combining co-linear segments.
 */

static struct {
    enum { EMPTY, LINE, VERTICAL } state;
    int                            x_start, y_start, x_end, y_end;
    int                            lc_min, lc_max;
#define prev_x2 lc_min                    // Alternate name
#ifdef LINE_COMPRESSION_CHECKS
    struct dvec                   *dv;    // Sanity checking.
#endif
} LC;

/* Flush pending line drawing. */

static void LC_flush(void)
{
    switch (LC.state) {
    case EMPTY:
        return;
    case LINE:
        DevDrawLine(LC.x_start, LC.y_start, LC.x_end, LC.y_end, FALSE);
        break;
    case VERTICAL:
        DevDrawLine(LC.x_start, LC.lc_min, LC.x_start, LC.lc_max, FALSE);
        break;
    }
    LC.state = EMPTY;
}

/* This replaces DevDrawLine() - low-level line drawing call. */

#ifdef LINE_COMPRESSION_CHECKS
static void drawLine(int x1, int y1, int x2, int y2, struct dvec *dv)
{
    if (LC.dv) {
        if (LC.dv != dv) {
            fprintf(cp_err, "LC: DV changed!\n");
            LC_flush();
            LC.dv = dv;
        }
    } else {
        LC.dv = dv;
        if (LC.state != EMPTY) {
            fprintf(cp_err, "LC: State %d but DV NULL.\n", (int)LC.state);
            LC_flush();
        }
    }
#else
static void drawLine(int x1, int y1, int x2, int y2)
{
#endif
    switch (LC.state) {
    refill:
        LC_flush();
        // Fall through ...
    case EMPTY:
        if (x1 == x2) {
            /* Vertical */

            LC.state = VERTICAL;
            LC.x_start = x1;
            LC.y_end = y2;
            if (y1 < y2) {
                LC.lc_min = y1;
                LC.lc_max = y2;
            } else {
                LC.lc_min = y2;
                LC.lc_max = y1;
            }
        } else {
            LC.state = LINE;
            LC.prev_x2 = x2;

            /* Store with LC.x_start < LC.x_end. */

            if (x1 < x2) {
                LC.x_start = x1;
                LC.y_start = y1;
                LC.x_end = x2;
                LC.y_end = y2;
            } else {
                LC.x_start = x2;
                LC.y_start = y2;
                LC.x_end = x1;
                LC.y_end = y1;
            }
        }
        break;
    case LINE:
        if ((int64_t)(x2 - x1) * (LC.y_end - LC.y_start) !=
            (int64_t)(y2 - y1) * (LC.x_end - LC.x_start)) {
            /* Not in line. */

            goto refill;
        }
        if (x1 != LC.prev_x2) {
            /* Not contiguous. */

            if (x1 > LC.x_end) {
                if (x2 > LC.x_end) {
                    /* Hole. */

                    goto refill;
                }
                LC.x_end = x1;
                LC.y_end = y1;
            } else if (x1 < LC.x_start) {
                if (x2 < LC.x_start) {
                    /* Hole. */

                    goto refill;
                }
                LC.x_start = x1;
                LC.y_start = y1;
            }
        }

        if (x2 > LC.x_end) {
            LC.x_end = x2;
            LC.y_end = y2;
        } else if (x2 < LC.x_start) {
            LC.x_start = x2;
            LC.y_start = y2;
        }
        LC.prev_x2 = x2;
        break;
    case VERTICAL:
        if (x1 != LC.x_start || x2 != LC.x_start)
            goto refill;
        if (y1 != LC.y_end) {
            /* Not contiguous, check for hole. */

            if (y1 < LC.lc_min) {
                if (y2 < LC.lc_min)
                    goto refill;
                LC.lc_min = y1;
            } else if (y1 > LC.lc_max) {
                if (y2 > LC.lc_max)
                    goto refill;
                LC.lc_max = y1;
            }
        }

        if (y2 < LC.lc_min)
            LC.lc_min = y2;
        else if (y2 > LC.lc_max)
            LC.lc_max = y2;
        LC.y_end = y2;
        break;
    }
}

/*
 *  Add a point to the curve we're currently drawing.
 *  Should be in between a gr_init() and a gr_end()
 *    except when iplotting, very bad hack
 *  Differences from old gr_point:
 *    We save points here, instead of in lower levels.
 *    Assume we are in right context
 *  Save points in data space (not screen space).
 *  We pass two points in so we can multiplex plots.
 *
 */
void gr_point(struct dvec *dv,
         double newx, double newy,
         double oldx, double oldy,
         int np)
{
    int oldtox, oldtoy;     /* value before clipping */

    char pointc[2];

    int fromx, fromy, tox, toy;
    int ymin, dummy;

    DatatoScreen(currentgraph, oldx, oldy, &fromx, &fromy);
    DatatoScreen(currentgraph, newx, newy, &tox, &toy);

    /* note: we do not particularly want to clip here */
    oldtox = tox; oldtoy = toy;
    if (!currentgraph->grid.circular) {
        if (clip_line(&fromx, &fromy, &tox, &toy,
                currentgraph->viewportxoff, currentgraph->viewportyoff,
                currentgraph->viewport.width + currentgraph->viewportxoff,
                currentgraph->viewport.height + currentgraph->viewportyoff)) {
            return;
        }
    }
    else {
        if (clip_to_circle(&fromx, &fromy, &tox, &toy,
                           currentgraph->grid.xaxis.circular.center,
                           currentgraph->grid.yaxis.circular.center,
                           currentgraph->grid.xaxis.circular.radius))
            return;
    }

    if (currentgraph->plottype != PLOT_POINT) {
        SetLinestyle(dv->v_linestyle);
    }
    else {
        /* if PLOT_POINT,
           don't want to plot an endpoint which have been clipped */
        if (tox != oldtox || toy != oldtoy) {
            return;
        }
    }
    SetColor(dv->v_color);

    switch (currentgraph->plottype) {
        double    *tics;
    case PLOT_LIN:
    case PLOT_RETLIN:
        /* If it's a linear plot, ignore first point since we don't
           want to connect with oldx and oldy. */
        if (np) {
#ifdef LINE_COMPRESSION_CHECKS
            drawLine(fromx, fromy, tox, toy, dv);
#else
            drawLine(fromx, fromy, tox, toy);
#endif
        } else {
            LC_flush(); // May be retrace with non-monotonic x-axis
        }

        if ((tics = currentgraph->ticdata) != NULL) {
            for (; *tics < HUGE; tics++)
                if (*tics == (double) np) {
                    DevDrawText(currentgraph->ticchar, (int) (tox - currentgraph->fontwidth / 2),
                                (int) (toy - currentgraph->fontheight / 2), 0);
                    break;
                }
        }
        else if ((currentgraph->ticmarks >0) && (np > 0) &&
                   (np % currentgraph->ticmarks == 0)) {
            /* Draw an 'x' */
            DevDrawText(currentgraph->ticchar, (int) (tox - currentgraph->fontwidth / 2),
                        (int) (toy - currentgraph->fontheight / 2), 0);
        }
        break;
    case PLOT_COMB:
        DatatoScreen(currentgraph,
                     0.0, currentgraph->datawindow.ymin,
                     &dummy, &ymin);
#ifdef LINE_COMPRESSION_CHECKS
        drawLine(tox, ymin, tox, toy, dv);
#else
        drawLine(tox, ymin, tox, toy);
#endif
        break;
    case PLOT_POINT:
        /* Here, gi_linestyle is the character used for the point.  */
        pointc[0] = (char) dv->v_linestyle;
        pointc[1] = '\0';
        DevDrawText(pointc, (int) (tox - currentgraph->fontwidth / 2),
                    (int) (toy - currentgraph->fontheight / 2), 0);
    default:
        break;
    }
}


static void gr_start_internal(struct dvec *dv, bool copyvec)
{
    struct dveclist *link;

    /* Do something special with poles and zeros.  Poles are 'x's, and
     * zeros are 'o's.  */
    if (dv->v_type == SV_POLE) {
        dv->v_linestyle = 'x';
        return;
    }
    else if (dv->v_type == SV_ZERO) {
        dv->v_linestyle = 'o';
        return;
    }

    /* Find a (hopefully) new line style and color. */
    if (currentgraph->plottype == PLOT_POINT) {
        if (pointchars[cur.linestyle - 1]) {
            cur.linestyle++;
        }
        else {
            cur.linestyle = 2;
        }
    }
    else if ((cur.linestyle > 0) &&
            (++cur.linestyle == dispdev->numlinestyles)) {
        cur.linestyle = 2;
    }

    if ((cur.color > 0) && (++cur.color == dispdev->numcolors))
        cur.color = (((currentgraph->grid.gridtype == GRID_SMITH ||
                currentgraph->grid.gridtype == GRID_SMITHGRID) &&
                (dispdev->numcolors > 3)) ? 4 : 2);

    if (currentgraph->plottype == PLOT_POINT) {
        dv->v_linestyle = pointchars[cur.linestyle - 2];
    }
    else {
        dv->v_linestyle = cur.linestyle;
    }

    dv->v_color = cur.color;

    /* Save the data so we can refresh */
    link = TMALLOC(struct dveclist, 1);
    link->next = currentgraph->plotdata;

    /* Either reuse input vector or copy depnding on copyvec */
    if (copyvec) {
        link->vector = vec_copy(dv);
        /* vec_copy doesn't set v_color or v_linestyle */
        link->vector->v_color = dv->v_color;
        link->vector->v_linestyle = dv->v_linestyle;
        link->vector->v_flags |= VF_PERMANENT;
        link->f_own_vector = TRUE;
    }
    else {
        link->vector = dv;
        link->f_own_vector = FALSE;
    }

    currentgraph->plotdata = link;

    /* Copy the scale vector, add it to the vector as v_scale
     * and use the copy instead of the original scale vector if requested */
    {
        struct dvec * const custom_scale = dv->v_scale;
        if (custom_scale != (struct dvec*) NULL) {
            if (copyvec) {
                currentgraph->plotdata->vector->v_scale = vec_copy(dv->v_scale);
                currentgraph->plotdata->vector->v_scale->v_flags |= VF_PERMANENT;
            }
        }
    }

    /* Put the legend entry on the screen. */
    if (!currentgraph->nolegend)
        drawlegend(currentgraph, cur.plotno++, dv);
}


/* Start one plot of a graph */
void gr_start(struct dvec *dv)
{
    gr_start_internal(dv, TRUE);
} /* end of function gr_start */



/* make sure the linestyles in this graph don't exceed the number of
   linestyles available in the current display device */
void
gr_relinestyle(GRAPH *graph)
{
    struct dveclist *link;

    for (link = graph->plotdata; link; link = link->next) {
        if (graph->plottype == PLOT_POINT)
            continue;
        if (!(link->vector->v_linestyle < dispdev->numlinestyles))
            link->vector->v_linestyle %= dispdev->numlinestyles;
        if (!(link->vector->v_color < dispdev->numcolors))
            link->vector->v_color %= dispdev->numcolors;
    }
}


/* PN  static */
void drawlegend(GRAPH *graph, int plotno, struct dvec *dv)
{
    const int x = (plotno % 2) ?
            graph->viewportxoff : (graph->viewport.width / 2);
    const int x_base = x + graph->viewport.width / 20;
    const int y = graph->absolute.height - graph->fontheight
        - ((plotno + 2) / 2) * (graph->fontheight);
    const int i = y + graph->fontheight / 2 + 1;
    SetColor(dv->v_color);
    if (graph->plottype == PLOT_POINT) {
        char buf[16];
        (void) sprintf(buf, "%c : ", dv->v_linestyle);
        DevDrawText(buf, x_base - 3 * graph->fontwidth, y, 0);
    }
    else {
        SetLinestyle(dv->v_linestyle);
        DevDrawLine(x, i, x + graph->viewport.width / 20, i, FALSE);
    }
    SetColor(1);
    DevDrawText(dv->v_name, x + graph->viewport.width / 20
                + graph->fontwidth, y, 0);
}


/* end one plot of a graph */
void gr_end(struct dvec *dv)
{
    LC_flush();
#ifdef LINE_COMPRESSION_CHECKS
    if (LC.dv && LC.dv != dv)
        fprintf(cp_err, "LC: DV changed in gr_end()!\n");
    else
        LC.dv = NULL;
#else
    NG_IGNORE(dv);
#endif
    DevUpdate();
}


/* Print text in the bottom line. */

void gr_pmsg(char *text)
{
    char buf[BSIZE_SP];
    buf[0] = '\0';

    DevUpdate();

    if (cp_getvar("device", CP_STRING, buf, sizeof(buf)) && !(strcmp("/dev/tty", buf) == 0))
        fprintf(cp_err, "%s", text);
    else if (currentgraph->grid.xlabel)
        /* MW. grid.xlabel may be NULL */
        DevDrawText(text, currentgraph->viewport.width -
                    (int) (strlen(currentgraph->grid.xlabel) + 3) *
                    currentgraph->fontwidth,
                    currentgraph->absolute.height - currentgraph->fontheight, 0);
    else
        fprintf(cp_err, " %s \n", text);

    DevUpdate();
}


void gr_clean(void)
{
    DevUpdate();
}


/* call this routine after viewport size changes */
void gr_resize(GRAPH *graph)
{
    double oldxratio, oldyratio;
    double scalex, scaley;
    struct _keyed *k;

    oldxratio = graph->aspectratiox;
    oldyratio = graph->aspectratioy;

    graph->grid.xsized = 0;
    graph->grid.ysized = 0;

    gr_resize_internal(graph);

    /* scale keyed text */
    scalex = oldxratio / graph->aspectratiox;
    scaley = oldyratio / graph->aspectratioy;

    for (k = graph->keyed; k; k = k->next) {
        k->x = (int)((k->x - graph->viewportxoff) * scalex + graph->viewportxoff);
        k->y = (int)((k->y - graph->viewportyoff) * scaley + graph->viewportyoff);
    }

    /* X also generates an expose after a resize.

       This is handled in X10 by not redrawing on resizes and waiting
       for the expose event to redraw.  In X11, the expose routine
       tries to be clever and only redraws the region specified in an
       expose event, which does not cover the entire region of the
       plot if the resize was from a small window to a larger window.
       So in order to keep the clever X11 expose event handling, we
       have the X11 resize routine pull out expose events for that
       window, and we redraw on resize also.  */
#ifdef X_DISPLAY_MISSING
    gr_redraw(graph);
#endif
}


/* PN static */
void gr_resize_internal(GRAPH *graph)
{
    if (!graph->grid.xsized)
        graph->viewport.width = (int)(graph->absolute.width -
                                      1.4 * graph->viewportxoff);
    if (!graph->grid.ysized)
        graph->viewport.height = graph->absolute.height -
            2 * graph->viewportyoff;

    gr_fixgrid(graph, graph->grid.xdelta, graph->grid.ydelta,
               graph->grid.xdatatype, graph->grid.ydatatype);

    /* cache width and height info to make DatatoScreen go fast */
    /* note: XXX see if this is actually used anywhere */
    graph->datawindow.width = graph->datawindow.xmax -
        graph->datawindow.xmin;
    graph->datawindow.height = graph->datawindow.ymax -
        graph->datawindow.ymin;

    /* cache (datawindow size) / (viewport size) */
    graph->aspectratiox = graph->datawindow.width / graph->viewport.width;
    graph->aspectratioy = graph->datawindow.height / graph->viewport.height;
}


/* redraw everything in struct graph */
void gr_redraw(GRAPH *graph)
{
    struct dveclist *link;

    /* establish current graph so default graphic calls will work right */
    PushGraphContext(graph);

    DevClear();

    /* redraw grid */
    gr_redrawgrid(graph);

    cur.plotno = 0;
    for (link = graph->plotdata; link; link = link->next) {
        /* redraw legend */
        if (!graph->nolegend)
            drawlegend(graph, cur.plotno++, link->vector);

        /* replot data
           if onevalue, pass it a NULL scale
           otherwise, if vec has its own scale, pass that
           else pass vec's plot's scale
        */
        ft_graf(link->vector,
                graph->onevalue ? NULL :
                (link->vector->v_scale ?
                 link->vector->v_scale :
                 link->vector->v_plot->pl_scale),
                TRUE);
    }

    gr_restoretext(graph);

    PopGraphContext();
}


void gr_restoretext(GRAPH *graph)
{
    struct _keyed *k;

    /* restore text */
    for (k = graph->keyed; k; k = k->next) {
        SetColor(k->colorindex);
        DevDrawText(k->text, k->x, k->y, 0);
    }
}


/* Do some incremental plotting. There are 3 cases:
 *
 * First, if length < IPOINTMIN, don't do anything.
 *
 * Second, if length = IPOINTMIN, plot what we have so far. This step
 * is essentially the initializaiton for the graph.
 *
 * Third, if length > IPOINTMIN, plot the last points and resize if
 * needed.
 *
 * Note we don't check for pole / zero because they are of length 1.
 *
 * FIXME: there is a problem with multiple iplots that use the same
 * vector, namely, that vector has the same color throughout.  This is
 * another reason why we need to pull color and linestyle out of dvec
 * XXX Or maybe even something more drastic ??
 * It would be better to associate a color with an instance using a
 * vector than the vector itself, for which color is something artificial. */
static int iplot(struct plot *pl, struct dbcomm *db)
{
    double window;
    int    len = pl->pl_scale->v_length;

    if (ft_grdb) {
        fprintf(cp_err, "Entering iplot, len = %d\n", len);
    }

    /* Do simple check for exit first */

    window = db->db_value1;
    if (len < 2 || db->db_op > len) { /* Nothing yet */
        return 0;
    }

    struct dvec   *v, *xs = pl->pl_scale;
    double        *lims, dy;
    double         start, stop, step;
    bool           changed = FALSE;
    int            id, yt;
    double         xlims[2], ylims[2];
    static REQUEST reqst = { checkup_option, NULL };
    int            inited = 0;
    int            n_vec_plot = 0;

    /* Exit if nothing is being plotted */
    for (v = pl->pl_dvecs; v; v = v->v_next) {
        if (v->v_flags & VF_PLOT) {
            ++n_vec_plot;
        }
    }

    if (n_vec_plot == 0) {
        return 0;
    }

    id = db->db_graphid;
    if (!id) { /* Do initialization */
        unsigned int  index, node_len;
        char          commandline[4196];

        strcpy(commandline, "plot ");
        index = 5;
        resumption = FALSE;
        /* Draw the grid for the first time, and plot everything. */
        lims = ft_minmax(xs, TRUE);
        xlims[0] = lims[0];
        xlims[1] = lims[1];
        if (window) {
            if (xlims[1] - xlims[0] > window) {
                xlims[1] += window / 3.0;       // Assume increasing scale.
                xlims[0] = xlims[1] - window;
            } else {
                xlims[1] = xlims[0] + window;
            }
        }
        ylims[0] = HUGE;
        ylims[1] = -ylims[0];
        for (v = pl->pl_dvecs; v; v = v->v_next) {
            if (v->v_flags & VF_PLOT) {
                lims = ft_minmax(v, TRUE);
                if (ylims[0] > lims[0]) {
                    ylims[0] = lims[0];
                }
                if (ylims[1] < lims[1]) {
                    ylims[1] = lims[1];
                }
                node_len = (unsigned int)snprintf(commandline + index,
                                                  (sizeof commandline) - index,
                                                  "%s ", v->v_name);
                if (commandline[index + node_len - 1] == ' ') // Not truncated
                    index += node_len;
                else
                    commandline[index] = '\0';           // Crop partial name
            }
        }

        /* Generate a small difference between ymin and ymax
           to catch the y=const case */
        if (ylims[0] == ylims[1]) {
            ylims[1] += 1e-9;
        }

        if (ft_grdb) {
            fprintf(cp_err,
                    "iplot: at start xlims = %G, %G, ylims = %G, %G\n",
                    xlims[0], xlims[1], ylims[0], ylims[1]);
        }

        for (yt = pl->pl_dvecs->v_type, v = pl->pl_dvecs->v_next; v;
                v = v->v_next) {
            if ((v->v_flags & VF_PLOT) && ((int) v->v_type != yt)) {
                yt = SV_NOTYPE;
                break;
            }
        }

        (void) gr_init(xlims, ylims, xs->v_name,
                pl->pl_title, NULL, n_vec_plot, 0.0, 0.0,
                GRID_LIN, PLOT_LIN, xs->v_name, "V", xs->v_type, yt,
                plot_cur->pl_typename, commandline, 0);

        for (v = pl->pl_dvecs; v; v = v->v_next) {
            if (v->v_flags & VF_PLOT) {
                gr_start_internal(v, FALSE);
                ft_graf(v, xs, TRUE);
            }
        }
        inited = 1;

    } else {
        /* plot the last points and resize if needed */

        Input(&reqst, NULL);

        /* Window was closed? */

        if (!currentgraph)
            return 0;

        /* First see if we have to make the screen bigger */

        dy = (isreal(xs) ? xs->v_realdata[len - 1] :
              realpart(xs->v_compdata[len - 1]));
        if (ft_grdb) {
            fprintf(cp_err, "x = %G\n", dy);
        }
        if (!if_tranparams(ft_curckt, &start, &stop, &step) ||
            !ciprefix("tran", pl->pl_typename)) {
            stop = HUGE;
            start = - stop;
        }

        /* checking for x lo */

        if (dy < currentgraph->data.xmin) {
            changed = TRUE;
            if (ft_grdb) {
                fprintf(cp_err, "resize: xlo %G -> %G\n",
                        currentgraph->data.xmin,
                        currentgraph->data.xmin -
                        (currentgraph->data.xmax - currentgraph->data.xmin)
                        * XFACTOR);
            }

            /* set the new x lo value */

            if (window) {
                currentgraph->data.xmin = dy - (window / 3.0);
            } else {
                currentgraph->data.xmin -=
                    (currentgraph->data.xmax - currentgraph->data.xmin) *
                        XFACTOR;
            }
            if (currentgraph->data.xmin < start)
                currentgraph->data.xmin = start;
        }

        /* checking for x hi */

        if (window && changed) {
            currentgraph->data.xmax = currentgraph->data.xmin + window;
        } else if (dy > currentgraph->data.xmax) {
            changed = TRUE;
            if (ft_grdb) {
                fprintf(cp_err, "resize: xhi %G -> %G\n",
                        currentgraph->data.xmax,
                        currentgraph->data.xmax +
                        (currentgraph->data.xmax - currentgraph->data.xmin)
                        * XFACTOR);
            }

            /* set the new x hi value */

            if (window) {
                currentgraph->data.xmax = dy + (window / 3.0);
                currentgraph->data.xmin = currentgraph->data.xmax - window;
            } else {
                currentgraph->data.xmax +=
                    (currentgraph->data.xmax - currentgraph->data.xmin) *
                        XFACTOR;
            }
            if (currentgraph->data.xmax > stop)
                currentgraph->data.xmax = stop;
        }

        if (currentgraph->data.xmax < currentgraph->data.xmin)
            currentgraph->data.xmax = currentgraph->data.xmin;

        /* checking for all y values */

        for (v = pl->pl_dvecs; v; v = v->v_next) {
            if (!(v->v_flags & VF_PLOT)) {
                continue;
            }
            dy = (isreal(v) ? v->v_realdata[len - 1] :
                  realpart(v->v_compdata[len - 1]));
            if (ft_grdb) {
                fprintf(cp_err, "y = %G\n", dy);
            }
            /* checking for y lo */
            while (dy < currentgraph->data.ymin) {
                changed = TRUE;
                if (ft_grdb) {
                    fprintf(cp_err, "resize: ylo %G -> %G\n",
                            currentgraph->data.ymin,
                            currentgraph->data.ymin -
                            (currentgraph->data.ymax - currentgraph->data.ymin)
                            * YFACTOR);
                }
                /* set the new y lo value */
                currentgraph->data.ymin -=
                    (currentgraph->data.ymax - currentgraph->data.ymin)
                    * YFACTOR;
                /* currentgraph->data.ymin +=
                  (dy - currentgraph->data.ymin) * YFACTOR;*/
                /* currentgraph->data.ymin = dy;
                  currentgraph->data.ymin *= (1 + YFACTOR); */
            }

            /* checking for y hi */

            while (dy > currentgraph->data.ymax) {
                changed = TRUE;
                if (ft_grdb) {
                    fprintf(cp_err, "resize: yhi %G -> %G\n",
                            currentgraph->data.ymax,
                            currentgraph->data.ymax +
                            (currentgraph->data.ymax - currentgraph->data.ymin)
                            * YFACTOR);
                }
                /* set the new y hi value */
                currentgraph->data.ymax +=
                    (currentgraph->data.ymax - currentgraph->data.ymin) *
                        YFACTOR;
                /* currentgraph->data.ymax +=
                  (dy - currentgraph->data.ymax) * YFACTOR;*/
                /* currentgraph->data.ymax = dy;
                  currentgraph->data.ymax *= (1 + YFACTOR); */
            }
        }

        if (currentgraph->data.ymax < currentgraph->data.ymin)
            currentgraph->data.ymax = currentgraph->data.ymin;

        if (changed) {
            /* Redraw everything. */
            gr_pmsg("Resizing screen");
            gr_resize(currentgraph);
#ifndef X_DISPLAY_MISSING
            gr_redraw(currentgraph);
#endif
        }
        else {
            /* Just connect the last two points. This won't be done
             * with curve interpolation, so it might look funny.  */
            for (v = pl->pl_dvecs; v; v = v->v_next) {
                if (v->v_flags & VF_PLOT) {
                    gr_point(v,
                             (isreal(xs) ? xs->v_realdata[len - 1] :
                              realpart(xs->v_compdata[len - 1])),
                             (isreal(v) ? v->v_realdata[len - 1] :
                              realpart(v->v_compdata[len - 1])),
                             (isreal(xs) ? xs->v_realdata[len - 2] :
                              realpart(xs->v_compdata[len - 2])),
                             (isreal(v) ? v->v_realdata[len - 2] :
                              realpart(v->v_compdata[len - 2])),
                             len - 1);
                    LC_flush();    // Disable line compression here ..
#ifdef LINE_COMPRESSION_CHECKS
                    LC.dv = NULL;  // ... and suppress warnings.
#endif
                }
            }
        }
    }
    DevUpdate();
    return inited;
}


static void set(struct plot *plot, struct dbcomm *db, bool value, short mode)
{
    struct dvec *v;
    struct dbcomm *dc;

    if (db->db_type == DB_IPLOTALL || db->db_type == DB_TRACEALL) {
        for (v = plot->pl_dvecs; v; v = v->v_next)
            if (value)
                v->v_flags |= mode;
            else
                v->v_flags &= (short) ~mode;
        return;
    }

    for (dc = db; dc; dc = dc->db_also) {
        if (dc->db_nodename1 == NULL)
            continue;
        v = vec_fromplot(dc->db_nodename1, plot);
        if (!v || v->v_plot != plot) {
            if (!eq(dc->db_nodename1, "0") && value) {
                fprintf(cp_err, "Warning: node %s non-existent in %s.\n",
                        dc->db_nodename1, plot->pl_name);
                /* note: XXX remove it from dbs, so won't get further errors */
            }
            continue;
        }
        if (value)
            v->v_flags |= mode;
        else
            v->v_flags &= (short) ~mode;
    }
}


static char *getitright(char *buf, double num)
{
    char *p;
    int k;

    sprintf(buf, "    % .5g", num);
    p = strchr(buf, '.');

    if (p) {
        return p - 4;
    } else {
        k = (int) strlen(buf);
        if (k > 8)
            return buf + 4;
        else /* k >= 4 */
            return buf + k - 4;
    }
}


static int hit, hit2;


void reset_trace(void)
{
    hit = -1;
    hit2 = -1;
}


void gr_iplot(struct plot *plot)
{
    struct dbcomm *db;
    int dontpop;        /* So we don't pop w/o push. */
    char buf[30];

    hit = 0;
    for (db = dbs; db; db = db->db_next) {
        if (db->db_type == DB_IPLOT || db->db_type == DB_IPLOTALL) {
            if (db->db_graphid) {
                GRAPH *gr;

                gr = FindGraph(db->db_graphid);
                if (!gr)
                    continue;
                PushGraphContext(gr);
            }

            /* Temporarily set plot flag on matching vector. */

            set(plot, db, TRUE, VF_PLOT);

            dontpop = 0;
            if (iplot(plot, db)) {
                /* graph just assigned */
                db->db_graphid = currentgraph->graphid;
                dontpop = 1;
            }

            set(plot, db, FALSE, VF_PLOT);

            if (!dontpop && db->db_graphid)
                PopGraphContext();

        } else if (db->db_type == DB_TRACENODE || db->db_type == DB_TRACEALL) {

            struct dvec *v, *u;
            int len;

            set(plot, db, TRUE, VF_PRINT);

            len = plot->pl_scale->v_length;

            dontpop = 0;
            for (v = plot->pl_dvecs; v; v = v->v_next) {
                if (v->v_flags & VF_PRINT) {
                    u = plot->pl_scale;
                    if (len <= 1 || hit <= 0 || hit2 < 0) {
                        if (len <= 1 || hit2 < 0) {
                            term_clear();
                        }
                        else {
                            term_home();
                        }
                        hit = 1;
                        hit2 = 1;
                        printf(
                            "\tExecution trace (remove with the \"delete\" command)");
                        term_cleol();
                        printf("\n");

                        if (u) {
                            printf("%12s:", u->v_name);
                            if (isreal(u)) {
                                printf("%s",
                                       getitright(buf, u->v_realdata[len - 1]));
                            }
                            else {
                                /* MW. Complex data here, realdata is NULL
                                   (why someone use realdata here again) */
                                printf("%s",
                                       getitright(buf, u->v_compdata[len - 1].cx_real));
                                printf(", %s",
                                       getitright(buf, u->v_compdata[len - 1].cx_imag));
                            }
                            term_cleol();
                            printf("\n");
                        }
                    }
                    if (v == u) {
                        continue;
                    }
                    printf("%12s:", v->v_name);
                    if (isreal(v)) {
                        printf("%s", getitright(buf, v->v_realdata[len - 1]));
                    }
                    else {
                        /* MW. Complex data again */
                        printf("%s", getitright(buf, v->v_compdata[len - 1].cx_real));
                        printf(", %s", getitright(buf, v->v_compdata[len - 1].cx_imag));
                    }
                    term_cleol();
                    printf("\n");
                }
            }
            set(plot, db, FALSE, VF_PRINT);
        }
    }
}


/* This gets called after iplotting is done.  We clear out the
 * db_graphid fields.  Copy the dvecs, which we referenced by
 * reference, so DestroyGraph gets to free its own copy.
 *
 * Note: This is a clear case for separating the linestyle and color
 * fields from dvec.  */

void gr_end_iplot(void)
{
    struct dbcomm *db, *prev, *next;
    GRAPH *graph;
    struct dveclist *link;
    struct dvec *dv;

    prev = NULL;
    for (db = dbs; db; prev = db, db = next) {
        next = db->db_next;
        if (db->db_type == DB_DEADIPLOT) {
            if (db->db_graphid) {
                DestroyGraph(db->db_graphid);
                if (prev)
                    prev->db_next = next;
                else
                    ft_curckt->ci_dbs = dbs = next;
                dbfree1(db);
            }
        }
        else if (db->db_type == DB_IPLOT || db->db_type == DB_IPLOTALL) {
            if (db->db_graphid) {

                /* get private copy of dvecs */
                graph = FindGraph(db->db_graphid);

                link = graph->plotdata;

                while (link) {
                    dv = link->vector;
                    link->vector = vec_copy(dv);
                    /* vec_copy doesn't set v_color or v_linestyle */
                    link->vector->v_color = dv->v_color;
                    link->vector->v_linestyle = dv->v_linestyle;
                    link->vector->v_flags |= VF_PERMANENT;
                    link = link->next;
                }

                db->db_graphid = 0;
            } else {
                /* warn that this wasn't plotted */
                fprintf(cp_err, "Warning: iplot %d was not executed.\n",
                        db->db_number);
            }
        }
    }
}


double *readtics(char *string)
{
    int k;
    char *words, *worde;
    double *tics, *ticsk;

    tics = TMALLOC(double, MAXTICS);
    ticsk = tics;
    words = string;

    for (k = 0; *words && k < MAXTICS; words = worde) {

        words = skip_ws(words);

        worde = words;
        while (isalpha_c(*worde) || isdigit_c(*worde))
            worde++;

        if (*worde)
            *worde++ = '\0';

        sscanf(words, "%lf", ticsk++);

        k++;

    }
    *ticsk = HUGE;
    return (tics);
}
