/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/graph.h"
#include "ngspice/ftedev.h"

#include "plot5.h"
#include "graf.h"
#include "ngspice/fteext.h"


static FILE *plotfile;

#define putsi(a)                                \
    do {                                        \
        putc((char) (a), plotfile);             \
        putc((char) ((a) >> 8), plotfile);      \
    } while(0)


#define SOLID 0

static char *linestyle[] = {
    "solid", "dotted", "longdashed", "shortdashed", "dotdashed" };

static int currentlinestyle = SOLID;


int Plt5_Init(void)
{
    dispdev->numlinestyles = 4;
    dispdev->numcolors = 2;

    /* arbitrary */
    dispdev->width = 1000;
    dispdev->height = 1000;

    return 0;
}


int Plt5_NewViewport(GRAPH *graph)
{
    if ((plotfile = fopen((char*) graph->devdep, "w")) == NULL) {
        perror((char *) graph->devdep);
        free(graph->devdep);
        graph->devdep = NULL;
        graph->n_byte_devdep = 0;
        return 1;
    }

    if (graph->absolute.width) {

        /* hardcopying from the scree,
           ie, we are passed a copy of an existing graph */
        putc('s', plotfile);
        putsi(0);
        putsi(0);
        putsi(graph->absolute.width);
        putsi(graph->absolute.height);

        /* re-scale linestyles */
        gr_relinestyle(graph);

    }
    else {
        /* scale space */
        putc('s', plotfile);
        putsi(0);
        putsi(0);
        putsi(dispdev->width);
        putsi(dispdev->height);

        /* reasonable values, used in gr_ for placement */
        graph->fontwidth = 12;
        graph->fontheight = 24;

        graph->absolute.width = dispdev->width;
        graph->absolute.height = dispdev->height;

    }

    /* set to NULL so graphdb doesn't incorrectly de-allocate it */
    graph->devdep = NULL;
    graph->n_byte_devdep = 0;

    return 0;
}


int Plt5_Close(void)
{
    /* in case Plt5_Close is called as part of an abort,
       w/o having reached Plt5_NewViewport */
    if (plotfile)
        fclose(plotfile);

    return 0;
}


int Plt5_Clear(void)
{
    /* do nothing */
    return 0;
}


int
Plt5_DrawLine(int x1, int y1, int x2, int y2, bool isgrid)
{
    NG_IGNORE(isgrid);

    putc('l', plotfile);
    putsi(x1);
    putsi(y1);
    putsi(x2);
    putsi(y2);

    return 0;
}


int Plt5_Arc(int xc, int yc, int radius, double theta, double delta_theta, bool isgrid)
{
    NG_IGNORE(isgrid);
    int x0, y0, x1, y1;

    if (delta_theta < 0) {
        theta += delta_theta;
        delta_theta = -delta_theta;
    }

    if ((2*M_PI - delta_theta)*radius < 0.5) {

        putc('c', plotfile);
        putsi(xc);
        putsi(yc);
        putsi(radius);

        return 0;
    }

    while (delta_theta*radius > 0.5) {

        double delta_phi = M_PI/2;

        if (delta_phi > delta_theta)
            delta_phi = delta_theta;

        x0 = xc + (int)(radius * cos(theta));
        y0 = yc + (int)(radius * sin(theta));
        x1 = xc + (int)(radius * cos(theta + delta_phi));
        y1 = yc + (int)(radius * sin(theta + delta_phi));

        putc('a', plotfile);
        putsi(xc);
        putsi(yc);
        putsi(x0);
        putsi(y0);
        putsi(x1);
        putsi(y1);

        delta_theta -= delta_phi;
        theta       += delta_phi;
    }

    return 0;
}


int Plt5_Text(const char *text, int x, int y, int angle)
{
    int savedlstyle;
    NG_IGNORE(angle);

    /* set linestyle to solid
       or may get funny color text on some plotters */
    savedlstyle = currentlinestyle;
    Plt5_SetLinestyle(SOLID);

    /* move to (x, y) */
    putc('m', plotfile);
    putsi(x);
    putsi(y);

    /* use the label option */
    fprintf(plotfile, "t%s\n", text);

    /* restore old linestyle */
    Plt5_SetLinestyle(savedlstyle);

    return 0;
}


int Plt5_SetLinestyle(int linestyleid)
{
    if (linestyleid < 0 || linestyleid > dispdev->numlinestyles) {
        internalerror("bad linestyleid");
        return 0;
    }
    putc('f', plotfile);
    fprintf(plotfile, "%s\n", linestyle[linestyleid]);
    currentlinestyle = linestyleid;

    return 0;
}


/* ARGSUSED */

int Plt5_SetColor(int colorid)
{
    NG_IGNORE(colorid);

    /* do nothing */
    return 0;
}


int Plt5_Update(void)
{
    fflush(plotfile);
    return 0;
}
