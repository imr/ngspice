/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include <ngspice.h>
#include <graph.h>
#include <ftedev.h>

#include "plot5.h"

static FILE *plotfile;

#define putsi(a)    putc((char) (a), plotfile); \
            putc((char) ((a) >> 8), plotfile)

#define SOLID 0
static char *linestyle[] = { "solid", "dotted", "longdashed", "shortdashed",
        "dotdashed" };
static int currentlinestyle = SOLID;


extern void gr_relinestyle (GRAPH *graph);
extern void internalerror (char *message);

int 
Plt5_Init(void)
{

    dispdev->numlinestyles = 4;
    dispdev->numcolors = 2;

    /* arbitrary */
    dispdev->width = 1000;
    dispdev->height = 1000;

    return(0);

}

int
Plt5_NewViewport(GRAPH *graph)
{

    if (!(plotfile = fopen(graph->devdep, "w"))) {
      graph->devdep = (char *) NULL;
      perror(graph->devdep);
      return(1);
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

    } else {
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
    graph->devdep = (char *) NULL;

    return(0);

}

void
Plt5_Close(void)
{

    /* in case Plt5_Close is called as part of an abort,
            w/o having reached Plt5_NewViewport */
    if (plotfile)
        fclose(plotfile);

}

void
Plt5_Clear(void)
{

    /* do nothing */

}

void
Plt5_DrawLine(int x1, int y1, int x2, int y2)
{

    putc('l', plotfile);
    putsi(x1);
    putsi(y1);
    putsi(x2);
    putsi(y2);

}

/* ARGSUSED */ /* until some code gets written */
void
Plt5_Arc(int x0, int y0, int radius, double theta1, double theta2)
{


}

void
Plt5_Text(char *text, int x, int y)
{

    int savedlstyle;

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

}

int
Plt5_SetLinestyle(int linestyleid)
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
void
Plt5_SetColor(int colorid)
{

    /* do nothing */

}

void
Plt5_Update(void)
{

    fflush(plotfile);

}

