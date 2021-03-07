/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jeffrey M. Hsu
**********/

/*
    This file contains the graph structure.
*/

#ifndef ngspice_GRAPH_H
#define ngspice_GRAPH_H

#include "ngspice/typedefs.h"
#include "ngspice/grid.h"
#include "ngspice/plot.h"
#include "ngspice/dvec.h"            /* for struct dvec */

struct _keyed;

#define GRAPH_UNITS_LENGTH  20

/* Device-independent data structure for plots. */

struct graph {
    int graphid;
    struct dveclist *plotdata;  /* normalized data */
    char *plotname;         /* name of plot this graph is in */
    int onevalue;           /* boolean variable,
                                 true if plotting one value
                                 against itself (real vs imaginary) */
    int degree;     /* degree of polynomial interpretation */

    int currentcolor;
    int linestyle;

    bool nolegend, nounits;

    struct {
        int height, width;
    } viewport;
    int viewportxoff;   /* x offset of viewport w/in graph */
    int viewportyoff;   /* y offset of viewport w/in graph */

    struct {
        int xpos;   /* x position of graph in screen coord */
        int ypos;   /* y position of graph in screen coord */
        int width;  /* width of window on screen */
        int height; /* height of window on screen */
    } absolute;

    struct {
        double xmin, ymin, xmax, ymax;
    } data;

    struct {
        double xmin, ymin, xmax, ymax;
        /* cache:  width = xmax - xmin  height = ymax - ymin */
        double width, height;
    } datawindow;

    /* cache (datawindow size) / (viewport size) */
    double aspectratiox, aspectratioy;

    int ticmarks;           /* mark every ticmark'th point */
    double *ticdata;
    char ticchar[2];
    int fontwidth, fontheight;  /* for use in grid */

    PLOTTYPE plottype;      /* defined in FTEconstant.h */
    struct {
      GRIDTYPE gridtype;        /* defined in FTEconstant.h */
      int circular;         /* TRUE if circular plot area */
      union {
        struct {
                char units[GRAPH_UNITS_LENGTH];  /* unit labels */
                int     spacing, numspace;
                double  distance, lowlimit, highlimit;
                int     mult;
                int     onedec;     /* a boolean */
                int     hacked;     /* true if hi - lo already hacked up */
                double  tenpowmag;
                double  tenpowmagx;
                int     digits;
        } lin;
        struct {
                char units[GRAPH_UNITS_LENGTH];  /* unit labels */
                int hmt, lmt, decsp, subs, pp;
        } log;
        struct {
                char units[GRAPH_UNITS_LENGTH];  /* unit labels */
                int radius, center;
                double mrad;
                int lmt;
                int hmt, mag; /* added, p.w.h. */
        } circular;     /* bogus, rework when write polar grids, etc */
      } xaxis, yaxis;
      int xdatatype, ydatatype;
      int xsized, ysized;
      double xdelta, ydelta; /* if non-zero, user-specified deltas */
      char *xlabel, *ylabel;
    } grid;

    /* characters the user typed on graph */
    /* note: think up better names */
    struct _keyed *keyed;

    /* for zoomin */
    char *commandline;

    /* colors used */
    unsigned long colorarray[25];

    /* we have a mother graph */
    int mgraphid;

    /* linewidths */
    int graphwidth;
    int gridwidth;

    /* Space here is allocated by NewViewport
        and de-allocated by DestroyGraph.
    */
    void *devdep;
    size_t n_byte_devdep; /* Size of devdep. Needed to allow copying */
};


/* characters the user typed on graph */
/* note: think up better names */

struct _keyed {
    char *text;
    int x, y;
    int colorindex;       /* index into colors array */
    struct _keyed *next;
};


#define rnd(x)  (int) ((x)+0.5)

#endif
