/*************
 * Header file for graf.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_GRAF_H
#define ngspice_GRAF_H

#include "ngspice/graph.h"

int gr_init(double *xlims, double *ylims,
        const char *xname,
        const char *plotname,
        const char *hcopy,
        int nplots,
        double xdelta, double ydelta,
        GRIDTYPE gridtype,
        PLOTTYPE plottype,
        const char *xlabel,
        const char *ylabel, /* Labels for axes. */
        int xtype, int ytype,
        const char *pname, const char *commandline,
        int prevgraph);
void gr_point(struct dvec *dv,
        double newx, double newy,
        double oldx, double oldy, int np);
void gr_start(struct dvec *dv);
void gr_relinestyle(GRAPH *graph);
void drawlegend(GRAPH *graph, int plotno, struct dvec *dv);
void gr_end(struct dvec *dv);
void gr_pmsg(char *text);
void gr_clean(void);
void gr_resize(GRAPH *graph);
void gr_resize_internal(GRAPH *graph);
void gr_redraw(GRAPH *graph);
void gr_restoretext(GRAPH *graph);
void reset_trace(void);
void gr_iplot(struct plot *plot);
void gr_end_iplot(void);
double *readtics(char *string);

#endif
