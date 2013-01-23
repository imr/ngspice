/*************
 * Header file for xgraph.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_XGRAPH_H
#define ngspice_XGRAPH_H

void ft_xgraph(double *xlims, double *ylims, char *filename, char *title,
               char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype,
               struct dvec *vecs);

#endif
