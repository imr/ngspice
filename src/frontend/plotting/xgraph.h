/*************
 * Header file for xgraph.c
 * 1999 E. Rouat
 ************/

#ifndef XGRAPH_H_INCLUDED
#define XGRAPH_H_INCLUDED

void ft_xgraph(double *xlims, double *ylims, char *filename, char *title,
               char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype,
               struct dvec *vecs);

#endif
