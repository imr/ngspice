/*************
 * Header file for agraf.c
 * 1999 E. Rouat
 ************/

#ifndef AGRAF_H_INCLUDED
#define AGRAF_H_INCLUDED

void ft_agraf(double *xlims, double *ylims, struct dvec *xscale, struct plot *plot, 
	      struct dvec *vecs, double xdel, double ydel, bool xlog, bool ylog, 
	      bool nointerp);

#endif
