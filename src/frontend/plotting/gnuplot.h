/*************
 * Header file for gnuplot.c
 * 2008 Stefano Pedretti
 ************/

#ifndef GNUPLOT_H_INCLUDED
#define GNUPLOT_H_INCLUDED

void ft_gnuplot(double *xlims, double *ylims, char *filename, char *title,
               char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype,
               struct dvec *vecs);


void ft_writesimple(double *xlims, double *ylims, char *filename, char *title,
               char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype,
               struct dvec *vecs);

#endif
