/*************
 * Header file for gnuplot.c
 * 2008 Stefano Pedretti
 ************/

#ifndef ngspice_GNUPLOT_H
#define ngspice_GNUPLOT_H

void ft_gnuplot(double *xlims, double *ylims, char *filename, char *title,
               char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype,
               struct dvec *vecs);


void ft_writesimple(double *xlims, double *ylims, char *filename, char *title,
               char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype,
               struct dvec *vecs);

#endif
