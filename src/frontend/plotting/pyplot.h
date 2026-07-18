/*************
 * Header file for pyplot.c (Enhancement-94)
 ************/

#ifndef ngspice_PYPLOT_H
#define ngspice_PYPLOT_H

void ft_pyplot(double *xlims, double *ylims,
        double xdel, double ydel,
        const char *filename, const char *title,
        const char *xlabel, const char *ylabel,
        GRIDTYPE gridtype, PLOTTYPE plottype,
        struct dvec *vecs);

#endif
