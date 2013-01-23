/*************
 * Header file for points.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_POINTS_H
#define ngspice_POINTS_H

double * ft_SMITHminmax(struct dvec *v, bool yval);
int SMITH_tfm(double re, double im, double *x, double *y);



#endif
