/*************
 * Header file for points.c
 * 1999 E. Rouat
 ************/

#ifndef POINTS_H_INCLUDED
#define POINTS_H_INCLUDED

double * ft_minmax(struct dvec *v, bool real);
int ft_findpoint(double pt, double *lims, int maxp, int minp, bool islog);
double * ft_SMITHminmax(struct dvec *v, bool yval);
int SMITH_tfm(double re, double im, double *x, double *y);



#endif
